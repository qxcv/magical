import argparse
import os
import pathlib
import openai
import json
import base64
import re
import random
from tenacity import retry, retry_if_exception_type, stop_after_attempt

class PromptTemplate:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, before_img, after_img):
        raise NotImplementedError
    
    def parse_output(self, output):
        raise NotImplementedError


class ImagesAtEndTemplate(PromptTemplate):
    
    def __init__(self, resolution):
        super().__init__(resolution)
        self.prompt = """
The following pictures are from an RL environment. It's a multitask benchmark where a robot must move blocks of different colours and shapes in a 2D square to achieve different goals. The goals are to be inferred from demonstrations, and are easy to grasp for humans.
The first picture is the starting state of the environment. The second picture is the goal state of the environment, where the objective was achieved. Each block is labeled by a number starting from B1, to B2, to B3, etc. to help you keep track of them. There are three possible kind of objects in this environment. The grey circular object labelled R is a robot with two arms that can move around and move blocks (if there are any) to achieve the objective. The robot is always present. There can be movable blocks, which have different shapes and colours, they all have a solid outline. The possible shapes are circle, square and star, no other shapes exist. The possible colours are blue, pink, and green. Blocks do not disappear. There can also be special areas in the environment, these are light coloured with dashed outlines. Special areas can only be rectangular. They are labelled SA1, SA2, etc. to help you distinguish them from movable blocks. The robot can move inside the special areas and place blocks (if there are any) inside them. There may or may not be any special areas. There is a 3x3 grid with rows A,B,C and columns 1,2,3 to help you refer to the position of the objects, but it is only a visual help for you. The objective does not include any reference to the grid. Your job is to be an expert on this environment and correctly identify the objective based on the start and end states. The goals should be as general and high-level as possible. Think out loud step-by-step by
1, identifying how many objects there are on the first picture. Each object is labeled by a number starting from 1 to help you keep track of them.
2, identify the robot on the first picture. It is always present, has a number and is grey circular with two arms.
3, identify the special areas on the first picture if there are any. These are rectangles and have labels with SA. Name their colour too.
4, identify the movable blocks if there are any. These are the numbered shapes labeled B. Name their shape and colour.
5, describing the position of the objects in the first picture by referring to the grid. For movable blocks, state explicitly whether they are in a special area or not. For special areas, state whether they contain blocks or not.
6, naming the objects in the second picture. These should be the same as in the first picture with the same numeric labels, objects do not disappear and new ones do not appear. Do not refer back to the first picture when doing this.
7, describing their position in the second picture. Do not refer back to the first picture when doing this. Don't compare and say what changed from the first picture, just describe the position of the objects. Don’t use words like “changed”, “remains”, “still” or any words that would refer back to the first picture.
8, comparing the image descriptions. Do not compare the pictures directly.
Output the goal as a short, descriptive sentence on a new line after the words "GOAL:" Do not mention the 3x3 grid in your final answer.
"""
        
    
    def __call__(self, before_img, after_img):
        return [
            {
                "type": "text",
                "text": self.prompt,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{before_img}",
                    "detail": self.resolution,
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{after_img}",
                    "detail": self.resolution,
                },
            },
        ]

    def parse_output(self, text):
        # Find any lines that start with "GOAL:"
        goal_lines = [line for line in text.split("\n") if line.startswith("GOAL:")]
        # Remove the "GOAL:" prefix
        goals = [line.replace("GOAL:", "") for line in goal_lines]
        return {'goals': goals}


PROMPT_TEMPLATES = {
    "images_at_end": ImagesAtEndTemplate,
}


@retry(retry=retry_if_exception_type(openai.RateLimitError), stop=stop_after_attempt(5))
def call_vlm(message_content, temperature, max_tokens, cache):
    client = openai.OpenAI(api_key="")
    vlm_args = {
        "model": "gpt-4-vision-preview",
        # "model":"gpt-3.5-turbo",
        "messages":[
            {
                "role": "user",
                "content": message_content,
            }
        ],
        "max_tokens":max_tokens,
        "temperature":temperature,
    }
    serialized_args = json.dumps(vlm_args)
    # check if we have a cached result
    if serialized_args in cache:
        print("Using cached result")
        return cache[serialized_args]
    
    response = client.chat.completions.create(
        **vlm_args
    )
    # TODO: might want to return the whole response, so we can cehck for finish_reason, etc.
    response = response.choices[0].message.content
    cache[serialized_args] = response
    save_cache(cache)
    return response


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_cache():
    try:
        with open("cache.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache):
    with open("cache.json", "w") as f:
        json.dump(cache, f, indent=4)


def query_vlm(start_img_file, end_img_file, args, out_file_name, cache):
    
    # Load the images in base64
    start_img = encode_image(start_img_file)
    end_img = encode_image(end_img_file)

    # Get the prompt
    # TODO: make this an option
    prompt_template = PROMPT_TEMPLATES[args.prompt_template](args.resolution)
    messages = prompt_template(before_img=start_img, after_img=end_img)
    llm_completion = call_vlm(messages, temperature=args.temperature, max_tokens=args.max_tokens, cache=cache)
    
    parsed_goals = prompt_template.parse_output(llm_completion)

    out_dict = {
        "messages": messages,
        "llm_completion": llm_completion,
        "parsed_goals": parsed_goals,
    }
    output_folder = pathlib.Path(args.output_folder)
    output_file = output_folder / (out_file_name.name + ".json")
    with open(output_file, 'w') as f:
        json.dump(out_dict, f, indent=4)
    print(f"Wrote {output_file}")
    return out_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_files', type=int, default=None, help=f'If `all_traj_folder` is provided, this is the number of files within the folder to process. (Defaults to using them all.)')
    parser.add_argument('--traj_folder', type=str, default=None, help=f'Folder containing images from a trajectory.')
    parser.add_argument('--all_traj_folder', type=str, default=None, help=f'Folder containing many subfolders, each of which contains a directory. The subfolders must be in the same format as specified for `traj_folder`. `traj_folder` and `all_traj_folder` can\'t be provided at the same time')
    parser.add_argument('--output_folder', type=str, required=True, help=f'Folder where analyses will be saved.')
    parser.add_argument('--prompt_template', type=str, default='images_at_end', help=f'Template of interleaved text and images. Choices are {PROMPT_TEMPLATES.keys()}')
    parser.add_argument('--resolution', type=str, default='high', choices=['high', 'low'])
    parser.add_argument('--start_frame', type=str, default='frame-allo-score-1-first.png')
    parser.add_argument('--end_frame', type=str, default='frame-allo-score-1-last.png')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--max_tokens', type=int, default=1000)
    args = parser.parse_args()

    cache = load_cache()
    
    # Confirm we either have traj_folder or all_traj_folder, but not both
    assert (args.traj_folder is not None) != (args.all_traj_folder is not None), "Must provide either `traj_folder` or `all_traj_folder`"
    
    output_folder = pathlib.Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    if args.traj_folder:
        # Get the list of files in the data folder
        data_folder = pathlib.Path(args.traj_folder)
        # check that it's a folder that exists
        assert data_folder.is_dir(), f"{data_folder} is not a folder that exists"
        start_file = data_folder / args.start_frame
        end_file = data_folder / args.end_frame
        query_vlm(start_file, end_file, args, data_folder, cache)
    else:
        all_traj_folder = pathlib.Path(args.all_traj_folder)
        # check that it's a folder that exists
        assert all_traj_folder.is_dir(), f"{all_traj_folder} is not a folder that exists"
        # Get the list of folders in the data folder
        traj_folders = [folder for folder in all_traj_folder.iterdir() if folder.is_dir()]
        if args.num_files is not None:
            traj_folders = traj_folders[:args.num_files]
        results_list = []
        for folder in traj_folders:
            print(f"Processing {folder}")
            start_file = folder / args.start_frame
            end_file = folder / args.end_frame
            results_list.append(query_vlm(start_file, end_file, args, folder, cache))
    print("Done!")


if __name__ == "__main__":
    main()

