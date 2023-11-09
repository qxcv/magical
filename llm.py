import argparse
import os
import pathlib
import openai
import json
import base64
import re
import random

parser = argparse.ArgumentParser()
parser.add_argument('--num_files', type=int, default=None)
parser.add_argument('--data_folder', type=str, default=None)
parser.add_argument('--output_folder', type=str, default=None)
parser.add_argument('--prompt_template', type=str, default=None)
args = parser.parse_args()


def prompt_template_1(before_img, after_img):
    prompt = """
The following pictures are from an RL environment. It's a multitask benchmark where an agent must move blocks of different colours and shapes in a 2D square to achieve different goals.
The goals are to be inferred from demonstrations, and are easy to grasp for humans.

The first picture is the starting state of the environment. The second picture is the goal state of the environment, where the objective was achieved.
There are three possible kind of objects in this environment. The grey circular object is an agent with two arms that can move around and move blocks (if there are any) to achieve the objective. The agent is always present.
There can be movable blocks, which have different shapes and colours, they all have a solid back outline. The possible shapes are circle, square and star, no other shapes exist, the possible colour are pink, green and blue. Blocks are not always present.
There can also be special areas in the environment, these are light coloured with dashed outlines. Special areas can only be rectangular. The agent can move inside the special areas and place blocks (if there are any) inside them.  Special areas are not always present.
Your job is to be an expert on this environment and correctly identify the objective based on the start and end states. The goals should be as general and high-level as possible.
Think out loud step-by-step by

1. identify the agent on the first picture. It is always present, has a number and is grey circular with two arms.
2. identify the special areas on the first picture if there are any, stating their colour.
3. identify the movable blocks if there are any. Name their shape and colour.
4. describing the position of the objects in the first picture by referring to the grid. For movable blocks, state explicitly whether they are in a special area or not. For special areas, state whether they contain blocks or not.
5. naming the objects in the second picture. These should be the same as in the first picture with the same numeric labels, objects do not disappear and new ones do not appear. Do not refer back to the first picture when doing this.
6. describing their position in the second picture. Do not refer back to the first picture when doing this. Don't compare and say what changed from the first picture, just describe the position of the objects. Don't use words like “changed”, “remains”, “still” or any words that would refer back to the first picture.
7. comparing the image descriptions. Do not compare the pictures directly.

Next, a list of possible goals the agent may have had. Format these as follows:
\possible_goal{goal description 1}
\possible_goal{goal description 2}
etc.
Suggest up to 5 possible goals.

Finally, output the most likely goal as a short, descriptive sentence in this format:
\goal{goal description}
"""
    return images_last_prompt_template(prompt, before_img, after_img)


def images_last_prompt_template(prompt, before_img, after_img):
    resolution = "low"
    template =[
        {
            "type": "text",
            "text": prompt,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{before_img}",
                "detail": resolution,
            },
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{after_img}",
                "detail": resolution,
            },
        },
    ]
    return template



def call_llm(messages):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": messages,
            }
        ],
        max_tokens=1000,
    )
    # TODO: might want to return the whole response, so we can cehck for finish_reason, function calls, etc.
    return response.choices[0].message.content


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


def parse_goals(text):
    # Define regex patterns for possible goals and goals
    possible_goal_pattern = r"\\possible_goal\{(.*?)\}"
    goal_pattern = r"\\goal\{(.*?)\}"
    
    # Find all matches for possible goals and goals
    possible_goals = re.findall(possible_goal_pattern, text)
    goals = re.findall(goal_pattern, text)
    
    return possible_goals, goals


def construct_summarize_prompt(results_list):
    random.seed(0)
    # shuffle the list
    random.shuffle(results_list)
    intro = """
Here are several predictions about the goal an agent was trying to complete. Each prediction consists of several guesses at possible made from a single video of an agent accomplishing the task. Read each, then make a guess overall at the agent's most likely overall goal. 
Remember these hints:
- The goal is exactly same across all trials of the task. 
- The guesses at "possible goals" and "most likely goal" may be incorrect. Summarize all the guesses to make your own guess at the most likely goal.
- The goal is always short, common-sense, and easy for a human to understand.
- The guesses were made from only seeing a single trial, so they may be overly specific and refer to the specific trial, rather than the overall goal.
"""

    ending = """
List possible overall goals the agent may have had. Format these as follows:
\possible_goal{goal description 1}
\possible_goal{goal description 2}
etc.
Suggest up to 5 possible goals.

Finally, output the most likely overall goal as a short, descriptive sentence in this format:
\goal{goal description}
""" 


    lines = [intro]
    for i in range(len(results_list)):
        lines.append(f'======== Trial {i+1} ========')
        for j, possible_goal in enumerate(results_list[i]['possible_goals']):
            lines.append(f"Possible goal {j+1}: {possible_goal}")
        lines.append(f"Most likely goal: {results_list[i]['goals'][0]}")
        lines.append("")
    lines.append(ending)
    prompt = "\n".join(lines)
    template =[
        {
            "type": "text",
            "text": prompt,
        },
    ]
    return template

def main():
    cache = load_cache()
    # Get the list of files in the data folder
    data_folder = pathlib.Path(args.data_folder)
    # check that it's a folder that exists
    assert data_folder.is_dir(), f"{data_folder} is not a folder that exists"
    end_frame_files = [file for file in data_folder.iterdir() if file.name.endswith(".png") and not 'start_frame' in file.name]
    if args.num_files is not None:
        end_frame_files = end_frame_files[:args.num_files]

    # Create the output folder, if it doesn't exist
    output_folder = pathlib.Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    before_img = encode_image(data_folder / "start_frame.png")
    
    results_list = []

    # Loop through the files
    for file in end_frame_files:
        # Get the  after image
        after_img = encode_image(file)

        # Get the prompt
        # TODO: make this an option
        messages = prompt_template_1(before_img=before_img, after_img=after_img)
        messages_serialized = json.dumps(messages)
        if messages_serialized in cache:
            llm_completion = cache[messages_serialized]
            print(f"Using cached result for {file.name}")
        else:
            llm_completion = call_llm(messages)
            cache[messages_serialized] = llm_completion
            save_cache(cache)
        possible_goals, goals = parse_goals(llm_completion)

        out_dict = {
            "messages": messages,
            "llm_completion": llm_completion,
            "possible_goals": possible_goals,
            "goals": goals,
        }
        results_list.append(out_dict)
        output_file = output_folder / file.name.replace(".png", ".json") 
        with open(output_file, 'w') as f:
            json.dump(out_dict, f, indent=4)
        print(f"Wrote {output_file}")
    
    summarize_messages = construct_summarize_prompt(results_list)
    summarize_result = call_llm(summarize_messages)
    possible_goals, goals = parse_goals(summarize_result)
    out_dict = {
        "messages": summarize_messages,
        "llm_completion": summarize_result,
        "possible_goals": possible_goals,
        "goals": goals,
    }
    output_file = output_folder / "summarize.json"
    with open(output_file, 'w') as f:
        json.dump(out_dict, f, indent=4)
        
    print("Done!")



if __name__ == "__main__":
    main()