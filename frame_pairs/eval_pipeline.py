import argparse
import os
import pathlib
from typing import Dict, List, Literal, Optional, Union
import openai
import json
import base64
import re
import random
import datetime
from tenacity import retry, retry_if_exception_type, stop_after_attempt
import glob
from guidance import models, gen, select, system, user, assistant
import matplotlib.pyplot as plt



@retry(retry=retry_if_exception_type(openai.RateLimitError), stop=stop_after_attempt(5))
def call_vlm(message_content, temperature, max_tokens, cache):
    """Call the VLM API with the given message content and return the response"""
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    vlm_args = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": message_content,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    serialized_args = json.dumps(vlm_args)
    # check if we have a cached result
    if serialized_args in cache:
        print("Using cached result")
        return cache[serialized_args]
    print("Not using cached result")
    response = client.chat.completions.create(**vlm_args)
    response = response.choices[0].message.content
    cache[serialized_args] = response
    save_cache(cache)
    return response


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_cache():
    try:
        with open("cache.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("No cache found")
        return {}


def save_cache(cache):
    with open("cache.json", "w") as f:
        json.dump(cache, f, indent=4)


def generate_single_traj(
    imgs: List[str],
    output_file: str,
    cache,
    prompt_template,
    temperature=0.5,
    resolution="high",
    max_tokens=1000,
    **prompt_kwargs,
):
    """
    Constructs a prompt and calls the VLM API. Returns a dictionary of the results.

    Args:
        imgs: A list of image paths
        output_file: The path to write the output to
        cache: A dictionary of cached results
        prompt_template: The prompt template to use
        temperature: The temperature to use for VLM
        resolution: The resolution of the images
        max_tokens: The maximum number of tokens to use for VLM
        prompt_kwargs: Additional keyword arguments to pass to the prompt template

    """

    # Load the images in base64
    encoded_imgs = []
    for img in imgs:
        print(f"Encoding {img}")
        encoded_imgs.append(encode_image(img))

    prompt_template_obj = PROMPT_TEMPLATES[prompt_template](resolution)
    messages = prompt_template_obj(encoded_imgs, **prompt_kwargs)
    llm_completion = call_vlm(
        messages, temperature=temperature, max_tokens=max_tokens, cache=cache
    )

    parsed_goals = prompt_template_obj.parse_output(llm_completion)

    out_dict = {
        "args": {
            "prompt_template": prompt_template,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "resolution": resolution,
            "prompt_kwargs": prompt_kwargs,
            "imgs": imgs,
        },
        "messages": messages,
        "llm_completion": llm_completion,
        "parsed_goals": parsed_goals,
    }

    with open(output_file, "w") as f:
        json.dump(out_dict, f, indent=4)
    print(f"Wrote {output_file}")
    return out_dict

def get_imgs_from_folder(img_pair_folder):
    imgs = glob.glob(f"{img_pair_folder}/*.png")
        # make sure they are sorted according to frame number
        # img names are of he form frame-PushToRedRegionTestAllv020231114T10_41_05.pkl.gz-0.png
        # so split on the - to get the frame number and cut off the .png
    frames = {int(img.split("-")[-1].split(".")[0]): img for img in imgs}
        # sort the frames by frame number
    imgs = [frames[key] for key in sorted(frames.keys())]
        #img_pair_folder is a path, so get last two folders
    return imgs

def get_ground_truths(images_folders):
    """ Returns a dictionary of the {img pair path: [ground_truths]}."""
    ground_truths = {}
    for folder in images_folders:
        assert os.path.isdir(
            folder
        ), f"{folder} is not a folder that exists"
        truth_files = glob.glob(f"{folder}/*truth.json")
        for truth_file in truth_files:
            with open(truth_file, "r") as f:
                truths = json.load(f)
            for pair_idx in truths.keys():
            # get the pair name
                pair_folder = os.path.join(
                    folder, f"pair{pair_idx}"
                )
                assert os.path.isdir(
                    pair_folder
                ), f"{pair_folder} is not a folder that exists"
                if pair_folder not in ground_truths:
                    ground_truths[pair_folder] = []
                ground_truths[pair_folder].append(truths[pair_idx])
    return ground_truths

def get_goals(images_folders, GOAL_FN, GOAL_FN_KWARGS):
    """ Returns a dictionary of the {img pair path: [goals]} by calling the goal function on the image pairs."""
    goals = {}
    # get all the image pair folders
    all_img_pair_folders = []
    for images_folder in images_folders:
        assert os.path.isdir(
            images_folder
        ), f"{images_folder} is not a folder that exists"
        # folders inside images_folder of the form pair1, pair2, pair3, etc.
        # get a list of the full paths to the pair folders
        img_pair_folders = [
            os.path.join(images_folder, folder)
            for folder in os.listdir(images_folder)
            if folder.startswith("pair")
        ]
        all_img_pair_folders.extend(img_pair_folders)
    for img_pair_folders in all_img_pair_folders:
        goal = GOAL_FN(img_pair_folders, **GOAL_FN_KWARGS)
        goals[img_pair_folders] = goal
    return goals

def final_answers(goals, ground_truths, out_file):
    """ 
    Takes in the goals and ground truths and writes them to a json file.
    Returns a dictionary of the {img pair path: {'ground_truths': [truth], 'prediction': pred}}."""
    final_answers = {}
    for img_pair_folder in goals.keys():
        final_answers[img_pair_folder] = {}
        final_answers[img_pair_folder]['ground_truths'] = ground_truths[img_pair_folder]
        final_answers[img_pair_folder]['prediction'] = goals[img_pair_folder]
    with open(out_file, "w") as f:
        json.dump(final_answers, f, indent=4)
    print(f"Wrote {out_file}")
    return final_answers

def evaluate(final_answers, out_file):
    SimilarityLevel = Union[Literal["Incorrect"], Literal["Correct"], Literal["Unknown"]]
    similarity_levels = {
    "Incorrect": (
        "An agent that tries to achieve the predicted "
        "goal will fail in achieving the real goal "
        "because there is a meaningful difference "
        "between the two goals."
    ),
    "Correct": (
        "An agent that tries to achieve the predicted "
        "goal will succeed in achieving the real goal, "
        "even if there are slight differences in phrasing "
        "that does not affect the meaning of the goal."
    ),
    "Unknown": (
        "There is not enough information to determine "
        "whether the predicted goal is correct or incorrect "
        " without seeing the frames."
    ),
    }
    def get_goal_similarity_chat(chat_llm: models.Model, ground_truths: List[str], predicted_goal: str, out_file) -> SimilarityLevel: # type: ignore
        
        cache = load_cache()
        # check if in cache
        serialized_args = json.dumps({'truths': truths, 'pred': pred})
        if serialized_args in cache:
            print("Using cached result")
            res = cache[serialized_args]
        else:
        
            sim_levels = "\n".join(
                f"({num}) {level}: {desc}:" for num, (level, desc) in enumerate(similarity_levels.items(), start=1)
            )
            str_ground_truths ="\n".join(ground_truths)
            with system():
                res = chat_llm + f"""You are a helpful assistant. Your job is to measure the similarity a predicted goal and a set of ground truth observations for a reinforcement learning benchmark environment, where the aim is to manipulate blocks by having a robot push them around in various ways. The set of ground truths are all correct, but were described by different people. You can assume that they all lead to the same outcome, so if the predicted goal describes the same as at least one of them, it's correct. The best predicted goal is one that tells the agent enough information to achieve the true goal, and no more. Note that the shapes, colors, and locations of blocks are significant. e.g. "Move the blue block" is different from "Move the red block". The blocks are labeled starting with B. There are also goal regions with labels containing SA of different colors, and their colors are significant too. Sometimes the goals are formulated in the first person, e.g. "move upwards" or in the third person e.g. "move R upwards". Here R refers to the label of robot you control, therefore these two would be equivalent. It does not matter if the prediction does not mention color or shape as long as the label is correct. Remember, a predicted goal is correct if it would lead to the same outcome as the set of ground truths, even if the wording is different; a goal is different if it would lead to a different outcome, even if the wording is similar. If you cannot determine whether the predicted goal is correct or incorrect without seeing the pictures of the environment, say "unknown".

        The {len(similarity_levels)} possible levels of simlarity are:

        {sim_levels}

        Here is the ground truth goals:

        --- BEGIN GROUND TRUTH GOAL ---
        {str_ground_truths}
        --- END GROUND TRUTH GOAL ---

        Here is the predicted goal:

        --- BEGIN PREDICTED GOAL ---
        {predicted_goal}
        --- END PREDICTED GOAL ---

        How similar are these goals? Take a deep breath and reason step by step. Reason without answering first. Be concise."""

            # do think step-by-step step
            with assistant():
                res = res + gen(name = "cot")

            with user():
                res = res + f"""Now, what is the final answer? Select from one of the three answers (spelled in exactly this way: {', '.join(similarity_levels)}). Say only one of the options and nothing else."""

            # final answer
            with assistant():
                res = res + select(similarity_levels.keys(), name="sim_level")
            
            # save to cache
            cache[serialized_args] = {"cot": res.get("cot"), "sim_level": res.get("sim_level")}
            save_cache(cache)

        # save the output
        with open(out_file, "w") as f:
            json.dump({'truths': truths,'pred':predicted_goal,"cot": res.get("cot"), "sim_level": res.get("sim_level")}, f, indent=4)
        return res.get("sim_level")

    for img_pair_folder in final_answers.keys():
        file_name = f"{img_pair_folder.split('/')[-3]}_{img_pair_folder.split('/')[-2]}__{img_pair_folder.split('/')[-1]}_eval.json"
        eval_out_file = os.path.join(os.path.dirname(out_file), file_name)
        truths = final_answers[img_pair_folder]['ground_truths']
        pred = final_answers[img_pair_folder]['prediction']

        sim=get_goal_similarity_chat(
        models.OpenAIChat("gpt-4-1106-preview", temperature=0),
        truths,
        pred, 
        eval_out_file)
        final_answers[img_pair_folder]['similarity'] = sim
    with open(out_file, "w") as f:
        json.dump(final_answers, f, indent=4)
    print(f"Wrote {out_file}")
    return final_answers

def count_evals(evald_answers):
    counts = {}
    for img_pair_folder in evald_answers.keys():
        task = img_pair_folder.split('/')[-3]
        frame_pair_dist = img_pair_folder.split('/')[-2].split('-')[0]
        if task not in counts:
            counts[task] = {}
        if frame_pair_dist not in counts[task]:
            counts[task][frame_pair_dist] = {"Correct": 0, "Incorrect": 0, "Unknown": 0}
        sim = evald_answers[img_pair_folder]['similarity']
        counts[task][frame_pair_dist][sim] += 1
    return counts

def plot_evals(counts, out_file):
    out_folder = os.path.dirname(out_file)
    for task in counts.keys():
        labels = counts[task].keys()
        correct = [counts[task][label]['Correct'] for label in labels]
        unknown = [counts[task][label]['Unknown'] for label in labels]

        # Calculating the total heights
        total_heights = [base + add for base, add in zip(correct, unknown)]

        # Setting up the plot
        fig, ax = plt.subplots()

        # Creating the base bars
        base_bars = ax.bar(labels, correct, color='blue')

        # Creating the additional bars on top of the base bars
        additional_bars = ax.bar(labels, total_heights, alpha=0.3, color='red')
        plt.legend((base_bars, additional_bars), ('Correct', 'Unknown'))
        # Setting the y-axis limit
        ax.set_ylim(0, 10)
        ax.set_xlabel("Frame difference")
        ax.set_ylabel("# of pairs")
        # Adding a title
        ax.set_title(task)
        out_file = os.path.join(out_folder, f'{task}.png')
        plt.savefig(out_file, bbox_inches='tight', dpi=300)
        print(f"Saved {out_file}")



 ################################################################
############### PROMPT TEMPLATES ################################################################
 ################################################################
# Constructs the prompt to send to the VLM, and parse the goal from the answer

class PromptTemplate:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, imgs):
        raise NotImplementedError
    
    def parse_output(self, output):
        raise NotImplementedError
    
# this is for determining the subgoal from two frames
TWO_FRAME_SHORT_GOAL_SPEC_LABEL_PROMPT = """
The following pictures are from an RL environment. It's a multitask benchmark where a robot must move blocks of different colours and shapes in a 2D square to achieve different goals. The goals are to be inferred from demonstrations, and are easy to grasp for humans. You are given two frames from a part of the trajectory of the agent in the process of completing the goal. There are three possible kind of objects in this environment. The grey circular object labelled R is a robot with two arms that can move around and move blocks (if there are any) to achieve the objective. The robot is always present. There can be movable blocks, which have different shapes and colours, they all have a solid outline. The possible shapes are circle, triangle and square, no other shapes exist. The possible colours are blue, pink, and green. Blocks do not disappear. Each block is labeled by a number starting from B1, to B2, to B3, etc. to help you keep track of them. There can also be special areas in the environment, these are light coloured with dashed outlines. Special areas can only be rectangular. They are labelled SA1, SA2, etc. to help you distinguish them from movable blocks. The robot can move inside the special areas and place blocks (if there are any) inside them. There may or may not be any special areas. There is a 3x3 grid with rows A,B,C and columns 1,2,3 to help you refer to the position of the objects. Column A is the left of the arena, and column C is the right of the arena. Row 1 is the top of the arena, Row 3 is the bottom of the arena. The grid is only a visual help for you, the objective does not include any reference to the grid. Your job is to be an expert on this environment and by looking at the given two trajectory frames, identify what subgoal the agent was following between the two frames. Think out loud step-by-step by
1, identify the robot on the first picture. It is always present, has a number and is grey circular with two arms. It is labeled R.
2, identify the special areas on the first picture if there are any. These are rectangles and have labels with SA and a number starting from 1. Name their colour too.
3, identify the movable blocks if there are any. These are the numbered shapes labeled B with a number starting from 1. Name their shape and colour.
5, describing the position of the objects in the first picture by referring to the grid. For movable blocks, state explicitly whether they are in a special area or not. For special areas, state whether they contain blocks or not.
6, naming the objects in the second picture. These should be the same as in the first picture with the same numeric labels, objects do not disappear and new ones do not appear. Do not refer back to the first picture when doing this.
7, describing their position in the second picture. Do not refer back to the first picture when doing this. Don't compare and say what changed from the first picture, just describe the position of the objects. Don’t use words like “changed”, “remains”, “still” or any words that would refer back to the first picture.
8, comparing the image descriptions. Do not compare the pictures directly.
Output the agent's subgoal as a short, descriptive sentence on a new line after the words "SUBGOAL:" based on the changes between the frames. Do not mention the 3x3 grid in your final answer, for example instead of "move to C3" describe parts of the area as "move to bottom right", "move to top left", "move to block B3" etc.  Don't be vague, for example instead of  "move to a new location", be specific and say "move to the bottom of the arena" if that's the case. If you refer to blocks or special areas in your subgoal, refer to them by label only, for example "move B3 to SA2" instead of "move the blue square block to the special area". If the robot is carrying a block between two frames, don't forget to mention the block moving too, not just the robot.
"""
class TwoFrameShortGoalSpecLabelTemplate(PromptTemplate):
    
    def __init__(self, resolution):
        super().__init__(resolution)
        self.prompt = TWO_FRAME_SHORT_GOAL_SPEC_LABEL_PROMPT
        
    
    def __call__(self,imgs):
        assert len(imgs) == 2, "Must provide two images"
        before_img, after_img = imgs
        return [
            {
                "type": "text",
                "text": self.prompt,
            },
            {
                "type": "text",
                "text": f"Frame 1:",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{before_img}",
                    "detail": self.resolution,
                },
            },
            {
                "type": "text",
                "text": f"Frame 2:",
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
        goal_lines = [line for line in text.split("\n") if line.startswith("SUBGOAL:")]
        # Remove the "GOAL:" prefix
        goals = [line.replace("SUBGOAL:", "") for line in goal_lines]
        return {'subgoals': goals}
    
# this is for picking the correct subgoal from a list of subgoals for two frames
SUMMARIZE_GOALS_PROMPT = """
The following pictures are from an RL environment. It's a multitask benchmark where a robot must move blocks of different colours and shapes in a 2D square to achieve different goals. The goals are to be inferred from demonstrations, and are easy to grasp for humans. You are given two frames from a part of the trajectory of the agent in the process of completing the goal. There are three possible kind of objects in this environment. The grey circular object labelled R is a robot with two arms that can move around and move blocks (if there are any) to achieve the objective. The robot is always present. There can be movable blocks, which have different shapes and colours, they all have a solid outline. The possible shapes are circle, triangle and square, no other shapes exist. The possible colours are blue, pink, and green. Blocks do not disappear. Each block is labeled by a number starting from B1, to B2, to B3, etc. to help you keep track of them. There can also be special areas in the environment, these are light coloured with dashed outlines. Special areas can only be rectangular. They are labelled SA1, SA2, etc. to help you distinguish them from movable blocks. The robot can move inside the special areas and place blocks (if there are any) inside them. There may or may not be any special areas. There is a 3x3 grid with rows A,B,C and columns 1,2,3 to help you refer to the position of the objects. Column A is the left of the arena, and column C is the right of the arena. Row 1 is the top of the arena, Row 3 is the bottom of the arena. The grid is only a visual help for you, the objective does not include any reference to the grid. Your job is to be an expert on this environment and by looking at the given two trajectory frames, identify what subgoal the agent was following between the two frames. The subgoals were produced by labelers using the following prompt:
'Output the agent's subgoal as a short, descriptive sentence on a new line after the words "SUBGOAL:" based on the changes between the frames. Do not mention the 3x3 grid in your final answer, for example instead of "move to C3" describe parts of the area as "move to bottom right", "move to top left", "move to block B3" etc.  Don't be vague, for example instead of  "move to a new location", be specific and say "move to the bottom of the arena" if that's the case. If you refer to blocks or special areas in your subgoal, refer to them by label only, for example "move B3 to SA2" instead of "move the blue square block to the special area". If the robot is carrying a block between two frames, don't forget to mention the block moving too, not just the robot.'

We provide you with a list of possible subgoals, and your job is to identify which subgoal the agent was following. It's possible that multiple are correct (for example if they describe the same thing differently), in that case pick any correct one. If none of the subgoals are correct, say "none of the above" and propose a new subgoal. Think out loud step-by-step and write your final answer on a new line after the words "FINAL:"

The possible subgoals are:
"""
class SummarizeGoalsTemplate(PromptTemplate):
    
    def __init__(self, resolution):
        super().__init__(resolution)
        self.prompt = SUMMARIZE_GOALS_PROMPT
        
    
    def __call__(self, imgs, subgoals):
        assert len(imgs) == 2, "Must provide two images"
        before_img, after_img = imgs
        return [
            {
                "type": "text",
                "text": self.prompt,
            },
            {
                "type": "text",
                "text": subgoals,
            },

            {
                "type": "text",
                "text": f"Frame 1:",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{before_img}",
                    "detail": self.resolution,
                },
            },
            {
                "type": "text",
                "text": f"Frame 2:",
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
        goal_line = text.split("FINAL:")[-1].replace("\n", "")
        # Remove the "GOAL:" prefix
        return {'final': goal_line}


# add new ones here 
PROMPT_TEMPLATES = {
    "two_frame_short_goal_spec_label": TwoFrameShortGoalSpecLabelTemplate,    
    "summarize_goals": SummarizeGoalsTemplate,
}
  
############### GET GOAL FUNCTIONS #####################
# these should take in a folder of images, other kwargs and return a goal

class GetGoal:
    def __init__(self,):
        pass

    def __call__(self, img_pair_folder, **kwargs):
        raise NotImplementedError

class GetGoalFromSinglePrompt(GetGoal):
    def __init__(self,):
        pass
    def __call__(self, img_pair_folder, output_folder, prompt_template, temperature=0.5, resolution="high", max_tokens=1000, **prompt_kwargs):
        imgs = get_imgs_from_folder(img_pair_folder)
        cache = load_cache()
        output_folder = pathlib.Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        time_str = now.strftime("%FT%H:%M:%S")
        file_name = f"{img_pair_folder.split('/')[-3]}_{img_pair_folder.split('/')[-2]}__{img_pair_folder.split('/')[-1]}__{time_str}.json"
        output_path = os.path.join(
                output_folder, file_name
            )
        out_dict = generate_single_traj(
                imgs,
                output_path,
                cache,
                prompt_template,
                temperature=temperature,
                resolution=resolution,
                max_tokens=max_tokens,
                **prompt_kwargs,
            )
        goal = out_dict['parsed_goals']
        return goal

class GetGoalPickedFromDiffTemps(GetGoal):
    def __init__(self,):
        pass

    def __call__(self,img_pair_folder, output_folder, prompt_template, temperatures=[0,0.2,0.5,0.7], resolution="high", max_tokens=1000, **prompt_kwargs):
        imgs = get_imgs_from_folder(img_pair_folder)
        cache = load_cache()
        output_folder = pathlib.Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        time_str = now.strftime("%FT%H:%M:%S")
        temp_goals = {}
        for temperature in temperatures:
            file_name = f"{img_pair_folder.split('/')[-3]}_{img_pair_folder.split('/')[-2]}__{img_pair_folder.split('/')[-1]}__{time_str}__{temperature}.json"
            output_path = os.path.join(
                    output_folder, file_name
                )
            out_dict = generate_single_traj(
                    imgs,
                    output_path,
                    cache,
                    prompt_template,
                    temperature=temperature,
                    resolution=resolution,
                    max_tokens=max_tokens,
                    **prompt_kwargs,
                )
            assert len(out_dict['parsed_goals']) == 1
            goal = list(out_dict['parsed_goals'].values())[0][0]
            temp_goals[temperature] = goal
        goals = list(temp_goals.values())
        goals = "\n".join(goals)
        file_name = f"{img_pair_folder.split('/')[-3]}_{img_pair_folder.split('/')[-2]}__{img_pair_folder.split('/')[-1]}__{time_str}_summary.json"
        output_path = os.path.join(
                    output_folder, file_name
                )
        summary_out_dict = generate_single_traj(imgs, 
            output_path,
            cache,
            'summarize_goals',
            temperature=0.5,
            resolution=resolution,
            max_tokens=max_tokens, 
            subgoals=goals)
        final = summary_out_dict['parsed_goals']['final']
        return final  



def main(images_folders, GOAL_FN, GOAL_FN_KWARGS, out_file):
    ground_truths = get_ground_truths(images_folders)
    goals = get_goals(images_folders, GOAL_FN, GOAL_FN_KWARGS)
    answers = final_answers(goals, ground_truths, out_file)
    evald_answers = evaluate(answers, out_file)
    #evald_answers = json.load(open('/Users/alexandrasouly/code/chai/magical/frame_pairs/alex_test2/alex_test.json', "r"))
    counts = count_evals(evald_answers)
    plot_evals(counts, out_file)
    return evald_answers

if __name__ == "__main__":
    images_folders = ['/Users/alexandrasouly/code/chai/magical/frame_pairs/Task1/10-frame-pairs',
                      '/Users/alexandrasouly/code/chai/magical/frame_pairs/Task1/20-frame-pairs',
                      '/Users/alexandrasouly/code/chai/magical/frame_pairs/Task1/40-frame-pairs',
                      '/Users/alexandrasouly/code/chai/magical/frame_pairs/Task1/80-frame-pairs',
                      '/Users/alexandrasouly/code/chai/magical/frame_pairs/Task2/10-frame-pairs',
                      '/Users/alexandrasouly/code/chai/magical/frame_pairs/Task2/20-frame-pairs',
                      '/Users/alexandrasouly/code/chai/magical/frame_pairs/Task2/50-frame-pairs',
                      '/Users/alexandrasouly/code/chai/magical/frame_pairs/Task2/100-frame-pairs',
                      '/Users/alexandrasouly/code/chai/magical/frame_pairs/Task3/10-frame-pairs',
                      '/Users/alexandrasouly/code/chai/magical/frame_pairs/Task3/20-frame-pairs',
                      '/Users/alexandrasouly/code/chai/magical/frame_pairs/Task3/60-frame-pairs',
                      '/Users/alexandrasouly/code/chai/magical/frame_pairs/Task3/180-frame-pairs',
                      ]
    # GOAL_FN = GetGoalFromSinglePrompt()
    # GOAL_FN_KWARGS = {'output_folder': '/Users/alexandrasouly/code/chai/magical/frame_pairs/alex_test','prompt_template': 'two_frame_short_goal_spec_label', 'temperature': 0.5, 'resolution': 'high', 'max_tokens': 1000}
    GOAL_FN = GetGoalPickedFromDiffTemps()
    GOAL_FN_KWARGS = {'output_folder': '/Users/alexandrasouly/code/chai/magical/frame_pairs/temp_test2','prompt_template': 'two_frame_short_goal_spec_label', 'temperatures': [0,0.2,0.5,0.7], 'resolution': 'high', 'max_tokens': 1000}
    out_file = '/Users/alexandrasouly/code/chai/magical/frame_pairs/temp_test2/test.json'
    main(images_folders, GOAL_FN, GOAL_FN_KWARGS, out_file)


# HOW TO USE:
# Specify images_folders: the folder containing the images you want to evaluate in images_folders, these folders should be of the form Task1/10-frame-pairs, Task2/10-frame-pairs, etc. that contain folders of image pairs of the form pair1, pair2, pair3, etc.

# Specify GOAL_FN: the function you want to use to get the goal from the images. If you want to use the single prompt method, use GetGoalFromSinglePrompt(). If you want to use the method where you generate goals with different temperatures and ask GPT4 to pick one, use GetGoalPickedFromDiffTemps(). If you want to do something else, feel free to write your own method. it should inherit from GetGoal and implement the __call__ method that takes in an img_pair_folder, the kwargs you want and returns a str goal.

# These goal functions take in a template as a kwarg. The template is a string that specifies the prompt template you want to use. Currently 'two_frame_short_goal_spec_label' asks GPT4 to generate a goal from two frames. If you want to pass in a different text or eg use multiple images, you will need to write a new template.

# Provide an output file name in out_file. This will be a json file that contains the ground truths and the goals, ready for evaluation.