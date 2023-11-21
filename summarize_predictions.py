import json
import random
import argparse
import pathlib
from call_vlm import call_vlm as call_llm, load_cache

class SummarizePromptTemplate:

    def __call__(self, pred_list):
        raise NotImplementedError
    
    def parse_output(self, output):
        raise NotImplementedError


class SummarizePromptDefault(SummarizePromptTemplate):
    def __call__(self, pred_list):
        intro = """Here are several predictions about the goal an agent was trying to complete. Each prediction consists of several guesses at possible made from a single video of an agent accomplishing the task. Read each, then make a guess overall at the agent's most likely overall goal. 
Remember these hints:
- The goal is exactly same across all trials of the task. 
- The goal is always short, common-sense, and easy for a human to understand.
- The guesses were made from only seeing a single trial, so they may be overly specific and refer to the specific trial, rather than the overall goal.
- Some of the guesses may be incorrect.
- The goal should not mention the labels of the objects in the environment.
"""

        ending = """
Output the goal as a short, descriptive sentence on a new line that starts with the words "GOAL:".
""" 
        random.seed(0)
        random.shuffle(pred_list)
        
        lines = [intro]
        for i in range(len(pred_list)):
            lines.append(f'Trial {i+1}":"')
            lines.append(f"Most likely goal: {pred_list[i]['goals'][0]}")
            lines.append("")
        lines.append(ending)
        prompt = "\n".join(lines)
        return prompt
        # template =[
        #     {
        #         "text": prompt,
        #     },
        # ]
#         completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"}
#   ]
# )
        
        
#         vlm_args = {
#         "model": model_name,
#         "messages":[
#             {
#                 "role": "user",
#                 "content": message_content,
#             }
#         ],
#         "max_tokens":max_tokens,
#         "temperature":temperature,
#     }
        
        
        # return template

    def parse_output(self, text):
        # Find any lines that start with "GOAL:"
        goal_lines = [line for line in text.split("\n") if line.startswith("GOAL:")]
        # Remove the "GOAL:" prefix
        goals = [line.replace("GOAL:", "") for line in goal_lines]
        return {'goals': goals}
    
PROMPT_TEMPLATES = {
    "summarize_prompt_default": SummarizePromptDefault,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_files', type=str, nargs='+', default=None, help=f'List of files to summarize. `traj_files` and `all_traj_folder` can\'t be provided at the same time')
    parser.add_argument('--all_traj_folder', type=str, default=None, help=f'Folder containing many subfolders, each of which contains a directory. The subfolders must be in the same format as specified for `traj_files`. `traj_files` and `all_traj_folder` can\'t be provided at the same time')
    parser.add_argument('--output_folder', type=str, required=True, help=f'Folder where analyses will be saved.')
    parser.add_argument('--prompt_template', type=str, default='summarize_prompt_default', help=f'Prompt template for summarization.', choices=PROMPT_TEMPLATES.keys())
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=1000)
    args = parser.parse_args()
    
    # Confirm we either have traj_files or all_traj_folder, but not both
    assert (args.traj_files is not None) != (args.all_traj_folder is not None), "Must provide either `traj_files` or `all_traj_folder`"
    if args.traj_files is not None:
        all_json_files = [pathlib.Path(f) for f in args.traj_files]
    else:
        # Take all the json files in all_traj_folder
        all_json_files = list(pathlib.Path(args.all_traj_folder).glob("**/*.json"))
    
    # The predicted goals are all in the "parsed_goals" field of the json file
    pred_list = []
    for json_file in all_json_files:
        with open(json_file, 'r') as f:
            pred_list.append(json.load(f)['parsed_goals'])
    
    summarize_template = PROMPT_TEMPLATES[args.prompt_template]()
    summarization_messages = summarize_template(pred_list)
    cache = load_cache()
    summarization_result = call_llm(summarization_messages, args.temperature, args.max_tokens, cache, model_name="gpt-4-1106-preview")
    goals = summarize_template.parse_output(summarization_result)
    out_dict = {
        "messages": summarization_messages,
        "llm_completion": summarization_result,
        "goals": goals,
    }
    output_folder = pathlib.Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "summarize.json"
    with open(output_file, 'w') as f:
        json.dump(out_dict, f, indent=4)

if __name__ == "__main__":
    main()
