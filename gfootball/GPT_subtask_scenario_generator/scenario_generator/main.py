import logging
import os

from openai import OpenAI
from pathlib import Path
import shutil
import time
import re


client = OpenAI(api_key="YOUR_KEY")
ROOT_DIR = os.getcwd()

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

def main(model, iteration, sample, temperature):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    suffix = "GPT"
    logging.info(f"Using LLM: {model}")


    # env_init = f'{ROOT_DIR}/env_code/__init__.py'
    # env = f'{ROOT_DIR}/env_code/football_env.py'
    # env_core = f'{ROOT_DIR}/env_code/football_env_core.py'
    # action_set= f'{ROOT_DIR}/env_code/football_action_set.py'
    # observation_processor = f'{ROOT_DIR}/env_code/observation_processor.py'
    # scenario_builder = f'{ROOT_DIR}/env_code/scenario_builder.py'

    three_vs_one_with_keeper = f'{ROOT_DIR}/scenario_examples/academy_3_vs_1_with_keeper.py'
    corner = f'{ROOT_DIR}/scenario_examples/academy_corner.py'
    counterattack_easy = f'{ROOT_DIR}/scenario_examples/academy_counterattack_easy.py'
    counterattack_hard = f'{ROOT_DIR}/scenario_examples/academy_counterattack_hard.py'
    empty_goal = f'{ROOT_DIR}/scenario_examples/academy_empty_goal.py'
    empty_goal_close = f'{ROOT_DIR}/scenario_examples/academy_empty_goal_close.py'
    pass_and_shoot_with_keeper = f'{ROOT_DIR}/scenario_examples/academy_pass_and_shoot_with_keeper.py'
    run_pass_and_shoot_with_keeper = f'{ROOT_DIR}/scenario_examples/academy_run_pass_and_shoot_with_keeper.py'
    run_to_score = f'{ROOT_DIR}/scenario_examples/academy_run_to_score.py'
    run_to_score_with_keeper = f'{ROOT_DIR}/scenario_examples/academy_run_to_score_with_keeper.py'
    single_goal_versus_lazy = f'{ROOT_DIR}/scenario_examples/academy_single_goal_versus_lazy.py'
    five_vs_five = f'{ROOT_DIR}/scenario_examples/5_vs_5.py'

    three_vs_one_with_keeper_code_string = file_to_string(three_vs_one_with_keeper)
    corner_code_string = file_to_string(corner)
    counterattack_easy_code_string = file_to_string(counterattack_easy)
    counterattack_hard_code_string = file_to_string(counterattack_hard)
    empty_goal_code_string = file_to_string(empty_goal)
    empty_goal_close_code_string = file_to_string(empty_goal_close)
    pass_and_shoot_with_keeper_code_string = file_to_string(pass_and_shoot_with_keeper)
    run_pass_and_shoot_with_keeper_code_string = file_to_string(run_pass_and_shoot_with_keeper)
    run_to_score_code_string = file_to_string(run_to_score)
    run_to_score_with_keeper_code_string = file_to_string(run_to_score_with_keeper)
    single_goal_versus_lazy_code_string = file_to_string(single_goal_versus_lazy)
    five_vs_five_code_string = file_to_string(five_vs_five)


    # env_init_code_string  = file_to_string(env_init)
    # env_code_string = file_to_string(env)
    # env_core_code_string = file_to_string(env_core)
    # action_set_code_string = file_to_string(action_set)
    # observation_processor_code_string = file_to_string(observation_processor)
    # scenario_builder_code_string = file_to_string(scenario_builder)


    parent_dir = os.path.dirname(ROOT_DIR)
    parent_dir = os.path.dirname(parent_dir)
    output_file = f"{parent_dir}/scenarios/scenario_{suffix.lower()}.py"

    # Loading all text prompts
    prompt_dir = f'{ROOT_DIR}/utils/prompts'

    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')

    example = file_to_string(f'{prompt_dir}/example.txt')

    example = example.format(
    three_vs_one_with_keeper_code_string=three_vs_one_with_keeper_code_string,
    corner_code_string = corner_code_string,
    counterattack_easy_code_string = counterattack_easy_code_string,
    counterattack_hard_code_string = counterattack_hard_code_string,
    empty_goal_code_string = empty_goal_code_string,
    empty_goal_close_code_string = empty_goal_close_code_string,
    pass_and_shoot_with_keeper_code_string = pass_and_shoot_with_keeper_code_string,
    run_pass_and_shoot_with_keeper_code_string = run_pass_and_shoot_with_keeper_code_string,
    run_to_score_code_string = run_to_score_code_string,
    run_to_score_with_keeper_code_string = run_to_score_with_keeper_code_string,
    single_goal_versus_lazy_code_string = single_goal_versus_lazy_code_string
    )

    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')

    initial_system = initial_system.format(five_vs_five=five_vs_five_code_string) + code_output_tip + example

    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]


    # generation loop
    for iter in range(iteration):
        # Get response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = sample if "gpt-3.5" in model else 4

        logging.info(f"Iteration {iter}: Generating {sample} samples with {model}")

        while True:
            if total_samples >= sample:
                break
            for attempt in range(1000):
                print("ATTEMPT:",attempt)
                try:
                    response_cur = client.chat.completions.create(model=model,
                    messages=messages,
                    temperature=temperature,
                    n=chunk_size)
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur.choices)
            print("PESPONSES:",responses)
            prompt_tokens = response_cur.usage.prompt_tokens
            total_completion_token += response_cur.usage.completion_tokens
            total_token += response_cur.usage.total_tokens

        if sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

        code_runs = []
        for response_id in range(sample):

            response_cur = responses[response_id].message.content
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            print("Code String 1:",code_string)

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])

            print("Code String 2:", code_string)


            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w') as file:
                file.writelines("from . import *" + '\n')
                file.writelines(code_string + '\n')

            with open(f"outputs/env_iter{iter}_response{response_id}.py", 'w') as file:
                file.writelines("from . import *" + '\n')
                file.writelines(code_string + '\n')





if __name__ == "__main__":
    main(model="gpt-3.5-turbo",iteration=5, sample=10, temperature=0.2)