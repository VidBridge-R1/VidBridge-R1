# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import load_dataset

from math_verify import parse, verify
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from open_r1_video.trainer import Qwen25VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed as dist
from PIL import Image
from sympy.parsing.latex import parse_latex
import requests

def load_model_and_tokenizer(model_path, max_memory=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        max_memory=max_memory
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def critic_model(content, sol):
    sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
    GT = sol_match.group(1).strip() if sol_match else sol.strip()
    con_match = re.search(r"<answer>(.*?)(</answer>|$)", content, re.DOTALL)
    Res = con_match.group(1).strip() if con_match else content.strip()
    if Res == '':
        Res = 'None'
    prompt = f'''You are an evaluator tasked with determining if a given response matches the Ground Truth (GT) provided. Your job is to compare the response and GT carefully and return a value based on their consistency.
Instructions:
1. Read the Response and GT carefully: Ensure you understand both the response and the GT completely.
2. Evaluate the Consistency:
*Score 2: If the response semantically covers the GT entirely, even if the response is longer.
*Score 1: If the response partially covers the GT, but does not fully encompass it.
*Score 0: If the response is entirely different and irrelevant from the GT.

Response: {Res}
GT: {GT}

Your judgement:
'''
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda:6")
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if '2' in response:
        return 2.0
    elif '1' in response:
        return 1.0
    else:
        return 0.0

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )


def keywords_reward(content):
    good_words = ["start with", "starts with", "then", "next", "after", 
    "begin with", "begins with", "followed by", "following", 
    "subsequently", "thereafter", "later", "initially", 
    "first", "second", "third", "begin to", "begins to",
    "finally", "lastly", "to begin with", "as a starting point", 
    "in the beginning", "at the outset", "once", "when", 
    "whenever", "meanwhile", "simultaneously"]
    bad_words = ["possibly", "suggesting", "likely", "appears to", "appearing", 
    "designed to", "seems to", "might", "may", "could", 
    "potentially", "presumably", "probably", "perhaps", 
    "allegedly", "reportedly", "supposedly", "apparently", 
    "arguably", "tentatively", "hypothetically", "theoretically", 
    "in theory", "in some cases", "occasionally", "sometimes", 
    "rarely", "unclear", "ambiguous", "vague", "speculative", 
    "indicative of", "hinting at", "implying", "inferring", 
    "open to interpretation", "subject to change", "not necessarily"]

    reward = 0.0
    con_match = re.search(r"<answer>(.*?)(</answer>|$)", content, re.DOTALL)
    Res = con_match.group(1).strip() if con_match else ""
    if Res == "" or Res == " ":
        reward = 0.0
    else:
        for good in good_words:
            if good in Res.lower():
                reward += 0.2
        for bad in bad_words:
            if bad in Res.lower():
                reward = min(reward, 0.0)
                reward -= 0.2
        
        reward = min(reward, 0.4)

    return reward

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    format_r = format_reward(completions)
    for content, sol, problem_type in zip(contents, solution, kwargs['problem_type']):
        if problem_type == 'MCQA':
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r"<answer>(.*?)(</answer>|$)", content, re.DOTALL)
                    if content_match:
                        student_answer = content_match.group(1).strip()
                        if 'A.' in student_answer or '{A}' in student_answer or '\\boxed{A}' in student_answer:
                            student_answer = ['A']
                        elif 'B.' in student_answer or '{B}' in student_answer or '\\boxed{B}' in student_answer:
                            student_answer = ['B']
                        elif 'C.' in student_answer or '{C}' in student_answer or '\\boxed{C}' in student_answer:
                            student_answer = ['C']
                        elif 'D.' in student_answer or '{D}' in student_answer or '\\boxed{D}' in student_answer:
                            student_answer = ['D']
                        elif 'E.' in student_answer or '{E}' in student_answer or '\\boxed{E}' in student_answer:
                            student_answer = ['E']
                        elif 'F.' in student_answer or '{F}' in student_answer or '\\boxed{F}' in student_answer:
                            student_answer = ['F']
                        elif 'G.' in student_answer or '{G}' in student_answer or '\\boxed{G}' in student_answer:
                            student_answer = ['G']

                        elif 'a. ' in student_answer:
                            student_answer = ['A']
                        elif 'b. ' in student_answer:
                            student_answer = ['B']
                        elif 'c. ' in student_answer:
                            student_answer = ['C']
                        elif 'd. ' in student_answer:
                            student_answer = ['D']
                        elif 'e. ' in student_answer:
                            student_answer = ['E']
                        elif 'f. ' in student_answer:
                            student_answer = ['F']
                        elif 'g. ' in student_answer:
                            student_answer = ['G']
                        else:
                            student_answer = [student_answer]
                            
                        if student_answer[0] == ground_truth[0]:
                            reward = 1.0
                    else:
                        reward = 0.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
        elif problem_type in ['DarkEventInfer', 'MixVidQA']:
            question = kwargs["problem"][0].strip()
            response = requests.post(
                "http://localhost:5000/predict",
                json={"content": content, # model_genertion
                    "sol": sol, # GT
                    "problem_type": problem_type,
                    "question": question}
            )
            reward = response.json()["output"]
        elif problem_type == "caption":
            response = requests.post(
                "http://localhost:5200/predict",
                json={"content": content, # model_genertion
                    "sol": sol, # GT
                    "problem_type": problem_type}
            )
            recall = response.json()["recall"]
            precision = response.json()["precision"]
            keywords_r = keywords_reward(content)
            reward = recall + 0.5*precision + keywords_r

        rewards.append(reward)

    final_rewards = [x * y for x, y in zip(format_r, rewards)]

    return final_rewards


def format_reward(completions):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "accuracy": accuracy_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

from datasets import Dataset, DatasetDict
import json

def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)
    return DatasetDict({
        "train": base_dataset
    })


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    if script_args.jsonl_path:
        # # load dataset from jsonl
        dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path)
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = "{Question}\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>. "

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    elif "video" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(
            make_conversation_video,
        )
    else:
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")
    
    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
