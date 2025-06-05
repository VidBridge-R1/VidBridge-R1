from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
import torch
import re

app = Flask(__name__)

model_path = "path_to/Qwen2_5_72B_Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def critic_model(content, sol, problem_type, question=None):
    sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
    GT = sol_match.group(1).strip() if sol_match else "None"
    con_match = re.search(r"<answer>(.*?)(</answer>|$)", content, re.DOTALL)
    Res = con_match.group(1).strip() if con_match else ""
    if Res == "" or Res == " ":
        return 0.0

    if problem_type == 'DarkEventInfer':
        prompt = f'''You are an evaluator tasked with determining if a given response matches the Ground Truth (GT) provided. Your job is to compare the response and GT carefully and return a value based on their consistency.
Instructions:
1. Read the Response and GT carefully: Ensure you understand both the response and the GT completely.
2. Evaluate the Consistency:
*Score 2: If the response semantically covers the GT entirely, even if the response is longer.
*Score 1: If the response partially covers the GT, but does not fully encompass it.
*Score 0: If the response is entirely different and irrelevant from the GT or the response is None.

Response: {Res}
GT: {GT}

Your judgement:
'''
    elif problem_type == 'MixVidQA':
        prompt = f"""Given a question along with its ground truth, and a generated answer, please judge whether the generated answer is True or False. If the ground truth or the generated answer is ambiguous, consider it as False.

Question: {question}
Groung truth: {GT}
Generated answer: {Res}
Your judgement:
"""
    else:
        raise ValueError()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        use_cache=True,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if problem_type == 'DarkEventInfer':
        if '2' in response:
            return 2.0
        elif '1' in response:
            return 1.0
        else:
            return 0.0
    elif problem_type == 'MixVidQA':
        if 'True' in response or 'true' in response:
            return 1.0
        else:
            return 0.0

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    with torch.no_grad():
        output = critic_model(data["content"], data["sol"], data["problem_type"], data["question"])
    return jsonify({"output": output})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)