from flask import Flask, request, jsonify
import os
import torch
import re
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dream_metric import process_one_sample_pre_and_rec

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    sol_match = re.search(r"<answer>(.*?)</answer>", data["sol"], re.DOTALL)
    GT = sol_match.group(1).strip() if sol_match else "None"
    con_match = re.search(r"<answer>(.*?)(</answer>|$)", data["content"], re.DOTALL)
    Res = con_match.group(1).strip() if con_match else ""
    if Res == "" or Res == " ":
        return jsonify({"recall": 0.0, "precision": 0.0})
    data_eval = {
        "response": GT,
        "prediction": Res,
    }

    for i in range(5): # Recall and Precision
        try:
            recall, precision = process_one_sample_pre_and_rec((data_eval, 'gpt-35-turbo-0125', False))
            if recall is not None and precision is not None:
                return jsonify({"recall": recall, "precision": precision})
        except:
            continue

    return jsonify({"recall": 0.0, "precision": 0.0})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5200)