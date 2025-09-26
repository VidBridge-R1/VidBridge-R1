# VidBridge-R1: Bridging QA and Captioning for RL-based Video Understanding Models with Intermediate Proxy Tasks

[[🤗 Model](https://huggingface.co/VidBridge-R1/VidBridge-R1)] [[🤗 Training Data](https://huggingface.co/datasets/VidBridge-R1/VidBridge-R1_training_data)]

## 👀 Overview
The "Reason-Then-Respond" paradigm, enhanced by Reinforcement Learning, has shown great promise in advancing MLLMs. However, its application to the video domain has led to specialized models that excel at either question answering (QA) or captioning tasks, but struggle to master both. Naively combining reward signals from these tasks results in mutual performance degradation, which we attribute to a conflict between their opposing task natures. To address this challenge, we propose a novel training framework built upon two intermediate proxy tasks: 

- DarkEventInfer, which presents videos with masked event segments, requiring models to infer the obscured content based on contextual video cues;
- MixVidQA, which presents interleaved video sequences composed of two distinct clips, challenging models to isolate and reason about one while disregarding the other.

<img src="./assets/main.jpg"/>

These proxy tasks compel the model to simultaneously develop both holistic, divergent understanding and precise, convergent reasoning capabilities. Embodying this framework, we present VidBridge-R1, the first versatile video reasoning model that effectively bridges the paradigm conflict. Extensive experiments show that VidBridge-R1 achieves significant performance gains on both QA and captioning within one model, demonstrating the efficacy of our approach in fostering more generalizable and powerful video understanding models.

## 🔧 Set up
To get started, follow the steps below:
```bash
git clone https://github.com/VidBridge-R1/VidBridge-R1.git
cd VidBridge-R1
conda create -n VidBridge-R1 python=3.10
conda activate VidBridge-R1
pip3 install -e ".[dev]"
pip3 install flash_attn --no-build-isolation
cd qwen-vl-utils
pip install -e .
cd ..
```
Note: After downloading the training data, please update the root directory path in ``VidBridge-R1_training_data.jsonl`` and ``run_qwen25vl_train.sh``.

## 🚀 Training
> **Hardware Note:** Our training was conducted on a system with 8 × A800 (80GB) GPUs. For different hardware configurations, please adjust the corresponding settings accordingly—such as the deployment of judge models in ``src/qwen25_judge_service.py`` and GPU device configurations in ``run_qwen25vl_train.sh``.

### Step 1: Deploy Judge Models
Before starting training, deploy the judge models used for evaluation across DarkEventInfer, MixVidQA, and captioning tasks:
```python
python src/qwen25_judge_service.py
python src/gpt35_judge_service.py
```

### Step 2: Run Training Script
Once the judge models are ready, you can begin training by executing the provided shell script:

```bash
bash run_qwen25vl_train.sh
```

## 🙏 Acknowledgements
We sincerely appreciate the contributions of the open-source community. Our project builds upon [Open-R1-Video](https://github.com/Wang-Xiaodong1899/Open-R1-Video), [open-r1](https://github.com/huggingface/open-r1) and etc.

## 📚 Citation
If you find our work helpful for your research, please consider citing our work.   

```
@article{chen2025versavid,
  title={VersaVid-R1: A Versatile Video Understanding and Reasoning Model from Question Answering to Captioning Tasks},
  author={Chen, Xinlong and Zhang, Yuanxing and Guan, Yushuo and Zeng, Bohan and Shi, Yang and Yang, Sihan and Wan, Pengfei and Liu, Qiang and Wang, Liang and Tan, Tieniu},
  journal={arXiv preprint arXiv:2506.09079},
  year={2025}
}
```
