
# ⚡ EB-JEPA (Resumable Training Fork)

> [!IMPORTANT]
> **Fork Note:** This is a modified version of the [original EB-JEPA repository](https://github.com/facebookresearch/eb_jepa) by Meta AI Research.

---

## 🔄 Why this fork?
The primary goal of this fork is to **simplify the training process** for users with limited or shared hardware. While toy examples don't require the massive compute of full-scale JEPA models, training for several hours can still be a hurdle on unstable or time-limited hardware.

This version enables you to **interrupt and resume training** seamlessly, whether voluntarily or due to platform constraints.

### **Example Scenario: Surviving Timeouts**
If you are training on **Google Colab** and your session times out or you deplete your free GPU hours, you don't lose your progress. You can easily continue training on another platform (like Kaggle or Lightning.ai) by cloning this repo and running the resume command.

**Compatible Platforms:**
* Google Colab
* Lightning.ai
* Kaggle

---

## 🚀 Getting Started

Follow these steps to set up the environment and launch your first training session.

### **1. Installation**
```bash
# Clone the repository
git clone [https://github.com/Hardwarize/eb_jepa.git](https://github.com/Hardwarize/eb_jepa.git)
cd eb_jepa

# Install requirements in editable mode
pip install -qe .

# Set environment variables for data and checkpoints
export EBJEPA_DSETS=/datasets
export EBJEPA_CKPTS=/checkpoints
```

### **2. Launch Training**
To start a new training run (using the Video JEPA example):
```bash
python -m examples.video_jepa.main
```

> [!TIP]
> On the first run, the code will ask if you want to log to **Weights & Biases (W&B)**. Select **Option 2** and paste your API Token. This is required to track your run ID for future resumption.

---

## ⏯️ Resuming Training
If your training is interrupted, you can resume from the exact point it stopped—even in a completely different environment—using your W&B Run ID:

```bash
python -m examples.video_jepa.main --wandb_run_id <YOUR_WANDB_RUN_ID>
```

The script will automatically fetch the previous state and configuration, ensuring no progress is lost.

---

## 🛠️ Troubleshooting & Memory
Different platforms provide different GPU memory (VRAM) capacities. If you encounter a **CUDA Out of Memory (OOM)** error, reduce the batch size using the argument below:

```bash
# Reduce batch size to fit your GPU
python -m examples.video_jepa.main --data.batch_size 16
```

---

<h1 align="center">
    <p>⚡ <b>EB-JEPA</b></p>
</h1>

<h2 align="center">
    <p><i>Energy-Based Joint-Embedding Predictive Architectures</i></p>
</h2>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/facebookresearch/eb_jepa" target="_blank" style="margin: 2px;"><img alt="Github" src="https://img.shields.io/badge/Github-facebookresearch/eb__jepa-black?logo=github" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://arxiv.org/abs/2602.03604" target="_blank" style="margin: 2px;"><img alt="ArXiv" src="https://img.shields.io/badge/arXiv-2602.03604-b5212f?logo=arxiv" style="display: inline-block; vertical-align: middle;"/></a>
</div>

<br>

<p align="center">
  <b><a href="https://ai.facebook.com/research/">Meta AI Research, FAIR</a></b>
</p>

<p align="center">
  <a href="https://x.com/BasileTerv987">Basile Terver</a>,
  Randall Balestriero,
  Megi Dervishi,
  David Fan,
  Quentin Garrido,
  Tushar Nagarajan,
  <br>
  Koustuv Sinha,
  Wancong Zhang,
  Mike Rabbat,
  Yann LeCun,
  Amir Bar
</p>

<p align="center">
  An open source library and tutorial for learning representations for<br>
  prediction and planning using joint embedding predictive architectures.
</p>

<p align="center">
  <img src="docs/archi-schema-eb-jepa.png" alt="EB-JEPA Architecture" width="800">
</p>

> Each example is (almost) self-contained and training takes up to a few hours on a single GPU card.

---

## 📚 Examples

### [Image JEPA](examples/image_jepa/README.md)
Self-supervised representations from unlabeled images on CIFAR-10, evaluated on classification.

### [Video JEPA](examples/video_jepa/README.md)
Predict next image representation in a sequence.

### [AC Video JEPA](examples/ac_video_jepa/README.md)
JEPA for world modeling + planning in Two Rooms environment.

---