# MemeLens-VLM: Robust Multimodal Misogyny Classification with Qwen3-VL

Fine-tuning and robustness evaluation of **Qwen3-VL-8B-Instruct** for multimodal misogyny classification on the **MAMI (Multimedia Automatic Misogyny Identification)** dataset.

This project was developed for **Deep Learning (EE-559) at EPFL** and investigates whether a vision-language model can remain robust when visual content is deliberately obfuscated by perturbations not observed during evaluation.

## Overview

Multimodal content moderation models can exploit both textual and visual signals, but their predictions may be sensitive to relatively simple modifications of the image.

In this project, we fine-tuned **Qwen3-VL-8B-Instruct** using **LoRA** and built an evaluation pipeline for studying robustness to synthetic visual obfuscations.

The project focuses on three questions:

1. Can a large vision-language model be efficiently adapted to multimodal misogyny classification?
2. How much does its performance deteriorate under visual obfuscation?
3. Can training with controlled perturbations improve generalization to **previously unseen perturbation geometries**?

## Model and Training

The base model is:

**Qwen3-VL-8B-Instruct**

Fine-tuning was performed using parameter-efficient **Low-Rank Adaptation (LoRA)** rather than updating all model parameters.

Main tools:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT / LoRA
- scikit-learn
- NumPy / pandas

The model was adapted to the MAMI image-text classification task while retaining the pretrained multimodal representations of Qwen3-VL.

## Dataset

The experiments use the **MAMI — Multimedia Automatic Misogyny Identification** dataset.

Each example contains:

- an image;
- associated text;
- a misogyny label.

This makes the task particularly useful for studying multimodal robustness because predictions can rely on information contained in either modality or on their interaction.

## Robustness Experiment

The central part of the project is a controlled robustness experiment based on **synthetic visual obfuscation**.

Images are modified using sticker-like perturbations designed to partially obscure visual information while preserving the underlying semantic content.

Rather than evaluating only on perturbations resembling those used during training, the robustness evaluation includes **previously unseen sticker geometries**.

This tests whether the model learns robustness that transfers beyond the exact augmentation pattern used during training.

### Evaluation Settings

We evaluate robustness in two settings:

- **Image-only:** measuring sensitivity to changes in visual information.
- **Image + text:** evaluating whether multimodal information helps compensate for visual corruption.

Performance is measured using **macro-F1**, with particular attention to the degradation between clean and perturbed inputs.

## Results

The robustness-oriented fine-tuning procedure substantially reduced the degradation in macro-F1 caused by previously unseen visual obfuscations.

| Evaluation setting | Macro-F1 degradation before | Macro-F1 degradation after |
|---|---:|---:|
| Image only | 5.9% | **1.5%** |
| Image + text | 7.0% | **1.1%** |

The corresponding prediction files and experiment summaries are retained under `Hate_Project/results/`, allowing the clean, obfuscated and fine-tuned conditions to be inspected separately.

The experiments indicate that robustness improvements transfer beyond the precise perturbation geometry used during training, particularly in the multimodal image-text setting.

## Experimental Pipeline

The project can be summarized as:

```text
MAMI image + text samples
          |
          v
  Data preprocessing
          |
          v
 Qwen3-VL-8B-Instruct
          |
          v
     LoRA fine-tuning
          |
          +----------------------+
          |                      |
          v                      v
     Clean evaluation     Visual perturbation
                                 |
                                 v
                     Unseen sticker geometries
                                 |
                                 v
                       Robustness evaluation
                                 |
                                 v
                         Macro-F1 analysis
                         