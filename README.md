# SEM-Based Pore System Characterisation of Heterogeneous Reservoir Rocks Using Vision-Language Guided Structure-Enhanced Segmentation

This repository provides the official implementation of **VLSE-Net**, a vision-language guided segmentation framework for pore extraction from heterogeneous rock core SEM images.

VLSE-Net is developed for SEM-based digital rock analysis and aims to improve pore-system characterisation by addressing two major challenges:

1. **Pore–matrix semantic ambiguity** caused by heterogeneous mineral textures and similar grayscale appearances.
2. **Structural discontinuity** of elongated pores and weak-boundary pore structures.

The framework integrates two complementary modules:

- **LSCM (Language-Driven Semantic Calibration Module)**  
  Introduces vision-language semantic priors and cross-modal constraints to improve pore–matrix discrimination.

- **ASRM (Anisotropy-Aware Structural Refinement Module)**  
  Applies direction-sensitive structural modelling and adaptive refinement to preserve elongated and connectivity-related pore structures.

---

# Model Architecture

<p align="center">
<img src="figures/module.svg" width="900" alt="VLSE-Net overall architecture">
</p>

## LSCM: Language-Driven Semantic Calibration Module

<p align="center">
<img src="figures/LSCM_module.svg" width="900" alt="LSCM module">
</p>

LSCM introduces semantic information from a vision-language model to improve pore recognition under complex geological textures.

Main components:

- Text prior injection into visual features.
- Token-level cross-modal interaction.
- Region-level semantic alignment constraint.

## ASRM: Anisotropy-Aware Structural Refinement Module

<p align="center">
<img src="figures/ASRM.svg" width="700" alt="ASRM module">
</p>

ASRM improves structural representation for anisotropic pore morphologies.

Main components:

- Parallel directional branches for horizontal, vertical and isotropic patterns.
- Adaptive spatial gating for feature fusion.
- Residual refinement for structural continuity preservation.

---

# Installation

## Environment

The implementation was developed and tested with:

- Python >= 3.10
- PyTorch 2.5.1
- torchvision 0.20.1
- CUDA-compatible GPU

The experiments reported in the paper were conducted on:

- GPU: NVIDIA RTX 4090 (24 GB)

## Install Dependencies

Create a virtual environment:

```bash
conda create -n vlsenet python=3.10
conda activate vlsenet
```

Install required packages:

```bash
pip install -r requirements.txt
```

The frozen CLIP-RN50 weights are downloaded automatically on first use if they are not already available in the local CLIP cache.

---

# Dataset Availability

The datasets used in this study include both private and public SEM datasets.

## Private datasets

The SCD-1 and SCD-2 datasets used in the paper are not publicly available due to data-use restrictions.

## Public datasets

The following datasets are publicly available from their original sources:

- DRP-317
- cigRockSEM

Public datasets can be organized according to the directory structure below. The repository includes a small example subset of SEM images, binary masks and sample-specific textual prompts for functional testing.

## Dataset Text Generation

Sample-specific textual prompts used in the reported experiments were generated offline from image-derived statistical reports using **Qwen3.5-Plus**. The generated prompts were stored before training and used as fixed auxiliary textual inputs for training, validation and testing.

`dataset_builder` is not required when pre-generated text prompts are already available. It is provided to reproduce the offline prompt-construction procedure for prepared image patches. See [`dataset_builder/README.md`](dataset_builder/README.md) for details.

---

# Repository Structure

```text
VLSE-Net/
│
├── VLSENet.py              # VLSE-Net implementation
├── train_VLSENet.py        # VLSE-Net training script
├── infer_VLSENet.py        # VLSE-Net inference script
├── unet.py                 # U-Net baseline implementation
├── train_unet.py           # U-Net training script
├── DSConv_pro.py           # Direction-aware convolution module
├── feature_renorm.py       # Feature normalization utilities
│
├── clip/                   # CLIP-related components
├── dataset/                # Small example dataset
├── dataset_builder/        # Offline prompt-generation utilities
├── figures/                # Model visualization files
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Dataset Format

`train_VLSENet.py` expects the following directory structure:

```text
dataset/
│
├── patch_images/
│   ├── image_001.png
│   └── ...
│
├── patch_mask/
│   ├── image_001.png
│   └── ...
│
└── text/
    ├── image_001.txt
    └── ...
```

Description:

- `patch_images/`: SEM images.
- `patch_mask/`: binary pore masks.
- `text/`: sample-specific textual prompts for vision-language guidance.

Image, mask and text files should share the same filename stem. For example:

```text
image_001.png
image_001.txt
```

The example files distributed in `dataset/` are intended for functional testing only and are not the complete SCD-1 or SCD-2 datasets.

---

# Training

## Train VLSE-Net

```bash
python train_VLSENet.py \
    --data-root ./dataset
```

## Train U-Net Baseline

```bash
python train_unet.py \
    --data-root ./dataset
```

Training outputs include:

- model checkpoints;
- training logs;
- validation results.

The small example dataset is not sufficient to reproduce the quantitative results reported in the paper. It is provided to check the released implementation and data format.

## Quick Functional Check

A one-epoch run can be used to verify the training-to-inference workflow:

```bash
python train_VLSENet.py \
    --data-root ./dataset \
    --epochs 1 \
    --num-workers 0
```

For the first run, the best checkpoint is written to:

```text
reports/latest_run_text_guided_unet/best_text_guided_unet.pt
```

If that report directory already exists, the training script creates a suffixed run directory and prints the actual path.

---

# Configuration Parameters

Common parameters include:

| Parameter | Description |
|---|---|
| `--data-root` | Dataset directory |
| `--batch-size` | Training batch size |
| `--lr` | Initial learning rate |
| `--epochs` | Number of training epochs |
| `--scheduler` | Learning-rate scheduler |

Please refer to the training scripts for all available options.

---

# Inference

VLSE-Net inference uses an SEM image together with its pre-generated sample-specific textual prompt and produces a binary pore segmentation mask. The language model used during offline prompt preparation is **not** called during inference.

## Run Inference

After obtaining a checkpoint from `train_VLSENet.py`, run:

```bash
python infer_VLSENet.py \
    --image-dir ./dataset/patch_images \
    --text-dir ./dataset/text \
    --checkpoint ./reports/latest_run_text_guided_unet/best_text_guided_unet.pt \
    --output-dir ./outputs
```

Arguments:

- `--image-dir`: directory containing input SEM images.
- `--text-dir`: directory containing matching sample-specific `.txt` prompts.
- `--checkpoint`: checkpoint produced by `train_VLSENet.py`.
- `--output-dir`: directory for generated binary masks.

Each image must have a non-empty text file with the same filename stem. Missing prompts are treated as an input error rather than silently replaced by a generic prompt.

The generated results are organized as follows:

```text
outputs/
└── masks/
    ├── sample_001.png
    └── ...
```

The predicted binary masks can be further used for pore-system descriptor analysis.

A trained checkpoint is not bundled with this repository. Use a checkpoint generated by the training script or provide a compatible VLSE-Net checkpoint.

---

# Reproducibility

This repository provides:

- VLSE-Net implementation;
- U-Net baseline implementation;
- training and inference scripts;
- a small example image/mask/text subset for functional testing;
- offline prompt-construction utilities.

Because of data-use restrictions, the original SCD-1 and SCD-2 datasets cannot be redistributed. The example subset is intended to validate the released workflow, not to reproduce the paper's reported quantitative metrics.

---

# Experimental Results

On the main SEM dataset:

- Dice score improvement: **+3.12 percentage points** over U-Net.
- SEM-derived apparent porosity error reduction: **39.6%**.

VLSE-Net also improves the reliability of pore-system descriptors, including local porosity, connectivity and connected-component density.

---

# License

This project is released under the MIT License.
