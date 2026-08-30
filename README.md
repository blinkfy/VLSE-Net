# SEM-Based Pore System Characterisation of Heterogeneous Reservoir Rocks Using Vision-Language Guided Structure-Enhanced Segmentation

This repository provides the official implementation of **VLSE-Net**, a vision-language guided segmentation framework for pore extraction from heterogeneous rock core SEM images.

## Authors

Xinru Zhang, Yanfei Xu, Yachuan Li, Shaohua Cao, and Peigang Liu

Corresponding author: **Peigang Liu**

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
├── train_VLSENet.py        # VLSE-Net training/evaluation entry point
├── infer_VLSENet.py        # VLSE-Net inference script
├── unet.py                 # U-Net baseline implementation
├── train_unet.py           # U-Net baseline training/evaluation entry point
├── data_utils.py            # Dataset loading and split-manifest utilities
├── DSConv_pro.py           # Direction-aware convolution module
├── feature_renorm.py       # Feature normalization utilities
│
├── clip/                   # CLIP-related components
├── dataset/                # Small example dataset
├── dataset_builder/        # Offline prompt-generation utilities
├── figures/                # Model visualization files
├── splits/                 # Split-manifest format documentation
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Dataset Format

The training scripts expect the following directory structure:

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
- `text/`: sample-specific textual prompts used by VLSE-Net.

Image, mask and text files should share the same filename stem. For example:

```text
image_001.png
image_001.txt
```

The example files distributed in `dataset/` are intended for functional testing only and are not the complete SCD-1 or SCD-2 datasets.

---

# Dataset Splitting and Paper Reproduction

The experiments reported in the paper use a **70% / 15% / 15% train/validation/test split**. To prevent information leakage, partitioning is performed at the **original-image level** before patch assignment. For DRP-317, partitioning is performed at the slice level so that samples derived from the same source slice remain in the same subset.

The released training scripts support this protocol through explicit split manifests:

```text
splits_for_run/
├── train.txt
├── val.txt
└── test.txt
```

Each manifest contains one patch filename per line. The manifests must be prepared so that all patches originating from the same source image (or the same DRP-317 slice) are assigned to only one subset. The scripts validate that:

- all three manifests are non-empty;
- no sample appears in more than one subset;
- every dataset sample is assigned exactly once;
- every listed filename exists in the prepared dataset.

For the private SCD-1/SCD-2 datasets, the source-level manifests are generated during dataset preparation and are not distributed because the underlying data are restricted. The manifest format is documented in [`splits/README.md`](splits/README.md).

When `--split-dir` is omitted, the scripts create a deterministic **sample-level** 70/15/15 split using the supplied random seed. This fallback exists only for functional testing of public/example data and **must not be interpreted as the paper's source-level experimental split**.

Each training run writes the exact allocation it used to `<report_dir>/splits/`, allowing the run to be audited or repeated.

---

# Training

## Train VLSE-Net

For paper-protocol training with prepared source-level manifests:

```bash
python train_VLSENet.py \
    --data-root ./dataset \
    --split-dir ./splits_for_run
```

The public training entry point uses the released VLSE-Net configuration and sample-specific text prompts. It selects the best checkpoint using validation Dice and evaluates that checkpoint once on the held-out test subset.

## Train U-Net Baseline

Use the **same split manifests** for the U-Net baseline:

```bash
python train_unet.py \
    --data-root ./dataset \
    --split-dir ./splits_for_run
```

Using identical manifests ensures that VLSE-Net and U-Net are compared on the same train/validation/test samples.

Training outputs include:

- best and last model checkpoints;
- epoch-level training and validation logs;
- the exact split manifests used by the run;
- `summary.json`;
- final held-out `test_metrics.json`.

The small example dataset is not sufficient to reproduce the quantitative results reported in the paper. It is provided to check the released implementation and data format.

## Quick Functional Check

The three released example samples are sufficient to exercise the complete train/validation/test code path:

```bash
python train_VLSENet.py \
    --data-root ./dataset \
    --epochs 1 \
    --num-workers 0
```

Without `--split-dir`, this command intentionally uses the deterministic sample-level fallback and is **not** a reproduction of the paper's experimental partition.

For the first run, the best VLSE-Net checkpoint is written to:

```text
reports/latest_run_text_guided_unet/best_text_guided_unet.pt
```

If that report directory already exists, the training script creates a suffixed run directory and prints the actual path.

---

# Configuration Parameters

Common parameters include:

| Parameter | Description |
|---|---|
| `--data-root` | Prepared dataset directory |
| `--split-dir` | Directory containing `train.txt`, `val.txt` and `test.txt` |
| `--image-size` | Input image size; default `512` |
| `--batch-size` | Training batch size; default `8` |
| `--lr` | Initial learning rate; default `1e-4` |
| `--epochs` | Maximum number of training epochs; default `50` |
| `--seed` | Random seed used by the functional fallback split and training |
| `--num-workers` | DataLoader worker count |

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
- matched train/validation/test handling for VLSE-Net and U-Net;
- split-manifest validation for source-level experimental partitions;
- training, held-out test evaluation and inference scripts;
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
