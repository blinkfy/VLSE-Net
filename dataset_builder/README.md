# dataset_builder

`dataset_builder` reproduces the offline construction of sample-specific textual prompts used by VLSE-Net. It operates on **prepared SEM image patches** and does not upload raw images or masks to the language model. Instead, it extracts image-derived statistical descriptors locally, formats them as a structured report, and sends only that text report to an OpenAI-compatible language-model endpoint.

The prompts used in the experiments reported in the paper were generated with **Qwen3.5-Plus** and stored before model training. They were then used as fixed auxiliary textual inputs for the training, validation and test sets. The language model is not part of VLSE-Net and is not called during model training or inference.

## 1. Scripts

- `compute_features.py`: extracts image-derived statistical descriptors.
- `prompt_builder.py`: formats the descriptors into the structured report sent to the language model.
- `llm_api.py`: calls an OpenAI-compatible API and returns the generated pore description.
- `build_dataset.py`: performs feature extraction, prompt construction and text generation in batch.
- `requirements.txt`: dependencies required by this utility.

The current release assumes that image and mask patches have already been prepared. It does not include the original image-splitting pipeline.

## 2. Expected Directory Layout

From the repository root:

```text
dataset/
├── patch_images/
│   ├── image_001.png
│   └── ...
├── patch_mask/
│   ├── image_001.png
│   └── ...
└── text/
    ├── image_001.txt
    └── ...
```

`build_dataset.py` pairs image and mask patches by filename. Mask content is not used to construct the statistical report and is not sent to the language model.

## 3. Installation

Python 3.10+ is recommended. Install the builder-specific dependencies from the repository root:

```bash
pip install -r dataset_builder/requirements.txt
```

The main VLSE-Net environment is installed separately with:

```bash
pip install -r requirements.txt
```

## 4. Language-Model Configuration

The utility uses an OpenAI-compatible API. Configure the service through environment variables or pass the model explicitly with `--model`.

Required:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`, unless `--model` is supplied

Optional:

- `OPENAI_BASE_URL`: custom or self-hosted OpenAI-compatible endpoint

Example `.env` configuration:

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.example.com
OPENAI_MODEL=<provider-specific-Qwen3.5-Plus-model-id>
```

**Paper reproduction.** The sample-specific prompts used in the reported experiments were generated with **Qwen3.5-Plus**. The exact API model identifier depends on the service provider, so the repository does not hard-code a provider-specific identifier. If neither `--model` nor `OPENAI_MODEL` is supplied, the script stops with an explicit error instead of silently switching to a different language model.

For custom datasets, another OpenAI-compatible language model may be selected explicitly. Such outputs should not be assumed to reproduce the prompts used in the paper.

## 5. Processing Pipeline

For each prepared image patch, the released pipeline performs three steps:

1. Extract image-derived statistical descriptors locally.
2. Convert the descriptors into a structured textual report.
3. Send only that report to the configured language model and save the returned description as `<image_stem>.txt`.

No ground-truth mask, manually specified pore boundary or other pixel-level annotation is used in prompt construction.

## 6. Statistical Descriptors

`compute_features.py` extracts the descriptors used by the released prompt-construction procedure.

### Geometry

- `width`
- `height`

### Color and intensity

- RGB means: `red_mean`, `green_mean`, `blue_mean`
- RGB standard deviations: `red_std`, `green_std`, `blue_std`
- Grayscale statistics: `mean_intensity`, `std_intensity`, `min_intensity`, `max_intensity`, `median_intensity`, `p10_intensity`, `p90_intensity`, `entropy`

### Structure

- `edge_density`
- `gradient_strength`

Inspect one of the example images locally with:

```bash
python dataset_builder/compute_features.py \
    --image dataset/patch_images/Boye5-1HF-ID-2_1_006-Mag-20000x-Scale-5um-Depth-3729.2m-crop_1.png
```

## 7. Prompt Construction

`prompt_builder.py` organizes the descriptors into:

1. sample identifier;
2. geometry, color/intensity and structural statistics;
3. instructions constraining the generated description.

The language model is asked to produce one concise description of approximately 15–25 words focusing on pore distribution, connectivity, shape and spatial heterogeneity. The prompt explicitly instructs the model not to infer unsupported rock type or visual details.

## 8. Batch Generation

Basic command:

```bash
python dataset_builder/build_dataset.py \
    --patch-image-dir dataset/patch_images \
    --patch-mask-dir dataset/patch_mask \
    --text-dir dataset/text
```

Common arguments:

- `--patch-image-dir`: prepared image-patch directory.
- `--patch-mask-dir`: matching mask-patch directory used for pair validation.
- `--text-dir`: output directory for generated `.txt` files.
- `--model`: explicit API model identifier; overrides `OPENAI_MODEL`.
- `--workers`: number of parallel API requests.
- `--no-resume`: regenerate instead of skipping existing text files.
- `--force`: overwrite existing outputs.
- `--error-log`: custom JSONL error-log path.
- `--status-log`: custom JSONL status-log path.
- `--max-retries`: maximum retries for a failed request.
- `--retry-delay`: base retry delay in seconds.

Example with four workers:

```bash
python dataset_builder/build_dataset.py \
    --patch-image-dir dataset/patch_images \
    --patch-mask-dir dataset/patch_mask \
    --text-dir dataset/text \
    --workers 4
```

Explicit model selection:

```bash
python dataset_builder/build_dataset.py \
    --model <provider-specific-model-id>
```

## 9. Output Files

The output directory contains:

- `*.txt`: one sample-specific description per successfully processed image;
- `status.jsonl`: generation status for each image/mask pair.

If a request fails, `errors.jsonl` records the corresponding error details. Existing `.txt` files are skipped by default; use `--force` to regenerate them.

## 10. Reproducibility Notes

- The example prompts already distributed in `dataset/text/` can be used directly for training and inference; the language-model service is not required in that case.
- To reproduce the paper's offline prompt-generation procedure, configure Qwen3.5-Plus explicitly and keep the API/model version fixed.
- Prompt generation uses only locally extracted image statistics. Raw images and masks are not sent to the language model.
- Generated text can vary if the service provider, model version or decoding implementation changes.
