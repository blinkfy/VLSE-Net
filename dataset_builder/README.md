# dataset_builder

`dataset_builder` is used to construct vision-language training text for core SEM pore segmentation. The pipeline does not send raw images to the language model. Instead, it first extracts structured statistical descriptors from the original image and then organizes those descriptors into a prompt that drives text generation for training or augmentation.

## 1. Directory and Scripts

### 1.1 Script Overview

- `split_images.py`: splits the original image and its corresponding mask into patches using a fixed grid.
- `compute_features.py`: extracts image-only statistical descriptors, including geometric, intensity, color, and structural information.
- `prompt_builder.py`: formats the statistical descriptors into a structured prompt suitable for LLM input.
- `llm_api.py`: calls an OpenAI-compatible API and submits only the prompt, returning a pore description.
- `build_dataset.py`: runs feature extraction, prompt construction, and text generation in batch, producing `dataset/text/*.txt`.
- `requirements.txt`: minimal dependency list.

### 1.2 Recommended Directory Layout

Default input and output directories are:

- `dataset/patch_images/`: image patches
- `dataset/patch_mask/`: mask patches
- `dataset/text/`: generated text files

Run the scripts from the repository root , for example:

```python
python dataset_builder/build_dataset.py ...
```

or

```python
python dataset_builder/build_dataset.py ...
```

## 2. Processing Pipeline

The dataset construction pipeline follows four steps:

1. Split the original image and mask into patch-level samples.
2. Extract statistical descriptors from the original image only.
3. Convert the descriptors into a paper-style structured prompt.
4. Send the prompt to the LLM to generate a pore-structure description.

Steps 2 to 4 are the core of this tool. The LLM receives structured statistics only and does not receive the raw image or mask.

## 3. Installation and Dependencies

### 3.1 Python Dependencies

Python 3.10+ is recommended. Install the dependencies with:

```python
pip install -r requirements.txt
```

### 3.2 Model Service Configuration

OpenAI-compatible API settings must be provided through environment variables. A `.env` file is recommended under `dataset_builder/`.

Required variable:

- `OPENAI_API_KEY`

Optional variables:

- `OPENAI_BASE_URL`: for self-hosted or proxy-compatible endpoints
- `OPENAI_MODEL`: default model name when no command-line override is provided

Example `.env` file:

```env
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://api.example.com
OPENAI_MODEL=gpt-5.4-mini
```

## 4. Statistical Feature Extraction

`compute_features.py` is not a segmentation script. Its purpose is to compress each original image into a compact set of statistical descriptors suitable for language-model conditioning. The current feature set is organized into three levels:

### 4.1 Geometry Level

- `width`
- `height`

These values preserve the basic spatial scale of the input patch.

### 4.2 Intensity and Color Level

- RGB channel means: `red_mean`, `green_mean`, `blue_mean`
- RGB channel standard deviations: `red_std`, `green_std`, `blue_std`
- Grayscale statistics: `mean_intensity`, `std_intensity`, `min_intensity`, `max_intensity`, `median_intensity`, `p10_intensity`, `p90_intensity`, `entropy`

This level captures the overall brightness distribution, dispersion, and information entropy of the image, helping the model perceive global contrast between pore regions and background regions.

### 4.3 Structural Level

- `edge_density`
- `gradient_strength`

This level characterizes local texture variation, edge richness, and directional change, allowing the model to form an abstract impression of pore-structure complexity.

### 4.4 Usage

```python
python dataset_builder/compute_features.py --image dataset/patch_images/example_1.png
```

The output is a JSON-formatted feature summary that can be inspected for a single sample.

## 5. Prompt Construction Principles

`prompt_builder.py` organizes the descriptors into a three-part structure:

1. Sample identifier
2. Hierarchical statistical summary
3. Text-generation constraints

The prompt is written in a paper-style format so that the LLM generates the description based only on statistical information rather than the image itself. The key constraints are:

- generate text only from structured statistics
- do not display or reference the image or mask
- do not infer rock type when lithology is not provided
- focus on pore distribution, pore connectivity, pore shape, and spatial heterogeneity

## 6. Batch Text Dataset Generation

### 6.1 Basic Command

```python
python dataset_builder/build_dataset.py --patch-image-dir dataset/patch_images --patch-mask-dir dataset/patch_mask --text-dir dataset/text
```

### 6.2 Common Arguments

- `--patch-image-dir`: patch image directory
- `--patch-mask-dir`: patch mask directory
- `--text-dir`: output text directory
- `--model`: override the default model name
- `--workers`: number of parallel workers
- `--no-resume`: disable resume mode
- `--force`: overwrite existing outputs
- `--error-log`: error log path
- `--status-log`: status log path
- `--max-retries`: maximum number of retries for a single sample
- `--retry-delay`: base retry backoff in seconds

### 6.3 Examples

Run with 4 workers:

```python
python dataset_builder/build_dataset.py --patch-image-dir dataset/patch_images --patch-mask-dir dataset/patch_mask --text-dir dataset/text --workers 4
```

Run with higher concurrency and more retries:

```python
python dataset_builder/build_dataset.py --patch-image-dir DRP-317/patch_images --patch-mask-dir DRP-317/patch_mask --text-dir DRP-317/text --workers 16 --max-retries 5 --retry-delay 2
```

Force regeneration:

```python
python dataset_builder/build_dataset.py --patch-image-dir dataset/patch_images --patch-mask-dir dataset/patch_mask --text-dir dataset/text --workers 4 --force
```

Explicitly switch models:

```python
python dataset_builder/build_dataset.py --model gpt-4o-mini
```

## 7. Output Files

After a successful run, the `text_dir` will contain:

- `*.txt`: pore-description text for each patch
- `status.jsonl`: per-sample status records, including `generated-text`, `skipped`, and `failed`
- `errors.jsonl`: error information for failed samples

When resume mode is enabled, existing text files are skipped and will not be regenerated.

## 8. Design Notes

### 8.1 Why Only Statistics Are Sent

This release of `dataset_builder` is intended for paper-associated code publication, so reproducibility, interpretability, and data usage boundaries are emphasized. The LLM input is therefore restricted to structured statistical information, and raw images or masks are not uploaded.

### 8.2 Why Hierarchical Features Are Used

The hierarchical design breaks image information into three levels that are easier for a language model to consume:

- geometry provides a scale anchor
- intensity and color provide global distribution constraints
- structure provides texture and directional cues

This makes the prompt more stable and more consistent with a paper-style, statistics-driven text-generation pipeline.

### 8.3 Intended Use of Generated Text

The generated descriptions can be used for:

- vision-language joint training
- ablation studies on text-prior injection
- documenting the semantic calibration module in the paper

## 9. Reproducibility Suggestions

- Test `compute_features.py` and `build_dataset.py` on a small subset first.
- Start with a small `--workers` value and increase it gradually.
- If the API returns rate limits or transient failures, increase `--max-retries` and `--retry-delay`.
- For strict reproduction, fix `OPENAI_MODEL` and the API service version.

## 10. Notes

- This tool only handles text dataset construction; it does not include evaluation logic for the image-splitting step itself.
- Text quality depends heavily on the statistical features and prompt design. If you change the feature set later, update `prompt_builder.py` and this README accordingly.
