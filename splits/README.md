# Split Manifest Format

The training scripts support explicit `train.txt`, `val.txt`, and `test.txt` manifests so that the exact experimental partition can be reproduced without inferring source groups from patch filenames.

For the experiments reported in the paper, the 70% / 15% / 15% partition is created at the **original-image level** before patch assignment. For DRP-317, the equivalent grouping unit is the source slice. All patches derived from the same source image or source slice must therefore remain in the same subset.

## Directory Layout

```text
splits_for_run/
├── train.txt
├── val.txt
└── test.txt
```

Each file contains one prepared patch filename per line:

```text
sample_0001.png
sample_0002.png
sample_0003.png
```

Blank lines and lines beginning with `#` are ignored.

## Validation Performed by the Training Scripts

When `--split-dir` is supplied, the scripts require that:

- `train.txt`, `val.txt`, and `test.txt` all exist and are non-empty;
- every listed sample matches a prepared image in `patch_images/`;
- no sample is duplicated within a manifest;
- the three subsets do not overlap;
- the three manifests together cover the prepared dataset exactly.

The scripts intentionally do **not** infer source-image identity from a naming convention. Source-level grouping must be established during dataset preparation, where the original-image or slice provenance is known reliably.

## Usage

```bash
python train_VLSENet.py \
    --data-root ./dataset \
    --split-dir ./splits_for_run
```

Use the same manifest directory for the U-Net baseline:

```bash
python train_unet.py \
    --data-root ./dataset \
    --split-dir ./splits_for_run
```

Each run writes a copy of the exact sample allocation to its report directory under `splits/`.

## Public Example Data

The three samples included in `dataset/` are provided only for functional testing. Running a training script without `--split-dir` uses a deterministic sample-level fallback split and does not reproduce the paper's source-level partition.
