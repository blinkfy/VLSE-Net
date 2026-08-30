from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path
    prompt: str


class PoreDataset(Dataset):
    """SEM pore-segmentation dataset with optional sample-specific text prompts."""

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        text_dir: Path | None = None,
        image_size: int = 512,
        require_text: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.text_dir = Path(text_dir) if text_dir is not None else None
        self.image_size = int(image_size)
        self.require_text = bool(require_text)

        if not self.image_dir.is_dir():
            raise NotADirectoryError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise NotADirectoryError(f"Mask directory not found: {self.mask_dir}")
        if self.require_text and (self.text_dir is None or not self.text_dir.is_dir()):
            raise NotADirectoryError(
                f"Text directory is required for VLSE-Net training: {self.text_dir}"
            )

        image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        mask_map = {
            path.name: path
            for path in self.mask_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        }
        text_map: dict[str, Path] = {}
        if self.text_dir is not None and self.text_dir.is_dir():
            text_map = {
                path.stem: path
                for path in self.text_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".txt"
            }

        missing_masks: list[str] = []
        missing_prompts: list[str] = []
        self.samples: list[Sample] = []

        for image_path in image_paths:
            mask_path = mask_map.get(image_path.name)
            if mask_path is None:
                missing_masks.append(image_path.name)
                continue

            prompt = ""
            text_path = text_map.get(image_path.stem)
            if text_path is not None:
                prompt = text_path.read_text(encoding="utf-8").strip()

            if self.require_text and not prompt:
                missing_prompts.append(image_path.name)
                continue

            self.samples.append(
                Sample(
                    image_path=image_path,
                    mask_path=mask_path,
                    prompt=prompt,
                )
            )

        if missing_masks:
            preview = ", ".join(missing_masks[:5])
            raise FileNotFoundError(
                f"{len(missing_masks)} image(s) have no matching mask. "
                f"Examples: {preview}"
            )
        if missing_prompts:
            preview = ", ".join(missing_prompts[:5])
            raise FileNotFoundError(
                f"{len(missing_prompts)} image(s) have no non-empty matching text prompt. "
                f"Examples: {preview}"
            )
        if not self.samples:
            raise RuntimeError(
                f"No matched samples found in image_dir={self.image_dir}, "
                f"mask_dir={self.mask_dir}."
            )

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )
        self.mask_resize = transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        with Image.open(sample.image_path) as image:
            image_tensor = self.image_transform(image.convert("RGB"))

        with Image.open(sample.mask_path) as mask:
            resized_mask = self.mask_resize(mask.convert("L"))
        mask_array = np.asarray(resized_mask, dtype=np.float32)
        if mask_array.max(initial=0.0) > 1.0:
            mask_array = mask_array / 255.0
        mask_array = (mask_array >= 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        return image_tensor, mask_tensor, sample.prompt, sample.image_path.name


def collate_fn(batch):
    images, masks, prompts, names = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.stack(masks, dim=0),
        list(prompts),
        list(names),
    )


def _read_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    entries: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)

    if not entries:
        raise ValueError(f"Split manifest is empty: {path}")
    return entries


def _resolve_manifest_entries(dataset: PoreDataset, entries: Iterable[str], label: str) -> list[int]:
    name_to_index = {sample.image_path.name: i for i, sample in enumerate(dataset.samples)}
    stem_to_indices: dict[str, list[int]] = {}
    for i, sample in enumerate(dataset.samples):
        stem_to_indices.setdefault(sample.image_path.stem, []).append(i)

    resolved: list[int] = []
    seen: set[int] = set()
    missing: list[str] = []

    for entry in entries:
        name = Path(entry).name
        index = name_to_index.get(name)

        if index is None:
            stem = Path(name).stem
            candidates = stem_to_indices.get(stem, [])
            if len(candidates) == 1:
                index = candidates[0]
            elif len(candidates) > 1:
                raise ValueError(
                    f"Ambiguous stem '{stem}' in {label} manifest; use the full filename."
                )

        if index is None:
            missing.append(entry)
            continue
        if index in seen:
            raise ValueError(f"Duplicate sample '{entry}' in {label} manifest.")

        seen.add(index)
        resolved.append(index)

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} entrie(s) in {label} manifest do not match the dataset. "
            f"Examples: {preview}"
        )
    return resolved


def build_dataset_splits(
    dataset: PoreDataset,
    split_dir: Path | None,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, Subset], str]:
    """Create train/val/test subsets.

    For paper reproduction, ``split_dir`` should contain train.txt, val.txt and
    test.txt produced after source-level partitioning. Each file lists the patch
    filenames assigned to that subset.

    If no manifest directory is supplied, a deterministic sample-level split is
    created only as a convenient functional-test fallback.
    """

    if split_dir is not None:
        split_dir = Path(split_dir)
        train_indices = _resolve_manifest_entries(
            dataset, _read_manifest(split_dir / "train.txt"), "train"
        )
        val_indices = _resolve_manifest_entries(
            dataset, _read_manifest(split_dir / "val.txt"), "val"
        )
        test_indices = _resolve_manifest_entries(
            dataset, _read_manifest(split_dir / "test.txt"), "test"
        )

        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)

        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise ValueError("Split manifests overlap; each sample must belong to one subset only.")

        assigned = train_set | val_set | test_set
        expected = set(range(len(dataset)))
        if assigned != expected:
            missing = sorted(expected - assigned)
            extra = sorted(assigned - expected)
            detail = []
            if missing:
                preview = ", ".join(dataset.samples[i].image_path.name for i in missing[:5])
                detail.append(f"{len(missing)} unassigned sample(s), e.g. {preview}")
            if extra:
                detail.append(f"{len(extra)} invalid assigned index(es)")
            raise ValueError(
                "Split manifests must cover the dataset exactly: " + "; ".join(detail)
            )

        subsets = {
            "train": Subset(dataset, train_indices),
            "val": Subset(dataset, val_indices),
            "test": Subset(dataset, test_indices),
        }
        return subsets, "manifest"

    if len(dataset) < 3:
        raise ValueError(
            "At least three samples are required for the fallback train/val/test split."
        )
    if not (0.0 < val_ratio < 1.0 and 0.0 < test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must both be in (0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be smaller than 1.")

    n = len(dataset)
    val_len = max(1, int(round(n * val_ratio)))
    test_len = max(1, int(round(n * test_ratio)))
    train_len = n - val_len - test_len

    while train_len < 1:
        if val_len >= test_len and val_len > 1:
            val_len -= 1
        elif test_len > 1:
            test_len -= 1
        else:
            raise ValueError("Dataset is too small to create non-empty train/val/test subsets.")
        train_len = n - val_len - test_len

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator).tolist()
    train_indices = permutation[:train_len]
    val_indices = permutation[train_len : train_len + val_len]
    test_indices = permutation[train_len + val_len :]

    subsets = {
        "train": Subset(dataset, train_indices),
        "val": Subset(dataset, val_indices),
        "test": Subset(dataset, test_indices),
    }
    return subsets, "deterministic-sample-level"


def write_split_manifests(
    dataset: PoreDataset,
    subsets: dict[str, Subset],
    output_dir: Path,
) -> None:
    """Write the exact sample allocation used by a run."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label in ("train", "val", "test"):
        subset = subsets[label]
        names = [dataset.samples[int(i)].image_path.name for i in subset.indices]
        (output_dir / f"{label}.txt").write_text(
            "\n".join(names) + "\n",
            encoding="utf-8",
        )
