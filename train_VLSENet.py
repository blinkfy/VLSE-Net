from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import amp
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("VLSENet_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    # Ensure the module is registered to avoid dataclasses edge-cases when loaded dynamically.
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class Sample:
    image_path: Path
    mask_path: Path
    prompt: str


class PoreSegmentationWithTextDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        text_dir: Path | None = None,
        image_size: int = 224,
        prompt_templates: Sequence[str] | None = None,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.text_dir = text_dir
        self.image_size = image_size
        self.prompt_templates = list(prompt_templates or ["pore", "rock pore", "microscopic pore"])

        image_files = sorted([p for p in image_dir.iterdir() if p.is_file()])
        mask_files = {p.name: p for p in mask_dir.iterdir() if p.is_file()}
        text_files = {}
        if text_dir is not None and text_dir.exists():
            text_files = {p.stem: p for p in text_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

        self.samples: List[Sample] = []
        for image_path in image_files:
            mask_path = mask_files.get(image_path.name)
            if mask_path is None:
                continue

            prompt = None
            text_path = text_files.get(image_path.stem)
            if text_path is not None:
                text_content = text_path.read_text(encoding="utf-8").strip()
                if text_content:
                    prompt = text_content

            if prompt is None:
                prompt = self.prompt_templates[len(self.samples) % len(self.prompt_templates)]

            self.samples.append(Sample(image_path=image_path, mask_path=mask_path, prompt=prompt))

        if not self.samples:
            raise RuntimeError(f"No matched image/mask pairs found in {image_dir} and {mask_dir}.")

        image_ops = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        self.image_transform = transforms.Compose(image_ops)
        self.mask_resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            image = img.convert("RGB")
        return self.image_transform(image)

    def _load_mask(self, path: Path) -> torch.Tensor:
        with Image.open(path) as m:
            mask = m.convert("L")
        mask = self.mask_resize(mask)
        mask_np = np.array(mask, dtype=np.float32)
        if mask_np.max() > 1:
            mask_np = mask_np / 255.0
        mask_np = (mask_np >= 0.5).astype(np.float32)
        return torch.from_numpy(mask_np).unsqueeze(0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        sample = self.samples[index]
        image = self._load_rgb(sample.image_path)
        mask = self._load_mask(sample.mask_path)
        return image, mask, sample.prompt, sample.image_path.name


def collate_fn(batch):
    images, masks, prompts, names = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(masks, dim=0), list(prompts), list(names)


class TrainAugmentedSubset(Dataset):
    """Apply paired augmentation to a subset while keeping val/test path untouched."""

    def __init__(
        self,
        subset: Dataset,
        enable_augmentation: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.2,
        rotate_deg: float = 12.0,
        affine_translate: float = 0.04,
        affine_scale_min: float = 0.95,
        affine_scale_max: float = 1.05,
        color_jitter: float = 0.12,
        noise_std: float = 0.015,
        blur_prob: float = 0.1,
    ) -> None:
        self.subset = subset
        self.enable_augmentation = bool(enable_augmentation)
        self.hflip_prob = float(max(0.0, min(1.0, hflip_prob)))
        self.vflip_prob = float(max(0.0, min(1.0, vflip_prob)))
        self.rotate_deg = float(max(0.0, rotate_deg))
        self.affine_translate = float(max(0.0, min(0.3, affine_translate)))
        self.affine_scale_min = float(max(0.5, affine_scale_min))
        self.affine_scale_max = float(max(self.affine_scale_min, affine_scale_max))
        self.color_jitter = float(max(0.0, color_jitter))
        self.noise_std = float(max(0.0, noise_std))
        self.blur_prob = float(max(0.0, min(1.0, blur_prob)))

    def __len__(self) -> int:
        return len(self.subset)

    def _apply_paired_geom(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < self.vflip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if self.rotate_deg > 0.0:
            angle = random.uniform(-self.rotate_deg, self.rotate_deg)
            image = TF.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
            mask = TF.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST, fill=0.0)

        if self.affine_translate > 0.0 or self.affine_scale_max > self.affine_scale_min:
            max_dx = int(round(self.affine_translate * image.shape[-1]))
            max_dy = int(round(self.affine_translate * image.shape[-2]))
            translate = (
                random.randint(-max_dx, max_dx) if max_dx > 0 else 0,
                random.randint(-max_dy, max_dy) if max_dy > 0 else 0,
            )
            scale = random.uniform(self.affine_scale_min, self.affine_scale_max)
            image = TF.affine(
                image,
                angle=0.0,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            mask = TF.affine(
                mask,
                angle=0.0,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )
        return image, mask

    def _apply_image_only(self, image: torch.Tensor) -> torch.Tensor:
        if self.color_jitter > 0.0:
            b = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            c = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            s = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            h = random.uniform(-self.color_jitter * 0.25, self.color_jitter * 0.25)
            image = TF.adjust_brightness(image, b)
            image = TF.adjust_contrast(image, c)
            image = TF.adjust_saturation(image, s)
            image = TF.adjust_hue(image, h)

        if self.blur_prob > 0.0 and random.random() < self.blur_prob:
            image = TF.gaussian_blur(image, kernel_size=[3, 3], sigma=[0.1, 1.2])

        if self.noise_std > 0.0:
            image = image + torch.randn_like(image) * self.noise_std

        return image.clamp(0.0, 1.0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        image, mask, prompt, name = self.subset[index]
        if self.enable_augmentation:
            image, mask = self._apply_paired_geom(image, mask)
            image = self._apply_image_only(image)
            mask = (mask >= 0.5).float()
        return image, mask, prompt, name


def tensor_to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu()
    image_np = image.permute(1, 2, 0).numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    return image_np


def mask_to_uint8(mask_tensor: torch.Tensor) -> np.ndarray:
    mask_np = mask_tensor.detach().cpu().squeeze().numpy()
    mask_np = (mask_np >= 0.5).astype(np.uint8) * 255
    return mask_np


def save_visualizations(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    names: List[str],
    save_dir: Path,
    epoch: int,
    max_samples: int,
) -> int:
    save_dir.mkdir(parents=True, exist_ok=True)
    preds = (torch.sigmoid(logits) >= 0.5).float()
    num_saved = 0

    for i in range(min(images.shape[0], max_samples)):
        image_np = tensor_to_uint8_image(images[i])
        gt_np = mask_to_uint8(masks[i])
        pred_np = mask_to_uint8(preds[i])

        overlay_pred = image_np.copy()
        overlay_gt = image_np.copy()
        overlay_pred[pred_np > 0] = [255, 255, 0]
        overlay_gt[gt_np > 0] = [255, 0, 0]

        stem = Path(names[i]).stem
        Image.fromarray(image_np).save(save_dir / f"epoch{epoch:03d}_{stem}_image.png")
        Image.fromarray(gt_np).save(save_dir / f"epoch{epoch:03d}_{stem}_gt.png")
        Image.fromarray(pred_np).save(save_dir / f"epoch{epoch:03d}_{stem}_pred.png")
        Image.fromarray(overlay_gt).save(save_dir / f"epoch{epoch:03d}_{stem}_overlay_gt.png")
        Image.fromarray(overlay_pred).save(save_dir / f"epoch{epoch:03d}_{stem}_overlay_pred.png")
        num_saved += 1

    return num_saved


@torch.no_grad()
def compute_batch_iou_and_dice(logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float]:
    preds = (torch.sigmoid(logits) >= threshold).float()
    masks = (masks >= 0.5).float()
    preds = preds.flatten(1)
    masks = masks.flatten(1)
    intersection = (preds * masks).sum(dim=1)
    union = preds.sum(dim=1) + masks.sum(dim=1) - intersection
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    dice = ((2 * intersection + 1e-6) / (preds.sum(dim=1) + masks.sum(dim=1) + 1e-6)).mean().item()
    return iou, dice


@torch.no_grad()
def compute_batch_precision_recall_and_porosity_error(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    preds = (torch.sigmoid(logits) >= threshold).float()
    masks = (masks >= 0.5).float()

    preds = preds.flatten(1)
    masks = masks.flatten(1)

    tp = (preds * masks).sum(dim=1)
    fp = (preds * (1.0 - masks)).sum(dim=1)
    fn = ((1.0 - preds) * masks).sum(dim=1)

    precision = (tp / (tp + fp + 1e-6)).mean().item()
    recall = (tp / (tp + fn + 1e-6)).mean().item()
    pred_porosity = preds.mean(dim=1)
    gt_porosity = masks.mean(dim=1)
    porosity_error = (pred_porosity - gt_porosity).abs().mean().item()
    return precision, recall, porosity_error

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    use_alignment_loss: bool = False,
    alignment_weight: float = 0.0,
    save_visual_dir: Path | None = None,
    epoch: int = 0,
    total_epochs: int | None = None,
):
    model.eval()
    losses = []
    ious = []
    dices = []
    precisions = []
    recalls = []
    porosity_errors = []
    align_losses = []
    total_saved = 0

    for images, masks, prompts, names in tqdm(loader, desc="valid", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        with amp.autocast('cuda',enabled=(use_amp and device.type == "cuda")):
            if use_alignment_loss:
                outputs = model(images, prompts, region_mask=masks, return_aux=True)
                logits = outputs["logits"]
                alignment_loss = outputs.get("alignment_loss")
                loss = criterion(logits, masks)
                if alignment_loss is not None:
                    loss = loss + alignment_weight * alignment_loss
                    align_losses.append(alignment_loss.item())
            else:
                logits = model(images, prompts)
                loss = criterion(logits, masks)
        iou, dice = compute_batch_iou_and_dice(logits, masks)
        precision, recall, porosity_error = compute_batch_precision_recall_and_porosity_error(logits, masks)
        losses.append(loss.item())
        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        porosity_errors.append(porosity_error)

        # Only save a fixed number of visualization samples (80) in the last epoch.
        if save_visual_dir is not None and total_epochs is not None and epoch==total_epochs and total_saved < 80:
            remaining = 80 - total_saved
            total_saved += save_visualizations(
                images=images,
                masks=masks,
                logits=logits,
                names=names,
                save_dir=save_visual_dir,
                epoch=epoch,
                max_samples=remaining,
            )

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "porosity_error": float(np.mean(porosity_errors)) if porosity_errors else 0.0,
        "alignment_loss": float(np.mean(align_losses)) if align_losses else 0.0,
        "saved_visualizations": total_saved,
    }


@torch.no_grad()
def save_final_visualizations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_visual_dir: Path | None,
    epoch: int,
    use_alignment_loss: bool = False,
    use_amp: bool = False,
    alignment_weight: float = 0.0,
    max_samples: int = 80,
) -> int:
    if save_visual_dir is None:
        return 0
    model.eval()
    total_saved = 0
    for images, masks, prompts, names in tqdm(loader, desc="visual", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        with amp.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
            if use_alignment_loss:
                outputs = model(images, prompts, region_mask=masks, return_aux=True)
                logits = outputs["logits"]
                alignment_loss = outputs.get("alignment_loss")
                if alignment_loss is not None:
                    _ = alignment_weight * alignment_loss
            else:
                logits = model(images, prompts)
        remaining = max_samples - total_saved
        total_saved += save_visualizations(
            images=images,
            masks=masks,
            logits=logits,
            names=names,
            save_dir=save_visual_dir,
            epoch=epoch,
            max_samples=remaining,
        )
        if total_saved >= max_samples:
            break
    return total_saved


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    scaler: amp.GradScaler | None = None,
    use_alignment_loss: bool = False,
    alignment_weight: float = 0.0,
):
    model.train()
    losses = []
    ious = []
    dices = []
    align_losses = []

    for images, masks, prompts, names in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast('cuda',enabled=(scaler is not None and device.type == "cuda")):
            if use_alignment_loss:
                outputs = model(images, prompts, region_mask=masks, return_aux=True)
                logits = outputs["logits"]
                alignment_loss = outputs.get("alignment_loss")
                loss = criterion(logits, masks)
                if alignment_loss is not None:
                    loss = loss + alignment_weight * alignment_loss
                    align_losses.append(alignment_loss.item())
            else:
                logits = model(images, prompts)
                loss = criterion(logits, masks)

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        iou, dice = compute_batch_iou_and_dice(logits.detach(), masks)
        losses.append(loss.item())
        ious.append(iou)
        dices.append(dice)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "alignment_loss": float(np.mean(align_losses)) if align_losses else 0.0,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_repro_lite(seed: int) -> None:
    set_seed(seed)
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def sanitize_args_for_checkpoint(args_dict: dict) -> dict:
    sanitized = {}
    for key, value in args_dict.items():
        if isinstance(value, Path):
            sanitized[key] = str(value)
        else:
            sanitized[key] = value
    return sanitized


def _torch_load_compat(checkpoint_path: Path, map_location: object, weights_only: bool | None = None):
    load_kwargs = {"map_location": map_location}
    if weights_only is not None:
        load_kwargs["weights_only"] = bool(weights_only)
    try:
        return torch.load(checkpoint_path, **load_kwargs)
    except TypeError:
        # Compatibility for older torch versions that do not accept weights_only.
        load_kwargs.pop("weights_only", None)
        return torch.load(checkpoint_path, **load_kwargs)


def load_checkpoint_with_fallback(checkpoint_path: Path, map_location: object):
    attempts: List[str] = []
    for weights_only in (False, None, True):
        mode = "default" if weights_only is None else f"weights_only={weights_only}"
        try:
            ckpt = _torch_load_compat(checkpoint_path, map_location=map_location, weights_only=weights_only)
            if weights_only is not False:
                print(f"Checkpoint load fallback succeeded using {mode}: {checkpoint_path}")
            return ckpt
        except Exception as exc:
            attempts.append(f"{mode}: {exc}")

    detail = " | ".join(attempts)
    raise RuntimeError(
        f"Failed to load checkpoint '{checkpoint_path}'. The file may be corrupted or not a PyTorch checkpoint. "
        f"Tried modes -> {detail}"
    )


def resolve_resume_checkpoint_with_candidates(resume_path: Path, map_location: object):
    candidates: List[Path] = []
    seen: set[str] = set()

    def add_candidate(p: Path) -> None:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            return
        seen.add(key)
        candidates.append(p)

    add_candidate(resume_path)
    if resume_path.parent.exists():
        add_candidate(resume_path.parent / "last_text_guided_unet.pt")
        add_candidate(resume_path.parent / "best_text_guided_unet.pt")
        for p in sorted(resume_path.parent.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
            add_candidate(p)

    errors: List[str] = []
    for cand in candidates:
        if not cand.exists():
            errors.append(f"{cand}: file not found")
            continue
        try:
            ckpt = load_checkpoint_with_fallback(cand, map_location=map_location)
            if cand != resume_path:
                print(f"Warning: resume checkpoint unavailable/corrupted, fallback to: {cand}")
            return ckpt, cand
        except Exception as exc:
            errors.append(f"{cand}: {exc}")

    detail = " | ".join(errors)
    raise RuntimeError(
        "Failed to resume training because no valid checkpoint could be loaded. "
        f"Checked candidates -> {detail}"
    )


def atomic_torch_save(obj: dict, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, target_path)


def save_training_logs(log_records: List[dict], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "training_log.json"
    csv_path = save_dir / "training_log.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, indent=2, ensure_ascii=False)

    if log_records:
        headers = list(log_records[0].keys())
        lines = [",".join(headers)]
        for row in log_records:
            values = [str(row.get(h, "")) for h in headers]
            lines.append(",".join(values))
        csv_path.write_text("\n".join(lines), encoding="utf-8")


def update_run_report(report_dir: Path, summary: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    def _to_json_serializable(o):
        # Path -> str
        if isinstance(o, Path):
            return str(o)
        # numpy scalars / arrays
        try:
            import numpy as _np
        except Exception:
            _np = None
        if _np is not None:
            if isinstance(o, _np.generic):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()
        # torch tensors
        try:
            import torch as _torch
        except Exception:
            _torch = None
        if _torch is not None and isinstance(o, _torch.Tensor):
            if o.numel() == 1:
                return o.item()
            return o.detach().cpu().tolist()
        # dict / list / tuple
        if isinstance(o, dict):
            return {k: _to_json_serializable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_json_serializable(v) for v in o]
        # fallback
        return o

    safe_summary = _to_json_serializable(summary)
    (report_dir / "summary.json").write_text(json.dumps(safe_summary, indent=2, ensure_ascii=False), encoding="utf-8")


def count_model_parameters(model: nn.Module) -> dict:
    target = model.module if hasattr(model, "module") else model
    total = int(sum(p.numel() for p in target.parameters()))
    trainable = int(sum(p.numel() for p in target.parameters() if p.requires_grad))
    return {
        "total": total,
        "trainable": trainable,
        "frozen": int(total - trainable),
    }


def generate_training_curves(plot_script: Path, log_csv: Path, output_dir: Path, model_name: str) -> None:
    if not plot_script.exists() or not log_csv.exists():
        return
    subprocess.run([
        "python",
        str(plot_script),
        "--log-csv",
        str(log_csv),
        "--output-dir",
        str(output_dir),
        "--model-name",
        model_name,
    ], check=True)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Text-Guided U-Net on pore segmentation dataset")
    parser.add_argument("--data-root", type=Path, default=Path("./dataset"))
    parser.add_argument("--image-dir-name", type=str, default="patch_images")
    parser.add_argument("--mask-dir-name", type=str, default="patch_mask")
    parser.add_argument("--text-dir-name", type=str, default="text")
    parser.add_argument("--module-path", type=Path, default=Path(__file__).resolve().parent / "VLSENet.py")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["none", "plateau"], help="Learning rate scheduler")
    parser.add_argument("--scheduler-patience", type=int, default=2, help="Plateau scheduler patience (epochs)")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="Plateau scheduler lr decay factor")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repro-lite", dest="repro_lite", action="store_true", default=True, help="More reproducible results with minimal impact (disables cudnn benchmark).")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--use-augmentation", action="store_true", help="Enable train-only paired data augmentation for small datasets")
    parser.add_argument("--aug-hflip-prob", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--aug-vflip-prob", type=float, default=0.2, help="Vertical flip probability")
    parser.add_argument("--aug-rotate-deg", type=float, default=12.0, help="Max absolute rotation degree")
    parser.add_argument("--aug-affine-translate", type=float, default=0.04, help="Max translation ratio for random affine")
    parser.add_argument("--aug-affine-scale-min", type=float, default=0.95, help="Min scale for random affine")
    parser.add_argument("--aug-affine-scale-max", type=float, default=1.05, help="Max scale for random affine")
    parser.add_argument("--aug-color-jitter", type=float, default=0.12, help="Color jitter strength")
    parser.add_argument("--aug-noise-std", type=float, default=0.015, help="Std of additive Gaussian noise")
    parser.add_argument("--aug-blur-prob", type=float, default=0.1, help="Gaussian blur probability")
    parser.add_argument("--use-region-alignment", action="store_true", default=True, help="Enable lightweight region-text auxiliary alignment loss")
    parser.add_argument("--alignment-weight", type=float, default=0.07, help="Weight for auxiliary region-text alignment loss")
    # cross-attention spatial modeling
    parser.add_argument("--text-spatial-mode", type=str, default="cross_attention", choices=["none", "cross_attention"], help="Text spatial modeling mode: none or cross_attention")
    # gated skip connections
    parser.add_argument("--use-skip-attention", action="store_true", default=True, help="Enable lightweight channel+spatial gated skip connections in decoder")
    parser.add_argument("--multi-scale-fusion", action="store_true", default=True, help="Enable multi-scale text fusion by conditioning encoder skip features")
    parser.add_argument("--use-decoder-text-adapter", action="store_true", default=True, help="Enable lightweight DecoderTextAdapter residual refinement in decoder")
    parser.add_argument("--use-directional-refine", action="store_true", default=True, help="Enable deep visual directional refine (skip4 + bottleneck)")
    parser.add_argument("--use-bottleneck-context", action="store_true", default=True, help="Enable lightweight bottleneck multi-scale context")
    parser.add_argument("--use-decoder-directional-refine", action="store_true", default=True, help="Enable dec4 directional refine after first decoder fusion")
    parser.add_argument("--feature-renorm",type=str,default="match_input_std",
        choices=["none", "match_input_stats", "match_input_std"],help="Feature re-normalization after FiLM/text conditioning to reduce distribution drift")
    parser.add_argument("--visualize-val", action="store_true", default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--report-every-epoch", action="store_true", default=True)
    parser.add_argument("--plot-curves-every-epoch", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", default=True, help="Enable mixed precision (torch.cuda.amp) when CUDA is available")
    return parser


def parse_args() -> argparse.Namespace:
    return get_parser().parse_args()


def build_optimizer_param_groups(model: nn.Module, base_lr: float):
    clip_params = []
    other_params = []
    clip_param_ids = set()

    # Collect CLIP parameters regardless of current requires_grad to allow dynamic unfreeze.
    if hasattr(model, "text_encoder") and hasattr(model.text_encoder, "clip_model"):
        for param in model.text_encoder.clip_model.parameters():
            clip_params.append(param)
            clip_param_ids.add(id(param))

    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in clip_param_ids:
            continue
        other_params.append(param)

    param_groups = []
    if clip_params:
        param_groups.append({"params": clip_params, "lr": base_lr, "name": "clip"})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "other"})
    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")
    return param_groups


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    if args.repro_lite:
        configure_repro_lite(args.seed)
        repro_mode = "repro-lite"
    else:
        set_seed(args.seed)
        repro_mode = "default"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_unique_report_dir(base: Path) -> Path:
        if not base.exists():
            return base
        suffix = 1
        while True:
            cand = Path(str(base) + f"_{suffix}")
            if not cand.exists():
                return cand
            suffix += 1

    base_report_dir = Path(__file__).resolve().parent / "reports" / "latest_run_text_guided_unet"
    report_dir = make_unique_report_dir(base_report_dir)

    image_dir = args.data_root / args.image_dir_name
    mask_dir = args.data_root / args.mask_dir_name
    text_dir = args.data_root / args.text_dir_name

    report_dir.mkdir(parents=True, exist_ok=True)
    report_visual_dir = report_dir / "visualizations"
    report_curves_dir = report_dir / "training_curves"
    if args.visualize_val:
        report_visual_dir.mkdir(parents=True, exist_ok=True)
    report_curves_dir.mkdir(parents=True, exist_ok=True)

    module = load_module(args.module_path)
    dataset = PoreSegmentationWithTextDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        text_dir=text_dir,
        image_size=args.image_size,
        prompt_templates=["pore", "rock pore"],
    )

    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    if train_len <= 0:
        raise RuntimeError("Dataset too small after validation split.")

    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))

    use_pin_memory = args.pin_memory or device.type == "cuda"
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": use_pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_dataset = TrainAugmentedSubset(
        train_set,
        enable_augmentation=args.use_augmentation,
        hflip_prob=args.aug_hflip_prob,
        vflip_prob=args.aug_vflip_prob,
        rotate_deg=args.aug_rotate_deg,
        affine_translate=args.aug_affine_translate,
        affine_scale_min=args.aug_affine_scale_min,
        affine_scale_max=args.aug_affine_scale_max,
        color_jitter=args.aug_color_jitter,
        noise_std=args.aug_noise_std,
        blur_prob=args.aug_blur_prob,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    tg_kwargs = {
        "num_classes": 1,
        "freeze_text": True,
        "alignment_proj_dim": 256,
        "text_spatial_mode": args.text_spatial_mode,
        "use_skip_attention": bool(args.use_skip_attention),
        "multi_scale_fusion": bool(args.multi_scale_fusion),
        "use_decoder_text_adapter": bool(args.use_decoder_text_adapter),
        "use_directional_refine": bool(args.use_directional_refine),
        "use_bottleneck_context": bool(args.use_bottleneck_context),
        "use_decoder_directional_refine": bool(args.use_decoder_directional_refine),
        "feature_renorm": str(args.feature_renorm),
    }
    sig = inspect.signature(module.VLSENet)
    tg_kwargs = {k: v for k, v in tg_kwargs.items() if k in sig.parameters}
    model = module.VLSENet(**tg_kwargs).to(device)
    param_stats = count_model_parameters(model)
    criterion = module.DiceBCELoss()
    optimizer_param_groups = build_optimizer_param_groups(model, base_lr=args.lr)
    optimizer = AdamW(optimizer_param_groups, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )

    best_dice = -1.0
    log_records: List[dict] = []
    epochs_without_improvement = 0
    last_report_summary: dict | None = None

    print(f"Dataset size: {len(dataset)} | train: {len(train_set)} | val: {len(val_set)}")
    print(f"Using device: {device}")
    print(f"Repro mode: {repro_mode}")
    print(f"Report dir: {report_dir}")
    print(
        "Augmentation | "
        f"enabled={bool(args.use_augmentation)} hflip={args.aug_hflip_prob} vflip={args.aug_vflip_prob} "
        f"rotate={args.aug_rotate_deg} translate={args.aug_affine_translate} "
        f"scale=[{args.aug_affine_scale_min},{args.aug_affine_scale_max}] "
        f"jitter={args.aug_color_jitter} noise_std={args.aug_noise_std} blur_prob={args.aug_blur_prob}"
    )
    print(
        "DataLoader config | "
        f"workers={args.num_workers} pin_memory={use_pin_memory} "
        f"persistent_workers={bool(args.persistent_workers and args.num_workers > 0)}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        scaler = None
        if args.amp and device.type == "cuda":
            scaler = amp.GradScaler()
            # Some code paths expect optimizer to carry a reference to the scaler
            try:
                setattr(optimizer, "_amp_scaler", scaler)
            except Exception:
                print("Warning: failed to set _amp_scaler attribute on optimizer; AMP may not work properly in some code paths.")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            use_alignment_loss=args.use_region_alignment,
            alignment_weight=args.alignment_weight,
        )
        after_train = time.perf_counter()
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=args.amp,
            use_alignment_loss=args.use_region_alignment,
            alignment_weight=args.alignment_weight,
            save_visual_dir=report_visual_dir if args.visualize_val else None,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        after_val = time.perf_counter()

        if scheduler is not None:
            scheduler.step(val_metrics["dice"])

        last_ckpt_path = report_dir / "last_text_guided_unet.pt"
        atomic_torch_save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "args": sanitize_args_for_checkpoint(vars(args)),
        }, last_ckpt_path)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_metrics['loss']:.4f} dice {train_metrics['dice']:.4f} iou {train_metrics['iou']:.4f} align {train_metrics['alignment_loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} dice {val_metrics['dice']:.4f} iou {val_metrics['iou']:.4f} prec {val_metrics['precision']:.4f} rec {val_metrics['recall']:.4f} "
            f"por_err {val_metrics['porosity_error']:.4f} align {val_metrics['alignment_loss']:.4f} | "
            f"timing train {after_train - epoch_start:.1f}s val {after_val - after_train:.1f}s"
        )

        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_dice": round(train_metrics["dice"], 6),
            "train_iou": round(train_metrics["iou"], 6),
            "train_alignment_loss": round(train_metrics["alignment_loss"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_dice": round(val_metrics["dice"], 6),
            "val_iou": round(val_metrics["iou"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "val_porosity_error": round(val_metrics["porosity_error"], 6),
            "val_alignment_loss": round(val_metrics["alignment_loss"], 6),
            "saved_visualizations": int(val_metrics.get("saved_visualizations", 0)),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        save_training_logs(log_records, report_dir)

        if val_metrics["dice"] > best_dice + args.min_delta:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            ckpt_path = report_dir / "best_text_guided_unet.pt"
            atomic_torch_save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "args": sanitize_args_for_checkpoint(vars(args)),
            }, ckpt_path)
        else:
            epochs_without_improvement += 1

        if args.plot_curves_every_epoch:
            plot_script = Path(__file__).resolve().parent / "plot_training_curves.py"
            try:
                generate_training_curves(
                    plot_script=plot_script,
                    log_csv=report_dir / "training_log.csv",
                    output_dir=report_curves_dir,
                    model_name="Text-Guided U-Net",
                )
            except Exception as exc:
                print(f"Warning: failed to generate training curves automatically: {exc}")
        report_summary = {
            "best_dice": float(best_dice),
            "last_epoch": int(epoch),
            "epochs_without_improvement": int(epochs_without_improvement),
            "device": str(device),
            "repro_mode": repro_mode,
            "dataset_size": int(len(dataset)),
            "train_size": int(len(train_set)),
            "val_size": int(len(val_set)),
            "current_lr": float(optimizer.param_groups[0]["lr"]),
            "scheduler": str(args.scheduler),
            "scheduler_factor": float(args.scheduler_factor),
            "scheduler_patience": int(args.scheduler_patience),
            "use_augmentation": bool(args.use_augmentation),
            "aug_hflip_prob": float(args.aug_hflip_prob),
            "aug_vflip_prob": float(args.aug_vflip_prob),
            "aug_rotate_deg": float(args.aug_rotate_deg),
            "aug_affine_translate": float(args.aug_affine_translate),
            "aug_affine_scale_min": float(args.aug_affine_scale_min),
            "aug_affine_scale_max": float(args.aug_affine_scale_max),
            "aug_color_jitter": float(args.aug_color_jitter),
            "aug_noise_std": float(args.aug_noise_std),
            "aug_blur_prob": float(args.aug_blur_prob),
            "last_metrics": log_records[-1],
            "use_region_alignment": bool(args.use_region_alignment),
            "alignment_weight": float(args.alignment_weight),
            "text_spatial_mode": str(args.text_spatial_mode),
            "use_skip_attention": bool(args.use_skip_attention),
            "multi_scale_fusion": bool(args.multi_scale_fusion),
            "use_decoder_text_adapter": bool(args.use_decoder_text_adapter),
            "use_directional_refine": bool(args.use_directional_refine),
            "use_bottleneck_context": bool(args.use_bottleneck_context),
            "use_decoder_directional_refine": bool(args.use_decoder_directional_refine),
            "feature_renorm": str(args.feature_renorm),
            "parameter_count": param_stats,
        }

        last_report_summary = report_summary

        if args.report_every_epoch:
            update_run_report(report_dir, report_summary)

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch} after {epochs_without_improvement} epochs without validation Dice improvement.")
            if args.visualize_val:
                saved = save_final_visualizations(
                    model=model,
                    loader=val_loader,
                    device=device,
                    save_visual_dir=report_visual_dir,
                    epoch=epoch,
                    use_alignment_loss=args.use_region_alignment,
                    use_amp=args.amp,
                    alignment_weight=args.alignment_weight,
                )
            break

    if last_report_summary is not None:
        update_run_report(report_dir, last_report_summary)

    print(
        "Final model stats | "
        f"params(total/trainable/frozen)={param_stats['total']}/{param_stats['trainable']}/{param_stats['frozen']} "
        f"| best_dice={best_dice:.4f}"
    )


if __name__ == "__main__":
    main()
