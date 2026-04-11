from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import random
import shutil
import subprocess
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
import torch.amp as amp
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm


def load_unet_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("pure_unet_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        probs = probs.flatten(1)
        targets = targets.flatten(1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice.mean()
        return bce + dice_loss


@dataclass
class Sample:
    image_path: Path
    mask_path: Path
    prompt: str


class PoreSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        text_dir: Path | None = None,
        image_size: int = 224,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.text_dir = text_dir
        self.image_size = image_size

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

            prompt = "pore"
            text_path = text_files.get(image_path.stem)
            if text_path is not None:
                text_content = text_path.read_text(encoding="utf-8").strip()
                if text_content:
                    prompt = text_content

            self.samples.append(Sample(image_path=image_path, mask_path=mask_path, prompt=prompt))

        if not self.samples:
            raise RuntimeError(f"No matched image/mask pairs found in {image_dir} and {mask_dir}.")

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.mask_resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        # Use context manager to ensure file handle is closed promptly (avoids file corruption/locks on Windows)
        with Image.open(path) as img:
            image = img.convert("RGB")
        return self.image_transform(image)

    def _load_mask(self, path: Path) -> torch.Tensor:
        # Use context manager to ensure file handle is closed promptly
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


def tensor_to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
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


def _binary_erosion_2d(mask01: torch.Tensor) -> torch.Tensor:
    """Binary erosion for 2D masks using max-pooling trick.

    Args:
        mask01: float tensor in {0,1}, shape (B,1,H,W)

    Returns:
        eroded float tensor in {0,1}, same shape.
    """

    if mask01.dim() != 4:
        raise ValueError(f"Expected (B,1,H,W), got {tuple(mask01.shape)}")
    # erosion(x) == 1 - dilation(1-x)
    inv = 1.0 - mask01
    dil_inv = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - dil_inv
    return (eroded >= 0.5).float()


@torch.no_grad()
def compute_batch_boundary_f1(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    threshold: float = 0.5,
    radius: int = 2,
) -> float:
    """Boundary F1 score with a pixel tolerance (radius).

    Computes boundaries via (mask - erode(mask)), then matches boundaries within
    a square dilation window of size (2*radius+1).
    """

    preds = (torch.sigmoid(logits) >= threshold).float()
    gts = (masks >= 0.5).float()

    pred_eroded = _binary_erosion_2d(preds)
    gt_eroded = _binary_erosion_2d(gts)
    pred_b = (preds - pred_eroded).clamp(min=0.0)
    gt_b = (gts - gt_eroded).clamp(min=0.0)
    pred_b = (pred_b >= 0.5).float()
    gt_b = (gt_b >= 0.5).float()

    r = max(0, int(radius))
    if r > 0:
        k = 2 * r + 1
        gt_b_dil = F.max_pool2d(gt_b, kernel_size=k, stride=1, padding=r)
        pred_b_dil = F.max_pool2d(pred_b, kernel_size=k, stride=1, padding=r)
    else:
        gt_b_dil = gt_b
        pred_b_dil = pred_b

    eps = 1e-6
    pred_sum = pred_b.flatten(1).sum(dim=1)
    gt_sum = gt_b.flatten(1).sum(dim=1)

    match_pred = (pred_b * (gt_b_dil > 0)).flatten(1).sum(dim=1)
    match_gt = (gt_b * (pred_b_dil > 0)).flatten(1).sum(dim=1)

    precision = torch.where(pred_sum > 0, match_pred / (pred_sum + eps), torch.ones_like(pred_sum))
    recall = torch.where(gt_sum > 0, match_gt / (gt_sum + eps), torch.ones_like(gt_sum))
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    # If both boundaries empty, define F1=1.
    both_empty = (pred_sum <= 0) & (gt_sum <= 0)
    f1 = torch.where(both_empty, torch.ones_like(f1), f1)
    return float(f1.mean().item())


def _zs_thinning(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning for a 2D binary image.

    Input/Output are uint8 arrays with values {0,1}.
    """

    img = (binary > 0).astype(np.uint8)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={img.shape}")

    changed = True
    h, w = img.shape
    if h < 3 or w < 3:
        return img

    def _neighbors(padded: np.ndarray):
        p2 = padded[0:-2, 1:-1]
        p3 = padded[0:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, 0:-2]
        p8 = padded[1:-1, 0:-2]
        p9 = padded[0:-2, 0:-2]
        return p2, p3, p4, p5, p6, p7, p8, p9

    while changed:
        changed = False

        padded = np.pad(img, 1, mode="constant")
        p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors(padded)
        p1 = padded[1:-1, 1:-1]

        n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        s = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
        s += ((p3 == 0) & (p4 == 1)).astype(np.uint8)
        s += ((p4 == 0) & (p5 == 1)).astype(np.uint8)
        s += ((p5 == 0) & (p6 == 1)).astype(np.uint8)
        s += ((p6 == 0) & (p7 == 1)).astype(np.uint8)
        s += ((p7 == 0) & (p8 == 1)).astype(np.uint8)
        s += ((p8 == 0) & (p9 == 1)).astype(np.uint8)
        s += ((p9 == 0) & (p2 == 1)).astype(np.uint8)

        m1 = (p2 * p4 * p6) == 0
        m2 = (p4 * p6 * p8) == 0
        to_del = (p1 == 1) & (n >= 2) & (n <= 6) & (s == 1) & m1 & m2
        if np.any(to_del):
            img[to_del] = 0
            changed = True

        padded = np.pad(img, 1, mode="constant")
        p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors(padded)
        p1 = padded[1:-1, 1:-1]

        n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        s = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
        s += ((p3 == 0) & (p4 == 1)).astype(np.uint8)
        s += ((p4 == 0) & (p5 == 1)).astype(np.uint8)
        s += ((p5 == 0) & (p6 == 1)).astype(np.uint8)
        s += ((p6 == 0) & (p7 == 1)).astype(np.uint8)
        s += ((p7 == 0) & (p8 == 1)).astype(np.uint8)
        s += ((p8 == 0) & (p9 == 1)).astype(np.uint8)
        s += ((p9 == 0) & (p2 == 1)).astype(np.uint8)

        m1 = (p2 * p4 * p8) == 0
        m2 = (p2 * p6 * p8) == 0
        to_del = (p1 == 1) & (n >= 2) & (n <= 6) & (s == 1) & m1 & m2
        if np.any(to_del):
            img[to_del] = 0
            changed = True

    return img


@torch.no_grad()
def compute_batch_cldice(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    """Compute clDice for a batch (2D).

    clDice = 2 * tprec * tsens / (tprec + tsens)
      tprec = |S(P) ∩ G| / |S(P)|
      tsens = |S(G) ∩ P| / |S(G)|
    where S(·) is a skeletonization operator.

    Notes:
    - This implementation uses a CPU Zhang-Suen thinning algorithm for skeletons.
    - Intended for evaluation, not for every training epoch.
    """

    preds = (torch.sigmoid(logits) >= threshold).detach().cpu().numpy().astype(np.uint8)
    gts = (masks >= 0.5).detach().cpu().numpy().astype(np.uint8)

    # Expect (B,1,H,W)
    if preds.ndim != 4:
        raise ValueError(f"Expected preds with shape (B,1,H,W), got {preds.shape}")

    eps = 1e-6
    scores: List[float] = []
    for i in range(preds.shape[0]):
        p = preds[i, 0]
        g = gts[i, 0]

        sp = _zs_thinning(p)
        sg = _zs_thinning(g)

        sp_sum = float(sp.sum())
        sg_sum = float(sg.sum())

        # Handle empty cases robustly.
        if sp_sum <= 0 and sg_sum <= 0:
            # If both masks are also empty, perfect.
            if float(p.sum()) <= 0 and float(g.sum()) <= 0:
                scores.append(1.0)
            else:
                scores.append(0.0)
            continue

        tprec = float((sp & (g > 0)).sum()) / (sp_sum + eps) if sp_sum > 0 else 1.0
        tsens = float((sg & (p > 0)).sum()) / (sg_sum + eps) if sg_sum > 0 else 1.0
        cl = (2.0 * tprec * tsens) / (tprec + tsens + eps)
        scores.append(float(cl))

    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    save_visual_dir: Path | None = None,
    epoch: int = 0,
    total_epochs: int | None = None,
    compute_boundary_f1: bool = False,
    boundary_f1_radius: int = 2,
    compute_cldice: bool = False,
):
    model.eval()
    losses = []
    ious = []
    dices = []
    precisions = []
    recalls = []
    porosity_errors = []
    boundary_f1s = []
    cldices = []
    total_saved = 0

    for images, masks, _, names in tqdm(loader, desc="valid", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        with amp.autocast('cuda',enabled=(use_amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)
        iou, dice = compute_batch_iou_and_dice(logits, masks)
        precision, recall, porosity_error = compute_batch_precision_recall_and_porosity_error(logits, masks)
        if compute_boundary_f1:
            boundary_f1s.append(
                compute_batch_boundary_f1(logits, masks, threshold=0.5, radius=int(boundary_f1_radius))
            )
        if compute_cldice:
            cldices.append(compute_batch_cldice(logits, masks, threshold=0.5))
        losses.append(loss.item())
        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        porosity_errors.append(porosity_error)

        # Only save a fixed number of visualization samples (80) in the last epoch.
        if save_visual_dir is not None and total_epochs is not None and epoch == total_epochs and total_saved < 80:
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

    out = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "porosity_error": float(np.mean(porosity_errors)) if porosity_errors else 0.0,
        "saved_visualizations": total_saved,
    }

    if compute_boundary_f1:
        out["boundary_f1"] = float(np.mean(boundary_f1s)) if boundary_f1s else 0.0
        out["boundary_f1_radius"] = int(boundary_f1_radius)
    if compute_cldice:
        out["cldice"] = float(np.mean(cldices)) if cldices else 0.0

    return out


@torch.no_grad()
def save_final_visualizations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_visual_dir: Path | None,
    epoch: int,
    use_amp: bool = False,
    max_samples: int = 80,
) -> int:
    if save_visual_dir is None:
        return 0
    model.eval()
    total_saved = 0
    for images, masks, _, names in tqdm(loader, desc="visual", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        with amp.autocast('cuda', enabled=(use_amp and device.type == "cuda")):
            logits = model(images)
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
):
    model.train()
    losses = []
    ious = []
    dices = []

    for images, masks, _, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast('cuda', enabled=(scaler is not None and device.type == "cuda")):
            logits = model(images)
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
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism(enable: bool) -> None:
    """Best-effort determinism for fair comparisons."""
    if not enable:
        return
    try:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
            print(
                "Warning: --deterministic enabled but CUBLAS_WORKSPACE_CONFIG is not set. "
                "Some matmul ops may remain non-deterministic."
            )
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"Warning: failed to enable deterministic mode: {e}")


def configure_repro_lite(enable: bool) -> None:
    """Reduce run-to-run variance without forcing strict deterministic kernels."""
    if not enable:
        return
    try:
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        print(f"Warning: failed to enable repro-lite mode: {e}")


def sanitize_args_for_checkpoint(args_dict: dict) -> dict:
    sanitized = {}
    for key, value in args_dict.items():
        if isinstance(value, Path):
            sanitized[key] = str(value)
        else:
            sanitized[key] = value
    return sanitized


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


def update_run_report(
    report_dir: Path,
    summary: dict,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    def _to_json_serializable(o):
        if isinstance(o, Path):
            return str(o)
        try:
            import numpy as _np
        except Exception:
            _np = None
        if _np is not None:
            if isinstance(o, _np.generic):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()
        try:
            import torch as _torch
        except Exception:
            _torch = None
        if _torch is not None and isinstance(o, _torch.Tensor):
            if o.numel() == 1:
                return o.item()
            return o.detach().cpu().tolist()
        if isinstance(o, dict):
            return {k: _to_json_serializable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_json_serializable(v) for v in o]
        return o

    safe_summary = _to_json_serializable(summary)
    (report_dir / "summary.json").write_text(json.dumps(safe_summary, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_run_report(report_dir: Path, updates: dict) -> None:
    """Merge updates into existing summary.json (if any) and write back."""
    base: dict = {}
    summary_path = report_dir / "summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8-sig"))
            if isinstance(existing, dict):
                base = existing
        except Exception:
            base = {}
    merged = dict(base)
    merged.update(updates)
    update_run_report(report_dir=report_dir, summary=merged)


def count_model_parameters(model: nn.Module) -> dict:
    target = model.module if hasattr(model, "module") else model
    total = int(sum(p.numel() for p in target.parameters()))
    trainable = int(sum(p.numel() for p in target.parameters() if p.requires_grad))
    return {
        "total": total,
        "trainable": trainable,
        "frozen": int(total - trainable),
    }


@torch.no_grad()
def benchmark_inference_fps(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool = False,
    warmup_batches: int = 3,
    max_measure_batches: int = 30,
) -> dict:
    was_training = model.training
    model.eval()

    timings: List[float] = []
    batch_sizes: List[int] = []
    max_total_batches = max(1, int(warmup_batches) + max(1, int(max_measure_batches)))

    for images, _, _, _ in loader:
        images = images.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with amp.autocast('cuda', enabled=(use_amp and device.type == "cuda")):
            _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
        batch_sizes.append(int(images.shape[0]))
        if len(timings) >= max_total_batches:
            break

    if was_training:
        model.train()

    if not timings:
        return {
            "fps": 0.0,
            "elapsed_s": 0.0,
            "measured_images": 0,
            "measured_batches": 0,
            "warmup_batches_excluded": 0,
        }

    start = min(max(0, int(warmup_batches)), len(timings) - 1)
    measured_times = timings[start:]
    measured_sizes = batch_sizes[start:]
    elapsed = float(sum(measured_times))
    measured_images = int(sum(measured_sizes))
    fps = float(measured_images / max(elapsed, 1e-8))
    return {
        "fps": fps,
        "elapsed_s": elapsed,
        "measured_images": measured_images,
        "measured_batches": int(len(measured_times)),
        "warmup_batches_excluded": int(start),
    }


def generate_training_curves(plot_script: Path, log_csv: Path, output_dir: Path, model_name: str = "Pure U-Net") -> None:
    if not plot_script.exists() or not log_csv.exists():
        return
    subprocess.run(
        [
            "python",
            str(plot_script),
            "--log-csv",
            str(log_csv),
            "--output-dir",
            str(output_dir),
            "--model-name",
            model_name,
        ],
        check=True,
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train pure U-Net on data_3.8 pore segmentation dataset")
    parser.add_argument("--data-root", type=Path, default=Path(r"f:\program\pore\data_3.8"))
    parser.add_argument("--image-dir-name", type=str, default="patch_images")
    parser.add_argument("--mask-dir-name", type=str, default="patch_mask")
    parser.add_argument("--text-dir-name", type=str, default="text")
    parser.add_argument("--module-path", type=Path, default=Path(__file__).resolve().parent / "unet.py")
    parser.add_argument("--use-wconv", action="store_true", help="Use WConv2d for U-Net 3x3 convs (keeps 1x1 convs unchanged)")
    parser.add_argument("--wconv-den", type=float, nargs="+", default=[0.75], help="den list for WConv2d, e.g. --wconv-den 0.75")
    parser.add_argument("--use-directional-refine", action="store_true", help="Enable deep visual directional refine (skip4 + bottleneck)")
    parser.add_argument("--wo-adaptive-structural-fusion", action="store_true", help="Disable adaptive gate fusion in TriBranchDirectionalRefine and use fixed uniform fusion")
    parser.add_argument("--use-skip-attention", action="store_true", help="Enable attention-gated skip connections in decoder")
    parser.add_argument("--skip-attention-reduction", type=int, default=8, help="Reduction ratio for skip attention hidden channels")
    parser.add_argument("--directional-kernel-size", type=int, default=5, help="Kernel size for directional DSConv branches")
    parser.add_argument("--directional-max-res-scale", type=float, default=0.30, help="Max residual scale for directional refine")
    parser.add_argument("--directional-alpha-init", type=float, default=-2.0, help="Initial alpha for directional refine residual scaling")
    parser.add_argument("--directional-extend-scope", type=float, default=1.0, help="Extend scope for DSConv_pro in directional refine")
    parser.add_argument("--use-bottleneck-context", action="store_true", help="Enable lightweight bottleneck multi-scale context")
    parser.add_argument("--bottleneck-context-max-res-scale", type=float, default=0.15, help="Max residual scale for bottleneck context")
    parser.add_argument("--bottleneck-context-beta-init", type=float, default=-2.2, help="Initial beta for bottleneck context residual scaling")
    parser.add_argument("--use-decoder-directional-refine", action="store_true", help="Enable dec4 directional refine after first decoder fusion")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Enable best-effort deterministic training for reproducibility")
    parser.add_argument("--repro-lite", action="store_true", help="Reproducibility-lite: disables cuDNN benchmark without forcing deterministic algorithms")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent / "reports" / "latest_run_unet")
    parser.add_argument("--save-dir", type=Path, default=None, help="If provided, saves checkpoints here as well as in the report-dir")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-last", action="store_true", default=True)
    parser.add_argument("--visualize-val", action="store_true")
    parser.add_argument("--visual-dir", type=Path, default=None, help="If provided, saves visualizations here as well as in the report-dir")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["none", "plateau"])
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--curves-dir", type=Path, default=None, help="If provided, saves curves here as well as in the report-dir")
    parser.add_argument("--report-every-epoch", action="store_true")
    parser.add_argument("--copy-report-visualizations", action="store_true")
    parser.add_argument("--plot-curves-every-epoch", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (torch.cuda.amp) when CUDA is available")

    # Eval from an existing checkpoint (for post-hoc metric computation)
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="If set, loads this checkpoint and runs a single validation evaluation, then exits.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Alias: run eval-only (requires --eval-checkpoint).",
    )
    parser.add_argument(
        "--compute-boundary-f1",
        action="store_true",
        help="Compute Boundary F1 on validation (recommended for eval-only).",
    )
    parser.add_argument(
        "--boundary-f1-radius",
        type=int,
        default=2,
        help="Pixel tolerance radius for Boundary F1 matching (default: 2).",
    )
    parser.add_argument(
        "--compute-cldice",
        action="store_true",
        help="Compute skeleton clDice on validation (recommended for eval-only).",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return get_parser().parse_args()


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    if args.deterministic:
        configure_determinism(True)
        repro_mode = "deterministic"
    elif args.repro_lite:
        configure_repro_lite(True)
        repro_mode = "repro_lite"
    else:
        repro_mode = "off"
    set_seed(args.seed)

    eval_mode = (args.eval_checkpoint is not None) or bool(args.eval_only)
    if bool(args.eval_only) and args.eval_checkpoint is None:
        raise RuntimeError("--eval-only requires --eval-checkpoint")

    # compute explicit CLI args
    explicit_cli_args: dict = {}
    for k, v in vars(args).items():
        try:
            default_v = parser.get_default(k)
        except Exception:
            default_v = None
        if v != default_v:
            explicit_cli_args[k] = v

    # avoid overwriting report_dir when not resuming
    def make_unique_report_dir(base: Path) -> Path:
        if not base.exists():
            return base
        suffix = 1
        while True:
            cand = Path(str(base) + f"_{suffix}")
            if not cand.exists():
                return cand
            suffix += 1

    if (not eval_mode) and args.resume is None and args.report_dir.exists():
        new_dir = make_unique_report_dir(args.report_dir)
        if new_dir != args.report_dir:
            print(f"Report dir {args.report_dir} exists; using new report dir {new_dir} to avoid overwrite.")
            args.report_dir = new_dir

    image_dir = args.data_root / args.image_dir_name
    mask_dir = args.data_root / args.mask_dir_name
    text_dir = args.data_root / args.text_dir_name

    # Consolidate output directories. Always use report-dir as the primary output.
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_visual_dir = args.report_dir / "visualizations"
    report_curves_dir = args.report_dir / "training_curves"
    
    if args.visualize_val:
        report_visual_dir.mkdir(parents=True, exist_ok=True)
    report_curves_dir.mkdir(parents=True, exist_ok=True)

    module = load_unet_module(args.module_path)
    dataset = PoreSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        text_dir=text_dir,
        image_size=args.image_size,
    )

    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    if train_len <= 0:
        raise RuntimeError("Dataset too small after validation split.")

    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = args.pin_memory or device.type == "cuda"
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": use_pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    unet_kwargs = {"n_channels": 3, "n_classes": 1, "bilinear": False}
    sig = inspect.signature(module.UNet)
    if "use_directional_refine" in sig.parameters:
        unet_kwargs["use_directional_refine"] = bool(args.use_directional_refine)
    if "use_adaptive_structural_fusion" in sig.parameters:
        unet_kwargs["use_adaptive_structural_fusion"] = not bool(args.wo_adaptive_structural_fusion)
    if "use_skip_attention" in sig.parameters:
        unet_kwargs["use_skip_attention"] = bool(args.use_skip_attention)
    if "skip_attention_reduction" in sig.parameters:
        unet_kwargs["skip_attention_reduction"] = int(args.skip_attention_reduction)
    if "directional_kernel_size" in sig.parameters:
        unet_kwargs["directional_kernel_size"] = int(args.directional_kernel_size)
    if "directional_max_res_scale" in sig.parameters:
        unet_kwargs["directional_max_res_scale"] = float(args.directional_max_res_scale)
    if "directional_alpha_init" in sig.parameters:
        unet_kwargs["directional_alpha_init"] = float(args.directional_alpha_init)
    if "directional_extend_scope" in sig.parameters:
        unet_kwargs["directional_extend_scope"] = float(args.directional_extend_scope)
    if "use_bottleneck_context" in sig.parameters:
        unet_kwargs["use_bottleneck_context"] = bool(args.use_bottleneck_context)
    if "bottleneck_context_max_res_scale" in sig.parameters:
        unet_kwargs["bottleneck_context_max_res_scale"] = float(args.bottleneck_context_max_res_scale)
    if "bottleneck_context_beta_init" in sig.parameters:
        unet_kwargs["bottleneck_context_beta_init"] = float(args.bottleneck_context_beta_init)
    if "use_decoder_directional_refine" in sig.parameters:
        unet_kwargs["use_decoder_directional_refine"] = bool(args.use_decoder_directional_refine)
    if args.use_wconv:
        sig = inspect.signature(module.UNet)
        if "use_wconv" in sig.parameters:
            unet_kwargs["use_wconv"] = True
        if "wconv_den" in sig.parameters:
            unet_kwargs["wconv_den"] = args.wconv_den
    model = module.UNet(**unet_kwargs).to(device)
    param_stats = count_model_parameters(model)
    criterion = DiceBCELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )

    best_dice = -1.0
    start_epoch = 1
    log_records: List[dict] = []
    epochs_without_improvement = 0
    last_report_summary: dict | None = None

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_dice = float(checkpoint.get("val_metrics", {}).get("dice", -1.0))
        print(f"Resumed from checkpoint: {args.resume} (starting at epoch {start_epoch})")
        # Try to load previous training log. Prefer explicit save_dir if provided, else fall back to report_dir
        prev_log = (args.save_dir or args.report_dir) / "training_log.json"
        if prev_log is not None and prev_log.exists():
            try:
                with open(prev_log, "r", encoding="utf-8") as f:
                    prev_records = json.load(f)
                if isinstance(prev_records, list):
                    log_records.extend(prev_records)
                    print(f"Loaded {len(prev_records)} previous training log records from {prev_log}")
            except Exception as e:
                print(f"Warning: failed to load previous training log from {prev_log}: {e}")

    if args.eval_checkpoint is not None:
        ckpt = torch.load(args.eval_checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(state, strict=True)
            missing, unexpected = [], []
        except RuntimeError as e:
            print(f"Warning: strict checkpoint load failed: {e}")
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Warning: checkpoint loaded non-strict. missing={len(missing)} unexpected={len(unexpected)}")
        print(f"Loaded eval checkpoint: {args.eval_checkpoint}")

        # Default to computing both extra metrics for eval-only runs.
        compute_boundary = bool(args.compute_boundary_f1) or (not args.compute_boundary_f1 and not args.compute_cldice)
        compute_cl = bool(args.compute_cldice) or (not args.compute_boundary_f1 and not args.compute_cldice)

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=args.amp,
            save_visual_dir=report_visual_dir if args.visualize_val else None,
            epoch=0,
            total_epochs=None,
            compute_boundary_f1=compute_boundary,
            boundary_f1_radius=int(args.boundary_f1_radius),
            compute_cldice=compute_cl,
        )

        eval_summary = {
            "eval_from_checkpoint": str(Path(args.eval_checkpoint).resolve()),
            "eval_val_metrics": val_metrics,
            "parameter_count": param_stats,
            "model_name": "pure-unet",
            "repro_mode": repro_mode,
            "use_directional_refine": bool(args.use_directional_refine),
            "wo_adaptive_structural_fusion": bool(args.wo_adaptive_structural_fusion),
            "use_skip_attention": bool(args.use_skip_attention),
            "use_bottleneck_context": bool(args.use_bottleneck_context),
            "use_decoder_directional_refine": bool(args.use_decoder_directional_refine),
        }

        merge_run_report(report_dir=args.report_dir, updates={"eval_metrics": eval_summary})
        print(
            "Eval done | "
            f"loss {val_metrics.get('loss', 0.0):.4f} dice {val_metrics.get('dice', 0.0):.4f} iou {val_metrics.get('iou', 0.0):.4f} "
            f"boundary_f1 {val_metrics.get('boundary_f1', float('nan')):.4f} cldice {val_metrics.get('cldice', float('nan')):.4f}"
        )
        return

    print(f"Dataset size: {len(dataset)} | train: {len(train_set)} | val: {len(val_set)}")
    print(f"Using device: {device}")
    print(f"Repro mode: {repro_mode}")
    print(
        "DataLoader config | "
        f"workers={args.num_workers} pin_memory={use_pin_memory} "
        f"persistent_workers={bool(args.persistent_workers and args.num_workers > 0)}"
    )

    # initialize GradScaler once if AMP requested and CUDA available
    scaler = None
    if args.amp and device.type == "cuda":
        scaler = amp.GradScaler()
        setattr(optimizer, "_amp_scaler", scaler)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        after_train = time.perf_counter()
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=args.amp,
            save_visual_dir=report_visual_dir if args.visualize_val else None,
            epoch=epoch,
            total_epochs=args.epochs,
            compute_boundary_f1=bool(args.compute_boundary_f1),
            boundary_f1_radius=int(args.boundary_f1_radius),
            compute_cldice=bool(args.compute_cldice),
        )
        if args.visual_dir is not None:
             args.visual_dir.mkdir(parents=True, exist_ok=True)
             for item in report_visual_dir.glob(f"epoch{epoch:03d}*"):
                 shutil.copy2(item, args.visual_dir / item.name)
        after_val = time.perf_counter()

        if args.save_last:
            last_ckpt_path = args.report_dir / "last_unet.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "args": sanitize_args_for_checkpoint(vars(args)),
                },
                last_ckpt_path,
            )
            if args.save_dir is not None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(last_ckpt_path, args.save_dir / "last_unet.pt")

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_metrics['loss']:.4f} dice {train_metrics['dice']:.4f} iou {train_metrics['iou']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} dice {val_metrics['dice']:.4f} iou {val_metrics['iou']:.4f} prec {val_metrics['precision']:.4f} rec {val_metrics['recall']:.4f} "
            f"por_err {val_metrics['porosity_error']:.4f} | "
            f"timing train {after_train - epoch_start:.1f}s val {after_val - after_train:.1f}s"
        )

        log_records.append(
            {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_dice": round(train_metrics["dice"], 6),
                "train_iou": round(train_metrics["iou"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_dice": round(val_metrics["dice"], 6),
                "val_iou": round(val_metrics["iou"], 6),
                "val_precision": round(val_metrics["precision"], 6),
                "val_recall": round(val_metrics["recall"], 6),
                "val_porosity_error": round(val_metrics["porosity_error"], 6),
                **(
                    {"val_boundary_f1": round(float(val_metrics.get("boundary_f1", 0.0)), 6)}
                    if "boundary_f1" in val_metrics
                    else {}
                ),
                **(
                    {"val_cldice": round(float(val_metrics.get("cldice", 0.0)), 6)}
                    if "cldice" in val_metrics
                    else {}
                ),
                "saved_visualizations": int(val_metrics.get("saved_visualizations", 0)),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        save_training_logs(log_records, args.report_dir)
        if args.save_dir is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            for log_name in ["training_log.csv", "training_log.json"]:
                if (args.report_dir / log_name).exists():
                    shutil.copy2(args.report_dir / log_name, args.save_dir / log_name)

        if scheduler is not None:
            scheduler.step(val_metrics["dice"])

        if val_metrics["dice"] > best_dice + args.min_delta:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            ckpt_path = args.report_dir / "best_unet.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "args": sanitize_args_for_checkpoint(vars(args)),
                },
                ckpt_path,
            )
            if args.save_dir is not None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ckpt_path, args.save_dir / "best_unet.pt")
        else:
            epochs_without_improvement += 1

        if args.plot_curves_every_epoch:
            plot_script = Path(__file__).resolve().parent / "plot_training_curves.py"
            try:
                generate_training_curves(
                    plot_script=plot_script,
                    log_csv=args.report_dir / "training_log.csv",
                    output_dir=report_curves_dir,
                    model_name="Pure U-Net",
                )
                if args.curves_dir is not None:
                    args.curves_dir.mkdir(parents=True, exist_ok=True)
                    for item in report_curves_dir.iterdir():
                        if item.is_file():
                            shutil.copy2(item, args.curves_dir / item.name)
            except Exception as exc:
                print(f"Warning: failed to generate training curves automatically: {exc}")

        # include explicit CLI args and resumed checkpoint args (if any)
        resumed_checkpoint_args = None
        if args.resume is not None:
            try:
                ckpt = torch.load(args.resume, map_location="cpu")
                resumed_checkpoint_args = ckpt.get("args")
            except Exception:
                resumed_checkpoint_args = None

        report_summary = {
            "best_dice": best_dice,
            "last_epoch": epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "device": str(device),
            "dataset_size": len(dataset),
            "train_size": len(train_set),
            "val_size": len(val_set),
            "scheduler": args.scheduler,
            "current_lr": float(optimizer.param_groups[0]["lr"]),
            "last_metrics": log_records[-1],
            "model_name": "pure-unet",
            "deterministic": bool(args.deterministic),
            "repro_mode": repro_mode,
            "use_directional_refine": bool(args.use_directional_refine),
            "wo_adaptive_structural_fusion": bool(args.wo_adaptive_structural_fusion),
            "use_skip_attention": bool(args.use_skip_attention),
            "skip_attention_reduction": int(args.skip_attention_reduction),
            "directional_kernel_size": int(args.directional_kernel_size),
            "directional_max_res_scale": float(args.directional_max_res_scale),
            "directional_alpha_init": float(args.directional_alpha_init),
            "directional_extend_scope": float(args.directional_extend_scope),
            "use_bottleneck_context": bool(args.use_bottleneck_context),
            "bottleneck_context_max_res_scale": float(args.bottleneck_context_max_res_scale),
            "bottleneck_context_beta_init": float(args.bottleneck_context_beta_init),
            "use_decoder_directional_refine": bool(args.use_decoder_directional_refine),
        }
        try:
            report_summary["explicit_cli_args"] = explicit_cli_args
        except Exception:
            report_summary["explicit_cli_args"] = {}
        if resumed_checkpoint_args is not None:
            report_summary["resumed_checkpoint_args"] = resumed_checkpoint_args

        last_report_summary = report_summary

        if args.report_every_epoch:
            update_run_report(
                report_dir=args.report_dir,
                summary=report_summary,
            )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch} after {epochs_without_improvement} epochs without validation Dice improvement."
            )
            if args.visualize_val:
                saved = save_final_visualizations(
                    model=model,
                    loader=val_loader,
                    device=device,
                    save_visual_dir=report_visual_dir,
                    epoch=epoch,
                    use_amp=args.amp,
                )
                if args.visual_dir is not None and saved > 0:
                    args.visual_dir.mkdir(parents=True, exist_ok=True)
                    for item in report_visual_dir.glob(f"epoch{epoch:03d}*"):
                        shutil.copy2(item, args.visual_dir / item.name)
            break

    fps_stats = benchmark_inference_fps(
        model=model,
        loader=val_loader,
        device=device,
        use_amp=bool(args.amp),
        warmup_batches=3,
        max_measure_batches=30,
    )
    print(
        "Final model stats | "
        f"params(total/trainable/frozen)={param_stats['total']}/{param_stats['trainable']}/{param_stats['frozen']} "
        f"| val_fps={fps_stats['fps']:.2f} "
        f"(warmup_excluded={fps_stats['warmup_batches_excluded']} measured_batches={fps_stats['measured_batches']})"
    )

    final_summary = dict(last_report_summary) if isinstance(last_report_summary, dict) else {
        "device": str(device),
        "model_name": "pure-unet",
        "dataset_size": len(dataset),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "best_dice": best_dice,
    }
    final_summary["parameter_count"] = param_stats
    final_summary["inference_fps"] = fps_stats
    final_summary["final_val_fps"] = float(fps_stats["fps"])
    update_run_report(report_dir=args.report_dir, summary=final_summary)


if __name__ == "__main__":
    main()
