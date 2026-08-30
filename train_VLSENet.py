from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from VLSENet import DiceBCELoss, VLSENet
from data_utils import PoreDataset, build_dataset_splits, collate_fn, write_split_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VLSE-Net for SEM pore segmentation."
    )
    parser.add_argument("--data-root", type=Path, default=Path("./dataset"))
    parser.add_argument("--image-dir-name", type=str, default="patch_images")
    parser.add_argument("--mask-dir-name", type=str, default="patch_mask")
    parser.add_argument("--text-dir-name", type=str, default="text")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing train.txt, val.txt and test.txt. "
            "For paper reproduction these manifests must be generated after "
            "source-image-level partitioning (slice-level for DRP-317)."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio for the functional-test fallback when --split-dir is omitted.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test ratio for the functional-test fallback when --split-dir is omitted.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--scheduler",
        choices=("none", "plateau"),
        default="plateau",
    )
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--region-alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the region-text alignment loss used by the released VLSE-Net configuration.",
    )
    parser.add_argument("--alignment-weight", type=float, default=0.07)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic mixed precision when CUDA is available.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("./reports/latest_run_text_guided_unet"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def make_unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    suffix = 1
    while True:
        candidate = Path(f"{base}_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
    )


def segmentation_metrics(logits: torch.Tensor, masks: torch.Tensor) -> dict[str, float]:
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    targets = (masks >= 0.5).float()

    pred_flat = predictions.flatten(1)
    target_flat = targets.flatten(1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    eps = 1e-6
    dice = (
        (2.0 * intersection + eps)
        / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps)
    ).mean()
    iou = ((intersection + eps) / (union + eps)).mean()

    tp = intersection
    fp = (pred_flat * (1.0 - target_flat)).sum(dim=1)
    fn = ((1.0 - pred_flat) * target_flat).sum(dim=1)
    precision = (tp / (tp + fp + eps)).mean()
    recall = (tp / (tp + fn + eps)).mean()

    porosity_error = (pred_flat.mean(dim=1) - target_flat.mean(dim=1)).abs().mean()

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "porosity_error": float(porosity_error.item()),
    }


def train_one_epoch(
    model: VLSENet,
    loader: DataLoader,
    criterion: DiceBCELoss,
    optimizer: AdamW,
    device: torch.device,
    scaler: amp.GradScaler,
    amp_enabled: bool,
    use_region_alignment: bool,
    alignment_weight: float,
) -> dict[str, float]:
    model.train()

    losses: list[float] = []
    dices: list[float] = []
    ious: list[float] = []
    alignments: list[float] = []

    for images, masks, prompts, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with amp.autocast("cuda", enabled=amp_enabled):
            if use_region_alignment:
                outputs = model(
                    images,
                    prompts=prompts,
                    region_mask=masks,
                    return_aux=True,
                )
                logits = outputs["logits"]
                alignment_loss = outputs.get("alignment_loss")
                loss = criterion(logits, masks)
                if alignment_loss is not None:
                    loss = loss + alignment_weight * alignment_loss
                    alignments.append(float(alignment_loss.detach().item()))
            else:
                logits = model(images, prompts=prompts)
                loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metrics = segmentation_metrics(logits.detach(), masks)
        losses.append(float(loss.detach().item()))
        dices.append(metrics["dice"])
        ious.append(metrics["iou"])

    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "alignment_loss": float(np.mean(alignments)) if alignments else 0.0,
    }


@torch.no_grad()
def evaluate(
    model: VLSENet,
    loader: DataLoader,
    criterion: DiceBCELoss,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()

    rows: list[dict[str, float]] = []
    losses: list[float] = []

    for images, masks, prompts, _ in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images, prompts=prompts)
            loss = criterion(logits, masks)

        losses.append(float(loss.item()))
        rows.append(segmentation_metrics(logits, masks))

    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean([row["dice"] for row in rows])),
        "iou": float(np.mean([row["iou"] for row in rows])),
        "precision": float(np.mean([row["precision"] for row in rows])),
        "recall": float(np.mean([row["recall"] for row in rows])),
        "porosity_error": float(np.mean([row["porosity_error"] for row in rows])),
    }


def save_log(records: list[dict], report_dir: Path) -> None:
    (report_dir / "training_log.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    if not records:
        return

    with (report_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def checkpoint_payload(
    model: VLSENet,
    optimizer: AdamW,
    epoch: int,
    val_metrics: dict[str, float],
    args: argparse.Namespace,
    split_source: str,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "val_metrics": val_metrics,
        "split_source": split_source,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    image_dir = args.data_root / args.image_dir_name
    mask_dir = args.data_root / args.mask_dir_name
    text_dir = args.data_root / args.text_dir_name

    dataset = PoreDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        text_dir=text_dir,
        image_size=args.image_size,
        require_text=True,
    )

    subsets, split_source = build_dataset_splits(
        dataset,
        args.split_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if split_source != "manifest":
        print(
            "Warning: --split-dir was not provided. A deterministic sample-level "
            "train/val/test split is being used for functional testing only. "
            "For reproducing the paper protocol, provide manifests generated "
            "after source-image-level partitioning (slice-level for DRP-317)."
        )

    report_dir = make_unique_dir(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifests(dataset, subsets, report_dir / "splits")

    train_loader = make_loader(
        subsets["train"], args.batch_size, True, args.num_workers, device
    )
    val_loader = make_loader(
        subsets["val"], args.batch_size, False, args.num_workers, device
    )
    test_loader = make_loader(
        subsets["test"], args.batch_size, False, args.num_workers, device
    )

    model = VLSENet(num_classes=1, freeze_text=True).to(device)
    criterion = DiceBCELoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )
        if args.scheduler == "plateau"
        else None
    )
    scaler = amp.GradScaler("cuda", enabled=amp_enabled)

    best_dice = float("-inf")
    epochs_without_improvement = 0
    records: list[dict] = []
    best_checkpoint = report_dir / "best_text_guided_unet.pt"
    last_checkpoint = report_dir / "last_text_guided_unet.pt"

    print(
        f"Dataset size: {len(dataset)} | "
        f"train: {len(subsets['train'])} | "
        f"val: {len(subsets['val'])} | "
        f"test: {len(subsets['test'])}"
    )
    print(f"Split source: {split_source}")
    print(f"Device: {device}")
    print(f"Report directory: {report_dir}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            use_region_alignment=args.region_alignment,
            alignment_weight=args.alignment_weight,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        if scheduler is not None:
            scheduler.step(val_metrics["dice"])

        payload = checkpoint_payload(
            model,
            optimizer,
            epoch,
            val_metrics,
            args,
            split_source,
        )
        torch.save(payload, last_checkpoint)

        if val_metrics["dice"] > best_dice + args.min_delta:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            torch.save(payload, best_checkpoint)
        else:
            epochs_without_improvement += 1

        record = {
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
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        records.append(record)
        save_log(records, report_dir)

        print(
            f"Epoch {epoch:03d} | "
            f"train dice={train_metrics['dice']:.4f} "
            f"val dice={val_metrics['dice']:.4f} "
            f"val IoU={val_metrics['iou']:.4f} "
            f"val porosity error={val_metrics['porosity_error']:.6f}"
        )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                "Early stopping: "
                f"{epochs_without_improvement} epoch(s) without validation Dice improvement."
            )
            break

    if not best_checkpoint.is_file():
        raise RuntimeError("Training finished without producing a best checkpoint.")

    best_payload = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )

    summary = {
        "split_source": split_source,
        "dataset_size": len(dataset),
        "train_size": len(subsets["train"]),
        "val_size": len(subsets["val"]),
        "test_size": len(subsets["test"]),
        "best_epoch": int(best_payload["epoch"]),
        "best_val_metrics": best_payload["val_metrics"],
        "test_metrics": test_metrics,
        "best_checkpoint": str(best_checkpoint),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (report_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )

    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
