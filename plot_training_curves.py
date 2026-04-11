from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def read_training_log(csv_path: Path) -> Dict[str, List[float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training log not found: {csv_path}")

    records: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in records.keys():
                records[key].append(float(row[key]))

    if not records["epoch"]:
        raise RuntimeError(f"Training log is empty: {csv_path}")

    return records


def plot_single_curve(epochs, train_values, val_values, ylabel: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_values, marker="o", label=f"train {ylabel.lower()}")
    plt.plot(epochs, val_values, marker="s", label=f"val {ylabel.lower()}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_summary(records: Dict[str, List[float]], output_path: Path, model_name: str = "CLIP-U-Net") -> None:
    epochs = records["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    pairs = [
        ("loss", "Loss", records["train_loss"], records["val_loss"]),
        ("dice", "Dice", records["train_dice"], records["val_dice"]),
        ("iou", "IoU", records["train_iou"], records["val_iou"]),
    ]

    for ax, (_, label, train_vals, val_vals) in zip(axes, pairs):
        ax.plot(epochs, train_vals, marker="o", label=f"train {label.lower()}")
        ax.plot(epochs, val_vals, marker="s", label=f"val {label.lower()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.suptitle(f"{model_name} Training Curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from training_log.csv")
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "training_log.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "training_curves",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="CLIP-U-Net",
        help="Model name to display in plot titles",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_training_log(args.log_csv)
    epochs = records["epoch"]

    plot_single_curve(
        epochs,
        records["train_loss"],
        records["val_loss"],
        ylabel="Loss",
        title="Training vs Validation Loss",
        output_path=args.output_dir / "loss_curve.png",
    )
    plot_single_curve(
        epochs,
        records["train_dice"],
        records["val_dice"],
        ylabel="Dice",
        title="Training vs Validation Dice",
        output_path=args.output_dir / "dice_curve.png",
    )
    plot_single_curve(
        epochs,
        records["train_iou"],
        records["val_iou"],
        ylabel="IoU",
        title="Training vs Validation IoU",
        output_path=args.output_dir / "iou_curve.png",
    )
    plot_summary(records, args.output_dir / "training_curves_summary.png", model_name=args.model_name)

    #print(f"Saved training curves to {args.output_dir}")


if __name__ == "__main__":
    main()
