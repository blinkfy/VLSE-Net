"""Inference script for VLSE-Net.

Generate binary pore segmentation masks from SEM images using the
sample-specific textual prompts prepared offline.

Example:
python infer_VLSENet.py \
    --image-dir ./dataset/patch_images \
    --text-dir ./dataset/text \
    --checkpoint ./reports/latest_run_text_guided_unet/best_text_guided_unet.pt \
    --output-dir ./outputs
"""

from pathlib import Path
import argparse

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from VLSENet import VLSENet


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMAGE_SIZE = 512
MASK_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with VLSE-Net")
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing input SEM images.",
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        required=True,
        help="Directory containing sample-specific .txt prompts matched by filename stem.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a checkpoint produced by train_VLSENet.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Directory for predicted binary masks.",
    )
    return parser.parse_args()


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


def load_model(checkpoint_path: Path, device: torch.device) -> VLSENet:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Unsupported checkpoint format. Expected a dictionary containing "
            "'model_state_dict' as produced by train_VLSENet.py."
        )

    model = VLSENet(num_classes=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def collect_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(f"No supported image files found in: {image_dir}")
    return image_paths


def load_prompt(text_dir: Path, image_path: Path) -> str:
    prompt_path = text_dir / f"{image_path.stem}.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Missing prompt for '{image_path.name}': expected {prompt_path}"
        )

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return prompt


def save_mask(probability: torch.Tensor, output_path: Path) -> None:
    mask = probability.detach().cpu().numpy().squeeze()
    mask = (mask > MASK_THRESHOLD).astype("uint8") * 255
    Image.fromarray(mask).save(output_path)


def main() -> None:
    args = parse_args()

    if not args.text_dir.is_dir():
        raise NotADirectoryError(f"Text directory not found: {args.text_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    transform = build_transform()
    image_paths = collect_image_paths(args.image_dir)

    with torch.inference_mode():
        for image_path in tqdm(image_paths, desc="Inference", unit="image"):
            with Image.open(image_path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

            prompt = load_prompt(args.text_dir, image_path)
            logits = model(tensor, prompts=[prompt], text_input_mode="raw")
            probability = torch.sigmoid(logits)
            save_mask(probability[0], mask_dir / f"{image_path.stem}.png")

    print(f"Inference finished. Masks saved to: {mask_dir}")


if __name__ == "__main__":
    main()
