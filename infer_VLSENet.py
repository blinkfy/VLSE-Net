"""
Inference script for VLSE-Net.

Generate pore segmentation masks from SEM images.

Example:
python infer_VLSENet.py \
    --image-dir ./demo/images \
    --checkpoint ./checkpoints/best_VLSE-Net.pt \
    --output-dir ./outputs
"""

from pathlib import Path
import argparse

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from VLSENet import VLSE_Net


def parse_args():

    parser = argparse.ArgumentParser(
        description="Inference with VLSE-Net"
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Input SEM image directory"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="VLSE-Net checkpoint path"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Prediction output directory"
    )

    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Save segmentation overlay"
    )

    return parser.parse_args()


def build_transform():

    return transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ])


def load_model(checkpoint, device):

    model = VLSE_Net(
        num_classes=1,
        text_spatial_mode="cross_attention",
        use_skip_attention=True,
        multi_scale_fusion=True,
        use_decoder_text_adapter=True,
        use_directional_refine=True,
        use_bottleneck_context=True,
        use_decoder_directional_refine=True
    )


    state_dict = torch.load(
        checkpoint,
        map_location=device
    )

    model.load_state_dict(
        state_dict["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model



def save_mask(mask, path):

    mask = (
        mask.cpu()
        .numpy()
        .squeeze()
    )

    mask = (mask > 0.5) * 255

    Image.fromarray(
        mask.astype("uint8")
    ).save(path)



def main():

    args = parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    output_dir = Path(args.output_dir)

    mask_dir = output_dir / "masks"

    mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    if args.save_overlay:

        overlay_dir = output_dir / "overlay"

        overlay_dir.mkdir(
            parents=True,
            exist_ok=True
        )


    model = load_model(
        args.checkpoint,
        device
    )


    transform = build_transform()


    image_paths = sorted(
        Path(args.image_dir).glob("*")
    )


    with torch.no_grad():

        for image_path in tqdm(image_paths):

            image = Image.open(
                image_path
            ).convert("RGB")


            tensor = transform(image)

            tensor = (
                tensor
                .unsqueeze(0)
                .to(device)
            )


            # Default text prompt for pore extraction
            prompt = ["pore"]


            prediction = model(
                tensor,
                prompt
            )


            probability = torch.sigmoid(
                prediction
            )


            save_mask(
                probability[0],
                mask_dir / image_path.name
            )


    print(
        "Inference finished."
    )


if __name__ == "__main__":
    main()
