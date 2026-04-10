from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from typing import Iterable, Tuple

from compute_features import compute_features_from_image
from llm_api import generate_description
from prompt_builder import build_prompt
from dotenv import load_dotenv

# Load .env if present so environment variables are populated
load_dotenv()


def iter_patch_pairs(patch_image_dir: Path, patch_mask_dir: Path) -> Iterable[Tuple[Path, Path]]:
    mask_map = {p.name: p for p in patch_mask_dir.iterdir() if p.is_file()}
    for image_path in sorted(p for p in patch_image_dir.iterdir() if p.is_file()):
        mask_path = mask_map.get(image_path.name)
        if mask_path is not None:
            yield image_path, mask_path


def append_error_log(error_log_path: Path, payload: dict) -> None:
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    with error_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_status_log(status_log_path: Path, payload: dict) -> None:
    status_log_path.parent.mkdir(parents=True, exist_ok=True)
    with status_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_text_atomic(output_path: Path, content: str) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, output_path)


def process_single_pair(
    image_path: Path,
    mask_path: Path,
    text_dir: Path,
    model: str | None,
    resume: bool,
    force: bool,
    error_log_path: Path,
    status_log_path: Path,
    max_retries: int,
    retry_delay: float,
) -> tuple[str, Path, str | None]:
    output_path = text_dir / f"{image_path.stem}.txt"
    if output_path.exists() and resume and not force:
        append_status_log(
            status_log_path,
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "output": str(output_path),
                "status": "skipped",
                "attempts": 0,
            },
        )
        return "skipped", image_path, None

    try:
        features = compute_features_from_image(image_path)
        prompt = build_prompt(image_path.name, features)
        result = generate_description(
            prompt=prompt,
            model=model,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        write_text_atomic(output_path, result["text"])
        append_status_log(
            status_log_path,
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "output": str(output_path),
                "status": f"generated-{result['mode']}",
                "attempts": result["attempts"],
                "fallback_used": result["fallback_used"],
            },
        )
        return "generated", image_path, None
    except Exception as exc:
        append_error_log(
            error_log_path,
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "output": str(output_path),
                "error": repr(exc),
            },
        )
        append_status_log(
            status_log_path,
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "output": str(output_path),
                "status": "failed",
                "attempts": max_retries,
                "error": repr(exc),
            },
        )
        return "failed", image_path, repr(exc)


def build_text_dataset(
    patch_image_dir: Path,
    patch_mask_dir: Path,
    text_dir: Path,
    model: str | None = None,
    workers: int = 1,
    resume: bool = True,
    force: bool = False,
    error_log_path: Path | None = None,
    status_log_path: Path | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict[str, int]:
    text_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = error_log_path or (text_dir / "errors.jsonl")
    status_log_path = status_log_path or (text_dir / "status.jsonl")
    pairs = list(iter_patch_pairs(patch_image_dir, patch_mask_dir))

    summary = {"generated": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                process_single_pair,
                image_path,
                mask_path,
                text_dir,
                model,
                resume,
                force,
                error_log_path,
                status_log_path,
                max_retries,
                retry_delay,
            )
            for image_path, mask_path in pairs
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="processing", unit="item"):
            status, image_path, error = future.result()
            summary[status] += 1
            if status == "failed":
                tqdm.write(f"[FAILED] {image_path.name}: {error}")
            elif status == "skipped":
                tqdm.write(f"[SKIP]   {image_path.name}")
            else:
                pass

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VLGS text dataset for rock core pore segmentation")
    parser.add_argument("--patch-image-dir", type=Path, default=Path("dataset/patch_images"))
    parser.add_argument("--patch-mask-dir", type=Path, default=Path("dataset/patch_mask"))
    parser.add_argument("--text-dir", type=Path, default=Path("dataset/text"))
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL"))
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for LLM requests")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip existing output files")
    parser.add_argument("--force", action="store_true", help="Force regenerate outputs even if target files exist")
    parser.add_argument("--error-log", type=Path, default=None, help="Path to JSONL error log file")
    parser.add_argument("--status-log", type=Path, default=None, help="Path to JSONL status log file")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for transient LLM request failures")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Base retry delay in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_text_dataset(
        patch_image_dir=args.patch_image_dir,
        patch_mask_dir=args.patch_mask_dir,
        text_dir=args.text_dir,
        model=args.model,
        workers=args.workers,
        resume=not args.no_resume,
        force=args.force,
        error_log_path=args.error_log,
        status_log_path=args.status_log,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    print(
        f"Done. generated={summary['generated']}, skipped={summary['skipped']}, failed={summary['failed']} -> {args.text_dir}"
    )


if __name__ == "__main__":
    main()
