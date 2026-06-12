#!/usr/bin/env python3
"""
data_prep.py — One-time preparation of the raw DeepGlobe dataset.

Walks the raw dataset directories, resizes image-mask pairs to 512×512,
and splits them into train/val/test sets.

Usage:
    python data_prep.py --config config.yaml
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml
from tqdm import tqdm


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration values.
    """
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")


def collect_image_mask_pairs(raw_dir: Path) -> List[Tuple[Path, Path]]:
    """Collect all satellite image and mask pairs from raw directories.

    Walks data/raw/train/ and data/raw/valid/ to find *_sat.jpg + *_mask.png pairs.

    Args:
        raw_dir: Path to the raw data directory (e.g., data/raw/).

    Returns:
        List of (image_path, mask_path) tuples.
    """
    pairs = []
    search_dirs = []

    # Look for train and valid subdirectories
    for subdir_name in ["train", "valid", "test"]:
        subdir = raw_dir / subdir_name
        if subdir.exists():
            search_dirs.append(subdir)

    # Also check the raw_dir itself (in case images are directly there)
    if not search_dirs:
        search_dirs.append(raw_dir)

    for search_dir in search_dirs:
        sat_images = sorted(search_dir.glob("*_sat.jpg"))
        for sat_path in sat_images:
            # Derive mask filename: replace _sat.jpg with _mask.png
            stem = sat_path.stem.replace("_sat", "_mask")
            mask_path = sat_path.parent / f"{stem}.png"

            if mask_path.exists():
                pairs.append((sat_path, mask_path))
            else:
                print(f"Warning: No matching mask for {sat_path.name}, skipping.")

    if not pairs:
        print(f"Warning: No image-mask pairs found in {raw_dir}")
        print(f"  Searched directories: {[str(d) for d in search_dirs]}")
        print("  Expected pattern: *_sat.jpg + *_mask.png")

    return pairs


def resize_pair(
    image_path: Path,
    mask_path: Path,
    img_size: int,
) -> Tuple:
    """Resize an image-mask pair to the target size.

    Args:
        image_path: Path to the satellite image.
        mask_path: Path to the mask image.
        img_size: Target size (both width and height).

    Returns:
        Tuple of (resized_image, resized_mask) as numpy arrays.
    """
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        # Resize image with INTER_AREA (best for downsampling)
        resized_image = cv2.resize(
            image, (img_size, img_size), interpolation=cv2.INTER_AREA
        )

        # Resize mask with INTER_NEAREST (preserves class labels)
        resized_mask = cv2.resize(
            mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST
        )

        return resized_image, resized_mask

    except Exception as e:
        raise RuntimeError(f"Error processing {image_path.name}: {e}")


def split_data(
    pairs: List[Tuple[Path, Path]],
    train_split: float,
    val_split: float,
    random_seed: int,
) -> Dict[str, List[Tuple[Path, Path]]]:
    """Split data into train/val/test sets.

    Args:
        pairs: List of (image_path, mask_path) tuples.
        train_split: Fraction for training (e.g., 0.8).
        val_split: Fraction for validation (e.g., 0.1).
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to lists of pairs.
    """
    random.seed(random_seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    return splits


def process_and_save(
    pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    split_name: str,
    img_size: int,
) -> int:
    """Process and save image-mask pairs for a given split.

    Args:
        pairs: List of (image_path, mask_path) tuples.
        output_dir: Base output directory (e.g., data/processed/).
        split_name: Name of the split ('train', 'val', or 'test').
        img_size: Target image size.

    Returns:
        Number of successfully processed pairs.
    """
    img_dir = output_dir / split_name / "images"
    mask_dir = output_dir / split_name / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for img_path, mask_path in tqdm(pairs, desc=f"Processing {split_name}"):
        try:
            resized_img, resized_mask = resize_pair(img_path, mask_path, img_size)

            # Save as PNG
            out_img_path = img_dir / f"{img_path.stem}.png"
            out_mask_path = mask_dir / f"{mask_path.stem}.png"

            cv2.imwrite(str(out_img_path), resized_img)
            cv2.imwrite(str(out_mask_path), resized_mask)
            success_count += 1

        except Exception as e:
            print(f"  Error: {e}")

    return success_count


def main() -> None:
    """Main entry point for data preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare DeepGlobe dataset for AeroSeg training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    img_size = config["data"]["img_size"]
    train_split = config["data"]["train_split"]
    val_split = config["data"]["val_split"]
    random_seed = config["data"]["random_seed"]

    print("=" * 60)
    print("AeroSeg — Data Preparation")
    print("=" * 60)
    print(f"  Raw data directory:    {raw_dir}")
    print(f"  Output directory:      {processed_dir}")
    print(f"  Target image size:     {img_size}×{img_size}")
    print(f"  Train/Val/Test split:  {train_split}/{val_split}/{1 - train_split - val_split:.1f}")
    print(f"  Random seed:           {random_seed}")
    print()

    # Collect pairs
    print("Collecting image-mask pairs...")
    pairs = collect_image_mask_pairs(raw_dir)
    print(f"  Found {len(pairs)} image-mask pairs.\n")

    if not pairs:
        print("No data found. Please download the DeepGlobe dataset and place it in:")
        print(f"  {raw_dir}/train/  (with *_sat.jpg and *_mask.png files)")
        return

    # Split data
    splits = split_data(pairs, train_split, val_split, random_seed)

    # Process and save each split
    print("Resizing and saving...")
    summary = {}
    for split_name, split_pairs in splits.items():
        count = process_and_save(split_pairs, processed_dir, split_name, img_size)
        summary[split_name] = count

    # Print summary
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"  {'Split':<10} {'Images':>10}")
    print(f"  {'-' * 20}")
    for split_name, count in summary.items():
        print(f"  {split_name:<10} {count:>10}")
    print(f"  {'-' * 20}")
    print(f"  {'Total':<10} {sum(summary.values()):>10}")
    print(f"\nOutput saved to: {processed_dir}/")


if __name__ == "__main__":
    main()
