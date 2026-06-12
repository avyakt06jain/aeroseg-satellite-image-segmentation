#!/usr/bin/env python3
"""
src/predict.py — Inference module for AeroSeg.

Runs segmentation inference on individual images, generates colored overlays,
and computes per-class area statistics.

Usage:
    python src/predict.py --image path/to/image.jpg --checkpoint checkpoints/best.pth --config config.yaml --output outputs/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import get_model, load_checkpoint


def get_device() -> str:
    """Auto-detect the best available device.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict_image(
    image_input: Union[str, Path, np.ndarray],
    model: torch.nn.Module,
    config: Dict,
    device: str,
) -> Dict:
    """Run segmentation inference on a single image.

    Args:
        image_input: File path (str/Path) or numpy array (H×W×3, RGB, uint8).
        model: Trained segmentation model in eval mode.
        config: Configuration dictionary.
        device: Device to run inference on.

    Returns:
        Dictionary containing:
            - original_image: H×W×3 uint8 RGB
            - predicted_mask: H×W int (class indices)
            - overlay: H×W×3 uint8 RGB (blended overlay)
            - class_pixel_counts: {class_name: pixel_count}
            - class_area_pct: {class_name: percentage}
    """
    img_size = config["data"]["img_size"]
    class_names = config["class_names"]
    class_colors = config["class_colors_rgb"]
    alpha = config["inference"]["overlay_alpha"]
    num_classes = config["data"]["num_classes"]

    # Load image
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise ValueError(f"Could not read image: {image_path}")
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        original = image_input.copy()
        if original.ndim != 3 or original.shape[2] != 3:
            raise ValueError("Image array must be H×W×3")
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    orig_h, orig_w = original.shape[:2]

    # Preprocess: resize to model input size
    resized = cv2.resize(original, (img_size, img_size), interpolation=cv2.INTER_AREA)
    image_float = resized.astype(np.float32) / 255.0

    # Apply encoder-specific normalization
    preprocess_fn = smp.encoders.get_preprocessing_fn(
        config["model"]["encoder"], config["model"]["encoder_weights"]
    )
    image_normalized = preprocess_fn(image_float).astype(np.float32)

    # Convert to tensor: HWC → CHW
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()
    image_tensor = image_tensor.to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # Resize mask back to original dimensions
    pred_mask_full = cv2.resize(
        pred_mask.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)

    # Generate overlay
    overlay = generate_overlay(original, pred_mask_full, class_colors, alpha)

    # Compute area statistics
    total_pixels = orig_h * orig_w
    class_pixel_counts = {}
    class_area_pct = {}

    for i, name in enumerate(class_names):
        count = int(np.sum(pred_mask_full == i))
        class_pixel_counts[name] = count
        class_area_pct[name] = round(count / total_pixels * 100, 2)

    return {
        "original_image": original,
        "predicted_mask": pred_mask_full,
        "overlay": overlay,
        "class_pixel_counts": class_pixel_counts,
        "class_area_pct": class_area_pct,
    }


def generate_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    class_colors: list,
    alpha: float,
) -> np.ndarray:
    """Generate a colored segmentation overlay blended with the original image.

    Args:
        image: Original RGB image (H×W×3, uint8).
        mask: Class index mask (H×W, int).
        class_colors: List of [R, G, B] colors for each class.
        alpha: Blending factor for the overlay.

    Returns:
        Blended overlay image (H×W×3, uint8).
    """
    overlay = image.copy().astype(np.float32)
    color_mask = np.zeros_like(image, dtype=np.float32)

    for class_idx, color in enumerate(class_colors):
        binary = (mask == class_idx)
        if np.any(binary):
            for c in range(3):
                color_mask[:, :, c][binary] = color[c]

    # Blend: overlay = (1 - alpha) * original + alpha * color_mask
    # Only blend where there's a class prediction (not unknown)
    has_class = np.zeros(mask.shape, dtype=bool)
    for i in range(len(class_colors)):
        has_class |= (mask == i)

    overlay[has_class] = (
        (1 - alpha) * overlay[has_class] + alpha * color_mask[has_class]
    )

    return overlay.clip(0, 255).astype(np.uint8)


def main() -> None:
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="Run AeroSeg inference on an image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input satellite/aerial image.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pth",
        help="Path to model checkpoint (default: checkpoints/best.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/",
        help="Output directory (default: outputs/)",
    )
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    try:
        model = get_model(config)
        model = load_checkpoint(model, args.checkpoint, device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run inference
    try:
        print(f"Processing: {args.image}")
        result = predict_image(args.image, model, config, device)
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # Save outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.image).stem

    # Save mask
    mask_path = output_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), result["predicted_mask"].astype(np.uint8))
    print(f"  Saved mask: {mask_path}")

    # Save overlay
    overlay_path = output_dir / f"{stem}_overlay.png"
    overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(overlay_path), overlay_bgr)
    print(f"  Saved overlay: {overlay_path}")

    # Save area stats
    areas_path = output_dir / f"{stem}_areas.json"
    area_stats = {
        "class_pixel_counts": result["class_pixel_counts"],
        "class_area_pct": result["class_area_pct"],
    }
    with open(areas_path, "w") as f:
        json.dump(area_stats, f, indent=2)
    print(f"  Saved area stats: {areas_path}")

    # Print summary
    print("\nClass Area Summary:")
    print(f"  {'Class':<22} {'Pixels':>10} {'%':>7}")
    print(f"  {'-' * 40}")
    for name in config["class_names"]:
        pct = result["class_area_pct"][name]
        count = result["class_pixel_counts"][name]
        print(f"  {name:<22} {count:>10} {pct:>6.2f}%")


if __name__ == "__main__":
    main()
