#!/usr/bin/env python3
"""
src/evaluate.py — Evaluation module for AeroSeg.

Computes per-class and aggregate metrics (IoU, Dice, F1, Accuracy)
on a given dataset split using a trained model checkpoint.

Usage:
    python src/evaluate.py --config config.yaml --split test --checkpoint checkpoints/best.pth
"""

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassFBetaScore,
    MulticlassJaccardIndex,
)
from tqdm import tqdm
import yaml
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_dataloaders, DeepGlobeDataset
from src.model import get_model, load_checkpoint
from torch.utils.data import DataLoader


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


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> Dict:
    """Compute segmentation metrics from predictions and targets.

    Args:
        pred: Predicted class indices tensor of shape (N,) or (N, H, W).
        target: Ground truth class indices tensor, same shape as pred.
        num_classes: Number of segmentation classes.
        ignore_index: Index value to ignore in metric computation.

    Returns:
        Dictionary with per-class and aggregate metrics.
    """
    device = pred.device

    # Flatten tensors
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Per-class IoU
    iou_per_class = MulticlassJaccardIndex(
        num_classes=num_classes, ignore_index=ignore_index, average="none"
    ).to(device)
    iou_per_class.update(pred_flat, target_flat)
    per_class_iou = iou_per_class.compute().cpu().tolist()

    # Mean IoU
    iou_mean = MulticlassJaccardIndex(
        num_classes=num_classes, ignore_index=ignore_index, average="macro"
    ).to(device)
    iou_mean.update(pred_flat, target_flat)
    miou = iou_mean.compute().item()

    # Per-class Dice (F1 beta=1)
    dice_per_class = MulticlassFBetaScore(
        num_classes=num_classes, beta=1.0, ignore_index=ignore_index, average="none"
    ).to(device)
    dice_per_class.update(pred_flat, target_flat)
    per_class_dice = dice_per_class.compute().cpu().tolist()

    # Mean Dice
    dice_mean = MulticlassFBetaScore(
        num_classes=num_classes, beta=1.0, ignore_index=ignore_index, average="macro"
    ).to(device)
    dice_mean.update(pred_flat, target_flat)
    mean_dice = dice_mean.compute().item()

    # Per-class F1 (same as Dice for binary/multiclass)
    per_class_f1 = per_class_dice.copy()
    macro_f1 = mean_dice

    # Accuracy
    acc_metric = MulticlassAccuracy(
        num_classes=num_classes, ignore_index=ignore_index, average="micro"
    ).to(device)
    acc_metric.update(pred_flat, target_flat)
    accuracy = acc_metric.compute().item()

    return {
        "per_class_iou": per_class_iou,
        "miou": miou,
        "per_class_dice": per_class_dice,
        "mean_dice": mean_dice,
        "per_class_f1": per_class_f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    config: Dict,
    device: str,
) -> Dict:
    """Evaluate the model on a full dataset split.

    Accumulates predictions and computes metrics over the entire split.

    Args:
        model: The trained segmentation model (in eval mode).
        dataloader: DataLoader for the evaluation split.
        config: Configuration dictionary.
        device: Device to run on.

    Returns:
        Dictionary with per-class and aggregate metrics.
    """
    model.eval()
    num_classes = config["data"]["num_classes"]
    ignore_index = config["data"]["ignore_index"]
    class_names = config["class_names"]

    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_metrics(all_preds, all_targets, num_classes, ignore_index)

    # Print formatted table
    print("\n" + "=" * 55)
    print(f"{'Class':<22} {'IoU':>7} {'Dice':>7} {'F1':>7}")
    print("-" * 55)
    for i, name in enumerate(class_names):
        iou = metrics["per_class_iou"][i]
        dice = metrics["per_class_dice"][i]
        f1 = metrics["per_class_f1"][i]
        print(f"{name:<22} {iou:>7.3f} {dice:>7.3f} {f1:>7.3f}")
    print("-" * 55)
    print(
        f"{'Mean (mIoU)':<22} {metrics['miou']:>7.3f} "
        f"{metrics['mean_dice']:>7.3f} {metrics['macro_f1']:>7.3f}"
    )
    print(f"{'Accuracy:':<22} {metrics['accuracy']:>7.3f}")
    print("=" * 55)

    return metrics


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate AeroSeg model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pth",
        help="Path to model checkpoint (default: checkpoints/best.pth)",
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
    print(f"Evaluating on: {args.split} split")
    print(f"Checkpoint: {args.checkpoint}")

    # Create dataset and loader
    try:
        dataset = DeepGlobeDataset(split=args.split, config=config, augment=False)
        dataloader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if len(dataset) == 0:
        print(f"Error: {args.split} dataset is empty.")
        return

    print(f"Dataset size: {len(dataset)} samples")

    # Load model
    try:
        model = get_model(config)
        model = load_checkpoint(model, args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluate
    metrics = evaluate_dataset(model, dataloader, config, device)


if __name__ == "__main__":
    main()
