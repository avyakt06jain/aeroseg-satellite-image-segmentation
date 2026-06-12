#!/usr/bin/env python3
"""
src/train.py — Training loop for AeroSeg segmentation model.

Trains a U-Net model on the DeepGlobe dataset with combined CE + Dice loss,
mixed precision training, and checkpoint management.

Usage:
    python src/train.py --config config.yaml
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
import yaml
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_dataloaders
from src.model import get_model


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


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    ce_loss_fn: nn.Module,
    dice_loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
    use_amp: bool,
) -> float:
    """Train the model for one epoch.

    Args:
        model: The segmentation model.
        train_loader: Training data loader.
        ce_loss_fn: Cross-entropy loss function.
        dice_loss_fn: Dice loss function.
        optimizer: Optimizer.
        device: Device to run on.
        scaler: GradScaler for mixed precision (None if not using AMP).
        grad_clip: Maximum gradient norm for clipping.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="  Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                ce = ce_loss_fn(logits, masks)
                dice = dice_loss_fn(logits, masks)
                loss = 0.5 * ce + 0.5 * dice

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            ce = ce_loss_fn(logits, masks)
            dice = dice_loss_fn(logits, masks)
            loss = 0.5 * ce + 0.5 * dice

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    ce_loss_fn: nn.Module,
    dice_loss_fn: nn.Module,
    device: str,
    num_classes: int,
    ignore_index: int,
) -> Tuple[float, float]:
    """Validate the model on the validation set.

    Args:
        model: The segmentation model.
        val_loader: Validation data loader.
        ce_loss_fn: Cross-entropy loss function.
        dice_loss_fn: Dice loss function.
        device: Device to run on.
        num_classes: Number of segmentation classes.
        ignore_index: Index to ignore in metrics computation.

    Returns:
        Tuple of (average_val_loss, mean_iou).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    iou_metric = MulticlassJaccardIndex(
        num_classes=num_classes, ignore_index=ignore_index, average="macro"
    ).to(device)

    pbar = tqdm(val_loader, desc="  Val  ", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        ce = ce_loss_fn(logits, masks)
        dice = dice_loss_fn(logits, masks)
        loss = 0.5 * ce + 0.5 * dice

        total_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=1)
        iou_metric.update(preds, masks)

    avg_loss = total_loss / max(num_batches, 1)
    miou = iou_metric.compute().item()

    return avg_loss, miou


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    miou: float,
    path: Path,
) -> None:
    """Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state to save.
        epoch: Current epoch number.
        miou: Current mIoU score.
        path: Path to save the checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "miou": miou,
        },
        str(path),
    )


def init_metrics_log(log_path: Path) -> None:
    """Initialize the metrics CSV log file.

    Args:
        log_path: Path to the CSV log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_miou"])


def append_metrics(
    log_path: Path, epoch: int, train_loss: float, val_loss: float, val_miou: float
) -> None:
    """Append a row of metrics to the CSV log.

    Args:
        log_path: Path to the CSV log file.
        epoch: Current epoch number.
        train_loss: Training loss for this epoch.
        val_loss: Validation loss for this epoch.
        val_miou: Validation mIoU for this epoch.
    """
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_miou:.6f}"])


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train AeroSeg segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
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

    # Training params
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    grad_clip = config["training"]["grad_clip"]
    use_amp = config["training"]["mixed_precision"] and device == "cuda"
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    log_path = Path(config["training"]["log_file"])
    num_classes = config["data"]["num_classes"]
    ignore_index = config["data"]["ignore_index"]

    # Create data loaders
    print("Loading datasets...")
    try:
        train_loader, val_loader, _ = get_dataloaders(config)
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        print("Have you run data_prep.py first?")
        return

    if len(train_loader.dataset) == 0:
        print("Error: Training dataset is empty. Run data_prep.py first.")
        return

    # Create model
    print("Creating model...")
    model = get_model(config)
    model = model.to(device)

    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    dice_loss_fn = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Initialize logging
    init_metrics_log(log_path)

    # Training loop
    best_miou = 0.0
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, ce_loss_fn, dice_loss_fn,
            optimizer, device, scaler, grad_clip, use_amp,
        )

        # Validate
        val_loss, val_miou = validate(
            model, val_loader, ce_loss_fn, dice_loss_fn,
            device, num_classes, ignore_index,
        )

        # Step scheduler
        scheduler.step()

        # Log metrics
        append_metrics(log_path, epoch, train_loss, val_loss, val_miou)

        # Print epoch summary
        improved = ""
        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(model, optimizer, epoch, val_miou, checkpoint_dir / "best.pth")
            improved = " ★ New best!"

        # Always save latest
        save_checkpoint(model, optimizer, epoch, val_miou, checkpoint_dir / "last.pth")

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_miou:.4f}{improved}"
        )

    print("=" * 70)
    print(f"Training complete! Best mIoU: {best_miou:.4f}")
    print(f"Best checkpoint: {checkpoint_dir / 'best.pth'}")
    print(f"Metrics log: {log_path}")


if __name__ == "__main__":
    main()
