#!/usr/bin/env python3
"""
src/model.py — Model factory for AeroSeg.

Provides functions to create, configure, and load the segmentation model
using segmentation-models-pytorch (smp).
"""

from pathlib import Path
from typing import Callable, Dict

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def get_model(config: Dict) -> nn.Module:
    """Create the segmentation model from config.

    Args:
        config: Configuration dictionary with 'model' and 'data' sections.

    Returns:
        A segmentation model (smp.Unet) ready for training or inference.
    """
    model = smp.Unet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        classes=config["data"]["num_classes"],
        activation=None,
    )
    return model


def get_preprocessing_fn(config: Dict) -> Callable:
    """Get the encoder-specific preprocessing function.

    Returns a function that normalizes images using the encoder's
    pretrained statistics (e.g., ImageNet mean/std for ResNet34).

    Args:
        config: Configuration dictionary with 'model' section.

    Returns:
        Preprocessing function that accepts and returns numpy arrays.
    """
    return smp.encoders.get_preprocessing_fn(
        config["model"]["encoder"],
        config["model"]["encoder_weights"],
    )


def load_checkpoint(
    model: nn.Module, checkpoint_path: str, device: str
) -> nn.Module:
    """Load model weights from a checkpoint file.

    Args:
        model: The model to load weights into.
        checkpoint_path: Path to the .pth checkpoint file.
        device: Device to map the checkpoint to ('cuda', 'mps', or 'cpu').

    Returns:
        The model with loaded weights, set to eval mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location=device, weights_only=False
        )

        # Handle both raw state_dict and wrapped checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()
        return model

    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint {checkpoint_path}: {e}")
