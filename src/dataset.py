#!/usr/bin/env python3
"""
src/dataset.py — DeepGlobe Land Cover Dataset for AeroSeg.

Provides a PyTorch Dataset class for loading preprocessed satellite imagery
and converting RGB masks to class indices. Includes augmentation pipelines.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset


# RGB → class index lookup table
RGB_TO_CLASS = {
    (0, 255, 255): 0,    # Urban / Impervious
    (255, 255, 0): 1,     # Agriculture
    (255, 0, 255): 2,     # Rangeland
    (0, 255, 0): 3,       # Forest / Vegetation
    (0, 0, 255): 4,       # Water
    (255, 255, 255): 5,   # Barren Land
}
UNKNOWN_RGB = (0, 0, 0)  # Unknown → ignore_index


def rgb_mask_to_class_indices(
    mask_rgb: np.ndarray, ignore_index: int = 255
) -> np.ndarray:
    """Convert an RGB mask to a class index mask.

    Uses exact integer tuple matching for RGB → class index conversion.

    Args:
        mask_rgb: RGB mask array of shape (H, W, 3) with uint8 values.
        ignore_index: Value to assign to unknown/unmapped pixels.

    Returns:
        Class index mask of shape (H, W) with dtype int64.
    """
    h, w = mask_rgb.shape[:2]
    class_mask = np.full((h, w), fill_value=ignore_index, dtype=np.int64)

    for rgb, class_idx in RGB_TO_CLASS.items():
        # Create boolean mask for exact RGB match
        match = (
            (mask_rgb[:, :, 0] == rgb[0])
            & (mask_rgb[:, :, 1] == rgb[1])
            & (mask_rgb[:, :, 2] == rgb[2])
        )
        class_mask[match] = class_idx

    return class_mask


def get_augmentation_pipeline() -> A.Compose:
    """Get the training augmentation pipeline.

    Returns:
        Albumentations Compose pipeline for training augmentations.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
        ),
        A.RandomBrightnessContrast(p=0.3),
    ])


class DeepGlobeDataset(Dataset):
    """PyTorch Dataset for DeepGlobe Land Cover Classification.

    Loads preprocessed satellite images and RGB masks, converts masks to
    class indices, applies augmentations, and normalizes using encoder stats.

    Args:
        split: Dataset split ('train', 'val', or 'test').
        config: Configuration dictionary.
        augment: Whether to apply data augmentation.
    """

    def __init__(self, split: str, config: Dict, augment: bool = False) -> None:
        """Initialize the dataset.

        Args:
            split: Dataset split ('train', 'val', or 'test').
            config: Configuration dictionary.
            augment: Whether to apply data augmentation.
        """
        self.split = split
        self.config = config
        self.augment = augment
        self.ignore_index = config["data"]["ignore_index"]

        # Set up paths
        processed_dir = Path(config["data"]["processed_dir"])
        self.img_dir = processed_dir / split / "images"
        self.mask_dir = processed_dir / split / "masks"

        # Collect image files
        if self.img_dir.exists():
            self.image_files = sorted(self.img_dir.glob("*.png"))
        else:
            self.image_files = []
            print(f"Warning: Image directory not found: {self.img_dir}")

        # Get preprocessing function from encoder
        self.preprocess_fn = smp.encoders.get_preprocessing_fn(
            config["model"]["encoder"],
            config["model"]["encoder_weights"],
        )

        # Set up augmentation
        self.aug_pipeline = get_augmentation_pipeline() if augment else None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary with 'image' tensor (3, H, W) and 'mask' tensor (H, W).
        """
        img_path = self.image_files[idx]

        # Derive mask path from image path
        # Image: xxx_sat.png → Mask: xxx_mask.png
        mask_name = img_path.name.replace("_sat", "_mask")
        mask_path = self.mask_dir / mask_name

        try:
            # Load image as RGB float32
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0

            # Load mask as RGB
            mask_rgb = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            if mask_rgb is None:
                raise ValueError(f"Could not read mask: {mask_path}")
            mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

            # Convert RGB mask to class indices
            mask = rgb_mask_to_class_indices(mask_rgb, self.ignore_index)

        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx} ({img_path.name}): {e}")

        # Apply augmentations (if training)
        if self.aug_pipeline is not None:
            augmented = self.aug_pipeline(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Apply encoder-specific preprocessing (ImageNet normalization)
        # The preprocessing function expects HWC float array
        image = self.preprocess_fn(image).astype(np.float32)

        # Convert to tensors
        # Image: HWC → CHW
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()

        return {"image": image_tensor, "mask": mask_tensor}


def get_dataloaders(
    config: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, val, and test splits.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = DeepGlobeDataset(split="train", config=config, augment=True)
    val_dataset = DeepGlobeDataset(split="val", config=config, augment=False)
    test_dataset = DeepGlobeDataset(split="test", config=config, augment=False)

    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Dataset sizes — Train: {len(train_dataset)}, "
          f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
