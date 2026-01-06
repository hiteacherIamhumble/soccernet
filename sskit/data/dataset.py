"""
Dataset for Spiideo SoccerNet SynLoc task.

Loads images and annotations in COCO format for player localization training.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import numpy as np


class SynLocDataset(Dataset):
    """
    Dataset for SoccerNet SynLoc player localization.

    Loads images and annotations, returns data formatted for DETR-style training.
    Images are padded to be divisible by patch_size for VGGT encoder.

    Args:
        root_dir: Path to image directory (e.g., 'data/train')
        coco_json: Path to COCO-format annotation file
        target_size: Target (H, W) after resizing, or None to keep original
        patch_size: Patch size for ViT (images padded to be divisible by this)
        transform: Optional transform to apply to images
        max_players: Maximum number of players (for padding targets)
    """

    def __init__(
        self,
        root_dir: str,
        coco_json: str,
        target_size: Optional[Tuple[int, int]] = None,  # (H, W)
        patch_size: int = 14,
        transform: Optional[Callable] = None,
        max_players: int = 30,
    ):
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.transform = transform
        self.max_players = max_players

        # If target_size specified, round to nearest patch_size multiple
        if target_size is not None:
            H, W = target_size
            H = (H // patch_size) * patch_size
            W = (W // patch_size) * patch_size
            self.target_size = (H, W)
        else:
            self.target_size = None

        # Load COCO annotations
        with open(coco_json, 'r') as f:
            self.coco = json.load(f)

        # Build image_id -> annotations mapping
        self.img_to_anns: Dict[int, List] = defaultdict(list)
        for ann in self.coco['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)

        # Store image info
        self.images = self.coco['images']
        self.img_id_to_info = {img['id']: img for img in self.images}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get image and target.

        Returns:
            image: Tensor of shape [1, 3, H, W] (with sequence dim for VGGT)
            target: dict with:
                - 'positions': [N, 2] normalized positions in [0, 1]
                - 'positions_px': [N, 2] pixel positions
                - 'positions_world': [N, 2] world coordinates (meters)
                - 'num_players': int
                - 'image_id': int
                - 'camera_matrix': [3, 4] tensor
                - 'dist_poly': tensor of distortion coefficients
                - 'undist_poly': tensor of undistortion coefficients
                - 'image_size': (H, W) of processed image
                - 'original_size': (H, W) of original image
        """
        img_info = self.images[idx]
        img_id = img_info['id']

        # Load image
        img_path = self.root_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        orig_W, orig_H = image.size

        # Resize if target_size specified
        if self.target_size is not None:
            target_H, target_W = self.target_size
            image = image.resize((target_W, target_H), Image.BILINEAR)
            scale_x = target_W / orig_W
            scale_y = target_H / orig_H
        else:
            target_H, target_W = orig_H, orig_W
            scale_x, scale_y = 1.0, 1.0

        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Pad to be divisible by patch_size
        H, W = image.shape[1], image.shape[2]
        pad_H = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_W = (self.patch_size - W % self.patch_size) % self.patch_size

        if pad_H > 0 or pad_W > 0:
            image = F.pad(image, (0, pad_W, 0, pad_H), mode='constant', value=0)

        final_H, final_W = image.shape[1], image.shape[2]

        # Add sequence dimension for VGGT: [1, 3, H, W]
        image = image.unsqueeze(0)

        # Apply transform if any
        if self.transform is not None:
            image = self.transform(image)

        # Get annotations
        anns = self.img_to_anns[img_id]

        # Extract player positions
        positions_px = []
        positions_world = []

        for ann in anns:
            # Use ground projection keypoint (index 1) for foot position
            # keypoints format: [[pelvis_x, pelvis_y, vis], [ground_x, ground_y, vis]]
            if 'keypoints' in ann and len(ann['keypoints']) >= 2:
                kp = ann['keypoints'][1]  # Ground projection
                px, py = kp[0], kp[1]
            elif 'bbox' in ann:
                # Fallback: use bottom center of bbox
                x, y, w, h = ann['bbox']
                px, py = x + w / 2, y + h

            # Scale to resized image
            px = px * scale_x
            py = py * scale_y

            positions_px.append([px, py])

            # World coordinates
            if 'position_on_pitch' in ann:
                world_pos = ann['position_on_pitch']
                positions_world.append(world_pos)

        # Convert to tensors
        if len(positions_px) > 0:
            positions_px = torch.tensor(positions_px, dtype=torch.float32)
            # Normalize to [0, 1] relative to padded image size
            positions_norm = positions_px.clone()
            positions_norm[:, 0] = positions_norm[:, 0] / final_W
            positions_norm[:, 1] = positions_norm[:, 1] / final_H
            # Clamp to valid range (some annotations might be outside after padding)
            positions_norm = positions_norm.clamp(0, 1)
        else:
            positions_px = torch.zeros((0, 2), dtype=torch.float32)
            positions_norm = torch.zeros((0, 2), dtype=torch.float32)

        if len(positions_world) > 0:
            positions_world = torch.tensor(positions_world, dtype=torch.float32)
        else:
            positions_world = torch.zeros((0, 2), dtype=torch.float32)

        # Camera parameters
        camera_matrix = torch.tensor(img_info['camera_matrix'], dtype=torch.float32)
        dist_poly = torch.tensor(img_info.get('dist_poly', []), dtype=torch.float32)
        undist_poly = torch.tensor(img_info.get('undist_poly', []), dtype=torch.float32)

        target = {
            'positions': positions_norm,  # Normalized [0, 1]
            'positions_px': positions_px,
            'positions_world': positions_world,
            'num_players': len(positions_px),
            'image_id': img_id,
            'camera_matrix': camera_matrix,
            'dist_poly': dist_poly,
            'undist_poly': undist_poly,
            'image_size': (final_H, final_W),
            'original_size': (orig_H, orig_W),
            'scale': (scale_x, scale_y),
        }

        return image, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Custom collate function for DataLoader.

    Stacks images and keeps targets as list (variable number of players).

    Returns:
        images: [B, S, 3, H, W] where S=1 for single-frame detection
        targets: list of B target dicts
    """
    images, targets = zip(*batch)

    # Stack images along batch dim: [B, 1, 3, H, W]
    # Each image is [1, 3, H, W], stack to get [B, 1, 3, H, W]
    images = torch.stack(images, dim=0)  # [B, 1, 3, H, W]

    return images, list(targets)


class SynLocDatasetMini(SynLocDataset):
    """Mini dataset for testing (uses mini.json with fewer samples)."""

    def __init__(self, root_dir: str, **kwargs):
        coco_json = Path(root_dir).parent / 'annotations' / 'mini.json'
        super().__init__(root_dir, str(coco_json), **kwargs)


def create_dataloaders(
    data_root: str = 'data',
    batch_size: int = 4,
    target_size: Optional[Tuple[int, int]] = (1078, 1918),  # Divisible by 14
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_root: Root data directory containing train/, val/, annotations/
        batch_size: Batch size
        target_size: Target image size (H, W), or None for original
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader
    """
    data_root = Path(data_root)

    train_dataset = SynLocDataset(
        root_dir=str(data_root / 'train'),
        coco_json=str(data_root / 'annotations' / 'train.json'),
        target_size=target_size,
    )

    val_dataset = SynLocDataset(
        root_dir=str(data_root / 'val'),
        coco_json=str(data_root / 'annotations' / 'val.json'),
        target_size=target_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
