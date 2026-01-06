#!/usr/bin/env python3
"""
Training script for DETR-style player decoder with frozen VGGT encoder.

Usage:
    python scripts/train_decoder.py --data_root data --epochs 100 --batch_size 4
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sskit.models import DETRPlayerDecoder, HungarianMatcher, SetCriterion
from sskit.data import SynLocDataset
from sskit.data.dataset import collate_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train DETR player decoder')

    # Data
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root data directory')
    parser.add_argument('--target_size', type=int, nargs=2, default=[1078, 1918],
                        help='Target image size (H W), divisible by 14')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Decoder hidden dimension')
    parser.add_argument('--num_queries', type=int, default=30,
                        help='Number of object queries')
    parser.add_argument('--num_decoder_layers', type=int, default=6,
                        help='Number of transformer decoder layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=0.1,
                        help='Gradient clipping max norm')

    # Loss
    parser.add_argument('--weight_position', type=float, default=5.0,
                        help='Position loss weight')
    parser.add_argument('--weight_confidence', type=float, default=1.0,
                        help='Confidence loss weight')
    parser.add_argument('--weight_no_object', type=float, default=0.1,
                        help='No-object weight in confidence loss')

    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Log every N iterations')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--vggt_model', type=str, default='facebook/VGGT-1B',
                        help='VGGT model name or path')

    return parser.parse_args()


def load_vggt_encoder(model_name: str, device: torch.device):
    """Load frozen VGGT encoder."""
    logger.info(f'Loading VGGT encoder from {model_name}...')

    # Import VGGT
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'vggt'))
        from vggt.models.vggt import VGGT
    except ImportError:
        raise ImportError('VGGT not found. Make sure vggt/ directory exists.')

    # Load model
    model = VGGT.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    logger.info('VGGT encoder loaded and frozen.')
    return model


def create_dataloaders(args):
    """Create train and validation dataloaders."""
    data_root = Path(args.data_root)

    train_dataset = SynLocDataset(
        root_dir=str(data_root / 'train'),
        coco_json=str(data_root / 'annotations' / 'train.json'),
        target_size=tuple(args.target_size) if args.target_size else None,
    )

    val_dataset = SynLocDataset(
        root_dir=str(data_root / 'val'),
        coco_json=str(data_root / 'annotations' / 'val.json'),
        target_size=tuple(args.target_size) if args.target_size else None,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    logger.info(f'Train dataset: {len(train_dataset)} images')
    logger.info(f'Val dataset: {len(val_dataset)} images')

    return train_loader, val_loader


def train_one_epoch(
    encoder,
    decoder,
    criterion,
    dataloader,
    optimizer,
    device,
    epoch,
    args,
):
    """Train for one epoch."""
    decoder.train()

    total_loss = 0.0
    total_loss_pos = 0.0
    total_loss_conf = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for i, (images, targets) in enumerate(pbar):
        # Move to device
        images = images.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
                   for t in targets]

        # Forward through frozen encoder
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = encoder.aggregator(images)

        # Forward through decoder
        outputs = decoder(aggregated_tokens_list, patch_start_idx)

        # Compute loss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['loss']

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)

        optimizer.step()

        # Logging
        total_loss += loss.item()
        total_loss_pos += loss_dict['loss_position'].item()
        total_loss_conf += loss_dict['loss_confidence'].item()
        num_batches += 1

        if (i + 1) % args.log_freq == 0:
            pbar.set_postfix({
                'loss': f'{total_loss / num_batches:.4f}',
                'pos': f'{total_loss_pos / num_batches:.4f}',
                'conf': f'{total_loss_conf / num_batches:.4f}',
            })

    return {
        'loss': total_loss / num_batches,
        'loss_position': total_loss_pos / num_batches,
        'loss_confidence': total_loss_conf / num_batches,
    }


@torch.no_grad()
def validate(encoder, decoder, criterion, dataloader, device):
    """Validate the model."""
    decoder.eval()

    total_loss = 0.0
    total_loss_pos = 0.0
    total_loss_conf = 0.0
    num_batches = 0

    for images, targets in tqdm(dataloader, desc='Validation'):
        images = images.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
                   for t in targets]

        # Forward
        aggregated_tokens_list, patch_start_idx = encoder.aggregator(images)
        outputs = decoder(aggregated_tokens_list, patch_start_idx)

        # Compute loss
        loss_dict = criterion(outputs, targets)

        total_loss += loss_dict['loss'].item()
        total_loss_pos += loss_dict['loss_position'].item()
        total_loss_conf += loss_dict['loss_confidence'].item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'loss_position': total_loss_pos / num_batches,
        'loss_confidence': total_loss_conf / num_batches,
    }


def save_checkpoint(decoder, optimizer, scheduler, epoch, args, is_best=False):
    """Save checkpoint."""
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'args': vars(args),
    }

    # Save latest
    torch.save(checkpoint, save_dir / 'latest.pt')

    # Save periodic
    if (epoch + 1) % args.save_freq == 0:
        torch.save(checkpoint, save_dir / f'epoch_{epoch+1}.pt')

    # Save best
    if is_best:
        torch.save(checkpoint, save_dir / 'best.pt')

    logger.info(f'Checkpoint saved: epoch {epoch + 1}')


def main():
    args = parse_args()

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA not available, using CPU')
        args.device = 'cpu'
    device = torch.device(args.device)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load VGGT encoder
    encoder = load_vggt_encoder(args.vggt_model, device)

    # Create decoder
    decoder = DETRPlayerDecoder(
        dim_in=2048,  # VGGT concat dim
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    logger.info(f'Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}')

    # Create loss
    matcher = HungarianMatcher(
        cost_position=args.weight_position,
        cost_confidence=args.weight_confidence,
    )
    criterion = SetCriterion(
        matcher=matcher,
        weight_position=args.weight_position,
        weight_confidence=args.weight_confidence,
        weight_no_object=args.weight_no_object,
    )

    # Create optimizer
    optimizer = AdamW(
        decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_epochs],
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Resumed from epoch {start_epoch}')

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Training loop
    logger.info('Starting training...')

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_one_epoch(
            encoder, decoder, criterion, train_loader, optimizer, device, epoch, args
        )

        # Validate
        val_metrics = validate(encoder, decoder, criterion, val_loader, device)

        # Update scheduler
        scheduler.step()

        # Log
        logger.info(
            f'Epoch {epoch + 1}/{args.epochs} - '
            f'Train: loss={train_metrics["loss"]:.4f}, pos={train_metrics["loss_position"]:.4f}, conf={train_metrics["loss_confidence"]:.4f} | '
            f'Val: loss={val_metrics["loss"]:.4f}, pos={val_metrics["loss_position"]:.4f}, conf={val_metrics["loss_confidence"]:.4f} | '
            f'LR: {scheduler.get_last_lr()[0]:.6f}'
        )

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']

        save_checkpoint(decoder, optimizer, scheduler, epoch, args, is_best=is_best)

    logger.info('Training complete!')


if __name__ == '__main__':
    main()
