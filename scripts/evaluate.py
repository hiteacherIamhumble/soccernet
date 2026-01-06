#!/usr/bin/env python3
"""
Evaluation script for DETR-style player decoder.

Evaluates predictions using:
1. Hungarian matching metrics (detected, missed, false positives)
2. mAP-LocSim (COCO-style AP with location similarity)

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --data_root data
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sskit.models import DETRPlayerDecoder
from sskit.data import SynLocDataset
from sskit.data.dataset import collate_fn
from sskit.camera import image_to_ground
from sskit.metric import match

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DETR player decoder')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root data directory')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test', 'mini'],
                        help='Dataset split to evaluate')
    parser.add_argument('--target_size', type=int, nargs=2, default=[1078, 1918],
                        help='Target image size (H W)')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--vggt_model', type=str, default='facebook/VGGT-1B',
                        help='VGGT model name or path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for predictions (JSON)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--vis_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')

    return parser.parse_args()


def load_models(args, device):
    """Load VGGT encoder and decoder."""
    # Load VGGT
    logger.info(f'Loading VGGT encoder from {args.vggt_model}...')
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'vggt'))
        from vggt.models.vggt import VGGT
    except ImportError:
        raise ImportError('VGGT not found.')

    encoder = VGGT.from_pretrained(args.vggt_model)
    encoder = encoder.to(device)
    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    # Load decoder
    logger.info(f'Loading decoder from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get('args', {})

    decoder = DETRPlayerDecoder(
        dim_in=2048,
        hidden_dim=ckpt_args.get('hidden_dim', 256),
        num_queries=ckpt_args.get('num_queries', 30),
        num_decoder_layers=ckpt_args.get('num_decoder_layers', 6),
        num_heads=ckpt_args.get('num_heads', 8),
        dropout=0.0,  # No dropout during evaluation
    ).to(device)

    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()

    logger.info('Models loaded.')
    return encoder, decoder


def postprocess_predictions(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict],
    conf_threshold: float = 0.5,
) -> List[Dict]:
    """
    Postprocess decoder outputs to world coordinates.

    Args:
        outputs: dict with 'positions' [B, N, 2] and 'confidences' [B, N]
        targets: list of target dicts with camera params
        conf_threshold: confidence threshold

    Returns:
        List of prediction dicts with positions_px, positions_world, scores
    """
    B = outputs['positions'].shape[0]
    results = []

    for b in range(B):
        # Get predictions above threshold
        mask = outputs['confidences'][b] > conf_threshold
        pos_norm = outputs['positions'][b][mask]  # [N_det, 2]
        scores = outputs['confidences'][b][mask]  # [N_det]

        # Denormalize to pixel coordinates
        H, W = targets[b]['image_size']
        pos_px = pos_norm.clone()
        pos_px[:, 0] = pos_px[:, 0] * W
        pos_px[:, 1] = pos_px[:, 1] * H

        # Project to world coordinates
        if len(pos_px) > 0:
            camera_matrix = targets[b]['camera_matrix']
            undist_poly = targets[b]['undist_poly']

            # Normalize for sskit camera: (px - center) / W
            pos_normalized = pos_px.clone()
            pos_normalized[:, 0] = (pos_normalized[:, 0] - (W - 1) / 2) / W
            pos_normalized[:, 1] = (pos_normalized[:, 1] - (H - 1) / 2) / W  # Note: divide by W

            # Convert to world coordinates
            pos_world = image_to_ground(camera_matrix, undist_poly, pos_normalized)
            pos_world = pos_world[:, :2]  # [x, y] only
        else:
            pos_world = torch.zeros((0, 2), device=pos_px.device)

        results.append({
            'positions_px': pos_px.cpu(),
            'positions_world': pos_world.cpu(),
            'scores': scores.cpu(),
            'image_id': targets[b]['image_id'],
        })

    return results


def evaluate_hungarian(
    predictions: List[Dict],
    targets: List[Dict],
    threshold: float = 0.25,
) -> Dict:
    """
    Evaluate using Hungarian matching.

    Args:
        predictions: list of prediction dicts
        targets: list of target dicts
        threshold: distance threshold for matching (meters)

    Returns:
        dict with evaluation metrics
    """
    total_detected = 0
    total_missed = 0
    total_extra = 0
    total_gt = 0
    all_distances = []

    for pred, tgt in zip(predictions, targets):
        gt_pos = tgt['positions_world']  # [N_gt, 2]
        pred_pos = pred['positions_world']  # [N_pred, 2]

        if len(gt_pos) == 0:
            total_extra += len(pred_pos)
            continue

        if len(pred_pos) == 0:
            total_missed += len(gt_pos)
            total_gt += len(gt_pos)
            continue

        # Hungarian matching
        detected, missed, extra, distances, matches = match(
            gt_pos, pred_pos, threshold=threshold
        )

        total_detected += detected
        total_missed += missed
        total_extra += extra
        total_gt += len(gt_pos)

        if len(distances) > 0:
            all_distances.extend(distances.tolist())

    # Compute metrics
    precision = total_detected / (total_detected + total_extra + 1e-8)
    recall = total_detected / (total_gt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    avg_distance = np.mean(all_distances) if all_distances else 0.0

    return {
        'detected': total_detected,
        'missed': total_missed,
        'false_positives': total_extra,
        'total_gt': total_gt,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_distance': avg_distance,
    }


def evaluate_locsim(
    predictions: List[Dict],
    targets: List[Dict],
    coco_json: str,
) -> Dict:
    """
    Evaluate using mAP-LocSim (COCO-style).

    Args:
        predictions: list of prediction dicts
        targets: list of target dicts
        coco_json: path to COCO annotation file

    Returns:
        dict with COCO evaluation metrics
    """
    try:
        from xtcocotools.coco import COCO
        from sskit.coco import LocSimCOCOeval
    except ImportError:
        logger.warning('xtcocotools not available, skipping LocSim evaluation')
        return {}

    # Convert predictions to COCO format
    coco_results = []
    for pred in predictions:
        img_id = pred['image_id']
        for i in range(len(pred['scores'])):
            x, y = pred['positions_px'][i].tolist()
            world_x, world_y = pred['positions_world'][i].tolist()

            coco_results.append({
                'image_id': img_id,
                'category_id': 1,
                'keypoints': [x, y, 1, x, y + 50, 1],  # pelvis, ground
                'position_on_pitch': [world_x, world_y],
                'score': pred['scores'][i].item(),
            })

    if len(coco_results) == 0:
        logger.warning('No predictions to evaluate')
        return {}

    # Run COCO evaluation
    coco_gt = COCO(coco_json)
    coco_dt = coco_gt.loadRes(coco_results)

    coco_eval = LocSimCOCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.params.position_from_keypoint_index = 0
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        'mAP': coco_eval.stats[0],
        'mAP_50': coco_eval.stats[1],
        'mAP_75': coco_eval.stats[2],
    }


def visualize_predictions(
    image: torch.Tensor,
    predictions: Dict,
    targets: Dict,
    output_path: str,
):
    """Visualize predictions on image."""
    from sskit import Draw, imread

    # Convert image tensor to numpy
    img = image.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    # Create Draw object
    from PIL import Image
    pil_img = Image.fromarray(img)
    drw = Draw(np.array(pil_img))

    # Draw ground truth (green)
    gt_px = targets['positions_px']
    if len(gt_px) > 0:
        drw.circle(gt_px, 5, 'green')

    # Draw predictions (red)
    pred_px = predictions['positions_px']
    if len(pred_px) > 0:
        drw.circle(pred_px, 5, 'red')

    # Save
    drw.save(output_path)


@torch.no_grad()
def main():
    args = parse_args()

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA not available, using CPU')
        args.device = 'cpu'
    device = torch.device(args.device)

    # Load models
    encoder, decoder = load_models(args, device)

    # Create dataloader
    data_root = Path(args.data_root)
    if args.split == 'mini':
        coco_json = data_root / 'annotations' / 'mini.json'
        img_dir = data_root / 'train'
    else:
        coco_json = data_root / 'annotations' / f'{args.split}.json'
        img_dir = data_root / args.split

    dataset = SynLocDataset(
        root_dir=str(img_dir),
        coco_json=str(coco_json),
        target_size=tuple(args.target_size) if args.target_size else None,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    logger.info(f'Evaluating on {args.split} split ({len(dataset)} images)...')

    # Run inference
    all_predictions = []
    all_targets = []

    for images, targets in tqdm(dataloader, desc='Inference'):
        images = images.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
                   for t in targets]

        # Forward
        aggregated_tokens_list, patch_start_idx = encoder.aggregator(images)
        outputs = decoder(aggregated_tokens_list, patch_start_idx)

        # Postprocess
        predictions = postprocess_predictions(outputs, targets, args.conf_threshold)
        all_predictions.extend(predictions)
        all_targets.extend(targets)

    # Evaluate with Hungarian matching
    logger.info('Computing Hungarian matching metrics...')
    hungarian_metrics = evaluate_hungarian(all_predictions, all_targets)

    print('\n=== Hungarian Matching Results ===')
    print(f'Detected: {hungarian_metrics["detected"]}/{hungarian_metrics["total_gt"]}')
    print(f'Missed: {hungarian_metrics["missed"]}')
    print(f'False Positives: {hungarian_metrics["false_positives"]}')
    print(f'Precision: {hungarian_metrics["precision"]:.4f}')
    print(f'Recall: {hungarian_metrics["recall"]:.4f}')
    print(f'F1 Score: {hungarian_metrics["f1"]:.4f}')
    print(f'Avg Distance: {hungarian_metrics["avg_distance"]:.4f}m')

    # Evaluate with LocSim
    logger.info('Computing LocSim metrics...')
    locsim_metrics = evaluate_locsim(all_predictions, all_targets, str(coco_json))

    if locsim_metrics:
        print('\n=== LocSim Results ===')
        print(f'mAP: {locsim_metrics["mAP"]:.4f}')
        print(f'mAP@50: {locsim_metrics["mAP_50"]:.4f}')
        print(f'mAP@75: {locsim_metrics["mAP_75"]:.4f}')

    # Save predictions
    if args.output:
        logger.info(f'Saving predictions to {args.output}...')
        output_data = {
            'predictions': [
                {
                    'image_id': p['image_id'],
                    'positions_world': p['positions_world'].tolist(),
                    'scores': p['scores'].tolist(),
                }
                for p in all_predictions
            ],
            'metrics': {
                'hungarian': hungarian_metrics,
                'locsim': locsim_metrics,
            },
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

    # Visualize
    if args.visualize:
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving visualizations to {vis_dir}...')

        for i, (pred, tgt) in enumerate(zip(all_predictions[:20], all_targets[:20])):
            # Load original image
            img_path = img_dir / f'{pred["image_id"]:06d}.jpg'
            if img_path.exists():
                visualize_predictions(
                    dataset[i][0],
                    pred,
                    tgt,
                    str(vis_dir / f'{pred["image_id"]:06d}.png'),
                )

    logger.info('Evaluation complete!')


if __name__ == '__main__':
    main()
