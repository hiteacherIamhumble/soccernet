#!/usr/bin/env python3
"""
Single image inference comparison: New Model vs Baseline (YOLOv5)

Usage:
    python scripts/compare_inference.py --checkpoint checkpoints/best.pt --image example/rgb.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sskit import imread, Draw, image_to_ground, load_camera, imshape, world_to_image, match
from sskit.models import DETRPlayerDecoder
from sskit.data import SynLocDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Compare inference: New Model vs Baseline')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default='example/rgb.jpg',
                        help='Path to input image')
    parser.add_argument('--scene_dir', type=str, default=None,
                        help='Scene directory with camera params (default: image parent dir)')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--output', type=str, default='comparison.png',
                        help='Output visualization path')
    parser.add_argument('--vggt_model', type=str, default='facebook/VGGT-1B',
                        help='VGGT model name')
    return parser.parse_args()


def load_new_model(checkpoint_path, vggt_model, device):
    """Load VGGT encoder and DETR decoder."""
    print(f'Loading VGGT encoder from {vggt_model}...')
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'vggt'))
    from vggt.models.vggt import VGGT

    encoder = VGGT.from_pretrained(vggt_model)
    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    print(f'Loading decoder from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get('args', {})

    decoder = DETRPlayerDecoder(
        dim_in=2048,
        hidden_dim=ckpt_args.get('hidden_dim', 128),
        num_queries=ckpt_args.get('num_queries', 30),
        num_decoder_layers=ckpt_args.get('num_decoder_layers', 3),
        num_heads=ckpt_args.get('num_heads', 4),
        ffn_dim=ckpt_args.get('ffn_dim', 512),
        dropout=0.0,
    ).to(device)

    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()

    return encoder, decoder


def load_baseline_model():
    """Load YOLOv5 baseline model."""
    print('Loading YOLOv5x6 baseline...')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)
    return model


@torch.no_grad()
def run_new_model(encoder, decoder, image_path, camera_matrix, undist_poly, device, conf_threshold=0.5):
    """Run inference with new VGGT + DETR model."""
    from PIL import Image
    import torch.nn.functional as F

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    orig_W, orig_H = image.size

    # Resize to model input size (divisible by 14)
    target_H, target_W = 1078, 1918
    image_resized = image.resize((target_W, target_H), Image.BILINEAR)

    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0

    # Add sequence dimension for VGGT: [1, 1, 3, H, W]
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)

    # Forward through encoder
    aggregated_tokens_list, patch_start_idx = encoder.aggregator(img_tensor)

    # Forward through decoder
    outputs = decoder(aggregated_tokens_list, patch_start_idx)

    # Get predictions
    positions = outputs['positions'][0]  # [N, 2] normalized
    confidences = torch.sigmoid(outputs['confidences'][0])  # [N] apply sigmoid

    # Filter by confidence
    mask = confidences > conf_threshold
    positions = positions[mask]
    confidences = confidences[mask]

    if len(positions) == 0:
        return torch.zeros((0, 2)), torch.zeros((0, 2)), torch.zeros(0)

    # Denormalize to original image size
    pos_px = positions.clone()
    pos_px[:, 0] = pos_px[:, 0] * orig_W
    pos_px[:, 1] = pos_px[:, 1] * orig_H

    # Project to world coordinates
    pos_normalized = pos_px.clone()
    pos_normalized[:, 0] = (pos_normalized[:, 0] - (orig_W - 1) / 2) / orig_W
    pos_normalized[:, 1] = (pos_normalized[:, 1] - (orig_H - 1) / 2) / orig_W  # Note: divide by W

    pos_world = image_to_ground(camera_matrix, undist_poly, pos_normalized.cpu())
    pos_world = pos_world[:, :2]

    return pos_px.cpu(), pos_world, confidences.cpu()


def run_baseline(model, image_path, camera_matrix, undist_poly, conf_threshold=0.5):
    """Run inference with YOLOv5 baseline."""
    _, h, w = imshape(image_path)

    res = model(str(image_path), 1280)
    dets = []
    confs = []
    for x1, y1, x2, y2, conf, cls in res.xyxy[0]:
        if res.names[int(cls)] == 'person' and conf > conf_threshold:
            dets.append([(x1 + x2) / 2, y2])  # Bottom center
            confs.append(conf.item())

    if len(dets) == 0:
        return torch.zeros((0, 2)), torch.zeros((0, 2)), torch.zeros(0)

    dets = torch.tensor(dets)
    confs = torch.tensor(confs)

    # Project to world coordinates
    pos_normalized = (dets - torch.tensor([(w - 1) / 2, (h - 1) / 2])) / w
    pos_world = image_to_ground(camera_matrix, undist_poly, pos_normalized)
    pos_world = pos_world[:, :2]

    return dets, pos_world, confs


def main():
    args = parse_args()

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        args.device = 'cpu'
    device = torch.device(args.device)

    # Load scene data
    image_path = Path(args.image)
    scene_dir = Path(args.scene_dir) if args.scene_dir else image_path.parent

    camera_matrix, dist_poly, undist_poly = load_camera(scene_dir)
    _, h, w = imshape(image_path)

    # Load ground truth
    objects_file = scene_dir / 'objects.json'
    if objects_file.exists():
        objects = json.loads(objects_file.read_bytes())
        gt_world = torch.tensor([
            obj['keypoints'].get('pelvis')[:2]
            for obj in objects.values()
            if obj['class'] == 'human' and 'pelvis' in obj['keypoints']
        ])
        # Project GT to image
        gt_world_3d = torch.cat([gt_world, torch.zeros(len(gt_world), 1)], dim=1)
        gt_normalized = world_to_image(camera_matrix, dist_poly, gt_world_3d)
        gt_px = gt_normalized * w + torch.tensor([(w - 1) / 2, (h - 1) / 2])
    else:
        gt_world = torch.zeros((0, 2))
        gt_px = torch.zeros((0, 2))
        print('Warning: No objects.json found, skipping ground truth')

    # Run new model
    print('\n=== Running New Model (VGGT + DETR) ===')
    encoder, decoder = load_new_model(args.checkpoint, args.vggt_model, device)
    new_px, new_world, new_conf = run_new_model(
        encoder, decoder, image_path, camera_matrix, undist_poly, device, args.conf_threshold
    )
    print(f'Detections: {len(new_px)}')

    # Run baseline
    print('\n=== Running Baseline (YOLOv5x6) ===')
    baseline_model = load_baseline_model()
    base_px, base_world, base_conf = run_baseline(
        baseline_model, image_path, camera_matrix, undist_poly, args.conf_threshold
    )
    print(f'Detections: {len(base_px)}')

    # Evaluate both
    print('\n=== Evaluation Results ===')
    print(f'Ground Truth Players: {len(gt_world)}')

    if len(gt_world) > 0:
        # New model metrics
        if len(new_world) > 0:
            detected, missed, extra, distances, matches = match(gt_world, new_world)
            print(f'\nNew Model (VGGT + DETR):')
            print(f'  Detected: {detected}/{len(gt_world)}')
            print(f'  Missed: {missed}')
            print(f'  False Positives: {extra}')
            print(f'  Avg Distance: {distances.mean():.3f}m' if len(distances) > 0 else '  Avg Distance: N/A')
            new_matches = matches
        else:
            print(f'\nNew Model: No detections')
            new_matches = []

        # Baseline metrics
        if len(base_world) > 0:
            detected, missed, extra, distances, matches = match(gt_world, base_world)
            print(f'\nBaseline (YOLOv5x6):')
            print(f'  Detected: {detected}/{len(gt_world)}')
            print(f'  Missed: {missed}')
            print(f'  False Positives: {extra}')
            print(f'  Avg Distance: {distances.mean():.3f}m' if len(distances) > 0 else '  Avg Distance: N/A')
            base_matches = matches
        else:
            print(f'\nBaseline: No detections')
            base_matches = []

    # Visualization
    print(f'\nSaving visualization to {args.output}...')

    # Create side-by-side visualization
    from PIL import Image

    img = np.array(Image.open(image_path))

    # Left: New model, Right: Baseline
    combined = np.concatenate([img, img], axis=1)

    drw = Draw(combined)

    # Draw on left half (new model)
    offset_left = 0
    if len(gt_px) > 0:
        gt_left = gt_px.clone()
        gt_left[:, 0] += offset_left
        drw.circle(gt_left, 8, 'green')

    if len(new_px) > 0:
        new_left = new_px.clone()
        new_left[:, 0] += offset_left
        drw.circle(new_left, 8, 'red')

    # Draw matches for new model
    if len(gt_world) > 0 and len(new_world) > 0:
        for j, i in new_matches:
            drw.line([gt_px[i] + torch.tensor([offset_left, 0]),
                      new_px[j] + torch.tensor([offset_left, 0])], 'blue', 2)

    # Draw on right half (baseline)
    offset_right = w
    if len(gt_px) > 0:
        gt_right = gt_px.clone()
        gt_right[:, 0] += offset_right
        drw.circle(gt_right, 8, 'green')

    if len(base_px) > 0:
        base_right = base_px.clone()
        base_right[:, 0] += offset_right
        drw.circle(base_right, 8, 'red')

    # Draw matches for baseline
    if len(gt_world) > 0 and len(base_world) > 0:
        for j, i in base_matches:
            drw.line([gt_px[i] + torch.tensor([offset_right, 0]),
                      base_px[j] + torch.tensor([offset_right, 0])], 'blue', 2)

    # Add labels
    drw.text((50, 50), 'New Model (VGGT + DETR)', font_size=40)
    drw.text((w + 50, 50), 'Baseline (YOLOv5x6)', font_size=40)
    drw.text((50, 100), f'Detections: {len(new_px)}', font_size=30)
    drw.text((w + 50, 100), f'Detections: {len(base_px)}', font_size=30)

    drw.save(args.output)
    print(f'Done! Visualization saved to {args.output}')
    print('Legend: Green=GT, Red=Predictions, Blue=Matches')


if __name__ == '__main__':
    main()
