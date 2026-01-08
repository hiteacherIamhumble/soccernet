"""
DETR-style loss functions for set prediction.

Includes Hungarian matching and set prediction loss with position and confidence terms.
"""

from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from sskit.camera import image_to_ground


class HungarianMatcher(nn.Module):
    """
    Bipartite matching between predictions and ground truth using Hungarian algorithm.

    Computes assignment that minimizes total cost = position_cost + confidence_cost.
    """

    def __init__(
        self,
        cost_position: float = 5.0,
        cost_confidence: float = 1.0,
    ):
        """
        Args:
            cost_position: Weight for L1 position cost
            cost_confidence: Weight for confidence/classification cost
        """
        super().__init__()
        self.cost_position = cost_position
        self.cost_confidence = cost_confidence

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching.

        Args:
            outputs: dict with:
                - 'positions': [B, num_queries, 2] predicted positions (normalized [0,1])
                - 'confidences': [B, num_queries] predicted confidence scores
            targets: list of B dicts, each with:
                - 'positions': [N_gt, 2] ground truth positions (normalized [0,1])

        Returns:
            List of B tuples (pred_indices, gt_indices), where:
                - pred_indices: indices of matched predictions
                - gt_indices: indices of matched ground truth
        """
        B, num_queries = outputs['positions'].shape[:2]

        indices = []
        for b in range(B):
            pred_pos = outputs['positions'][b]  # [num_queries, 2]
            pred_conf = outputs['confidences'][b]  # [num_queries]
            gt_pos = targets[b]['positions']  # [N_gt, 2]

            N_gt = gt_pos.shape[0]

            if N_gt == 0:
                # No ground truth, return empty indices
                indices.append((
                    torch.tensor([], dtype=torch.int64, device=pred_pos.device),
                    torch.tensor([], dtype=torch.int64, device=pred_pos.device),
                ))
                continue

            # Convert to float32 for numerical stability (AMP can cause issues)
            pred_pos_f32 = pred_pos.float()
            gt_pos_f32 = gt_pos.float()
            pred_conf_f32 = pred_conf.float()

            # Compute L1 distance cost matrix
            # [num_queries, N_gt]
            cost_pos = torch.cdist(pred_pos_f32, gt_pos_f32, p=1)

            # Classification cost: -log(confidence) for positive matches
            # We want high confidence for matched predictions
            # Cost = -log(p) = log(1/p), lower confidence = higher cost
            # Note: pred_conf are logits, apply sigmoid to get probabilities
            pred_prob = torch.sigmoid(pred_conf_f32)
            # Clamp probability to avoid log(0) and extreme costs
            pred_prob = pred_prob.clamp(min=1e-6, max=1.0 - 1e-6)
            cost_conf = -torch.log(pred_prob.unsqueeze(1).expand(-1, N_gt))

            # Total cost matrix
            C = self.cost_position * cost_pos + self.cost_confidence * cost_conf

            # Handle NaN/inf values that may arise from numerical issues
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)
            C = C.cpu().numpy()

            # Hungarian algorithm (minimize total cost)
            row_ind, col_ind = linear_sum_assignment(C)

            indices.append((
                torch.tensor(row_ind, dtype=torch.int64, device=pred_pos.device),
                torch.tensor(col_ind, dtype=torch.int64, device=pred_pos.device),
            ))

        return indices


class SetCriterion(nn.Module):
    """
    DETR-style set prediction loss.

    Computes:
    - L1 loss on matched position predictions
    - BCE loss on confidence predictions (matched = 1, unmatched = 0)
    - Optional world coordinate loss for better evaluation alignment
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        weight_position: float = 5.0,
        weight_confidence: float = 1.0,
        weight_no_object: float = 0.1,
        weight_world: float = 0.0,
        aux_loss_weight: float = 0.5,
    ):
        """
        Args:
            matcher: HungarianMatcher instance
            weight_position: Weight for position loss
            weight_confidence: Weight for confidence loss
            weight_no_object: Weight for no-object class in BCE (handles class imbalance)
            weight_world: Weight for world coordinate loss (0 to disable)
            aux_loss_weight: Weight for auxiliary losses from intermediate decoder layers
        """
        super().__init__()
        self.matcher = matcher
        self.weight_position = weight_position
        self.weight_confidence = weight_confidence
        self.weight_no_object = weight_no_object
        self.weight_world = weight_world
        self.aux_loss_weight = aux_loss_weight

        # World coordinate loss (optional)
        self.world_loss = WorldCoordinateLoss(tau=5.0) if weight_world > 0 else None

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute set prediction loss.

        Args:
            outputs: dict with:
                - 'positions': [B, num_queries, 2]
                - 'confidences': [B, num_queries]
                - 'aux_positions': optional list of [B, num_queries, 2] from intermediate layers
                - 'aux_confidences': optional list of [B, num_queries] from intermediate layers
            targets: list of B dicts with:
                - 'positions': [N_gt, 2]
                - 'positions_world': [N_gt, 2] (for world coordinate loss)
                - 'image_size': (H, W)
                - 'camera_matrix': [3, 4]
                - 'undist_poly': distortion polynomial

        Returns:
            dict with loss values
        """
        # Compute matching
        indices = self.matcher(outputs, targets)

        # Position loss (L1 on matched pairs)
        loss_pos = self._compute_position_loss(outputs['positions'], targets, indices)

        # Confidence loss (BCE)
        loss_conf = self._compute_confidence_loss(outputs['confidences'], targets, indices)

        # World coordinate loss (optional)
        loss_world = outputs['positions'].new_zeros(())
        if self.world_loss is not None:
            loss_world = self.world_loss(outputs['positions'], targets, indices)

        # Total loss
        loss = (self.weight_position * loss_pos +
                self.weight_confidence * loss_conf +
                self.weight_world * loss_world)

        losses = {
            'loss': loss,
            'loss_position': loss_pos,
            'loss_confidence': loss_conf,
            'loss_world': loss_world,
        }

        # Auxiliary losses
        if 'aux_positions' in outputs and outputs['aux_positions']:
            aux_losses = self._compute_aux_losses(outputs, targets)
            losses.update(aux_losses)
            losses['loss'] = losses['loss'] + self.aux_loss_weight * aux_losses['loss_aux']

        return losses

    def _compute_position_loss(
        self,
        pred_positions: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute L1 position loss on matched pairs."""
        loss = pred_positions.new_zeros(())  # Maintains gradient graph
        num_matched = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            pred_pos = pred_positions[b][pred_idx]  # [N_matched, 2]
            gt_pos = targets[b]['positions'][gt_idx]  # [N_matched, 2]

            loss = loss + F.l1_loss(pred_pos, gt_pos, reduction='sum')
            num_matched += len(pred_idx)

        # Normalize by number of matched pairs * 2 (for x and y coordinates)
        if num_matched > 0:
            loss = loss / (num_matched * 2)

        return loss

    def _compute_confidence_loss(
        self,
        pred_confidences: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute BCE confidence loss using logits (AMP-safe)."""
        B, num_queries = pred_confidences.shape
        device = pred_confidences.device

        # Build target confidence (1 for matched, 0 for unmatched)
        target_conf = torch.zeros(B, num_queries, device=device)
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_conf[b, pred_idx] = 1.0

        # Count positives and negatives for proper weighting
        num_pos = target_conf.sum()
        num_neg = target_conf.numel() - num_pos

        # Compute BCE with logits
        loss = F.binary_cross_entropy_with_logits(
            pred_confidences, target_conf,
            reduction='none'
        )

        # Apply class weighting: weight negatives by weight_no_object
        # This encourages the model to output low confidence for unmatched queries
        weight = torch.ones_like(target_conf)
        weight[target_conf == 0] = self.weight_no_object  # Down-weight negatives

        loss = (loss * weight).sum() / (weight.sum() + 1e-8)

        return loss

    def _compute_aux_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses from intermediate decoder layers."""
        aux_positions = outputs['aux_positions']
        aux_confidences = outputs['aux_confidences']

        total_aux_loss = outputs['positions'].new_zeros(())  # Maintains gradient graph
        num_aux = len(aux_positions)

        for i in range(num_aux):
            aux_outputs = {
                'positions': aux_positions[i],
                'confidences': aux_confidences[i],
            }
            # Use same matcher for auxiliary outputs
            indices = self.matcher(aux_outputs, targets)

            loss_pos = self._compute_position_loss(aux_positions[i], targets, indices)
            loss_conf = self._compute_confidence_loss(aux_confidences[i], targets, indices)

            total_aux_loss = total_aux_loss + self.weight_position * loss_pos + self.weight_confidence * loss_conf

        if num_aux > 0:
            total_aux_loss = total_aux_loss / num_aux

        return {'loss_aux': total_aux_loss}


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in confidence prediction.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This implementation works with logits for numerical stability and AMP compatibility.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: Predicted logits [B, N] (NOT probabilities)
            target: Binary targets [B, N]
        """
        # Compute BCE with logits (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

        # Compute probabilities for focal weight
        pred_prob = torch.sigmoid(pred_logits)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        return loss.mean()


class SetCriterionWithFocal(SetCriterion):
    """Set criterion using focal loss instead of weighted BCE for confidence."""

    def __init__(
        self,
        matcher: HungarianMatcher,
        weight_position: float = 5.0,
        weight_confidence: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        aux_loss_weight: float = 0.5,
    ):
        super().__init__(
            matcher=matcher,
            weight_position=weight_position,
            weight_confidence=weight_confidence,
            weight_no_object=1.0,  # Not used with focal loss
            aux_loss_weight=aux_loss_weight,
        )
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def _compute_confidence_loss(
        self,
        pred_confidences: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute focal loss for confidence."""
        B, num_queries = pred_confidences.shape
        device = pred_confidences.device

        # Build target confidence
        target_conf = torch.zeros(B, num_queries, device=device)
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_conf[b, pred_idx] = 1.0

        return self.focal_loss(pred_confidences, target_conf)


class WorldCoordinateLoss(nn.Module):
    """
    Computes L1 loss in world coordinates (meters on pitch).

    Projects predicted image positions to world coordinates using camera parameters,
    then computes L1 distance to ground truth world positions. This aligns training
    with the mAP-LocSim evaluation metric which uses world-space distances.
    """

    def __init__(self, tau: float = 5.0):
        """
        Args:
            tau: Scale factor for loss normalization (matches mAP-LocSim tau=5m scale)
        """
        super().__init__()
        self.tau = tau

    def forward(
        self,
        pred_positions: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Convert predicted image positions to world coords and compute L1 loss.

        Args:
            pred_positions: [B, num_queries, 2] predicted positions in [0,1]
            targets: List of B dicts with 'positions_world', 'image_size',
                     'camera_matrix', 'undist_poly'
            indices: List of B tuples (pred_idx, gt_idx) from Hungarian matching

        Returns:
            Scalar L1 loss in world coordinates
        """
        loss = pred_positions.new_zeros(())
        num_matched = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            # Get matched predictions in [0,1] normalized coords
            pred_norm = pred_positions[b][pred_idx]  # [N, 2]

            # Convert to pixel coords
            H, W = targets[b]['image_size']
            pred_px = pred_norm.clone()
            pred_px[:, 0] = pred_px[:, 0] * W
            pred_px[:, 1] = pred_px[:, 1] * H

            # Convert to camera normalized coords (centered, normalized by width)
            center = torch.tensor([(W - 1) / 2, (H - 1) / 2], device=pred_px.device, dtype=pred_px.dtype)
            pred_cam = (pred_px - center) / W

            # Project to world coordinates using camera parameters
            camera_matrix = targets[b]['camera_matrix'].to(pred_px.device)
            undist_poly = targets[b]['undist_poly'].to(pred_px.device)

            # Handle case where undist_poly might be empty
            if undist_poly.numel() == 0:
                continue

            pred_world = image_to_ground(camera_matrix, undist_poly, pred_cam)  # [N, 3]
            pred_world_2d = pred_world[:, :2]  # [N, 2] - x, y on pitch

            # Ground truth world positions
            gt_world = targets[b]['positions_world'][gt_idx].to(pred_px.device)

            # L1 loss in meters
            loss = loss + F.l1_loss(pred_world_2d, gt_world, reduction='sum')
            num_matched += len(pred_idx)

        # Normalize by number of matched coordinates and scale by tau
        if num_matched > 0:
            loss = loss / (num_matched * 2 * self.tau)

        return loss
