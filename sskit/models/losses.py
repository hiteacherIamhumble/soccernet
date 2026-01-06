"""
DETR-style loss functions for set prediction.

Includes Hungarian matching and set prediction loss with position and confidence terms.
"""

from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


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

            # Compute L1 distance cost matrix
            # [num_queries, N_gt]
            cost_pos = torch.cdist(pred_pos, gt_pos, p=1)

            # Classification cost: -log(confidence) for positive matches
            # We want high confidence for matched predictions
            # Cost = -log(p) = log(1/p), lower confidence = higher cost
            cost_conf = -torch.log(pred_conf.unsqueeze(1).expand(-1, N_gt) + 1e-8)

            # Total cost matrix
            C = self.cost_position * cost_pos + self.cost_confidence * cost_conf
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
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        weight_position: float = 5.0,
        weight_confidence: float = 1.0,
        weight_no_object: float = 0.1,
        aux_loss_weight: float = 0.5,
    ):
        """
        Args:
            matcher: HungarianMatcher instance
            weight_position: Weight for position loss
            weight_confidence: Weight for confidence loss
            weight_no_object: Weight for no-object class in BCE (handles class imbalance)
            aux_loss_weight: Weight for auxiliary losses from intermediate decoder layers
        """
        super().__init__()
        self.matcher = matcher
        self.weight_position = weight_position
        self.weight_confidence = weight_confidence
        self.weight_no_object = weight_no_object
        self.aux_loss_weight = aux_loss_weight

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

        Returns:
            dict with loss values
        """
        # Compute matching
        indices = self.matcher(outputs, targets)

        # Position loss (L1 on matched pairs)
        loss_pos = self._compute_position_loss(outputs['positions'], targets, indices)

        # Confidence loss (BCE)
        loss_conf = self._compute_confidence_loss(outputs['confidences'], targets, indices)

        # Total loss
        loss = self.weight_position * loss_pos + self.weight_confidence * loss_conf

        losses = {
            'loss': loss,
            'loss_position': loss_pos,
            'loss_confidence': loss_conf,
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
        loss = torch.tensor(0.0, device=pred_positions.device)
        num_matched = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            pred_pos = pred_positions[b][pred_idx]  # [N_matched, 2]
            gt_pos = targets[b]['positions'][gt_idx]  # [N_matched, 2]

            loss = loss + F.l1_loss(pred_pos, gt_pos, reduction='sum')
            num_matched += len(pred_idx)

        # Normalize by number of matched pairs
        if num_matched > 0:
            loss = loss / num_matched

        return loss

    def _compute_confidence_loss(
        self,
        pred_confidences: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute BCE confidence loss."""
        B, num_queries = pred_confidences.shape
        device = pred_confidences.device

        # Build target confidence (1 for matched, 0 for unmatched)
        target_conf = torch.zeros(B, num_queries, device=device)
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_conf[b, pred_idx] = 1.0

        # Weighted BCE (lower weight for no-object to handle class imbalance)
        weight = torch.ones_like(target_conf)
        weight[target_conf == 0] = self.weight_no_object

        loss = F.binary_cross_entropy(pred_confidences, target_conf, weight=weight)

        return loss

    def _compute_aux_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses from intermediate decoder layers."""
        aux_positions = outputs['aux_positions']
        aux_confidences = outputs['aux_confidences']

        total_aux_loss = torch.tensor(0.0, device=outputs['positions'].device)
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
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities [B, N]
            target: Binary targets [B, N]
        """
        p_t = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        bce = F.binary_cross_entropy(pred, target, reduction='none')
        loss = focal_weight * bce

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
