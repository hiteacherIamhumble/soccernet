"""
DETR-style decoder for sparse player localization.

Takes frozen VGGT encoder features and predicts player positions in image coordinates.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple multi-layer perceptron (also called FFN)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.dropout(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        return x


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for spatial features."""

    def __init__(self, hidden_dim: int, temperature: float = 10000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C, H, W]
        Returns:
            Positional encoding of shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # Create position grids
        y_embed = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, W)

        # Normalize to [0, 1]
        y_embed = y_embed / (H - 1 + 1e-6)
        x_embed = x_embed / (W - 1 + 1e-6)

        dim_t = torch.arange(self.hidden_dim // 2, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.hidden_dim // 2))

        pos_x = x_embed.unsqueeze(-1) / dim_t  # [H, W, D/2]
        pos_y = y_embed.unsqueeze(-1) / dim_t  # [H, W, D/2]

        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)  # [H, W, D]
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)  # [H, W, D]

        pos = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, 2*D]
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2*D, H, W]

        # Truncate or pad to match hidden_dim
        if pos.shape[1] > C:
            pos = pos[:, :C]
        elif pos.shape[1] < C:
            pos = F.pad(pos, (0, 0, 0, 0, 0, C - pos.shape[1]))

        return pos


class DETRPlayerDecoder(nn.Module):
    """
    DETR-style decoder for sparse player localization.

    Takes aggregated tokens from VGGT encoder and predicts player positions
    and confidence scores using learnable object queries.

    Args:
        dim_in: Input dimension from VGGT encoder (2048 for concat frame+global)
        hidden_dim: Hidden dimension for decoder
        num_queries: Number of object queries (max players to detect)
        num_decoder_layers: Number of transformer decoder layers
        num_heads: Number of attention heads
        ffn_dim: FFN hidden dimension
        dropout: Dropout rate
        intermediate_layer_idx: Which VGGT layers to use for features
        use_multi_scale: Whether to use multi-scale feature fusion
    """

    def __init__(
        self,
        dim_in: int = 2048,
        hidden_dim: int = 128,
        num_queries: int = 30,
        num_decoder_layers: int = 3,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        intermediate_layer_idx: Optional[List[int]] = None,
        use_multi_scale: bool = False,
    ):
        super().__init__()

        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.use_multi_scale = use_multi_scale

        # Default to using only final layer
        if intermediate_layer_idx is None:
            intermediate_layer_idx = [23]  # Final layer
        self.intermediate_layer_idx = intermediate_layer_idx

        # Feature projection
        if use_multi_scale and len(intermediate_layer_idx) > 1:
            # Multi-scale: project each layer then sum
            self.input_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim_in, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in intermediate_layer_idx
            ])
        else:
            # Single scale: just project final layer
            self.input_proj = nn.Sequential(
                nn.Linear(dim_in, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(hidden_dim)

        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        nn.init.normal_(self.query_embed.weight, std=0.01)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Prediction heads (simple 2-layer MLPs)
        self.position_head = MLP(hidden_dim, hidden_dim, 2, num_layers=2)
        self.confidence_head = MLP(hidden_dim, hidden_dim, 1, num_layers=2)

        # Initialize
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            aggregated_tokens_list: List of 24 tensors from VGGT encoder,
                each of shape [B, S, P, 2048] where S=1 for single image
            patch_start_idx: Index where patch tokens start (typically 5)
            image_size: Optional (H, W) of input image for computing patch grid size

        Returns:
            dict with:
                - 'positions': [B, num_queries, 2] normalized [0, 1] coordinates
                - 'confidences': [B, num_queries] confidence scores
        """
        # Get features from specified layers
        if self.use_multi_scale and len(self.intermediate_layer_idx) > 1:
            # Multi-scale fusion
            features_list = []
            for i, layer_idx in enumerate(self.intermediate_layer_idx):
                tokens = aggregated_tokens_list[layer_idx]  # [B, S, P, 2048]
                B, S, P, C = tokens.shape
                # Remove special tokens (camera + register)
                patch_tokens = tokens[:, :, patch_start_idx:, :]  # [B, S, P_patches, 2048]
                # Squeeze sequence dim (S=1)
                patch_tokens = patch_tokens.squeeze(1)  # [B, P_patches, 2048]
                # Project
                features = self.input_projs[i](patch_tokens)  # [B, P_patches, hidden_dim]
                features_list.append(features)
            # Sum all scales
            features = sum(features_list) / len(features_list)
        else:
            # Single scale (final layer)
            layer_idx = self.intermediate_layer_idx[-1]
            tokens = aggregated_tokens_list[layer_idx]  # [B, S, P, 2048]
            B, S, P, C = tokens.shape
            # Remove special tokens
            patch_tokens = tokens[:, :, patch_start_idx:, :]  # [B, S, P_patches, 2048]
            # Squeeze sequence dim
            patch_tokens = patch_tokens.squeeze(1)  # [B, P_patches, 2048]
            # Project
            features = self.input_proj(patch_tokens)  # [B, P_patches, hidden_dim]

        B, P_patches, _ = features.shape

        # Compute patch grid size
        # For 1080p padded to 1078x1918: H_patch=77, W_patch=137
        # P_patches = H_patch * W_patch
        H_patch = int(math.sqrt(P_patches * 1080 / 1920))  # Approximate
        W_patch = P_patches // H_patch
        # Adjust if needed
        while H_patch * W_patch != P_patches:
            H_patch += 1
            W_patch = P_patches // H_patch
            if H_patch > P_patches:
                # Fallback: assume square-ish
                H_patch = int(math.sqrt(P_patches))
                W_patch = P_patches // H_patch
                break

        # Reshape to 2D for positional encoding
        features_2d = features.view(B, H_patch, W_patch, -1).permute(0, 3, 1, 2)  # [B, C, H, W]

        # Add positional encoding
        pos_embed = self.pos_encoding(features_2d)  # [B, C, H, W]

        # Flatten back to sequence
        features = features + pos_embed.permute(0, 2, 3, 1).flatten(1, 2)  # [B, P_patches, C]

        # Get query embeddings
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, C]

        # Transformer decoder
        # memory: [B, P_patches, C], tgt: [B, num_queries, C]
        decoder_output = self.decoder(queries, features)  # [B, num_queries, C]

        # Prediction heads
        positions = self.position_head(decoder_output)  # [B, num_queries, 2]
        positions = positions.sigmoid()  # Normalize to [0, 1]

        # Output logits for confidences (apply sigmoid during inference, not here)
        # This is required for AMP compatibility with BCE loss
        confidence_logits = self.confidence_head(decoder_output)  # [B, num_queries, 1]
        confidence_logits = confidence_logits.squeeze(-1)  # [B, num_queries]

        return {
            'positions': positions,
            'confidences': confidence_logits,  # Logits, apply sigmoid for probabilities
            'decoder_output': decoder_output,  # For auxiliary losses if needed
        }


class DETRPlayerDecoderWithAux(DETRPlayerDecoder):
    """
    DETR decoder with auxiliary losses at intermediate decoder layers.

    Returns predictions from all decoder layers for auxiliary supervision.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Auxiliary prediction heads for each decoder layer
        self.aux_position_heads = nn.ModuleList([
            MLP(self.hidden_dim, self.hidden_dim, 2, num_layers=3)
            for _ in range(self.num_decoder_layers - 1)
        ])
        self.aux_confidence_heads = nn.ModuleList([
            MLP(self.hidden_dim, self.hidden_dim, 1, num_layers=3)
            for _ in range(self.num_decoder_layers - 1)
        ])

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> dict:
        """Forward with auxiliary outputs."""
        # Get features (same as parent)
        if self.use_multi_scale and len(self.intermediate_layer_idx) > 1:
            features_list = []
            for i, layer_idx in enumerate(self.intermediate_layer_idx):
                tokens = aggregated_tokens_list[layer_idx]
                B, S, P, C = tokens.shape
                patch_tokens = tokens[:, :, patch_start_idx:, :].squeeze(1)
                features = self.input_projs[i](patch_tokens)
                features_list.append(features)
            features = sum(features_list) / len(features_list)
        else:
            layer_idx = self.intermediate_layer_idx[-1]
            tokens = aggregated_tokens_list[layer_idx]
            B, S, P, C = tokens.shape
            patch_tokens = tokens[:, :, patch_start_idx:, :].squeeze(1)
            features = self.input_proj(patch_tokens)

        B, P_patches, _ = features.shape

        # Compute patch grid
        H_patch = int(math.sqrt(P_patches * 1080 / 1920))
        W_patch = P_patches // H_patch
        while H_patch * W_patch != P_patches:
            H_patch += 1
            W_patch = P_patches // H_patch
            if H_patch > P_patches:
                H_patch = int(math.sqrt(P_patches))
                W_patch = P_patches // H_patch
                break

        # Positional encoding
        features_2d = features.view(B, H_patch, W_patch, -1).permute(0, 3, 1, 2)
        pos_embed = self.pos_encoding(features_2d)
        features = features + pos_embed.permute(0, 2, 3, 1).flatten(1, 2)

        # Queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Run decoder layer by layer to get intermediate outputs
        aux_positions = []
        aux_confidences = []
        output = queries

        for i, layer in enumerate(self.decoder.layers):
            output = layer(output, features)

            if i < self.num_decoder_layers - 1:
                # Auxiliary predictions
                aux_pos = self.aux_position_heads[i](output).sigmoid()
                aux_conf = self.aux_confidence_heads[i](output).squeeze(-1).sigmoid()
                aux_positions.append(aux_pos)
                aux_confidences.append(aux_conf)

        # Final predictions
        positions = self.position_head(output).sigmoid()
        confidences = self.confidence_head(output).squeeze(-1).sigmoid()

        return {
            'positions': positions,
            'confidences': confidences,
            'aux_positions': aux_positions,
            'aux_confidences': aux_confidences,
            'decoder_output': output,
        }
