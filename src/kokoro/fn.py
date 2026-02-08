"""
Helper functions for Kokoro.

Straight-Through Estimators and other mathematical utilities.
"""

import torch


def ste_sign(x: torch.Tensor, threshold: float = 0) -> torch.Tensor:
    """
    Straight-Through Estimator for sign function.

    Forward: sign(x - threshold)
    Backward: identity (gradient passes through unchanged)

    Args:
        x: Input tensor
        threshold: Threshold for sign function (default: 0)

    Returns:
        Ternary tensor {-1, 0, 1} with gradients flowing through
    """
    return x + (torch.sign(x - threshold) - x).detach()


def ste_clip(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Straight-Through Estimator for clamp function.

    Forward: clamp(x, min_val, max_val)
    Backward: identity (gradient passes through unchanged)

    Args:
        x: Input tensor
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped tensor with gradients flowing through
    """
    return x + (torch.clamp(x, min_val, max_val) - x).detach()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Position Embedding (RoPE) to input tensor.

    Rotates pairs of elements in the embedding dimension using precomputed
    cos/sin values based on position.

    Args:
        x: Input tensor of shape (batch_size, seq_len, n_heads, head_dim)
        cos: Cosine values of shape (seq_len, head_dim)
        sin: Sine values of shape (seq_len, head_dim)

    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    # Split the last dimension in half: x -> [x1, x2]
    x1, x2 = x.chunk(2, dim=-1)

    # Split cos/sin in half to match x1/x2 dimensions
    cos_half, _ = cos.chunk(2, dim=-1)  # Take first half
    sin_half, _ = sin.chunk(2, dim=-1)  # Take first half

    # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    # Broadcasting: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    cos_expanded = cos_half.unsqueeze(0).unsqueeze(2)
    sin_expanded = sin_half.unsqueeze(0).unsqueeze(2)

    rotated_x1 = x1 * cos_expanded - x2 * sin_expanded
    rotated_x2 = x1 * sin_expanded + x2 * cos_expanded

    return torch.cat([rotated_x1, rotated_x2], dim=-1)
