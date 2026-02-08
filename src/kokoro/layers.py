"""
Atomic layers for Kokoro.

Contains fundamental building blocks:
- BitLinear: 1.58-bit quantized linear layer
- RMSNorm: Root Mean Square normalization
- RotaryPositionEmbedding: RoPE for attention
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from . import fn


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    Instead of adding position information, RoPE rotates the query/key vectors by an angle
    proportional to their position. This allows the model to naturally encode relative positions
    through the dot product in attention.

    Args:
        dim: Dimension of each attention head (head_dim, must be even)
        max_seq_len: Maximum sequence length to precompute
        base: Base for computing rotation frequencies (default: 10000, as in the paper)
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands: theta_i = base^(-2i/dim) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin for all positions
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin values for all positions."""
        # Position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)

        # Compute angles: outer product of positions and frequencies
        # Shape: (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)

        # Concatenate to get full dimension: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Store cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin values for the sequence length of x.

        Args:
            x: Input tensor of shape (batch_size, seq_len, ...)

        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, dim)
        """
        seq_len = x.shape[1]

        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._update_cos_sin_cache(seq_len)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    More efficient than LayerNorm as it doesn't center the inputs.
    Used in modern transformers like LLaMA, GPT-NeoX, etc.

    Formula: x_norm = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        dim: The dimension to normalize (usually hidden_size)
        eps: Small constant for numerical stability (default: 1e-6)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor of same shape as input
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x_norm = x / rms
        return self.weight * x_norm

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dim={self.weight.shape[0]}, eps={self.eps})'


class BitLinear(nn.Module):
    """
    BitLinear layer with 1.58-bit quantization.

    Implements ternary weight quantization {-1, 0, +1} and activation quantization
    with proper scaling factors. This allows for extreme compression while maintaining
    performance.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias (default: True)
        act_bits: Activation quantization bits (default: 4)
        eps: Small constant for numerical stability (default: 1e-5)
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            act_bits: int = 4,
            eps: float = 1e-5
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters similar to nn.Linear"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weight binarization and activation quantization.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Weight binarization: center weights and binarize to {-1, +1}
        w_centered = self.weight - self.weight.mean(dim=1, keepdim=True)
        w_bin = fn.ste_sign(w_centered)

        # Compute weight scale factor (beta): shape (1, out_features) for broadcasting
        beta = self.weight.abs().mean(dim=1, keepdim=True).t()

        # Normalize activations using LayerNorm
        xln = F.layer_norm(x, (x.shape[-1],), eps=self.eps)

        # Activation quantization: compute quantization range and scale factor
        qb = 2 ** (self.act_bits - 1) - 1  # Quantization range: e.g., 127 for 8-bit
        gama = xln.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps)  # Per-sample gamma

        # Quantize activations to integer range [-qb, qb-1]
        xq = xln * (qb / gama)
        xq = fn.ste_clip(xq, -qb + self.eps, qb - self.eps)

        # Binary matmul (without bias - we'll add it after rescaling)
        y = F.linear(xq, w_bin, None)

        # Rescale output: y = y / (wscale * xscale) = y * beta * gama / qb
        y = y * (beta * gama / qb)

        # Add bias after rescaling (if present)
        if self.bias is not None:
            y = y + self.bias

        return y

    def __repr__(self) -> str:
        """String representation showing layer configuration"""
        return (f'{self.__class__.__name__}('
                f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'act_bits={self.act_bits})')
