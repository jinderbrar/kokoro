"""
Composite blocks for Kokoro.

Contains transformer components built from atomic layers:
- MultiHeadAttention: Self-attention with RoPE
- SwiGLU: Gated activation function
- FeedForward: FFN with SwiGLU
- TransformerBlock: Complete transformer block
"""

import torch
from torch import nn
from torch.nn import functional as F

from . import fn
from .layers import RotaryPositionEmbedding, RMSNorm, BitLinear


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE and optional BitLinear.

    Implements scaled dot-product attention with multiple heads:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        hidden_size: Model dimension (must be divisible by n_heads)
        n_heads: Number of attention heads
        dropout: Dropout probability (default: 0.0)
        use_bitlinear: Use BitLinear instead of nn.Linear (default: False)
        act_bits: Activation bits for BitLinear if used (default: 8)
        max_seq_len: Maximum sequence length for RoPE (default: 2048)
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: float = 0.0,
        use_bitlinear: bool = False,
        act_bits: int = 8,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.use_bitlinear = use_bitlinear

        # Choose layer type
        LinearLayer = BitLinear if use_bitlinear else nn.Linear

        # Q, K, V projections
        if use_bitlinear:
            self.q_proj = LinearLayer(hidden_size, hidden_size, bias=False, act_bits=act_bits)
            self.k_proj = LinearLayer(hidden_size, hidden_size, bias=False, act_bits=act_bits)
            self.v_proj = LinearLayer(hidden_size, hidden_size, bias=False, act_bits=act_bits)
            self.o_proj = LinearLayer(hidden_size, hidden_size, bias=False, act_bits=act_bits)
        else:
            self.q_proj = LinearLayer(hidden_size, hidden_size, bias=False)
            self.k_proj = LinearLayer(hidden_size, hidden_size, bias=False)
            self.v_proj = LinearLayer(hidden_size, hidden_size, bias=False)
            self.o_proj = LinearLayer(hidden_size, hidden_size, bias=False)

        # RoPE for positional encoding
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len=max_seq_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask of shape (batch_size, seq_len) or (batch_size, 1, seq_len, seq_len)
                           True/1 for positions to attend, False/0 to mask

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, L, H)
        k = self.k_proj(x)  # (B, L, H)
        v = self.v_proj(x)  # (B, L, H)

        # Reshape to multi-head: (B, L, H) -> (B, L, n_heads, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to Q and K
        cos, sin = self.rope(x)
        q = fn.apply_rope(q, cos, sin)
        k = fn.apply_rope(k, cos, sin)

        # Transpose for attention: (B, L, n_heads, head_dim) -> (B, n_heads, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores: (B, n_heads, L, head_dim) @ (B, n_heads, head_dim, L) -> (B, n_heads, L, L)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Handle different mask shapes
            mask = attention_mask  # Type narrowing for checker
            if mask.dim() == 2:
                # (B, L) -> (B, 1, 1, L) for broadcasting
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (B, 1, L) -> (B, 1, 1, L)
                mask = mask.unsqueeze(1)

            # Convert boolean mask to additive mask: True->0, False->-inf
            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            else:
                attn_scores = attn_scores + mask

        # Compute attention weights: softmax over last dimension
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: (B, n_heads, L, L) @ (B, n_heads, L, head_dim) -> (B, n_heads, L, head_dim)
        out = torch.matmul(attn_weights, v)

        # Transpose back and reshape: (B, n_heads, L, head_dim) -> (B, L, n_heads, head_dim) -> (B, L, H)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection
        out = self.o_proj(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_size={self.hidden_size}, '
                f'n_heads={self.n_heads}, '
                f'head_dim={self.head_dim}, '
                f'use_bitlinear={self.use_bitlinear})')


class SwiGLU(nn.Module):
    """
    SwiGLU activation function from "GLU Variants Improve Transformer".

    SwiGLU(x) = Swish(xW + b) ⊗ (xV + c)
    where Swish(x) = x * sigmoid(x) and ⊗ is element-wise multiplication

    This is a gated activation that has shown better performance than ReLU/GELU
    in transformer models. Used in LLaMA, PaLM, and other modern LLMs.

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (typically 4 * dim or 8/3 * dim)
        use_bitlinear: Use BitLinear instead of nn.Linear (default: False)
        act_bits: Activation bits for BitLinear if used (default: 8)
        bias: Whether to use bias in projections (default: False)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        use_bitlinear: bool = False,
        act_bits: int = 8,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.use_bitlinear = use_bitlinear

        # Choose layer type
        if use_bitlinear:
            self.w = BitLinear(dim, hidden_dim, bias=bias, act_bits=act_bits)
            self.v = BitLinear(dim, hidden_dim, bias=bias, act_bits=act_bits)
        else:
            self.w = nn.Linear(dim, hidden_dim, bias=bias)
            self.v = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Output tensor of shape (..., hidden_dim)
        """
        # Swish activation: x * sigmoid(x)
        swish_w = F.silu(self.w(x))  # F.silu is the same as Swish
        v = self.v(x)
        return swish_w * v

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'dim={self.dim}, '
                f'hidden_dim={self.hidden_dim}, '
                f'use_bitlinear={self.use_bitlinear})')


class FeedForward(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation.

    Architecture: x -> SwiGLU -> Projection -> output
    With residual connection in the transformer block.

    Args:
        hidden_size: Model dimension
        ff_dim: Feed-forward hidden dimension (typically 4 * hidden_size)
        dropout: Dropout probability (default: 0.0)
        use_bitlinear: Use BitLinear instead of nn.Linear (default: False)
        act_bits: Activation bits for BitLinear if used (default: 8)
    """

    def __init__(
        self,
        hidden_size: int,
        ff_dim: int,
        dropout: float = 0.0,
        use_bitlinear: bool = False,
        act_bits: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ff_dim = ff_dim
        self.use_bitlinear = use_bitlinear

        # SwiGLU activation with projection
        self.swiglu = SwiGLU(
            dim=hidden_size,
            hidden_dim=ff_dim,
            use_bitlinear=use_bitlinear,
            act_bits=act_bits,
            bias=False,
        )

        # Output projection
        if use_bitlinear:
            self.proj = BitLinear(ff_dim, hidden_size, bias=False, act_bits=act_bits)
        else:
            self.proj = nn.Linear(ff_dim, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of same shape as input
        """
        x = self.swiglu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_size={self.hidden_size}, '
                f'ff_dim={self.ff_dim}, '
                f'use_bitlinear={self.use_bitlinear})')


class TransformerBlock(nn.Module):
    """
    Complete Transformer block with pre-norm architecture.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))

    This is the pre-norm variant which has better training stability.
    This block will be wrapped by RecursiveBlock for the recursive transformer.

    Args:
        hidden_size: Model dimension
        n_heads: Number of attention heads
        ff_dim: Feed-forward hidden dimension
        dropout: Dropout probability (default: 0.0)
        use_bitlinear: Use BitLinear instead of nn.Linear (default: False)
        act_bits: Activation bits for BitLinear if used (default: 8)
        max_seq_len: Maximum sequence length for RoPE (default: 2048)
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
        use_bitlinear: bool = False,
        act_bits: int = 8,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.ff_dim = ff_dim

        # Pre-normalization layers
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

        # Attention block
        self.attn = MultiHeadAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            dropout=dropout,
            use_bitlinear=use_bitlinear,
            act_bits=act_bits,
            max_seq_len=max_seq_len,
        )

        # Feed-forward block
        self.ff = FeedForward(
            hidden_size=hidden_size,
            ff_dim=ff_dim,
            dropout=dropout,
            use_bitlinear=use_bitlinear,
            act_bits=act_bits,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            Output tensor of same shape as input
        """
        # Attention with residual
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_size={self.hidden_size}, '
                f'n_heads={self.n_heads}, '
                f'ff_dim={self.ff_dim})')
