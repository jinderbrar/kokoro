"""
Recursive components for Kokoro.

Contains components specific to the recursive transformer architecture:
- LayerPositionEmbedding: Distinguishes between recursive iterations
- RecursiveBlock: Applies a block T times recursively
"""

import torch
from torch import nn


class LayerPositionEmbedding(nn.Module):
    """
    Learnable position embeddings for recursive transformer layers.

    Critical component for recursive transformers - helps the model distinguish
    which iteration of recursion it's currently in. Without this, all recursive
    iterations look identical and the model suffers from rank collapse.

    This is different from RotaryPositionEmbedding which encodes token positions
    within a sequence. LayerPositionEmbedding encodes which recursive iteration
    (layer) is being processed.

    Args:
        num_layers: Maximum number of recursive iterations (T)
        hidden_size: Dimension of the hidden states
    """

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Learnable embeddings: one for each layer position
        self.embeddings = nn.Parameter(torch.zeros(num_layers, hidden_size))

        # Initialize with small random values
        nn.init.normal_(self.embeddings, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Add layer position embedding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            layer_idx: Which recursive iteration (0 to num_layers-1)

        Returns:
            x + layer_position_embedding[layer_idx]
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} >= num_layers {self.num_layers}")

        # Broadcasting: (hidden_size,) -> (1, 1, hidden_size) -> adds to (B, L, H)
        return x + self.embeddings[layer_idx]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_layers={self.num_layers}, hidden_size={self.hidden_size})'


class RecursiveBlock(nn.Module):
    """
    Recursive wrapper for transformer blocks.

    Applies the same transformer block T times with layer position embeddings
    to distinguish between iterations. This is the core innovation - instead of
    stacking N different blocks, we use 1 block recursively T times.

    Benefits:
    - Memory efficient: 1 block instead of N blocks
    - Parameter efficient: Weights shared across all iterations
    - Can achieve similar performance with proper layer position embeddings

    Risks without LayerPositionEmbedding:
    - Rank collapse: hidden states converge to same subspace
    - Degraded performance: model can't utilize recursion depth

    Args:
        block: The transformer block to apply recursively
        num_iterations: Number of times to apply the block (T)
        hidden_size: Dimension of hidden states (for layer position embeddings)
        use_layer_pos_emb: Whether to use layer position embeddings (default: True)
    """

    def __init__(
        self,
        block: nn.Module,
        num_iterations: int,
        hidden_size: int,
        use_layer_pos_emb: bool = True
    ):
        super().__init__()
        self.block = block
        self.num_iterations = num_iterations
        self.hidden_size = hidden_size
        self.use_layer_pos_emb = use_layer_pos_emb

        # Layer position embeddings (critical for preventing rank collapse)
        if use_layer_pos_emb:
            self.layer_pos_emb = LayerPositionEmbedding(num_iterations, hidden_size)
        else:
            self.layer_pos_emb = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply the block recursively T times.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            **kwargs: Additional arguments to pass to the block (e.g., attention_mask)

        Returns:
            Output after T iterations of the block
        """
        for t in range(self.num_iterations):
            # Add layer position embedding before block
            if self.layer_pos_emb is not None:
                x_in = self.layer_pos_emb(x, t)
            else:
                x_in = x

            # Apply the transformer block
            x = self.block(x_in, **kwargs)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'num_iterations={self.num_iterations}, '
                f'hidden_size={self.hidden_size}, '
                f'use_layer_pos_emb={self.use_layer_pos_emb})')
