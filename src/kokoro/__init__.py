"""
Kokoro - Recursive Ternary Transformer

A memory-efficient language model using:
- 1.58-bit quantization (ternary weights)
- Recursive transformer architecture
- <10M parameters, <250MB memory
"""

# Atomic layers
from .layers import (
    BitLinear,
    RMSNorm,
    RotaryPositionEmbedding,
)

# Composite blocks
from .blocks import (
    MultiHeadAttention,
    SwiGLU,
    FeedForward,
    TransformerBlock,
)

# Recursive components
from .recursive import (
    LayerPositionEmbedding,
    RecursiveBlock,
)

# Helper functions
from . import fn

# Model and config
from .model import KokoroConfig, KokoroLM

# Training
from .train import Trainer

__all__ = [
    # Layers
    "BitLinear",
    "RMSNorm",
    "RotaryPositionEmbedding",
    # Blocks
    "MultiHeadAttention",
    "SwiGLU",
    "FeedForward",
    "TransformerBlock",
    # Recursive
    "LayerPositionEmbedding",
    "RecursiveBlock",
    # Functions
    "fn",
    # Model
    "KokoroConfig",
    "KokoroLM",
    # Training
    "Trainer",
]


def hello():
    """Print Kokoro info."""
    print("Kokoro - Recursive Ternary Transformer")
    print("Recursive • Ternary • Tiny")
