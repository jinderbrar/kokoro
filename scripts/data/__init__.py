"""
Data loading utilities for training.

This module handles dataset loading, preprocessing, and DataLoader creation.
Each dataset has its own module (e.g., tinystories.py, wikipedia.py).
"""

from .tinystories import load_tinystories

__all__ = [
    'load_tinystories',
]
