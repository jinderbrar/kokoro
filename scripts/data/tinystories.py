"""
TinyStories dataset loader.

Loads the TinyStories dataset from HuggingFace and creates PyTorch DataLoaders.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional


class TextDataset(Dataset):
    """PyTorch Dataset for tokenized text."""

    def __init__(self, tokenized_data):
        """
        Args:
            tokenized_data: List of dicts with 'input_ids', 'attention_mask', 'labels'
        """
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def tokenize_dataset(dataset, tokenizer, max_length: int, num_samples: Optional[int] = None):
    """
    Tokenize a HuggingFace dataset.

    Args:
        dataset: HuggingFace dataset with 'text' field
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        num_samples: Optional limit on number of samples

    Returns:
        List of tokenized samples
    """
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    tokenized = []

    print(f"Tokenizing {len(dataset):,} samples...")

    for i, example in enumerate(dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,}/{len(dataset):,} samples")

        text = example['text']

        # Tokenize
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Create labels (same as input_ids for language modeling)
        tokenized.append({
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0),
        })

    print(f"✓ Tokenization complete: {len(tokenized):,} samples")
    return tokenized


def load_tinystories(
    tokenizer_name: str = "gpt2",
    batch_size: int = 16,
    max_length: int = 512,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    cache_dir: str = "./data/cache",
    num_workers: int = 0,
):
    """
    Load TinyStories dataset and create DataLoaders.

    Args:
        tokenizer_name: Tokenizer to use (default: gpt2)
        batch_size: Batch size for DataLoaders
        max_length: Maximum sequence length
        max_train_samples: Limit training samples (None = all)
        max_val_samples: Limit validation samples (None = all)
        cache_dir: Directory to cache dataset
        num_workers: Number of DataLoader workers

    Returns:
        Tuple of (train_loader, val_loader, tokenizer)
    """
    print("="*70)
    print("LOADING TINYSTORIES DATASET")
    print("="*70)

    # Load tokenizer
    print(f"\n1. Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Set pad token if not set (GPT2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   Set pad_token = eos_token = '{tokenizer.eos_token}'")

    print(f"   Vocab size: {len(tokenizer):,}")
    print(f"   PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    # Load dataset from HuggingFace
    print(f"\n2. Loading TinyStories from HuggingFace")
    print(f"   Cache directory: {cache_dir}")

    try:
        dataset = load_dataset(
            'roneneldan/TinyStories',
            cache_dir=cache_dir,
        )
        print(f"   ✓ Dataset loaded successfully")
        print(f"   Train samples: {len(dataset['train']):,}")
        print(f"   Validation samples: {len(dataset['validation']):,}")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        print(f"\n   NOTE: TinyStories requires internet connection on first load")
        print(f"   Attempting to download from HuggingFace Hub...")
        raise

    # Tokenize datasets
    print(f"\n3. Tokenizing datasets (max_length={max_length})")

    train_tokenized = tokenize_dataset(
        dataset['train'],
        tokenizer,
        max_length,
        max_train_samples
    )

    val_tokenized = tokenize_dataset(
        dataset['validation'],
        tokenizer,
        max_length,
        max_val_samples
    )

    # Create PyTorch Datasets
    train_dataset = TextDataset(train_tokenized)
    val_dataset = TextDataset(val_tokenized)

    # Create DataLoaders
    print(f"\n4. Creating DataLoaders (batch_size={batch_size})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    print("\n" + "="*70)
    print("DATA LOADING COMPLETE")
    print("="*70)

    return train_loader, val_loader, tokenizer
