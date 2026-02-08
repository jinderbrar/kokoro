"""
Kokoro Recursive Ternary Transformer Model.

Complete language model with:
- Token embeddings
- Recursive transformer with layer position embeddings
- Language modeling head
- Support for 1.58-bit BitLinear quantization
"""

import torch
from torch import nn
from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path

from .layers import RMSNorm, BitLinear
from .blocks import TransformerBlock
from .recursive import RecursiveBlock


@dataclass
class KokoroConfig:
    """
    Configuration for Kokoro model.

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Model dimension (must be divisible by n_heads)
        n_heads: Number of attention heads
        num_iterations: Number of recursive iterations (T)
        ff_dim: Feed-forward hidden dimension (typically 4 * hidden_size)
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        use_bitlinear: Use BitLinear quantization (default: False)
        act_bits: Activation bits for BitLinear (default: 8)
        use_layer_pos_emb: Use layer position embeddings in recursion (default: True)
        pad_token_id: Padding token ID (default: 0)
        tie_weights: Tie input/output embeddings (default: True)
    """
    vocab_size: int = 32000
    hidden_size: int = 512
    n_heads: int = 8
    num_iterations: int = 12
    ff_dim: Optional[int] = None  # Defaults to 4 * hidden_size
    max_seq_len: int = 2048
    dropout: float = 0.1
    use_bitlinear: bool = False
    act_bits: int = 8
    use_layer_pos_emb: bool = True
    pad_token_id: int = 0
    tie_weights: bool = True

    def __post_init__(self):
        """Compute derived values."""
        if self.ff_dim is None:
            self.ff_dim = 4 * self.hidden_size

        # Validate
        assert self.hidden_size % self.n_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by n_heads ({self.n_heads})"
        assert self.num_iterations > 0, "num_iterations must be positive"

    def save(self, path: str | Path):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> 'KokoroConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def count_parameters(self, use_bitlinear: bool = None) -> dict:
        """
        Estimate parameter count.

        Returns:
            Dictionary with parameter counts and memory estimates
        """
        if use_bitlinear is None:
            use_bitlinear = self.use_bitlinear

        # Token embeddings
        embed_params = self.vocab_size * self.hidden_size

        # Transformer block (single block, reused T times)
        # Attention: Q, K, V, O projections
        attn_params = 4 * (self.hidden_size * self.hidden_size)

        # Feed-forward: SwiGLU has 2 projections (W, V) + output projection
        ff_params = 2 * (self.hidden_size * self.ff_dim) + (self.ff_dim * self.hidden_size)

        # Norms: 2 RMSNorm per block (just gamma parameter)
        norm_params = 2 * self.hidden_size

        # Total for one block
        block_params = attn_params + ff_params + norm_params

        # Layer position embeddings (one per iteration)
        layer_pos_params = self.num_iterations * self.hidden_size

        # Final norm
        final_norm_params = self.hidden_size

        # LM head (if not tied)
        if self.tie_weights:
            lm_head_params = 0
        else:
            lm_head_params = self.vocab_size * self.hidden_size

        # Total
        total_params = (
            embed_params +
            block_params +
            layer_pos_params +
            final_norm_params +
            lm_head_params
        )

        # Memory estimate (in MB)
        if use_bitlinear:
            # BitLinear: ~2 bits per weight (ternary) + full precision scaling factors
            # Approximate as 1/8 of full precision for weights, plus activations
            bits_per_param = 2  # Ternary weights
            memory_mb = (total_params * bits_per_param) / (8 * 1024 * 1024)
        else:
            # Full precision: 4 bytes per float32
            memory_mb = (total_params * 4) / (1024 * 1024)

        return {
            "total": total_params,
            "embeddings": embed_params,
            "transformer_block": block_params,
            "layer_pos_emb": layer_pos_params,
            "lm_head": lm_head_params,
            "memory_mb": memory_mb,
            "bits_per_param": 2 if use_bitlinear else 32,
        }


class KokoroLM(nn.Module):
    """
    Kokoro Recursive Ternary Transformer Language Model.

    Architecture:
        x -> TokenEmbedding -> RecursiveTransformer(T iterations) -> FinalNorm -> LMHead -> logits

    The core innovation: 1 transformer block applied T times recursively with
    layer position embeddings, using 1.58-bit BitLinear quantization.

    Args:
        config: KokoroConfig instance
    """

    def __init__(self, config: KokoroConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        # Create the transformer block (will be reused recursively)
        transformer_block = TransformerBlock(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            use_bitlinear=config.use_bitlinear,
            act_bits=config.act_bits,
            max_seq_len=config.max_seq_len,
        )

        # Wrap in recursive block
        self.recursive_transformer = RecursiveBlock(
            block=transformer_block,
            num_iterations=config.num_iterations,
            hidden_size=config.hidden_size,
            use_layer_pos_emb=config.use_layer_pos_emb,
        )

        # Final normalization
        self.final_norm = RMSNorm(config.hidden_size)

        # Language modeling head
        if config.tie_weights:
            # Share weights with token embeddings
            self.lm_head = None
        else:
            # Separate LM head
            if config.use_bitlinear:
                self.lm_head = BitLinear(
                    config.hidden_size,
                    config.vocab_size,
                    bias=False,
                    act_bits=config.act_bits,
                )
            else:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Print model info
        param_info = config.count_parameters()
        print(f"Kokoro Model Initialized:")
        print(f"  Total parameters: {param_info['total']:,}")
        print(f"  Memory (approx):  {param_info['memory_mb']:.1f} MB")
        print(f"  Iterations:       {config.num_iterations}")
        print(f"  BitLinear:        {config.use_bitlinear}")

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional mask of shape (batch_size, seq_len)
            labels: Optional labels for computing loss, same shape as input_ids

        Returns:
            Dictionary with:
                - logits: (batch_size, seq_len, vocab_size)
                - loss: Optional, if labels provided
        """
        # Token embeddings
        hidden_states = self.token_embeddings(input_ids)  # (B, L, H)

        # Apply recursive transformer
        hidden_states = self.recursive_transformer(hidden_states, attention_mask=attention_mask)

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        # LM head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied weights: use embedding matrix
            logits = torch.matmul(hidden_states, self.token_embeddings.weight.t())

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        return {
            "logits": logits,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting tokens of shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens (optional)
            top_p: Nucleus sampling threshold (optional)

        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            input_ids_crop = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            outputs = self.forward(input_ids_crop)
            logits = outputs["logits"]

            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get number of parameters.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            if self.lm_head is not None:
                n_params -= self.lm_head.weight.numel() if hasattr(self.lm_head, 'weight') else 0
        return n_params

    def save_checkpoint(self, path: str | Path, **extra_data):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            **extra_data: Additional data to save (e.g., optimizer state, step, epoch)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': asdict(self.config),
            **extra_data
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = 'cpu') -> tuple['KokoroLM', dict]:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model on

        Returns:
            Tuple of (model, extra_data) where extra_data contains optimizer state, etc.
        """
        checkpoint = torch.load(path, map_location=device)

        # Recreate config
        config = KokoroConfig(**checkpoint['config'])

        # Create model
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        # Extract extra data
        extra_data = {k: v for k, v in checkpoint.items()
                     if k not in ['model_state_dict', 'config']}

        print(f"Checkpoint loaded from {path}")
        return model, extra_data
