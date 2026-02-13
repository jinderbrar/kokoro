"""
Training script for Kokoro on TinyStories dataset.

This is the first training run for Week 1 milestone!

Usage:
    python scripts/train_tinystories.py
    python scripts/train_tinystories.py --use-bitlinear --gradient-accumulation 4
"""

import argparse
import torch
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kokoro import (
    KokoroConfig,
    KokoroLM,
    Trainer,
)

# Import data loading from scripts
sys.path.insert(0, str(Path(__file__).parent))
from data.tinystories import load_tinystories


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Kokoro on TinyStories")

    # Model args
    parser.add_argument("--vocab-size", type=int, default=None,
                       help="Vocabulary size (default: from tokenizer)")
    parser.add_argument("--hidden-size", type=int, default=256,
                       help="Hidden dimension")
    parser.add_argument("--n-heads", type=int, default=4,
                       help="Number of attention heads")
    parser.add_argument("--num-iterations", type=int, default=6,
                       help="Number of recursive iterations")
    parser.add_argument("--max-seq-len", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--use-bitlinear", action="store_true",
                       help="Use BitLinear quantization")
    parser.add_argument("--act-bits", type=int, default=8,
                       help="Activation bits for BitLinear")

    # Data args
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="Tokenizer name")
    parser.add_argument("--max-train-samples", type=int, default=10000,
                       help="Max training samples (None = all)")
    parser.add_argument("--max-val-samples", type=int, default=1000,
                       help="Max validation samples")

    # Training args
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Gradient clipping threshold")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--mixed-precision", type=str, default=None,
                       choices=[None, "fp16", "bf16"],
                       help="Mixed precision training")

    # Other args
    parser.add_argument("--output-dir", type=str, default="./outputs/tinystories_run_1",
                       help="Output directory")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu, default: auto)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--eval-every", type=int, default=1,
                       help="Run evaluation every N epochs")
    parser.add_argument("--save-every", type=int, default=1,
                       help="Save checkpoint every N epochs")

    # Resume training
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume from checkpoint directory (e.g., outputs/run_1/checkpoint_epoch_5)")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Kokoro Training on TinyStories")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    print("Loading data...")
    train_loader, val_loader, tokenizer = load_tinystories(
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_len,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    # Create config
    print("\nCreating model...")
    config = KokoroConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        num_iterations=args.num_iterations,
        max_seq_len=args.max_seq_len,
        use_bitlinear=args.use_bitlinear,
        act_bits=args.act_bits,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Print config
    param_info = config.count_parameters()
    print(f"\nModel Configuration:")
    print(f"  Vocab size:      {config.vocab_size:,}")
    print(f"  Hidden size:     {config.hidden_size}")
    print(f"  Num heads:       {config.n_heads}")
    print(f"  Iterations:      {config.num_iterations}")
    print(f"  BitLinear:       {config.use_bitlinear}")
    print(f"\nParameter Count:")
    print(f"  Total:           {param_info['total']:,}")
    print(f"  Memory:          {param_info['memory_mb']:.1f} MB")
    print(f"  Bits/param:      {param_info['bits_per_param']}")

    # Check Phase 0 constraints
    if param_info['total'] >= 10_000_000:
        print(f"\n⚠️  WARNING: Model has {param_info['total']:,} params (target: <10M)")
    if param_info['memory_mb'] >= 250:
        print(f"\n⚠️  WARNING: Model uses {param_info['memory_mb']:.1f}MB (target: <250MB)")

    # Check if resuming
    if args.resume_from:
        # Resume from checkpoint
        print(f"\nResuming from checkpoint: {args.resume_from}")
        trainer = Trainer.from_checkpoint(
            checkpoint_path=args.resume_from,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
        )
    else:
        # Fresh training
        # Save config
        config_path = Path(args.output_dir) / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.save(config_path)
        print(f"\nConfig saved to: {config_path}")

        # Save run metadata
        run_metadata = {
            'run_name': Path(args.output_dir).name,
            'start_time': datetime.now().isoformat(),
            'command': ' '.join(sys.argv),
            'device': args.device,
            'dataset': 'TinyStories',
            'total_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'model_config': {
                'vocab_size': config.vocab_size,
                'hidden_size': config.hidden_size,
                'n_heads': config.n_heads,
                'num_iterations': config.num_iterations,
                'use_bitlinear': config.use_bitlinear,
            }
        }

        with open(Path(args.output_dir) / 'run_metadata.json', 'w') as f:
            json.dump(run_metadata, f, indent=2)

        print(f"Run metadata saved to: {Path(args.output_dir) / 'run_metadata.json'}")

        # Create model
        model = KokoroLM(config)

        # Create trainer
        print("\nInitializing trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation,
            mixed_precision=args.mixed_precision,
            device=args.device,
            output_dir=args.output_dir,
            warmup_steps=args.warmup_steps,
        )

    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    trainer.train(
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("\nWeek 1 Milestone Achieved! 🎉")


if __name__ == "__main__":
    main()
