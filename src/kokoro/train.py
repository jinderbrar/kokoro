"""
Training loop for Kokoro model.

Features:
- Gradient accumulation for large effective batch sizes
- Mixed precision training (FP16/BF16)
- Gradient clipping
- Learning rate scheduling
- Validation and checkpointing
- Progress tracking
- Directory-based checkpoints with rich metadata
- Resume training from checkpoints
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Optional, Literal
from tqdm import tqdm
import json
import time
from datetime import datetime

from .model import KokoroLM, KokoroConfig


class Trainer:
    """
    Trainer for Kokoro models.

    Args:
        model: KokoroLM model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        learning_rate: Peak learning rate (default: 3e-4)
        weight_decay: Weight decay for AdamW (default: 0.1)
        betas: AdamW betas (default: (0.9, 0.95))
        max_grad_norm: Gradient clipping threshold (default: 1.0)
        gradient_accumulation_steps: Accumulate gradients over N steps (default: 1)
        mixed_precision: Use mixed precision training ('fp16', 'bf16', or None)
        device: Device to train on
        output_dir: Directory for checkpoints and logs
        warmup_steps: Number of warmup steps (default: 100)
        max_steps: Maximum training steps (if None, train for full epochs)
    """

    def __init__(
        self,
        model: KokoroLM,
        train_loader,
        val_loader,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: Optional[Literal['fp16', 'bf16']] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = './outputs',
        warmup_steps: int = 100,
        max_steps: Optional[int] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training hyperparameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )

        # Calculate total steps
        if max_steps is None:
            steps_per_epoch = len(train_loader) // gradient_accumulation_steps
            total_steps = steps_per_epoch * 100  # Assume 100 epochs max
        else:
            total_steps = max_steps

        # Learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=learning_rate * 0.1,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # Mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision == 'fp16' else None
        self.autocast_dtype = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            None: torch.float32,
        }[mixed_precision]

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.samples_seen = 0
        self.start_time = time.time()

        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Mixed precision: {mixed_precision}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Effective batch size: {train_loader.batch_size * gradient_accumulation_steps}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Total steps: {total_steps if max_steps else 'unlimited'}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        train_loader,
        val_loader,
        device: Optional[str] = None,
    ):
        """
        Create Trainer from checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint directory (e.g., 'outputs/run_1/checkpoint_epoch_5')
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device override (None = auto-detect)

        Returns:
            Trainer instance with restored state
        """
        checkpoint_dir = Path(checkpoint_path)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

        print(f"\n{'='*70}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"{'='*70}")
        print(f"Checkpoint: {checkpoint_dir}")

        # Load model checkpoint
        model_path = checkpoint_dir / 'model.pt'
        checkpoint = torch.load(model_path, map_location='cpu')

        # Recreate model
        config = KokoroConfig(**checkpoint['config'])
        model = KokoroLM(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load training state
        with open(checkpoint_dir / 'training_state.json') as f:
            training_state = json.load(f)

        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create trainer with same hyperparameters
        # Note: We extract these from checkpoint or use defaults
        output_dir = checkpoint_dir.parent  # Parent directory of checkpoint

        trainer = cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=str(output_dir),
        )

        # Restore optimizer and scheduler state
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if checkpoint['scaler_state_dict'] and trainer.scaler:
            trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore training progress
        trainer.global_step = training_state['global_step']
        trainer.current_epoch = training_state['epoch']
        trainer.samples_seen = training_state['samples_seen']
        trainer.best_val_loss = training_state['best_val_loss']
        trainer.train_losses = training_state['train_losses']
        trainer.val_losses = training_state['val_losses']

        # Adjust start time to account for previous training
        trainer.start_time = time.time() - training_state['elapsed_time_seconds']

        print(f"\n✓ Checkpoint loaded successfully!")
        print(f"  Epoch: {trainer.current_epoch}")
        print(f"  Global step: {trainer.global_step}")
        print(f"  Samples seen: {trainer.samples_seen:,}")
        print(f"  Best val loss: {trainer.best_val_loss:.4f}")
        print(f"  Training will resume from epoch {trainer.current_epoch + 1}")
        print(f"{'='*70}\n")

        return trainer

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=True,
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.mixed_precision is not None, dtype=self.autocast_dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs['loss']

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights every N steps
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

            # Track samples and loss
            self.samples_seen += input_ids.size(0)
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'step': self.global_step,
            })

            # Stop if max_steps reached
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
        )

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with autocast(enabled=self.mixed_precision is not None, dtype=self.autocast_dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs['loss']

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': f'{total_loss / num_batches:.4f}'})

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)

        return avg_loss

    def save_checkpoint(self, name: str = 'checkpoint'):
        """
        Save checkpoint to directory with metadata.

        Creates a directory structure:
            checkpoint_epoch_N/
                ├── model.pt            # Model + optimizer + scheduler
                ├── metrics.json        # Epoch metrics
                └── training_state.json # Training progress
        """
        # Create checkpoint directory
        checkpoint_dir = self.output_dir / f'checkpoint_{name}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model + optimizer + scheduler
        model_path = checkpoint_dir / 'model.pt'
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config.__dict__,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }

        # Atomic write (write to temp, then rename)
        temp_path = model_path.with_suffix('.tmp')
        torch.save(checkpoint_data, temp_path)
        temp_path.replace(model_path)

        # Save metrics
        metrics = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'samples_seen': self.samples_seen,
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'best_val_loss': self.best_val_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'is_best': name == 'best',
        }

        with open(checkpoint_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save training state
        training_state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'samples_seen': self.samples_seen,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time_seconds': time.time() - self.start_time,
            'completed': False,
        }

        with open(checkpoint_dir / 'training_state.json', 'w') as f:
            json.dump(training_state, f, indent=2)

        print(f"  ✓ Checkpoint saved: {checkpoint_dir.name}")

    def train(self, num_epochs: int, eval_every: int = 1, save_every: int = 1):
        """
        Main training loop.

        Args:
            num_epochs: Total number of epochs to train (including already completed)
            eval_every: Run validation every N epochs
            save_every: Save checkpoint every N epochs
        """
        start_epoch = self.current_epoch + 1  # Resume from next epoch
        epochs_remaining = num_epochs - self.current_epoch

        print(f"\nStarting training...")
        print(f"  Start epoch: {start_epoch}")
        print(f"  Total epochs: {num_epochs}")
        print(f"  Epochs to train: {epochs_remaining}")
        print("=" * 60)

        for epoch in range(start_epoch - 1, num_epochs):  # -1 because current_epoch is 0-indexed
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")

            # Validate
            if (epoch + 1) % eval_every == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best')
                    print(f"New best model! Val loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}')

            # Stop if max_steps reached
            if self.max_steps is not None and self.global_step >= self.max_steps:
                print(f"Reached max_steps ({self.max_steps}), stopping training.")
                break

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save final checkpoint
        self.save_checkpoint('final')


def quick_train(
    config: KokoroConfig,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    output_dir: str = './outputs',
    **kwargs,
) -> Trainer:
    """
    Quick training function for convenience.

    Args:
        config: Model configuration
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use
        output_dir: Output directory
        **kwargs: Additional trainer arguments

    Returns:
        Trained Trainer instance
    """
    # Create model
    model = KokoroLM(config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_dir,
        **kwargs,
    )

    # Train
    trainer.train(num_epochs=num_epochs)

    return trainer
