"""
Text generation script for trained Kokoro models.

Usage:
    # Use pre-built test prompts
    python scripts/generate.py --checkpoint outputs/test_run_small/checkpoint_best --test-prompts

    # Custom prompt
    python scripts/generate.py --checkpoint outputs/test_run_small/checkpoint_best --prompt "Once upon a time"

    # Multiple custom prompts
    python scripts/generate.py --checkpoint outputs/test_run_small/checkpoint_best --prompts-file prompts.txt
"""

# Pre-built test prompts for evaluation
TEST_PROMPTS = [
    "Once upon a time",
    "The little girl was very",
    "In the forest, there was a",
    "One day, a boy found",
    "The cat and the dog",
    "It was a sunny day and",
    "Mom said to her son",
    "The magic tree could",
]

import argparse
import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kokoro import KokoroConfig, KokoroLM


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with Kokoro")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint directory")

    # Prompt options (mutually exclusive)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", type=str,
                             help="Single text prompt for generation")
    prompt_group.add_argument("--test-prompts", action="store_true",
                             help="Use pre-built test prompts")
    prompt_group.add_argument("--prompts-file", type=str,
                             help="File with prompts (one per line)")

    # Generation parameters
    parser.add_argument("--max-length", type=int, default=200,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")
    parser.add_argument("--num-samples", type=int, default=1,
                       help="Number of samples to generate per prompt")
    return parser.parse_args()


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
):
    """Generate text from a prompt."""
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate tokens one by one
    generated = input_ids

    for _ in range(max_length):
        # Forward pass
        outputs = model(generated)
        logits = outputs["logits"]

        # Get last token logits
        next_token_logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[:, indices_to_remove] = float('-inf')

        # Sample
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Stop if EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text


def main():
    args = parse_args()

    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Kokoro Text Generation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print()

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint) / "model.pt"
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    # Load config and model
    config = KokoroConfig(**checkpoint['config'])
    model = KokoroLM(config).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded: {config.vocab_size:,} vocab, {config.hidden_size} hidden")
    print()

    # Load tokenizer (assuming GPT-2 tokenizer was used)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine prompts to use
    prompts = []
    if args.test_prompts:
        prompts = TEST_PROMPTS
        print(f"Using {len(TEST_PROMPTS)} pre-built test prompts")
    elif args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    elif args.prompt:
        prompts = [args.prompt]
    else:
        prompts = ["Once upon a time"]
        print("No prompt specified, using default")

    print()

    # Generate for each prompt
    for i, prompt in enumerate(prompts, 1):
        if len(prompts) > 1:
            print("\n" + "=" * 60)
            print(f"PROMPT {i}/{len(prompts)}")
            print("=" * 60)

        print(f"Input: {prompt}")
        print("-" * 60)

        # Generate multiple samples if requested
        for sample_num in range(args.num_samples):
            if args.num_samples > 1:
                print(f"\n[Sample {sample_num + 1}/{args.num_samples}]")

            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
            )

            print(generated_text)

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
