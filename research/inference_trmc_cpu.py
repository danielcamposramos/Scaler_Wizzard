"""Optimized CPU Inference for TRMC Models.

This script provides a dedicated way to execute TRMC models on CPU-only
environments, including optimizations for thread count and dynamic quantization.
"""

import argparse
import sys
import os
import time
import torch

# Ensure the research directory is in the path so TRMCModel can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trmc_model import TRMCModel

def run_cpu_inference(
    checkpoint_path: str = None,
    num_threads: int = 4,
    optimize: bool = True,
    seq_len: int = 64,
    iterations: int = 8
):
    """Runs a benchmark/sample inference on CPU with optimizations."""

    # 1. CPU-specific global optimizations
    torch.set_num_threads(num_threads)
    print(f"Setting torch threads to {num_threads}")

    # 2. Model initialization/loading
    if checkpoint_path:
        print(f"Loading model from {checkpoint_path}...")
        # Since we don't have a full directory with config.json, we'll assume standard specs
        # if the path is just a .pt file, otherwise use from_pretrained
        if checkpoint_path.endswith(".pt"):
             # For this demo, we'll create a model with default specs and load state dict
             model = TRMCModel(vocab_size=32000, hidden_dim=256, num_experts=8, num_iterations=iterations)
             model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        else:
             model = TRMCModel.from_pretrained(checkpoint_path, map_location="cpu")
    else:
        print("Initializing new TRMC model with default specs for benchmarking...")
        model = TRMCModel(
            vocab_size=32000,
            hidden_dim=256,
            num_heads=8,
            num_experts=8,
            expert_dim=1024,
            num_iterations=iterations,
            max_seq_len=seq_len
        )

    # 3. Optimization
    if optimize:
        model.optimize_for_cpu()

    model.eval()

    # 4. Prepare input
    dummy_input = torch.randint(0, 32000, (1, seq_len))
    print(f"Input shape: {dummy_input.shape}")

    # 5. Inference loop
    print(f"Running inference ({iterations} recursive steps)...")
    start_time = time.time()
    with torch.no_grad():
        logits, _ = model(dummy_input)
    end_time = time.time() - start_time

    print(f"Inference completed in {end_time:.4f} seconds.")
    print(f"Throughput: {seq_len / end_time:.2f} tokens/sec")

    # 6. Sample output (top-5 tokens for the last position)
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 5)

    print("\nTop-5 predicted next tokens:")
    for i in range(5):
        print(f"  Token ID {top_indices[i].item()}: {top_probs[i].item():.4f}")

def main():
    parser = argparse.ArgumentParser(description="TRMC CPU Inference Tool")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--no-optimize", action="store_true", help="Disable dynamic quantization")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length for inference")
    parser.add_argument("--iterations", type=int, default=8, help="Number of recursive reasoning steps")

    args = parser.parse_args()

    run_cpu_inference(
        checkpoint_path=args.checkpoint,
        num_threads=args.threads,
        optimize=not args.no_optimize,
        seq_len=args.seq_len,
        iterations=args.iterations
    )

if __name__ == "__main__":
    main()
