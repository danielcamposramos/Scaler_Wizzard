"""Test script to verify CPU optimization of the TRMC model.
"""

import torch
import time
from research.trmc_model import TRMCModel

def test_cpu_optimization():
    print("Testing TRMC Model on CPU with optimization...")

    # 1. Initialize model on CPU
    vocab_size = 1000
    hidden_dim = 128
    num_heads = 4
    num_experts = 4
    expert_dim = 512
    num_iterations = 4

    model = TRMCModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_experts=num_experts,
        expert_dim=expert_dim,
        num_iterations=num_iterations,
        use_quantization=False
    ).to("cpu")

    # 2. Prepare dummy input
    batch_size = 1
    seq_len = 32
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 3. Optimize for CPU
    model.optimize_for_cpu()

    # 4. Run inference
    print(f"Running inference with batch_size={batch_size}, seq_len={seq_len}, iterations={num_iterations}...")
    start_time = time.time()
    with torch.no_grad():
        logits, latent = model(dummy_input)
    inference_time = time.time() - start_time

    print(f"Optimized inference time: {inference_time:.4f}s")
    print(f"Logits shape: {logits.shape}")
    print(f"Latent shape: {latent.shape}")

    # 5. Verify output
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert latent.shape == (batch_size, seq_len, hidden_dim)
    print("Optimized CPU execution test passed!")

if __name__ == "__main__":
    test_cpu_optimization()
