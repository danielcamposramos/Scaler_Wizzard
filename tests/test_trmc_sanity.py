"""Sanity check for TRMC model and dataset."""

import torch
import sys
import os

# Add research/ to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "research")))

from trmc_model import TRMCModel
from dataset_curator import TRMCDataset

def test_trmc_sanity():
    print("Running TRMC Sanity Check...")
    vocab_size = 1000
    hidden_dim = 128
    num_heads = 4
    num_experts = 4
    expert_dim = 256
    num_iterations = 4
    seq_len = 32

    model = TRMCModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_experts=num_experts,
        expert_dim=expert_dim,
        num_iterations=num_iterations,
        max_seq_len=seq_len
    )

    # Test forward pass (text only)
    x = torch.randint(0, vocab_size, (2, seq_len))
    logits, latent = model(x)
    assert logits.shape == (2, seq_len, vocab_size)
    assert latent.shape == (2, seq_len, hidden_dim)
    print("Forward pass (text only) successful.")

    # Test forward pass (with vision)
    images = torch.randn(2, 3, 224, 224)
    logits_v, latent_v = model(x, images=images)
    # Vision tokens (196) + text tokens (32) = 228
    assert logits_v.shape == (2, seq_len, vocab_size)
    assert latent_v.shape == (2, 228, hidden_dim)
    print("Forward pass (with vision) successful.")

    # Test dataset
    class DummyTokenizer:
        def __init__(self, vocab_size): self.vocab_size = vocab_size
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.randint(0, self.vocab_size, (1, kwargs.get("max_length", 32)))}

    tokenizer = DummyTokenizer(vocab_size)
    dataset = TRMCDataset(tokenizer, max_seq_len=seq_len, num_samples=5, include_vision=True)
    assert len(dataset) == 5
    x, y_p, y_n, img = dataset[0]
    assert x.shape == (seq_len,)
    assert y_p.shape == (seq_len,)
    assert y_n.shape == (3, seq_len)
    assert img.shape == (3, 224, 224)
    print("Dataset sanity check successful.")

if __name__ == "__main__":
    test_trmc_sanity()
