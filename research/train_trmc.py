"""Training script for the Tiny Recursive MoE Contrastive (TRMC) Model.

This notebook-style script handles dataset generation, model initialization,
and training on a reasoning task (e.g., synthetic Sudoku or Mazes).
Optimized for consumer hardware like the NVIDIA RTX 3070.
"""

import json
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset_curator import MultiDomainTRMCDataset
from trmc_model import TRMCModel


class LogicPuzzlesDataset(Dataset):
    """Synthetic dataset for simple logic puzzles with contrastive examples.

    This dataset provides (input, positive_label, negative_labels) triplets
    to facilitate supervised contrastive learning.
    """

    def __init__(
        self,
        size: int = 1000,
        seq_len: int = 16,
        vocab_size: int = 10,
        num_negatives: int = 5,
    ):
        """Initializes the LogicPuzzlesDataset.

        Args:
            size (int): The total number of samples. Defaults to 1000.
            seq_len (int): The length of each sample sequence. Defaults to 16.
            vocab_size (int): The number of possible tokens. Defaults to 10.
            num_negatives (int): Number of negative examples per positive.
        """
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_negatives = num_negatives

        self.data = []
        for _ in range(size):
            # Input sequence (random numbers)
            x = torch.randint(3, vocab_size, (seq_len,))
            # Positive label (e.g., the reversed sequence)
            y_pos = torch.flip(x, dims=[0])

            # Negative labels (e.g., slightly corrupted reversed sequences)
            y_negs = []
            for _ in range(num_negatives):
                # Create a negative example by perturbing the positive one
                y_neg = y_pos.clone()
                idx = torch.randint(0, seq_len, (max(1, seq_len // 4),))
                y_neg[idx] = torch.randint(3, vocab_size, (len(idx),))
                y_negs.append(y_neg)

            self.data.append((x, y_pos, torch.stack(y_negs)))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def train_trmc():
    """Main training loop for the TRMC model."""

    # 1. Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 32000
    seq_len = 128
    hidden_dim = 256
    num_heads = 8
    num_experts = 8
    expert_dim = 1024
    num_iterations = 8
    batch_size = 64
    epochs = 50
    learning_rate = 1e-4
    contrastive_weight = 0.2

    print(f"Training on: {device}")

    # 2. Dataset and DataLoader
    # Use MultiDomainTRMCDataset for more robust training simulation
    dataset = MultiDomainTRMCDataset(size=5000, seq_len=seq_len, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Model, Optimizer, and Loss
    model = TRMCModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_experts=num_experts,
        expert_dim=expert_dim,
        num_iterations=num_iterations,
        max_seq_len=seq_len
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    from trmc_model import contrastive_loss

    # 4. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (x, y_pos, y_negs) in enumerate(progress_bar):
            x, y_pos, y_negs = x.to(device), y_pos.to(device), y_negs.to(device)

            # Optional: Simulate vision input for multi-modal training (32x32 patches)
            # In real training, these would come from the dataset
            images = torch.randn(x.shape[0], 3, 32, 32).to(device)

            optimizer.zero_grad()

            # Forward pass through recursive core for the input
            logits, query_latent = model(x, images=images)

            # Forward pass to get latents for positive/negative examples
            # In a full MoCo setup, these would come from a momentum-updated encoder.
            # For this lightweight version, we reuse the same model.
            with torch.no_grad():
                _, pos_latent = model(y_pos, images=images)

                batch_size, num_negs, s_len = y_negs.shape
                # Flatten negatives to process in one batch pass
                _, neg_latent_all = model(y_negs.view(-1, s_len), images=images.repeat_interleave(num_negs, dim=0))
                neg_latents = neg_latent_all.view(batch_size, num_negs, -1, hidden_dim)

            # Compute standard cross-entropy prediction loss
            ce_loss = criterion(logits.view(-1, vocab_size), y_pos.view(-1))

            # Compute Matryoshka-aware contrastive loss on sequence-level latents (mean-pooled)
            # Query: (B, H), Pos: (B, H), Negs: (B, N, H)
            c_loss = contrastive_loss(
                query_latent.mean(dim=1),
                pos_latent.mean(dim=1),
                neg_latents.mean(dim=2),
                matryoshka_dims=model.matryoshka_dims
            )

            # Total loss
            loss = ce_loss + contrastive_weight * c_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # 5. Save the trained model
    os.makedirs("checkpoints/trmc", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/trmc/model_final.pt")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    # In a real notebook, we would run this in a cell.
    # To run on the console: python research/train_trmc.py
    train_trmc()
