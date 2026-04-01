"""Tiny Recursive MoE Contrastive (TRMC) Model Implementation.

This module provides a PyTorch implementation of a small-scale model that
combines recursive reasoning, Mixture of Experts (MoE) layers, and
a contrastive learning objective.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts (MoE) layer.

    This layer routes inputs to one of several "expert" networks based on a
    gating mechanism, reducing computation per input while increasing total
    parameter capacity.

    Attributes:
        num_experts (int): The total number of expert networks.
        hidden_dim (int): The input and output dimensionality.
        expert_dim (int): The intermediate dimensionality of each expert.
        top_k (int): The number of experts to activate for each input.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        expert_dim: int,
        top_k: int = 1,
    ) -> None:
        """Initializes the SparseMoELayer.

        Args:
            num_experts (int): The total number of expert networks.
            hidden_dim (int): The input and output dimensionality.
            expert_dim (int): The intermediate dimensionality of each expert.
            top_k (int): The number of experts to activate. Defaults to 1.
        """
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.top_k = top_k

        # Gating network
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the SparseMoELayer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: The output tensor of the same shape as input.
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # (batch_size * seq_len, hidden_dim)

        # Compute gate logits
        gate_logits = self.gate(x_flat)  # (B*L, num_experts)

        # Get top-k experts and their weights
        weights, selected_experts = torch.topk(
            F.softmax(gate_logits, dim=-1), self.top_k, dim=-1
        )  # (B*L, top_k)

        # Normalize weights
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Initialize output
        final_output = torch.zeros_like(x_flat)

        # Route inputs to experts and accumulate outputs
        for k in range(self.top_k):
            # Mask for inputs routed to expert i
            for i in range(self.num_experts):
                mask = (selected_experts[:, k] == i)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[i](expert_input)
                    final_output[mask] += weights[mask, k:k+1] * expert_output

        return final_output.view(batch_size, seq_len, hidden_dim)


class TRMCBlock(nn.Module):
    """A single transformer block with an MoE layer instead of a standard FFN.

    Attributes:
        attention (nn.MultiheadAttention): Multi-head self-attention mechanism.
        moe (SparseMoELayer): Sparse Mixture of Experts layer.
        norm1 (nn.LayerNorm): Layer normalization after attention.
        norm2 (nn.LayerNorm): Layer normalization after MoE.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_experts: int,
        expert_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the TRMCBlock.

        Args:
            hidden_dim (int): The dimensionality of the hidden states.
            num_heads (int): The number of attention heads.
            num_experts (int): The total number of experts in the MoE layer.
            expert_dim (int): The intermediate dimensionality of each expert.
            dropout (float): The dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.moe = SparseMoELayer(num_experts, hidden_dim, expert_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the TRMCBlock.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: The output tensor of the same shape as input.
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MoE layer with residual connection
        moe_out = self.moe(x)
        x = x + self.dropout(moe_out)
        x = self.norm2(x)

        return x


class TRMCModel(nn.Module):
    """Tiny Recursive MoE Contrastive (TRMC) Model.

    This model recursively applies a small transformer core to solve reasoning tasks.

    Attributes:
        embedding (nn.Embedding): Token embedding layer.
        core (TRMCBlock): The recursive transformer core block.
        prediction_head (nn.Linear): Maps latent state to output vocabulary.
        num_iterations (int): The number of recursive reasoning steps.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_experts: int = 8,
        expert_dim: int = 256,
        num_iterations: int = 8,
        max_seq_len: int = 64,
    ) -> None:
        """Initializes the TRMCModel.

        Args:
            vocab_size (int): The size of the input/output vocabulary.
            hidden_dim (int): The dimensionality of the hidden states. Defaults to 128.
            num_heads (int): The number of attention heads. Defaults to 4.
            num_experts (int): The total number of experts. Defaults to 8.
            expert_dim (int): The intermediate dimensionality of each expert. Defaults to 256.
            num_iterations (int): The number of recursive steps. Defaults to 8.
            max_seq_len (int): The maximum sequence length. Defaults to 64.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        # Recursive core
        self.core = TRMCBlock(hidden_dim, num_heads, num_experts, expert_dim)

        # Prediction head
        self.prediction_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the TRMCModel.

        Args:
            x (torch.Tensor): Input tokens of shape (batch_size, seq_len).
            num_steps (Optional[int]): Number of iterations to run. Defaults to self.num_iterations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Logits of shape (batch_size, seq_len, vocab_size) from the final step.
                - Final latent state of shape (batch_size, seq_len, hidden_dim).
        """
        batch_size, seq_len = x.shape
        num_steps = num_steps or self.num_iterations

        # Initial state: embed inputs and add positional encoding
        h = self.embedding(x) + self.pos_embedding[:, :seq_len, :]

        # Recursive reasoning steps
        for _ in range(num_steps):
            h = self.core(h)

        # Final prediction
        logits = self.prediction_head(h)

        return logits, h


def contrastive_loss(
    query_latent: torch.Tensor,
    positive_latent: torch.Tensor,
    negative_latents: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Calculates a supervised contrastive loss for the TRMC model.

    This function encourages the query latent state to be closer to the positive
    latent representation and farther from negative ones using InfoNCE loss.

    Args:
        query_latent (torch.Tensor): The current model state (B, H).
        positive_latent (torch.Tensor): Latent state of the positive example (B, H).
        negative_latents (torch.Tensor): Latent states of negative examples (B, N, H).
        temperature (float): The temperature for the contrastive loss. Defaults to 0.1.

    Returns:
        torch.Tensor: The calculated InfoNCE contrastive loss.
    """
    # Normalize latents to the unit hypersphere
    query = F.normalize(query_latent, p=2, dim=-1)
    pos = F.normalize(positive_latent, p=2, dim=-1)
    negs = F.normalize(negative_latents, p=2, dim=-1)

    # Compute positive similarity: (B, 1)
    pos_sim = torch.sum(query * pos, dim=-1, keepdim=True) / temperature

    # Compute negative similarities: (B, N)
    # negs: (B, N, H), query: (B, H) -> unsqueeze query to (B, 1, H)
    neg_sims = torch.bmm(negs, query.unsqueeze(-1)).squeeze(-1) / temperature

    # Concatenate positive and negative similarities: (B, 1 + N)
    logits = torch.cat([pos_sim, neg_sims], dim=-1)

    # The target is always index 0 (the positive sample)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)


if __name__ == "__main__":
    # Quick sanity check
    model = TRMCModel(vocab_size=10, hidden_dim=64, num_experts=4, num_iterations=4)
    dummy_input = torch.randint(0, 10, (2, 16))
    logits, latent = model(dummy_input)
    print(f"Logits shape: {logits.shape}")
    print(f"Latent shape: {latent.shape}")

    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
