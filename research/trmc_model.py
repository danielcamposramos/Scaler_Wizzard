"""Tiny Recursive MoE Contrastive (TRMC) Model Implementation.

This module provides a PyTorch implementation of a small-scale model that
combines recursive reasoning, Mixture of Experts (MoE) layers, and
a contrastive learning objective.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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


class VisionEncoder(nn.Module):
    """Lightweight vision encoder inspired by DeepSeek-VL.

    This encoder projects image-like spatial data into the transformer's latent space.
    For this 'Tiny' implementation, we use a simple CNN-based encoder.
    """
    def __init__(self, hidden_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        # Simple projection from image patches to hidden_dim
        self.proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Projects images to visual tokens.

        Args:
            images (torch.Tensor): Shape (B, 3, H, W).

        Returns:
            torch.Tensor: Visual tokens (B, L_v, hidden_dim).
        """
        x = self.proj(images)  # (B, hidden_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, L_v, hidden_dim)
        return self.norm(x)


class TRMCModel(nn.Module):
    """Tiny Recursive MoE Contrastive (TRMC) Model with Matryoshka and Vision.

    Attributes:
        embedding (nn.Embedding): Token embedding layer.
        vision_encoder (VisionEncoder): Optional lightweight vision encoder.
        core (TRMCBlock): The recursive transformer core block.
        prediction_head (nn.Linear): Maps latent state to output vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_experts: int = 8,
        expert_dim: int = 1024,
        num_iterations: int = 8,
        max_seq_len: int = 128,
        matryoshka_dims: Optional[List[int]] = None,
        use_vision: bool = True,
    ) -> None:
        """Initializes the TRMCModel.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_dim (int): Dimensionality of the hidden states. Defaults to 256.
            num_heads (int): Number of attention heads. Defaults to 8.
            num_experts (int): Number of experts in MoE. Defaults to 8.
            expert_dim (int): Intermediate dimensionality of experts. Defaults to 1024.
            num_iterations (int): Number of recursive iterations. Defaults to 8.
            max_seq_len (int): Maximum sequence length. Defaults to 128.
            matryoshka_dims (Optional[List[int]]): Dimensions for Matryoshka embeddings.
            use_vision (bool): Whether to include a vision encoder. Defaults to True.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.max_seq_len = max_seq_len
        # Default matryoshka dims: quarter, half, and full hidden_dim
        self.matryoshka_dims = matryoshka_dims or [hidden_dim // 4, hidden_dim // 2, hidden_dim]

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        # Vision/OCR Awareness
        self.use_vision = use_vision
        if use_vision:
            self.vision_encoder = VisionEncoder(hidden_dim)
            self.vision_pos_embedding = nn.Parameter(torch.zeros(1, 256, hidden_dim))

        # Recursive core
        self.core = TRMCBlock(hidden_dim, num_heads, num_experts, expert_dim)
        self.prediction_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the TRMCModel.

        Args:
            x (torch.Tensor): Input tokens of shape (batch_size, seq_len).
            images (Optional[torch.Tensor]): Input images of shape (B, 3, H, W).
            num_steps (Optional[int]): Number of iterations to run. Defaults to self.num_iterations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Logits of shape (batch_size, seq_len, vocab_size) from the final step.
                - Final latent state of shape (batch_size, total_seq_len, hidden_dim).
        """
        batch_size, seq_len = x.shape
        num_steps = num_steps or self.num_iterations

        # Initial state: embed inputs
        h = self.embedding(x) + self.pos_embedding[:, :seq_len, :]

        # Add vision tokens if available
        if self.use_vision and images is not None:
            v = self.vision_encoder(images)
            v = v + self.vision_pos_embedding[:, :v.shape[1], :]
            h = torch.cat([v, h], dim=1)

        # Recursive reasoning steps
        for _ in range(num_steps):
            h = self.core(h)

        # Final prediction (only on text tokens if vision tokens were prepended)
        # Assuming we want to predict text tokens, which are at the end.
        logits = self.prediction_head(h[:, -seq_len:, :])

        return logits, h


def contrastive_loss(
    query_latent: torch.Tensor,
    positive_latent: torch.Tensor,
    negative_latents: torch.Tensor,
    temperature: float = 0.1,
    matryoshka_dims: Optional[List[int]] = None,
) -> torch.Tensor:
    """Calculates a Matryoshka-aware supervised contrastive loss.

    Args:
        query_latent (torch.Tensor): The current model state (B, H).
        positive_latent (torch.Tensor): Latent state of the positive example (B, H).
        negative_latents (torch.Tensor): Latent states of negative examples (B, N, H).
        temperature (float): The temperature for the contrastive loss. Defaults to 0.1.
        matryoshka_dims (Optional[List[int]]): Dimensions for Matryoshka loss.

    Returns:
        torch.Tensor: The calculated Matryoshka InfoNCE loss.
    """
    dims = matryoshka_dims or [query_latent.shape[-1]]
    total_loss = 0.0

    for dim in dims:
        # Truncate latents for this dimension
        q = query_latent[..., :dim]
        p = positive_latent[..., :dim]
        n = negative_latents[..., :dim]

        # Normalize truncated latents
        q = F.normalize(q, p=2, dim=-1)
        p = F.normalize(p, p=2, dim=-1)
        n = F.normalize(n, p=2, dim=-1)

        # Compute InfoNCE for this dimension
        pos_sim = torch.sum(q * p, dim=-1, keepdim=True) / temperature
        neg_sims = torch.bmm(n, q.unsqueeze(-1)).squeeze(-1) / temperature

        logits = torch.cat([pos_sim, neg_sims], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        total_loss += F.cross_entropy(logits, labels)

    return total_loss / len(dims)


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
