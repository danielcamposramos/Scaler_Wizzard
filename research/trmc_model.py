"""Tiny Recursive MoE Contrastive (TRMC) Model Implementation.

This module provides a PyTorch implementation of a small-scale model that
combines recursive reasoning, Mixture of Experts (MoE) layers, and
a contrastive learning objective.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts (MoE) layer.

    This layer routes inputs to one of several "expert" networks based on a
    gating mechanism, reducing computation per input while increasing total
    parameter capacity.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        expert_dim: int,
        top_k: int = 1,
        use_quantization: bool = False,
    ) -> None:
        """Initializes the SparseMoELayer.

        Args:
            num_experts (int): The total number of expert networks.
            hidden_dim (int): The input and output dimensionality.
            expert_dim (int): The intermediate dimensionality of each expert.
            top_k (int): The number of experts to activate. Defaults to 1.
            use_quantization (bool): Whether to use 4-bit quantization for expert weights.
        """
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.top_k = top_k

        # Gating network
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Expert networks
        if use_quantization and HAS_BNB:
            # Check if we are on a GPU, bitsandbytes 4bit needs it
            if torch.cuda.is_available():
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        bnb.nn.Linear4bit(hidden_dim, expert_dim, bias=False),
                        nn.GELU(),
                        bnb.nn.Linear4bit(expert_dim, hidden_dim, bias=False)
                    ) for _ in range(num_experts)
                ])
            elif torch.backends.mps.is_available():
                print("Warning: bitsandbytes 4bit requested but on Apple Silicon (MPS). bitsandbytes doesn't support MPS. Falling back to standard Linear.")
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_dim, expert_dim),
                        nn.GELU(),
                        nn.Linear(expert_dim, hidden_dim)
                    ) for _ in range(num_experts)
                ])
            else:
                print("Warning: bitsandbytes 4bit requested but no GPU available. Falling back to standard Linear.")
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_dim, expert_dim),
                        nn.GELU(),
                        nn.Linear(expert_dim, hidden_dim)
                    ) for _ in range(num_experts)
                ])
        else:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, expert_dim),
                    nn.GELU(),
                    nn.Linear(expert_dim, hidden_dim)
                ) for _ in range(num_experts)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the SparseMoELayer."""
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
        # Optimized routing for small models
        for i in range(self.num_experts):
            mask = (selected_experts == i)
            if mask.any():
                # Find which items in the batch are routed to this expert
                item_indices, k_indices = torch.where(mask)
                expert_input = x_flat[item_indices]
                expert_output = self.experts[i](expert_input)
                final_output[item_indices] += weights[item_indices, k_indices].unsqueeze(-1) * expert_output

        return final_output.view(batch_size, seq_len, hidden_dim)


class TRMCBlock(nn.Module):
    """A single transformer block with an MoE layer instead of a standard FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_experts: int,
        expert_dim: int,
        dropout: float = 0.1,
        use_quantization: bool = False,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.moe = SparseMoELayer(num_experts, hidden_dim, expert_dim, use_quantization=use_quantization)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """Lightweight vision encoder inspired by DeepSeek-VL."""
    def __init__(self, hidden_dim: int, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        # Simple projection from image patches to hidden_dim
        self.proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Projects images to visual tokens. (B, 3, H, W) -> (B, L_v, hidden_dim)"""
        x = self.proj(images)  # (B, hidden_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, L_v, hidden_dim)
        return self.norm(x)


class TRMCModel(nn.Module):
    """Tiny Recursive MoE Contrastive (TRMC) Model with Matryoshka and Vision.
    Optimized for quick training and large context.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_experts: int = 16,
        expert_dim: int = 2048,
        num_iterations: int = 12,
        max_seq_len: int = 4096,  # Increased to match big labs AIs
        matryoshka_dims: Optional[List[int]] = None,
        use_vision: bool = True,
        use_quantization: bool = False,
    ) -> None:
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_experts": num_experts,
            "expert_dim": expert_dim,
            "num_iterations": num_iterations,
            "max_seq_len": max_seq_len,
            "matryoshka_dims": matryoshka_dims or [hidden_dim // 4, hidden_dim // 2, hidden_dim],
            "use_vision": use_vision,
            "use_quantization": use_quantization
        }

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.max_seq_len = max_seq_len
        self.matryoshka_dims = self.config["matryoshka_dims"]

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # Use Sinusoidal or learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        # Vision/OCR Awareness
        self.use_vision = use_vision
        if use_vision:
            self.vision_encoder = VisionEncoder(hidden_dim)
            # Default vision sequence length for 224x224 image with patch_size 16 is (224/16)^2 = 196
            # We use a larger buffer for vision position embeddings (up to 4096 tokens)
            self.vision_pos_embedding = nn.Parameter(torch.zeros(1, 4096, hidden_dim))

        # Recursive core
        self.core = TRMCBlock(
            hidden_dim, num_heads, num_experts, expert_dim, use_quantization=use_quantization
        )

        if use_quantization and HAS_BNB and torch.cuda.is_available():
             self.prediction_head = bnb.nn.Linear4bit(hidden_dim, vocab_size, bias=False)
        elif use_quantization and torch.backends.mps.is_available():
             # MPS doesn't support bitsandbytes yet, but we can still use the model
             self.prediction_head = nn.Linear(hidden_dim, vocab_size)
        else:
             self.prediction_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape
        num_steps = num_steps or self.num_iterations

        # Initial state: embed inputs
        h = self.embedding(x) + self.pos_embedding[:, :seq_len, :]

        # Add vision tokens if available
        if self.use_vision and images is not None:
            v = self.vision_encoder(images)
            # Ensure v.shape[1] does not exceed vision_pos_embedding size
            v_len = v.shape[1]
            if v_len > self.vision_pos_embedding.shape[1]:
                 # Trim or interpolate if necessary, here we trim for simplicity
                 v = v[:, :self.vision_pos_embedding.shape[1], :]
                 v_len = v.shape[1]

            v = v + self.vision_pos_embedding[:, :v_len, :]
            h = torch.cat([v, h], dim=1)

        # Recursive reasoning steps
        for _ in range(num_steps):
            h = self.core(h)

        # Final prediction (only on text tokens)
        logits = self.prediction_head(h[:, -seq_len:, :])

        return logits, h

    def save_pretrained(self, save_directory: str):
        """Saves the model and its configuration."""
        os.makedirs(save_directory, exist_ok=True)
        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
        # Save state dict
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_directory: str, map_location: str = "cpu"):
        """Loads the model from a directory.

        Args:
            load_directory (str): Directory containing config.json and pytorch_model.bin.
            map_location (str): Device to load the model to (e.g., 'cpu', 'cuda', or 'mps').
        """
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(
            torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location=map_location)
        )
        return model

    def optimize_for_cpu(self):
        """Applies CPU-specific optimizations such as dynamic quantization."""
        print("Optimizing TRMC Model for CPU execution...")
        # Apply dynamic quantization to the core and prediction head
        # This can significantly speed up inference on many CPUs
        self.eval()
        # MultiheadAttention quantization can be tricky in some versions of PyTorch
        # Let's start with Linear layers first which are the most common
        self.core = torch.quantization.quantize_dynamic(
            self.core, {nn.Linear}, dtype=torch.qint8
        )
        if isinstance(self.prediction_head, nn.Linear):
             self.prediction_head = torch.quantization.quantize_dynamic(
                 self.prediction_head, {nn.Linear}, dtype=torch.qint8
             )
        print("Model optimized for CPU.")
        return self


def contrastive_loss(
    query_latent: torch.Tensor,
    positive_latent: torch.Tensor,
    negative_latents: torch.Tensor,
    temperature: float = 0.1,
    matryoshka_dims: Optional[List[int]] = None,
) -> torch.Tensor:
    """Calculates a Matryoshka-aware supervised contrastive loss."""
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
        # Positive similarity (B, 1)
        pos_sim = torch.sum(q * p, dim=-1, keepdim=True) / temperature
        # Negative similarities (B, N)
        neg_sims = torch.bmm(n, q.unsqueeze(-1)).squeeze(-1) / temperature

        logits = torch.cat([pos_sim, neg_sims], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        total_loss += F.cross_entropy(logits, labels)

    return total_loss / len(dims)


if __name__ == "__main__":
    # Quick sanity check
    model = TRMCModel(vocab_size=10, hidden_dim=128, num_experts=4, num_iterations=4)
    dummy_input = torch.randint(0, 10, (2, 16))
    logits, latent = model(dummy_input)
    print(f"Logits shape: {logits.shape}")
    print(f"Latent shape: {latent.shape}")

    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
