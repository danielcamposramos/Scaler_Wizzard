# Tiny Recursive MoE Contrastive (TRMC) Model

The TRMC model is a small-scale, high-capacity AI architecture designed for multi-step reasoning tasks. It combines recursive processing with a Sparse Mixture of Experts (MoE) and a contrastive learning paradigm.

## Key Features

- **Recursive Core**: A small transformer block that reuses its own weights across multiple iterations, similar to the Tiny Recursive Model (TRM) from Samsung Research.
- **Sparse MoE**: Replaces traditional feed-forward networks with a gating mechanism that routes inputs to specialized experts, increasing model capacity without proportional compute costs.
- **Contrastive Learning**: Implements a supervised contrastive objective (InfoNCE) that optimizes the model's final latent state against positive and negative examples, helping it better distinguish correct reasoning paths.
- **Consumer Hardware Friendly**: Targeted at GPUs like the RTX 3070 (8GB VRAM), with low parameter counts (7M-20M) and memory-efficient recursive state.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- tqdm (for progress bars)

```bash
pip install torch tqdm
```

### Running the Implementation

1. **Architecture**: The core model is defined in `research/trmc_model.py`.
2. **Training**: You can run the training script or the interactive notebook to see the model solve a synthetic reasoning task (sequence reversal).

```bash
python research/train_trmc.py
```

Or open `research/train_trmc.ipynb` in a Jupyter environment.

## Model Architecture

The `TRMCModel` class in `research/trmc_model.py` includes:
- **Embedding Layer**: Token and positional embeddings.
- **TRMCBlock**: A transformer block with Multi-head Attention and a `SparseMoELayer`.
- **Recursive Loop**: Applies the `TRMCBlock` $N$ times (default 8).
- **Prediction Head**: A linear layer mapping to the output vocabulary.

## Research Background

This implementation was inspired by the paper "Less is More: Recursive Reasoning with Tiny Networks" (2025) and enhanced with modern MoE techniques to bridge the gap between tiny models and large language models (LLMs) on logic-intensive tasks.
