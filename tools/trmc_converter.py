"""TRMC GGUF Mapping Logic.

This module provides the core weight mapping and metadata handling
for TRMC (Tiny Recursive MoE Contrastive) model conversion.
"""

import torch
import numpy as np
from gguf import GGUFWriter


def convert_trmc_to_gguf(checkpoint_path: str, gguf_path: str, model_config: dict = None):
    """Converts a TRMC PyTorch checkpoint to GGUF format.

    Args:
        checkpoint_path (str): Path to the .pt checkpoint.
        gguf_path (str): Path to save the .gguf file.
        model_config (dict): Optional dictionary containing model hyperparameters.
    """
    print(f"Loading TRMC checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", model_config or {})
    else:
        state_dict = checkpoint
        config = model_config or {}

    writer = GGUFWriter(gguf_path, arch="trmc")

    # 1. Architectural Metadata
    writer.add_name("TRMC-Recursive-MoE")
    writer.add_description("Tiny Recursive MoE Contrastive Model")

    hidden_dim = config.get("hidden_dim", 128)
    num_heads = config.get("num_heads", 4)
    num_experts = config.get("num_experts", 8)
    num_iterations = config.get("num_iterations", 8)
    vocab_size = config.get("vocab_size", 32000)
    expert_dim = config.get("expert_dim", 256)

    writer.add_uint32("trmc.hidden_dim", hidden_dim)
    writer.add_uint32("trmc.head_count", num_heads)
    writer.add_uint32("trmc.expert_count", num_experts)
    writer.add_uint32("trmc.iteration_count", num_iterations)
    writer.add_uint32("trmc.expert_hidden_dim", expert_dim)
    writer.add_uint32("trmc.vocab_size", vocab_size)

    # 2. Tensor Mapping
    for name, tensor in state_dict.items():
        data = tensor.detach().cpu().numpy()
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        gguf_name = name
        if "embedding.weight" in name:
            gguf_name = "token_embd.weight"
        elif "pos_embedding" in name:
            gguf_name = "pos_embd.weight"
        elif "prediction_head.weight" in name:
            gguf_name = "output.weight"
        elif "prediction_head.bias" in name:
            gguf_name = "output.bias"
        elif "core.attention" in name:
            gguf_name = name.replace("core.attention", "blk.0.attn")
        elif "core.moe.gate" in name:
            gguf_name = "blk.0.moe.gate.weight"
        elif "core.moe.experts" in name:
            parts = name.split(".")
            expert_idx = parts[3]
            layer_idx = parts[4]
            suffix = parts[5]
            if layer_idx == "0":
                gguf_name = f"blk.0.moe.expert.{expert_idx}.ffn_up.{suffix}"
            else:
                gguf_name = f"blk.0.moe.expert.{expert_idx}.ffn_down.{suffix}"
        elif "core.norm" in name:
            gguf_name = name.replace("core.norm", "blk.0.norm")
        elif "vision_encoder" in name:
             gguf_name = name.replace("vision_encoder", "venc")

        writer.add_tensor(gguf_name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Successfully converted TRMC model to {gguf_path}")
