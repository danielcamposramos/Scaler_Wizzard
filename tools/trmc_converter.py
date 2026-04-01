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

    # Spoof architecture to llama for Ollama/llama.cpp compatibility
    writer = GGUFWriter(gguf_path, arch="llama")

    # 1. Architectural Metadata
    writer.add_name("TRMC-Recursive-MoE")
    writer.add_description("Tiny Recursive MoE Contrastive Model (Llama-Spoofed)")

    hidden_dim = config.get("hidden_dim", 256)
    num_heads = config.get("num_heads", 8)
    num_experts = config.get("num_experts", 8)
    num_iterations = config.get("num_iterations", 8)
    vocab_size = config.get("vocab_size", 32000)
    expert_dim = config.get("expert_dim", 1024)

    # Standard Llama KV pairs for runner compatibility
    writer.add_context_length(config.get("max_seq_len", 128))
    writer.add_embedding_length(hidden_dim)
    writer.add_head_count(num_heads)
    writer.add_feed_forward_length(expert_dim)
    writer.add_block_count(1) # We use 1 recursive block
    writer.add_expert_count(num_experts)

    # TRMC Specific Metadata
    writer.add_uint32("trmc.iteration_count", num_iterations)

    # 2. Tensor Mapping
    for name, tensor in state_dict.items():
        data = tensor.detach().cpu().numpy()
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        gguf_name = name
        # Llama backend compatible mapping
        if "embedding.weight" in name:
            gguf_name = "token_embd.weight"
        elif "pos_embedding" == name:
            # Main text position embedding
            gguf_name = "trmc.pos_embd_text.weight"
        elif "vision_pos_embedding" == name:
            # Vision position embedding
            gguf_name = "trmc.pos_embd_vision.weight"
        elif "prediction_head.weight" in name:
            gguf_name = "output.weight"
        elif "prediction_head.bias" in name:
            gguf_name = "output.bias"
        elif "core.attention" in name:
            # Map core attention to blk.0 for llama runner
            gguf_name = name.replace("core.attention", "blk.0.attn")
        elif "core.moe.gate" in name:
            gguf_name = "blk.0.ffn_gate_inp.weight" # Llama MoE gate name
        elif "core.moe.experts" in name:
            parts = name.split(".")
            expert_idx = parts[3]
            layer_idx = parts[4]
            suffix = parts[5]
            # Llama MoE expert mapping
            if layer_idx == "0":
                gguf_name = f"blk.0.ffn_up.{expert_idx}.{suffix}"
            else:
                gguf_name = f"blk.0.ffn_down.{expert_idx}.{suffix}"
        elif "core.norm1" in name:
            gguf_name = "blk.0.attn_norm." + name.split(".")[-1]
        elif "core.norm2" in name:
            gguf_name = "blk.0.ffn_norm." + name.split(".")[-1]
        elif "vision_encoder" in name:
             gguf_name = name.replace("vision_encoder", "venc")

        print(f"Mapping tensor '{name}' -> '{gguf_name}'")
        writer.add_tensor(gguf_name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Successfully converted TRMC model to {gguf_path}")
