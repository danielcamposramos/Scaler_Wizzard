"""Ollama Export Tool for TRMC Models.

This script handles the preparation of trained TRMC models for use within Ollama.
It uses a specialized converter (trmc_converter.py) to generate GGUF files and
references the C++ architectural definition (trmc.cpp).
"""

import argparse
import os
import torch
import numpy as np
from trmc_converter import convert_trmc_to_gguf


def create_modelfile(gguf_path: str, modelfile_path: str):
    """Creates a template Modelfile for the TRMC model.

    Args:
        gguf_path (str): Path to the converted GGUF file.
        modelfile_path (str): Path where the Modelfile should be saved.
    """
    content = f"""# TRMC Modelfile
# Generated for Ollama integration

FROM ./{os.path.basename(gguf_path)}

# TRMC Specific Parameters
PARAMETER architecture trmc
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"

# System prompt for reasoning
SYSTEM \"\"\"You are a TRMC reasoning agent.
Think step-by-step using your recursive core.
Perform contrastive verification of your logic.\"\"\"
"""

    with open(modelfile_path, "w") as f:
        f.write(content)

    print(f"Successfully created Modelfile at {modelfile_path}")


def main():
    parser = argparse.ArgumentParser(description="Export TRMC model to Ollama")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pt checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/ollama",
        help="Directory to save the export artifacts"
    )
    # Architectural parameters
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=8)
    parser.add_argument("--expert_dim", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_name = os.path.basename(args.checkpoint)
    gguf_name = checkpoint_name.replace(".pt", ".gguf")
    if not gguf_name.endswith(".gguf"):
        gguf_name += ".gguf"

    gguf_path = os.path.join(args.output_dir, gguf_name)
    modelfile_path = os.path.join(args.output_dir, "Modelfile")

    print(f"Exporting checkpoint: {args.checkpoint}")

    # 1. Convert to GGUF
    config = {
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_experts": args.num_experts,
        "num_iterations": args.num_iterations,
        "expert_dim": args.expert_dim,
        "vocab_size": args.vocab_size
    }

    try:
        convert_trmc_to_gguf(args.checkpoint, gguf_path, config)
    except Exception as e:
        print(f"Error during GGUF conversion: {e}")
        print("Note: Conversion requires the 'gguf' python package and a valid .pt file.")
        return

    # 2. Create the Modelfile
    create_modelfile(gguf_path, modelfile_path)

    print("\n--- NEXT STEPS ---")
    print(f"1. Compile the architectural converter: g++ tools/trmc.cpp -o trmc_converter")
    print(f"2. Run: ollama create trmc-model -f {modelfile_path}")
    print("3. Run: ollama run trmc-model")


if __name__ == "__main__":
    main()
