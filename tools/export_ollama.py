"""Ollama Export Tool for TRMC Models.

This script handles the preparation of trained TRMC models for use within Ollama.
It converts PyTorch checkpoints to GGUF format and generates a Modelfile.
"""

import argparse
import os
import torch
import numpy as np
from gguf import GGUFWriter


def convert_to_gguf(checkpoint_path: str, gguf_path: str):
    """Converts a TRMC PyTorch checkpoint to GGUF format.

    Args:
        checkpoint_path (str): Path to the .pt checkpoint.
        gguf_path (str): Path to save the .gguf file.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Initialize GGUF writer
    writer = GGUFWriter(gguf_path, arch="trmc")

    # Add metadata
    writer.add_name("TRMC-Model")
    writer.add_description("Tiny Recursive MoE Contrastive Model")

    print("Writing tensors to GGUF...")
    for name, tensor in state_dict.items():
        # Convert torch tensor to numpy
        data = tensor.detach().cpu().numpy()

        # GGUF expects specific naming and types
        # For this implementation, we preserve the original names
        # and ensure data is in float32 for maximum compatibility
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Successfully converted to {gguf_path}")


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
    try:
        convert_to_gguf(args.checkpoint, gguf_path)
    except Exception as e:
        print(f"Error during GGUF conversion: {e}")
        print("Note: Conversion requires the 'gguf' python package and a valid .pt file.")
        return

    # 2. Create the Modelfile
    create_modelfile(gguf_path, modelfile_path)

    print("\n--- NEXT STEPS ---")
    print(f"1. Run: ollama create trmc-model -f {modelfile_path}")
    print("2. Run: ollama run trmc-model")


if __name__ == "__main__":
    main()
