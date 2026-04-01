# Getting Started with Tiny Recursive MoE Contrastive (TRMC) Models

This guide provides a comprehensive walkthrough for setting up your environment on SparkyLinux (a Debian-based distribution) and training the TRMC model from scratch.

## 1. Hardware Requirements

To train TRMC models effectively on consumer-grade hardware, we recommend the following:

- **GPU**: NVIDIA RTX 3060/3070 (8GB VRAM) or better. The architecture is optimized for 8GB-12GB VRAM.
- **Apple Silicon**: M1/M2/M3 (Pro/Max/Ultra) with 16GB+ Unified Memory.
- **CPU**: 8+ core processor (e.g., AMD Ryzen 7 or Intel Core i7).
- **RAM**: 16GB minimum (32GB recommended for large datasets).
- **Storage**: 50GB+ free SSD space for datasets and checkpoints.

## 2. Operating System Setup (SparkyLinux)

SparkyLinux is a fast, lightweight, and fully customizable Debian-based Linux distribution.

### Installation Steps:
1. **Download**: Obtain the latest SparkyLinux ISO (Stable or Semi-Rolling) from [sparkylinux.org](https://sparkylinux.org/download/).
2. **Flash**: Use a tool like Etcher or `dd` to create a bootable USB drive.
3. **Install**: Boot from the USB and follow the Calamares installer.
4. **Update**: Once installed, open a terminal and run:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

### NVIDIA Driver Installation (for Linux):
SparkyLinux provides `sparky-aptus` to simplify driver installation:
1. Open **Sparky APTus**.
2. Navigate to **Graphics** -> **NVIDIA Drivers**.
3. Select the latest proprietary driver and follow the prompts.
4. Reboot your system.
5. Verify installation by running `nvidia-smi` in the terminal.

### Apple Silicon Setup (macOS):
Ensure you have the latest macOS and Xcode Command Line Tools installed:
1. Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. Install Python: `brew install python`

## 3. Software Environment Setup

### Prerequisites
Install basic development tools:
```bash
sudo apt install git python3-pip python3-venv build-essential -y
```

### Repository Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/scaler-wizard.git
   cd scaler-wizard
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:

   **For NVIDIA GPU (Linux/Windows):**
   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install tqdm pyyaml gputil psutil jsonschema ipykernel
   ```
   *Note: Adjust the CUDA version (cu118) according to your installed driver.*

   **For Apple Silicon (macOS):**
   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio
   pip install tqdm pyyaml gputil psutil jsonschema ipykernel
   ```

## 4. Understanding TRMC Architecture

The TRMC model (Tiny Recursive MoE Contrastive) is designed for high reasoning capacity with low parameter counts (7M-20M), drawing inspiration from several cutting-edge architectures:

- **Recursive Core (TRM)**: Inspired by Samsung Research, it reuses a single transformer block $N$ times (default 8) per forward pass. This allows the model to "think" deeper without increasing the number of physical parameters.
- **Sparse MoE (Mixture of Experts)**: Uses a gating mechanism to route tokens to specialized experts. This provides the capacity of a much larger model while keeping the computational cost (FLOPs) equivalent to a tiny model.
- **Matryoshka Embeddings**: Inspired by Qwen, these allow for efficient multi-scale representations. The model can be used at different "resolutions" (e.g., 32, 64, or 128 dimensions) depending on compute constraints.
- **Vision Encoder**: A lightweight projection layer inspired by DeepSeek-VL that allows the model to process spatial/visual data alongside text, enabling multi-modal reasoning.
- **Contrastive Learning**: Implements a supervised InfoNCE objective to align latent states of successful reasoning paths, distinguishing them from "negative" or incorrect logic.
- **Adaptive Context**: A multi-tier system (VRAM -> RAM -> Disk) inspired by Ollama and Clawdbot, allowing the model to handle context windows that exceed physical hardware limits.

## 5. Starting and Training the TRMC Model

### Data Curation
The training process begins with high-quality data. The `TRMCDatasetCurator` in `research/dataset_curator.py` is responsible for preparing:
- **Wikipedia**: General knowledge.
- **The Stack**: Multi-language code for logic.
- **Math Datasets**: For structured reasoning.
- **Synthetic Logic Puzzles**: Sequence reversal, Sudoku-style tasks, etc.

To run a curation preview:
```bash
python3 research/dataset_curator.py
```

### Training on Apple Silicon
The TRMC model fully supports Apple Silicon (MPS). The training script `research/train_trmc.py` will automatically detect and use the MPS device if available.

### Training via Script
The primary way to train is using the `train_trmc.py` script. To ensure imports work correctly, run it from the project root:

```bash
# Add the research directory to PYTHONPATH so trmc_model can be imported
export PYTHONPATH=$PYTHONPATH:$(pwd)/research
python3 research/train_trmc.py
```

This script will:
1. Initialize a `TRMCModel` with configured hyperparameters.
2. Generate synthetic logic puzzles for training.
3. Perform a training loop with combined Cross-Entropy and Contrastive loss.
4. Save the final model to `checkpoints/trmc/model_final.pt`.

### Interactive Training via Jupyter
For a more hands-on approach, use the provided notebook:
1. Start the Jupyter server:
   ```bash
   jupyter notebook research/train_trmc.ipynb
   ```
2. Follow the cells to:
   - Understand the `LogicPuzzlesDataset`.
   - Configure the `TRMCModel` architecture.
   - Run the training loop with integrated Cross-Entropy and Contrastive loss.
   - Visualize loss curves using `matplotlib`.
   - Export your model for deployment.

## 6. Deployment with Ollama

Ollama is a powerful tool for running LLMs locally. You can use it to run TRMC models once they are trained.

### 1. Export the Model
After training, use the provided export tool to convert the model to GGUF and generate a Modelfile:
```bash
python tools/export_ollama.py --checkpoint checkpoints/trmc/model_final.pt
```
This script uses the specialized `trmc_converter.py` to preserve recursive MoE metadata and generates a `Modelfile` in `checkpoints/ollama/`.

### 2. Compile the Architectural Converter
The `trmc.cpp` file provides the architectural definition for converting TRMC models into the GGUF format:
```bash
g++ tools/trmc.cpp -o trmc_converter
./trmc_converter input.pt model.gguf
```
This program outlines how the recursive MoE layers and architectural metadata are mapped to the GGUF format for use with Ollama.

### 3. Create the Ollama Model
```bash
ollama create trmc-model -f checkpoints/ollama/Modelfile
```

### 4. Run the Model
```bash
ollama run trmc-model
```

## 7. Advanced Usage

### Monitoring
The `components/telemetry` module logs training metrics. You can view these logs to ensure the model is converging correctly.

### Safety Circuit Breaker
The `components/safety` module monitors VRAM and temperature. If hardware limits are reached, it will automatically pause or halt training to prevent damage or crashes.

---
For more technical details on the TRMC design, see `research/trmc_design.md`.
