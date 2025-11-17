# Scaler Wizard

Scaler Wizard is the implementation hub for Daniel Ramos' Multi-Vibe Code In Chain initiative to extend TransformerLab with accessible model scaling workflows. It provides a suite of tools and components to enhance machine learning models through **parameter-efficient fine-tuning (LoRA)**, **context window extension**, and **knowledge distillation**, with a focus on safety, efficiency, and developer experience.

## What Scaler Wizard Does

Scaler Wizard **increases model behavioral capacity** through:
- **LoRA Adaptation**: Adds trainable adapter parameters (<1% of base model size) to specialize models for specific tasks
- **Context Extension**: Expands context windows (2K → 32K+ tokens) using RoPE/NTK scaling
- **Knowledge Distillation**: Transfers capabilities from larger teacher models

**Important**: Scaler Wizard uses LoRA (Low-Rank Adaptation), which does **not** increase the base model's parameter count (e.g., it doesn't expand a 3B model to 7B parameters). Instead, it adds small, efficient adapter layers that increase the model's effective capacity for specific tasks while maintaining the original architecture.

For users seeking larger base models, we recommend using pretrained 7B/13B models (Llama 2, Mistral, Phi-3) with LoRA adaptation rather than attempting to expand smaller models.

See `research/parameter_expansion_analysis.md` for detailed technical analysis of LoRA vs architectural expansion.

## Hardware Requirements

Scaler Wizard is designed to run on **consumer hardware** using efficient techniques.

### Recommended Hardware by Use Case

#### LoRA Fine-tuning 3B Models
- **GPU**: 8-12 GB VRAM (e.g., RTX 3060 12GB, RTX 4060 Ti 16GB)
- **RAM**: 16 GB system memory
- **Storage**: 50 GB free space
- **Training Time**: Hours to days (depending on dataset size)

#### LoRA Fine-tuning 7B Models
- **GPU**: 16-24 GB VRAM (e.g., RTX 4090 24GB, RTX 4060 Ti 16GB with QLoRA)
- **RAM**: 32 GB system memory
- **Storage**: 100 GB free space
- **Training Time**: Hours to days

#### Context Extension (2K → 8K)
- **GPU**: 12-16 GB VRAM
- **RAM**: 24 GB system memory
- **Additional**: ~1.5x VRAM overhead during training

#### Context Extension (2K → 32K)
- **GPU**: 24-48 GB VRAM (gradient checkpointing required)
- **RAM**: 32-64 GB system memory
- **Additional**: Batch size reduction needed

### Consumer GPU Compatibility

✅ **Fully Supported**:
- NVIDIA RTX 4090 (24GB) - Best for 7B models
- NVIDIA RTX 4060 Ti (16GB) - Good for 3B models, 7B with QLoRA
- NVIDIA RTX 3090 (24GB) - Excellent all-around
- NVIDIA RTX 3060 (12GB) - Entry level for 3B models

⚠️ **Partially Supported** (requires QLoRA/optimizations):
- NVIDIA RTX 4070 Ti (12GB)
- NVIDIA RTX 3070 (8GB) - 3B models only, limited context

❌ **Not Recommended**:
- GPUs with <8GB VRAM
- AMD GPUs (ROCm support not yet tested)
- Apple Silicon (MPS backend not yet optimized)

**Note**: True parameter expansion (architectural growth like 3B → 7B) is **not currently supported** as it requires multi-GPU clusters (60+ GB VRAM) and extensive pretraining. See `docs/architecture/expansion_gap_analysis.md` for details on what would be required.

## Getting Started

Follow these instructions to get a local copy of the project up and running for development and testing purposes.

### Prerequisites

This project uses Python and Node.js. You will need to have them installed on your system.

*   Python (3.8+ recommended)
*   Node.js and npm (for UI components)
*   CUDA 11.8+ (for GPU acceleration)
*   cuDNN 8.0+ (recommended)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/scaler-wizard.git
    cd scaler-wizard
    ```

2.  **Set up the Python environment:**
    It is recommended to use a virtual environment.
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
    Install the required Python packages. While a `requirements.txt` is not yet present, the following packages are used by various components:
    ```sh
    pip install pyyaml gputil psutil jsonschema
    ```

3.  **Set up the UI environment (optional):**
    If you are working on the UI components, navigate to the `ui` directory and install the dependencies.
    ```sh
    cd ui
    npm install
    cd ..
    ```

## Usage

Scaler Wizard is a collection of components designed to be integrated into a larger model scaling workflow. Here are some of the key parts and how to use them:

*   **Scaling Engine (`components/scaling_engine`):** Provides adaptive heuristics to suggest LoRA configurations based on your hardware and scaling targets.
*   **Safety Circuit Breaker (`components/safety`):** Monitors training metrics and can automatically halt or warn if performance degrades, preventing wasted resources.
*   **Profile Recommender (`components/scaling`):** Detects your hardware and recommends a suitable scaling profile from a set of templates.
*   **Rollback Tool (`tools/rollback.py`):** A command-line script to revert a training run to a previous, known-good phase.
    ```sh
    python tools/rollback.py --run-dir /path/to/your/run --target-phase 1
    ```

## Project Structure

The repository is organized into the following main directories:

-   `components/`: Core Python modules for different aspects of the scaling workflow.
    -   `cache/`: Caching for training checkpoints.
    -   `distillation/`: Scaffolding for knowledge distillation.
    -   `evaluation/`: Utilities for progressive context evaluation.
    -   `safety/`: The circuit breaker and other safety components.
    -   `scaling/`: The hardware-aware profile recommender.
    -   `scaling_engine/`: The core adaptive scaling logic.
    -   `telemetry/`: A lightweight metrics logger.
    -   `validation/`: JSON schema validation helpers.
-   `docs/`: Project documentation, including architecture, process, and backlog.
-   `specs/`: Formal specifications for APIs, data schemas (`cockpit_schema.json`), and interfaces.
-   `tools/`: Standalone scripts for managing scaling runs, like `rollback.py`.
-   `ui/`: Frontend components for the "cockpit" dashboard.
    -   `services/`: Client-side services like the `voice_pipeline.js` for audio alerts.
-   `research/`: Research notes and comparisons of different techniques.
-   `benchmarks/`: Datasets and results for evaluating scaling performance.
-   `community/`: A place for shared patterns and community contributions.

## Frequently Asked Questions

### Q: Can Scaler Wizard expand a 3B model to 7B parameters?

**A**: No, not currently. Scaler Wizard uses **LoRA (Low-Rank Adaptation)**, which adds small adapter parameters to enhance model behavior without changing the base architecture. True parameter expansion (growing from 3B to 7B) would require:
- Multi-GPU cluster (60+ GB VRAM total)
- Extensive continuation pretraining (100B+ tokens)
- Significant implementation of expansion operators
- Weeks of training time

For users who need a 7B model, we **strongly recommend** downloading a pretrained 7B model (Llama 2, Mistral, Phi-3) and fine-tuning it with LoRA.

See `research/parameter_expansion_analysis.md` for comprehensive technical analysis.

### Q: What does LoRA actually do?

**A**: LoRA adds small trainable matrices to each layer of a frozen base model. For a 7B model, LoRA adapters are typically <50MB (vs 23GB for the base model). This provides:
- Task-specific specialization
- Multi-task capability via adapter swapping
- Efficient fine-tuning on consumer GPUs
- Composability (merge multiple adapters)

### Q: How much does LoRA improve model performance?

**A**: LoRA can increase effective capacity by 5-30% for specific tasks, matching or exceeding full fine-tuning performance while using <1% of the parameters. However, it doesn't fundamentally change what the base model can learn - it specializes existing capabilities.

### Q: Can I extend context from 2K to 128K tokens?

**A**: Theoretically yes, but practical limits exist:
- 2K → 8K: ✅ Well-tested, works on consumer hardware
- 8K → 32K: ✅ Possible with gradient checkpointing
- 32K → 64K: ⚠️ Requires high VRAM (48GB+), quality may degrade
- 64K → 128K+: ⚠️ Research territory, significant quality challenges

Context extension beyond 32K often requires specialized techniques (LongRoPE, Position Interpolation variants) and extensive validation.

### Q: What's the best starting model for Scaler Wizard?

**A**: Recommendations by use case:
- **General**: Llama 2 7B, Mistral 7B (excellent base quality)
- **Code**: CodeLlama 7B, Phi-3-mini-4K (code-optimized)
- **Budget hardware**: TinyLlama 1.1B, Phi-2 (2.7B) - run on 8GB GPUs
- **Reasoning**: Llama 2 13B (requires 24GB+ VRAM with QLoRA)

Choose models with RoPE positional encoding for best context extension support.

### Q: Is true parameter expansion planned for the future?

**A**: It's under research consideration. Implementation would require:
- 6-12 month development effort
- Access to multi-GPU clusters
- Large-scale pretraining datasets
- Research-level expertise

We're monitoring recent research (LEMON, LESA, progressive growth) and will reassess as methods mature. For now, focus remains on accessible LoRA-based workflows.

See `docs/architecture/expansion_gap_analysis.md` for the full roadmap and decision points.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

For questions about Multi-Vibe Code In Chain collaboration workflow, see `docs/process/multi_vibe_chain.md`.

## License

This project is licensed under the terms of the LICENSE file.

## Additional Resources

- **Technical Documentation**: See `CLAUDE.md` for comprehensive AI assistant guide
- **Architecture**: `docs/architecture/scaler_wizard_overview.md`
- **Research**: `research/` directory for PEFT comparisons and technical analyses
- **Specifications**: `specs/` directory for formal schemas and interfaces