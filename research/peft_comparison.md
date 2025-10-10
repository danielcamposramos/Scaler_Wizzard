# Parameter-Efficient Fine-Tuning Comparison

| Technique       | Pros | Cons | Recommended Use |
|-----------------|------|------|-----------------|
| **LoRA**        | Mature ecosystem, good balance between compute and quality | Adapter size grows with rank | Default strategy for Scaler Wizard |
| **QLoRA**       | Memory efficient via 4-bit base model | More sensitive to hyperparameters | Use on consumer GPUs with <24 GB memory |
| **IA³**         | Minimal additional parameters | Limited open-source tooling | Consider for inference-time adaptation |
| **Prefix Tuning** | Simple to integrate, controllable behavior | Less expressive for large rewrites | Prompt personalization and style transfer |
| **Adapter Fusion** | Combine multiple skills | Requires multiple source adapters | Future roadmap for recipe sharing |

## Selection Guidelines
1. Prefer **LoRA** or **QLoRA** for balanced workloads.  
2. Enable **Prefix Tuning** templates for rapid experimentation.  
3. Log adapter metadata in telemetry to build pattern library.  
4. Evaluate **Adapter Fusion** once community contributions grow.
