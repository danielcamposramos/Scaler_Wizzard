# Scaler Wizard CLI Prototype Spec

## Purpose
Provide a command-line interface that mirrors the planned TransformerLab plugin so partners can exercise the scaling pipeline without UI dependencies.

## Command
```bash
python -m scaler_wizard.cli \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset local://data/instruction.jsonl \
  --target-context 16384 \
  --template balanced_scaling \
  --output-dir runs/tinyllama-balanced
```

## Arguments
- `--base-model` (required): Hugging Face model id or path.
- `--tokenizer`: Optional tokenizer override.
- `--template`: Scaling template id defined in `components/templates/scaling_profiles.yaml`.
- `--lora-rank`: Manual override; otherwise determined via adaptive algorithm.
- `--target-context`: Desired maximum sequence length.
- `--context-schedule`: Comma-separated list for progressive checkpoints.
- `--dataset`: Dataset reference. Supports `hf://`, `local://`, `synthetic://`.
- `--weak-supervision`: Toggle Snorkel-based augmentation.
- `--output-dir`: Where to store adapters, tokenizer, logs, and telemetry.
- `--dry-run`: Validate configuration without executing tasks.

## Workflow
1. **Resolve Configuration**  
   - Load template defaults.  
   - Merge CLI overrides.  
   - Validate against plugin schema.  
   - Load `config/human_contract.yaml` and verify signature/version.

2. **Bootstrap Runtime**  
   - Detect hardware capabilities (GPU count, memory).  
   - Configure Accelerate for distributed runs.  
   - Initialize telemetry stream (local jsonl by default).  
   - Load cockpit JSON schema for real-time validation of outbound events.

3. **Model Preparation**  
   - Load base model/tokenizer with 8-bit or 4-bit quantization if required.  
   - Apply adaptive LoRA rank from `components/scaling_engine/adaptive_scaling.py`.  
   - Register circuit breaker callbacks.

4. **Context Expansion**  
   - Compute progression schedule.  
   - Apply NTK-RoPE scaling or YaRN modules from `research/context_extension_survey.md`.  
   - Optionally run calibration pass with synthetic prompts.

5. **Training Loop**  
   - Use Hugging Face Trainer with callbacks for telemetry, circuit breaker, and evaluation checkpoints.  
   - At each scheduled context step, run probes defined in `components/evaluation/progressive_context.py`.  
   - Route alert messages through `components/ui/cognitive_throttle.py` before engaging voice or visual channels.  
   - Use `ui/services/voice_pipeline.js` (via frontend bridge) to narrate canonical sentence when voice modality is granted.

6. **Knowledge Preservation**  
   - Distill from teacher if configured via `components/distillation/knowledge_transfer.py`.  
   - Log metrics to `benchmarks/results`.

7. **Export & Reporting**  
   - Save adapters, tokenizer, and metadata JSON.  
   - Emit cockpit telegram summarizing status for the dashboard.  
   - Attach signed human contract to run artifacts.  
   - Record rollback checkpoint pointer for `tools/rollback.py`.

## Telemetry Artifacts
- `metrics.jsonl`: Stream of metric events.  
- `telegram.json`: Latest cockpit status with action (`continue|warn|stop`).  
- `config.resolved.yaml`: Final configuration snapshot.  
- `hardware.json`: Hardware inventory used for adaptive decisions.
- `run_metadata.json`: Phase tracker and checkpoint map consumed by `tools/rollback.py`.
- `cockpit_payload.json`: Last validated telemetry snapshot matching `specs/cockpit_schema.json`.
