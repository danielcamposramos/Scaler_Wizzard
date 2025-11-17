# CLAUDE.md - AI Assistant Guide for Scaler Wizard

## Project Overview

Scaler Wizard is the implementation hub for Daniel Ramos' Multi-Vibe Code In Chain initiative, designed to extend TransformerLab with accessible model scaling workflows. The project provides tools and components to automate and manage machine learning model scaling through LoRA-based parameter expansion, context extension, and dataset augmentation.

**Core Philosophy**: Multi-Vibe Code In Chain treats every collaborator—AI or human—as a peer contributing original insight. This repository captures and executes ideas from the "swarm" (collaborative AI partners).

**Project Goals**:
- Empower creators to "grow" smaller base models by scaling effective parameters and context windows
- Provide safety-first automation with circuit breakers and human oversight
- Enable accessible model scaling without heavy infrastructure requirements
- Integrate with TransformerLab as a plugin while maintaining standalone CLI capability

## Codebase Structure

```
Scaler_Wizzard/
├── components/              # Core Python modules for scaling workflow
│   ├── cache/              # Checkpoint caching (checkpoint_cache.py)
│   ├── distillation/       # Knowledge transfer scaffolding (knowledge_transfer.py)
│   ├── evaluation/         # Progressive context evaluation (progressive_context.py)
│   ├── safety/             # Circuit breaker and human contract enforcement
│   │   ├── circuit_breaker.py    # Training safety monitoring
│   │   └── human_contract.md     # Contract specification
│   ├── scaling/            # Hardware-aware profile recommender (profile_recommender.py)
│   ├── scaling_engine/     # Adaptive LoRA scaling logic (adaptive_scaling.py)
│   ├── telemetry/          # Lightweight metrics logging (metrics_logger.py)
│   ├── ui/                 # UI utilities (cognitive_throttle.py)
│   ├── validation/         # JSON schema validation (schema_validator.py)
│   └── templates/          # Scaling profile templates (scaling_profiles.yaml)
├── docs/                   # Project documentation
│   ├── architecture/       # Architecture specs and overviews
│   ├── process/            # Process documentation (Multi-Vibe chain)
│   └── backlog/            # Roadmap and sprint planning
├── specs/                  # Formal specifications and schemas
│   ├── cockpit_schema.json       # Telemetry payload schema
│   ├── plugin_interface.yaml     # TransformerLab plugin spec
│   ├── scaling_engine.md         # Scaling engine architecture spec
│   └── cli_prototype.md          # CLI specification
├── tools/                  # Standalone utility scripts
│   └── rollback.py         # Checkpoint rollback utility
├── ui/                     # Frontend components
│   ├── components/         # React components for cockpit dashboard
│   ├── services/           # Client services (voice_pipeline.js)
│   └── mockups/            # UI design mockups
├── tests/                  # Test suite
│   └── integration/        # Integration tests (test_pipeline_safety.py)
├── research/               # Research notes and comparisons
├── benchmarks/             # Performance evaluation datasets and results
├── community/              # Community contributions and patterns
├── checkpoints/            # Training checkpoint storage
└── Multi-Vibe_Coding_Chains/  # Collaboration workflow documentation
```

## Key Components and Their Roles

### 1. Scaling Engine (`components/scaling_engine/adaptive_scaling.py`)

**Purpose**: Determines optimal LoRA adapter parameters based on hardware, base model, and scaling targets.

**Key Classes**:
- `HardwareProfile`: Describes runtime environment (VRAM, GPU count, CPU memory)
- `ScalingTargets`: Specifies capacity and context goals
- `LoRAConfigSuggestion`: Recommended LoRA configuration (rank, alpha, dropout, use_qlora)

**Key Functions**:
- `suggest_lora_config()`: Main heuristic that generates LoRA configs
- `calculate_optimal_lora_rank()`: Backward-compatible rank calculation

**Design Principles**:
- Model-agnostic: uses billions-of-parameters metric
- Hardware-aware: adapts to VRAM and GPU availability
- Context-pressure aware: reduces rank when context grows aggressively
- Automatic QLoRA: switches on for low-memory setups (<20GB VRAM)

### 2. Safety Circuit Breaker (`components/safety/circuit_breaker.py`)

**Purpose**: Monitors training metrics and automatically halts or warns when performance degrades.

**Key Classes**:
- `CBConfig`: Configuration for the circuit breaker (window size, thresholds, stop behavior)
- `MetricSnapshot`: Training step metrics (perplexity, accuracy)
- `CircuitBreaker`: Main monitoring class

**Safety Conditions**:
- Perplexity slope: checks for rapid increases over a sliding window
- Accuracy floor: ensures minimum acceptable accuracy threshold

**Integration**: Emits "telegrams" (JSON status messages) to stdout and optional file paths for cockpit consumption.

### 3. Human Contract System (`components/safety/human_contract.md`)

**Purpose**: Ensures all automated actions have explicit human authorization from Daniel Ramos (visionary architect).

**Contract Fields**:
- `contract_version`: Semantic version of the contract
- `authorized_by`: Human approver name
- `approved_at`: ISO datetime stamp
- `session_id`: Ties consent to specific job
- `scope`: Summary of intended action
- `rollback_plan`: Checkpoint location for recovery
- `signoff`: Explicit acceptance string

**Enforcement**: Pipeline constructor validates contract before execution; missing or mismatched versions raise `ContractNotAcceptedError`.

### 4. Telemetry and Cockpit (`specs/cockpit_schema.json`)

**Purpose**: Canonical payload schema for backend-to-dashboard communication.

**Required Fields**:
- `schemaVersion`, `runId`, `timestamp`
- `phase`: Training phase (P0-P4)
- `status`: Current state (initializing, running, warning, stopped, etc.)
- `qualityMetrics`: Perplexity, accuracy, trend
- `resourceUsage`: GPU/CPU memory, throughput, temperature
- `humanContract`: Contract validation status
- `actions`: Approve/Abort/Rollback button states
- `alerts`: Array of notifications (voice, visual, telegram modalities)

**Alert Modalities**:
- `voice`: Spoken notifications with language and persona
- `visual`: Dashboard indicators
- `telegram`: File-based status messages

### 5. Rollback Tool (`tools/rollback.py`)

**Purpose**: Command-line script to revert training to a previous phase.

**Usage**:
```bash
python tools/rollback.py --run-dir /path/to/run --target-phase 1
```

**Operations**:
1. Loads run metadata from `run_metadata.json`
2. Unloads current adapters
3. Restores checkpoint files from target phase
4. Updates metadata to reflect rollback

### 6. Scaling Profiles (`components/templates/scaling_profiles.yaml`)

**Available Templates**:
- `balanced_scaling`: General-purpose instruction tuning (16GB VRAM, 4x context)
- `max_context`: Long-form and RAG workloads (20GB VRAM, 16x context)
- `max_parameters`: Complex reasoning tasks (18GB VRAM, high LoRA rank)
- `budget_context`: Limited VRAM setups (8GB VRAM)
- `reasoning_boost`: Analytical tasks (16GB VRAM)
- `creative_fusion`: Stylistic flexibility (24GB VRAM, 6x context)

**Template Fields**: LoRA rank/alpha, context multiplier, distillation flag, telemetry profile, minimum VRAM.

## Development Workflows

### Multi-Vibe Code In Chain Process

This is the collaborative workflow that governs the project:

**Roles**:
- **Visionary Architect (Daniel Ramos)**: Curates partner prompts, stitches contributions, ensures coherence
- **AI Partners**: Expand on each other's thoughts, propose improvements, surface risks
- **Implementation Partner (Codex/Claude)**: Only agent with repository access; translates deliberation into code

**Collaboration Loop**:
1. Daniel gathers inputs from AI partners (ideas, code, sketches)
2. Implementation partner materializes agreed work in the repo with rationale and file references
3. Daniel signs the human contract in the cockpit before execution
4. Daniel validates with the swarm and returns new directives
5. Iterate until feature set and documentation meet collective vision

**Operating Principles**:
- Prefer transparent, modular files over monolithic specs
- Note open decisions explicitly for parallel swarm processing
- Treat experiments as first-class citizens; capture hypotheses, not only conclusions
- Every update cites files touched and highlights pending questions

### Git Workflow

**Branch Strategy**:
- Main development occurs on feature branches prefixed with `claude/`
- Branch names include session identifiers for tracking
- Example: `claude/add-claude-documentation-01QuLiS33m3wNyQTsWbDp8bP`

**Commit Practices**:
- Clear, descriptive commit messages
- Focus on "why" rather than "what"
- Reference relevant files and components
- Follow existing commit message style in repository

**Push Requirements**:
- Always use `git push -u origin <branch-name>`
- Branch must start with `claude/` and end with matching session ID
- Retry on network errors with exponential backoff (2s, 4s, 8s, 16s)

### Documentation Standards

**Location Strategy**:
- Roadmaps and planning: `docs/backlog/`
- Architecture notes: `docs/architecture/`
- Process documentation: `docs/process/`
- Technical specs: `specs/`

**Signals**:
- Every code update must cite files touched
- Highlight pending questions explicitly
- Document open decisions for swarm discussion
- Include usage examples in component documentation

## Coding Conventions

### Python Style

**Module Structure**:
```python
"""Module docstring explaining purpose and key insights.

References to partner contributions (e.g., "Implements Kimi's eagle-vision insight")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
```

**Docstring Format** (Google Style):
```python
def function_name(param: Type) -> ReturnType:
    """Short description of what the function does.

    Longer explanation if needed, including context about design decisions
    or partner insights that influenced the implementation.

    Args:
        param (Type): Description of parameter.

    Returns:
        ReturnType: Description of return value.

    Raises:
        ErrorType: When this error occurs.
    """
```

**Class Documentation**:
- Use `@dataclass` for data containers with clear attribute documentation
- Document validation rules in class docstrings
- Include invariants and assumptions

**Error Handling**:
- Raise descriptive errors with context
- Use specific exception types
- Validate inputs at boundaries
- Document error conditions in docstrings

### Configuration Files

**YAML Format**:
- Use consistent indentation (2 spaces)
- Include description fields for templates and options
- Group related settings logically
- Document units (GB, tokens/sec, etc.)

**JSON Schema**:
- Follow JSON Schema Draft 07 specification
- Mark required fields explicitly
- Use enums for constrained values
- Include descriptions for all properties
- Specify validation constraints (minimum, maximum, pattern)

### File Organization

**Naming Conventions**:
- Python modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Configuration: `descriptive_name.yaml` or `.json`

**Import Organization**:
1. Future imports (`from __future__ import annotations`)
2. Standard library imports
3. Third-party imports (alphabetically)
4. Local imports (alphabetically)

**Module Exports**:
- Use `__all__` to explicitly declare public API
- List exports at module bottom

## Working with the Codebase

### Prerequisites

**Python Environment**:
- Python 3.8+ required
- Virtual environment recommended
- Key dependencies: pyyaml, gputil, psutil, jsonschema

**Setup**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install pyyaml gputil psutil jsonschema
```

**UI Environment (Optional)**:
```bash
cd ui
npm install
```

### Common Development Tasks

#### 1. Adding a New Scaling Profile

**File**: `components/templates/scaling_profiles.yaml`

```yaml
new_profile_name:
  description: Clear description of use case
  lora_rank: 16
  lora_alpha: 48
  context_multiplier: 4
  use_distillation: true
  telemetry_profile: standard
  min_vram_gb: 12
```

**Validation**: Ensure profile aligns with hardware constraints in `adaptive_scaling.py`.

#### 2. Modifying Circuit Breaker Thresholds

**File**: `components/safety/circuit_breaker.py`

Key parameters in `CBConfig`:
- `window_steps`: Sliding window size (default: 10)
- `max_perplexity_delta`: Maximum perplexity increase threshold (default: 0.15)
- `min_accuracy`: Minimum acceptable accuracy (default: 0.5)
- `hard_stop`: Whether to stop (True) or warn (False)

**Example**:
```python
config = CBConfig(
    window_steps=15,
    max_perplexity_delta=0.10,
    min_accuracy=0.60,
    hard_stop=True,
    telegram_path=Path("./telemetry/status.json")
)
breaker = CircuitBreaker(config)
```

#### 3. Extending Telemetry Schema

**File**: `specs/cockpit_schema.json`

When adding new fields:
1. Update schema with proper type definitions
2. Mark as required if mandatory
3. Update validation in `components/validation/schema_validator.py`
4. Document new fields in schema description
5. Update UI components to consume new data

#### 4. Creating Integration Tests

**File**: `tests/integration/test_*.py`

Follow existing patterns:
- Test circuit breaker integration
- Validate schema compliance
- Test rollback functionality
- Verify human contract enforcement

### Understanding the Training Pipeline

**Job Orchestration Flow**:
1. Model load & tokenizer preparation
2. LoRA adapter injection (via PEFT library)
3. Optional vocabulary resize
4. Positional embedding scaling (RoPE/NTK strategy)
5. Dataset loading (local or Hugging Face hub)
6. Fine-tuning with adaptive LoRA rank resolution
7. Knowledge distillation (optional)
8. Progressive context checkpoints
9. Contract validation and fail-fast enforcement
10. Telemetry emission and circuit breaker evaluation
11. Artifact export (model, adapters, tokenizer, logs)

**Safety Integration Points**:
- Pre-flight: Human contract validation
- Training loop: Circuit breaker metric registration
- Checkpoint: Metadata persistence for rollback
- Post-training: Artifact audit trail

## Safety and Security

### Circuit Breaker Integration

**When to Use**:
- All training runs (mandatory)
- Custom fine-tuning workflows
- Experimental scaling strategies

**Best Practices**:
```python
# Initialize with appropriate thresholds
breaker = CircuitBreaker(CBConfig(
    window_steps=10,
    max_perplexity_delta=0.15,
    min_accuracy=0.5,
    hard_stop=True
))

# Register metrics at each training step
telegram = breaker.register(
    step=current_step,
    perplexity=current_perplexity,
    accuracy=current_accuracy
)

# Check action and respond
if telegram["action"] == "stop":
    # Halt training, trigger rollback
    logger.error(f"Circuit breaker triggered: {telegram['reason']}")
    raise TrainingHaltedException(telegram["reason"])
elif telegram["action"] == "warn":
    # Log warning, continue with caution
    logger.warning(f"Circuit breaker warning: {telegram['reason']}")
```

### Human Contract Enforcement

**Required Before**:
- Starting any training run
- Deploying new scaling configurations
- Modifying production models

**Contract Validation**:
```python
def validate_contract(contract_data: dict) -> None:
    """Validates human contract before execution.

    Raises:
        ContractNotAcceptedError: If contract is invalid or not signed.
    """
    if contract_data.get("version") != EXPECTED_CONTRACT_VERSION:
        raise ContractNotAcceptedError("Contract version mismatch")

    if contract_data.get("signoff") != "I accept responsibility for this run":
        raise ContractNotAcceptedError("Contract not properly signed")

    if not contract_data.get("approved_at"):
        raise ContractNotAcceptedError("Contract not timestamped")
```

### Rollback Procedures

**When to Rollback**:
- Circuit breaker triggers hard stop
- Unexpected performance degradation
- User-initiated abort from cockpit
- Contract violation or safety concern

**Rollback Execution**:
1. Identify target phase from run metadata
2. Unload current adapters
3. Restore checkpoint from target phase
4. Update run metadata to reflect state
5. Log rollback event in telemetry

**Manual Rollback**:
```bash
python tools/rollback.py \
  --run-dir /path/to/training/run \
  --target-phase 1
```

## TransformerLab Plugin Integration

### Plugin Interface

**Configuration**: `specs/plugin_interface.yaml`

**Required Packages**:
- transformers >= 4.40.0
- peft >= 0.11.0
- datasets >= 2.18.0
- accelerate >= 0.30.0
- bitsandbytes >= 0.43.0
- flash-attn >= 2.5.6

**Frontend Routes**:
- Path: `/scaler-wizard`
- Permissions: `models:write`, `datasets:read`
- Entry component: `ScalerWizardTab`

**Backend Integration**:
- Task runner: Python
- Entrypoint: `scaler_wizard.runner:run_job`
- Job schema validation required before execution

### Job Submission Schema

**Required Sections**:
1. **model**: Base model, tokenizer, runtime configuration
2. **scaling**: Strategy, adaptive rank, target context, template selection
3. **contract**: Human authorization and rollback plan
4. **dataset**: Source, identifier, split configuration
5. **safety**: Circuit breaker and telemetry settings

**Example Job**:
```yaml
model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  tokenizer: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  runtime: single_gpu

scaling:
  strategy: lora
  adaptive_rank: true
  target_context: 8192
  template: balanced_scaling

contract:
  version: "1.0.0"
  authorized_by: "Daniel Ramos"
  signoff: "I accept responsibility for this run"
  approved_at: "2025-01-15T10:30:00Z"
  rollback_plan: "./checkpoints/phase_0"
  session_id: "abc-123-def-456"

dataset:
  source: huggingface_hub
  identifier: "HuggingFaceH4/ultrachat_200k"
  split: train

safety:
  circuit_breaker:
    window_steps: 10
    max_perplexity_delta: 0.15
    min_accuracy: 0.5
    hard_stop: true
  telemetry:
    emit_metrics: true
    destination: local
```

## Testing Strategy

### Integration Tests

**Location**: `tests/integration/`

**Coverage Areas**:
- Pipeline safety (circuit breaker, contract validation)
- Schema validation (cockpit payload, job submission)
- Rollback functionality
- Telemetry emission
- Cognitive load throttle behavior

**Running Tests**:
```bash
pytest tests/integration/ -v
```

### Manual Testing Checklist

Before committing changes:
- [ ] Circuit breaker responds to metric degradation
- [ ] Contract validation rejects invalid contracts
- [ ] Rollback restores correct checkpoint
- [ ] Telemetry schema validates successfully
- [ ] Scaling profiles produce valid LoRA configs
- [ ] Voice/visual/telegram alerts fire correctly
- [ ] Cognitive throttle prioritizes high-severity alerts

## Extensibility Points

### 1. Scaling Strategies

**Current**: LoRA, QLoRA

**Extension Interface**: Add new parameter-growth techniques (prefix-tuning, adapter fusion)

**Location**: `components/scaling/`

### 2. Context Engines

**Current**: RoPE scaling

**Extension Interface**: Abstraction for longer-context patches (RoPE v1/v2, ALiBi, etc.)

**Location**: `components/scaling_engine/`

### 3. Dataset Recipes

**Extension Interface**: Compose dataset transformations (filtering, synthetic generation, weak supervision)

**Location**: `components/distillation/` or new `components/datasets/`

### 4. Telemetry Destinations

**Current**: Local file, stdout

**Extension Interface**: Plug in additional sinks (websocket, remote collector, HTTP endpoints)

**Location**: `components/telemetry/metrics_logger.py`

### 5. Alert Modalities

**Current**: Voice, visual, telegram

**Extension Interface**: Add new notification channels while respecting cognitive load throttle

**Location**: `ui/services/voice_pipeline.js`, `components/ui/cognitive_throttle.py`

## Milestone Roadmap

### Sprint 0 (Prototype)
- ✅ CLI version of growth pipeline
- ✅ Configuration schema and metrics capture
- ✅ Circuit breaker integration
- ✅ Contract validation and rollback scaffolding
- ✅ Cockpit JSON schema and validator

### Sprint 1 (TransformerLab Integration)
- React tab with backend endpoint wiring
- Real-time status updates and error surfacing
- Cockpit telemetry strip rendering
- Cognitive load throttle with voice narration
- Integration tests

### Sprint 2 (Community Toolkit)
- Dataset creation templates
- Tutorial notebooks for scaling recipes
- Community pattern library

### Sprint 3+ (Advanced Scaling)
- FlashAttention integration
- Multi-turn dataset ingestion
- Evaluation harness
- Alternative scaling methods (prefix tuning, adapter fusion)
- Telemetry-driven meta-model for recipe recommendations

## Common Questions and Answers

### Q: How do I determine the right scaling profile for my use case?

**A**: Use the profile recommender in `components/scaling/profile_recommender.py`, which detects hardware and suggests templates. Alternatively, reference `components/templates/scaling_profiles.yaml` descriptions:
- General purpose: `balanced_scaling`
- Long context: `max_context`
- Complex reasoning: `max_parameters` or `reasoning_boost`
- Limited VRAM: `budget_context`

### Q: What happens when the circuit breaker triggers?

**A**: Depends on the `hard_stop` configuration:
- `hard_stop=True`: Training halts immediately, rollback can be initiated
- `hard_stop=False`: Warning logged, training continues with monitoring

The circuit breaker emits a "telegram" (JSON status message) that the cockpit dashboard consumes to enable user actions.

### Q: How do I add a new telemetry field?

**A**:
1. Update `specs/cockpit_schema.json` with the new field definition
2. Modify `components/telemetry/metrics_logger.py` to emit the field
3. Update `components/validation/schema_validator.py` if validation logic changes
4. Update UI components in `ui/components/` to display the new data
5. Document the field in schema description

### Q: What's the difference between `rank` and `alpha` in LoRA configs?

**A**:
- `rank`: Dimensionality of low-rank matrices; higher rank = more capacity, more memory
- `alpha`: Scaling factor applied to LoRA updates; typically 2-4x the rank
- The ratio `alpha/rank` controls the learning rate for adapter parameters

### Q: When should I use QLoRA vs regular LoRA?

**A**: The scaling engine automatically suggests QLoRA when:
- `prefer_low_memory=True` in `suggest_lora_config()`
- Available VRAM < 20GB
- Use QLoRA for 4-bit quantization with minimal quality loss on memory-constrained hardware

### Q: How do phases (P0-P4) work in the training pipeline?

**A**: Phases represent progressive training stages:
- **P0**: Initialization and model loading
- **P1**: Initial fine-tuning with base context
- **P2**: First context expansion
- **P3**: Further context scaling
- **P4**: Final training and export

Each phase creates a checkpoint for rollback capability.

## Best Practices for AI Assistants

### 1. Always Validate Safety

Before suggesting or implementing changes:
- Check circuit breaker integration
- Ensure human contract validation
- Verify rollback checkpoints are created
- Confirm telemetry emission

### 2. Document Open Questions

When encountering ambiguity or multiple valid approaches:
- Note the question explicitly in code comments or docs
- Reference relevant sections of architecture documents
- Suggest gathering swarm input for significant decisions
- Document assumptions made

### 3. Cite File References

When discussing or modifying code:
- Use format `file_path:line_number` for specific references
- Example: "The circuit breaker is initialized in components/safety/circuit_breaker.py:58"
- List all files touched in commit messages

### 4. Respect the Multi-Vibe Process

Remember:
- Daniel Ramos is the visionary architect with final authority
- AI partners contribute ideas; implementation partner executes
- Human contract must be signed before execution
- Experiments are first-class citizens

### 5. Maintain Modularity

When adding features:
- Keep files focused and transparent
- Avoid monolithic implementations
- Use existing abstractions and interfaces
- Document extensibility points

### 6. Prioritize Backward Compatibility

When modifying existing components:
- Maintain backward compatibility when possible
- Document breaking changes clearly
- Provide migration paths for existing configs
- Version schemas appropriately

## Additional Resources

### Internal Documentation
- Architecture overview: `docs/architecture/scaler_wizard_overview.md`
- Scaling engine specification: `specs/scaling_engine.md`
- Multi-Vibe process: `docs/process/multi_vibe_chain.md`
- Roadmap: `docs/backlog/roadmap.md`
- Safety specification: `docs/architecture/safety.md`

### External Dependencies
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- PEFT (Parameter-Efficient Fine-Tuning): https://huggingface.co/docs/peft
- Datasets library: https://huggingface.co/docs/datasets

### Community
- Contribution guidelines: `community/contributions/README.md`
- Pattern library: `community/patterns/README.md`

---

**Document Version**: 1.0.1
**Last Updated**: 2025-11-17
**Maintained By**: Multi-Vibe Code In Chain Swarm
**Questions**: Document open questions in code comments and surface to Daniel Ramos for swarm discussion
