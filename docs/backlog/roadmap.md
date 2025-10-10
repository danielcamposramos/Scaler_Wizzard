# Scaler Wizard Backlog

## Immediate (Sprint 0)
- Produce CLI prototype spec for scaling pipeline.
- Define configuration schema (`config/scaler_wizard.yaml`) capturing model, LoRA, context, dataset options.
- Collect partner input on priority base models and datasets.
- Implement safety circuit breaker scaffold and document guardrails.
- Draft telemetry event schema for cockpit dashboard.
- Draft human contract template and embed versioning checks.
- Prototype rollback script linked to checkpoints metadata.

## Near Term (Sprint 1)
- Implement CLI prototype with logging and artifact export.
- Draft React UI wireframes for the TransformerLab tab.
- Document RoPE scaling approach and validation strategy.
- Integrate circuit breaker and telemetry into training loop.
- Wire contract validator and rollback hook into CLI/pipeline start-up.
- Finalise cognitive-load throttle behaviour across voice, visual, telegram.
- Add integration tests covering circuit breaker, schema validation, throttle, and rollback.

## Mid Term (Sprint 2)
- Integrate prototype into TransformerLab backend service layer.
- Build dataset ingestion helpers (local uploads + HF hub).
- Add automated tests for scaling pipeline components.
- Deliver cockpit dashboard MVP with Approve/Abort/Rollback controls.
- Implement contract audit trail and voice alert integration.

## Long Term (Beyond Sprint 2)
- Support alternative scaling methods (prefix tuning, adapter fusion).
- Integrate FlashAttention or equivalent efficient attention kernels.
- Establish community template library for contributed recipes.
- Train telemetry-driven meta-model to recommend scaling recipes.
- Automate rollback recommendation engine using telemetry meta-model.
