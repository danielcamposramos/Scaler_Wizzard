# Scaling Engine Specification

## Overview
The scaling engine orchestrates parameter growth, context extension, and quality preservation for Scaler Wizard. It is composed of modular services that can be invoked independently or as part of the full pipeline.

## Modules
1. **Adaptive Rank Resolver**  
   - Input: base model metadata, target capacity multiplier, hardware profile.  
   - Output: LoRA configuration (rank, alpha, dropout).  
   - Implementation: `components/scaling_engine/adaptive_scaling.py`.

2. **Context Scheduler**  
   - Input: target context, base context, optional schedule.  
   - Output: progressive steps with evaluation checkpoints.  
   - Implementation: `components/evaluation/progressive_context.py`.

3. **Knowledge Preservation**  
   - Input: teacher model reference, student model, dataset.  
   - Output: distillation configs and callbacks.  
   - Implementation: `components/distillation/knowledge_transfer.py`.

4. **Safety Envelope**  
   - Input: rolling metrics (perplexity, accuracy, throughput).  
   - Output: cockpit telegram with `continue|warn|stop`.  
   - Implementation: `components/safety/circuit_breaker.py`.

5. **Human Contract Gatekeeper**  
   - Input: signed contract document.  
   - Output: approval token enabling pipeline start.  
   - Implementation: `components/safety/human_contract.md` (runtime checker TBD).

6. **Telemetry Sink**  
   - Input: metric events, config snapshots, hardware inventory.  
   - Output: JSON Lines stream for dashboards and meta-learning.  
   - Implementation: `components/telemetry/metrics_logger.py`.

7. **Cognitive Load Throttle**  
   - Input: alert requests from safety, telemetry, or UI modules.  
   - Output: permission to engage voice/visual/telegram channels.  
   - Implementation: `components/ui/cognitive_throttle.py`.

8. **Profile Recommender**  
   - Input: target use-case, detected hardware profile.  
   - Output: Scaling template with adjusted LoRA/context settings.  
   - Implementation: `components/scaling/profile_recommender.py`.

9. **Checkpoint Cache**  
   - Input: checkpoint directories and phase metadata.  
   - Output: Locally cached snapshots for fast rollback operations.  
   - Implementation: `components/cache/checkpoint_cache.py`.

## Data Contracts
- **MetricEvent**  
  ```json
  {
    "step": 123,
    "metric": "eval/perplexity",
    "value": 5.42,
    "timestamp": "2024-05-26T18:42:00Z"
  }
  ```
- **TelemetryEnvelope**  
  ```json
  {
    "config": {...},
    "hardware": {...},
    "metrics_path": "runs/2024-05-26/metrics.jsonl",
    "telegram_path": "runs/2024-05-26/telegram.json"
  }
  ```

## Extensibility
- Register new scaling strategies by implementing `ScalerStrategy` protocol and exposing via entry points.
- Support additional context methods (ALiBi, linear positional extrapolation) through plugin registry.
- Allow custom evaluation probes by dropping new classes into `components/evaluation/probes`.

## Open Questions
- Minimum hardware profile required for 64K context training?  
- Best practice for normalising metrics across heterogeneous datasets?  
- Should telemetry support opt-in remote aggregation for community meta-learning?
