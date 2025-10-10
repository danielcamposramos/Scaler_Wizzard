# Safety Envelope Architecture

## Purpose
Ensure the Scaler Wizard never blindly ships degraded models by pairing automated checks with human oversight in the Multi-Vibe chain.

## Circuit Breaker
- Implemented in `components/safety/circuit_breaker.py`.
- Monitors rolling perplexity slope and accuracy floors.
- Emits cockpit telegrams with `continue|warn|stop` directives.
- Writes the latest telegram to disk for the dashboard to consume.

## Human Contract
- Contract template recorded in `components/safety/human_contract.md`.
- Cockpit collects consent and stores `config/human_contract.yaml` per run.
- Pipeline refuses to start when contract version mismatches (`contract_version = 1.0.0`).

## Telemetry Stream
- Metrics logger persists JSONL events via `components/telemetry/metrics_logger.py`.
- Supports local-only storage by default; remote sinks can be added later.
- Metrics include training/eval loss, accuracy, throughput, and hardware utilisation.
- Schema validator (`components/validation/schema_validator.py`) guards cockpit updates using `specs/cockpit_schema.json`.

## Human-in-the-Loop Flow
1. Telemetry events update the cockpit dashboard (to be defined).  
2. Circuit breaker signals if thresholds are violated.  
3. Daniel reviews dashboard and confirms `Approve`, `Abort`, or `Rollback`.  
4. Contract consent modal reminds Daniel of rollback plan before confirming.  
5. Decision is logged back into telemetry for traceability.

## Configuration Knobs
- Safety thresholds surfaced in the UI with sensible defaults.  
- Templates configure different safety personas (conservative, balanced, experimental).  
- All safety decisions recorded in run metadata for audits.

## Open Questions
- What additional probes (toxicity, hallucination) should trigger warnings?  
- Should the cockpit block downloads when safety signals are unresolved?  
- How do we version safety policies as the community contributes new rules?
- How do we expose contract history and signature logs without cluttering the cockpit?
- What automated test coverage is needed to keep the safety chain trustworthy across updates?
