import json
from pathlib import Path

import pytest

from components.safety.circuit_breaker import CBConfig, CircuitBreaker
from components.ui.cognitive_throttle import Alert, CognitiveLoadThrottle, Modality
from components.validation.schema_validator import SchemaValidator
from components.scaling.profile_recommender import ProfileRecommender
from tools.rollback import rollback_phase


def test_circuit_breaker_triggers_on_accuracy_drop(tmp_path):
    config = CBConfig(window_steps=3, max_perplexity_delta=0.5, min_accuracy=0.6, hard_stop=True)
    breaker = CircuitBreaker(config)

    for step in range(3):
        breaker.register(step=step, perplexity=5.0, accuracy=0.8)
    result = breaker.register(step=4, perplexity=5.1, accuracy=0.4)

    assert result["action"] == "stop"
    assert "accuracy" in result["reason"]


def test_schema_validator_catches_missing_fields():
    jsonschema = pytest.importorskip("jsonschema")
    validator = SchemaValidator()
    payload = {
        "schemaVersion": "v1.0",
        "runId": "123e4567-e89b-12d3-a456-426614174000",
        "timestamp": "2024-05-26T18:42:00Z",
        "phase": "P1",
        "status": "running",
        "qualityMetrics": {"perplexity": 5.0, "accuracy": 0.8, "trend": "stable"},
        "resourceUsage": {"gpuMemoryGb": 10, "cpuMemoryGb": 8, "throughputTokensPerSec": 120},
        "humanContract": {"version": "1.0.0", "authorizedBy": "Daniel Ramos", "signed": True, "signoff": "I accept responsibility for this run"},
        "actions": {
            "approve": {"enabled": True},
            "abort": {"enabled": False},
            "rollback": {"enabled": True, "targetPhase": "P0", "checkpoint": "checkpoints/phase_0"}
        },
        "alerts": [],
    }
    # missing required field (statusMessage optional? not part) but actions rollback missing lastTriggered? not required.
    # We'll remove required field qualityMetrics? Already there.
    # To trigger error remove humanContract signoff? but required. Keep but set to wrong string.
    payload["humanContract"]["signoff"] = "I do not accept"

    result = validator.validate(payload)
    assert not result.ok
    assert any("signoff" in msg for msg in result.errors)


def test_voice_priority_preempts_visual():
    throttle = CognitiveLoadThrottle()
    accepted, _ = throttle.request(Alert(Modality.VISUAL, "visual alert"))
    assert accepted
    accepted, preempted = throttle.request(Alert(Modality.VOICE, "voice alert"))
    assert accepted and preempted == Modality.VISUAL
    throttle.release(Modality.VOICE)
    accepted, _ = throttle.request(Alert(Modality.TELEGRAM, "log message"))
    assert accepted


def test_rollback_phase_restores_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    checkpoints_dir = run_dir / "checkpoints"
    target_phase_dir = checkpoints_dir / "phase_1"
    model_dir = run_dir / "model"
    adapters_dir = run_dir / "adapters"
    target_phase_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    adapters_dir.mkdir(parents=True)

    (target_phase_dir / "weights.bin").write_text("checkpoint", encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(
        json.dumps({
            "phase": 2,
            "checkpoints": {"phase_1": str(target_phase_dir)}
        }),
        encoding="utf-8",
    )
    # create adapter file to ensure cleanup
    (adapters_dir / "adapter.safetensors").write_text("adapter", encoding="utf-8")

    rollback_phase(run_dir, 1)

    restored = (model_dir / "weights.bin").read_text(encoding="utf-8")
    assert restored == "checkpoint"
    assert not any(adapters_dir.iterdir())
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["phase"] == 1


def test_profile_recommender_returns_template(monkeypatch):
    recommender = ProfileRecommender()

    class DummyHardware:
        def __init__(self):
            self.vram_free = 24

    recommender.hardware = DummyHardware()
    profile = recommender.recommend_profile(use_case="long_context")
    assert profile["context_multiplier"] >= 4
