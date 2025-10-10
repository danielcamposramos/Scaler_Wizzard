"""Knowledge distillation scaffolding for Scaler Wizard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DistillationConfig:
    teacher_model: str
    temperature: float = 1.5
    alpha_ce: float = 0.7
    alpha_student: float = 0.3
    match_intermediate_layers: bool = True
    attention_transfer: bool = False


def build_distillation_kwargs(config: DistillationConfig) -> Dict[str, Any]:
    """Translate config into Trainer kwargs (placeholder until wiring)."""

    return {
        "teacher_model": config.teacher_model,
        "temperature": config.temperature,
        "alpha_ce": config.alpha_ce,
        "alpha_student": config.alpha_student,
        "match_intermediate_layers": config.match_intermediate_layers,
        "attention_transfer": config.attention_transfer,
    }


def maybe_configure_distillation(
    teacher_model: Optional[str],
    overrides: Optional[Dict[str, Any]] = None,
) -> Optional[DistillationConfig]:
    """Generate a config if a teacher model is provided."""

    if not teacher_model:
        return None

    config = DistillationConfig(teacher_model=teacher_model)
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    return config


__all__ = [
    "DistillationConfig",
    "build_distillation_kwargs",
    "maybe_configure_distillation",
]
