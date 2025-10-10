"""Knowledge distillation scaffolding for Scaler Wizard.

This module provides data structures and helper functions for configuring
knowledge distillation, a process where a smaller "student" model learns
from a larger "teacher" model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DistillationConfig:
    """Configuration for the knowledge distillation process.

    Attributes:
        teacher_model (str): The identifier of the teacher model to use.
        temperature (float): The temperature for softening the teacher's logits.
        alpha_ce (float): The weight for the cross-entropy loss between student
                          predictions and ground truth labels.
        alpha_student (float): The weight for the distillation loss, which aligns
                               the student's logits with the teacher's.
        match_intermediate_layers (bool): Whether to match hidden states between
                                          teacher and student.
        attention_transfer (bool): Whether to use attention-based transfer.
    """
    teacher_model: str
    temperature: float = 1.5
    alpha_ce: float = 0.7
    alpha_student: float = 0.3
    match_intermediate_layers: bool = True
    attention_transfer: bool = False


def build_distillation_kwargs(config: DistillationConfig) -> Dict[str, Any]:
    """Translates a DistillationConfig into a dictionary of keyword arguments.

    This is a placeholder function to demonstrate how the config might be
    consumed by a hypothetical training framework.

    Args:
        config (DistillationConfig): The distillation configuration object.

    Returns:
        Dict[str, Any]: A dictionary of arguments for a trainer.
    """
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
    """Creates a DistillationConfig if a teacher model is specified.

    This function serves as a factory for `DistillationConfig`. If a teacher model
    is provided, it constructs a config, optionally applying any overrides.

    Args:
        teacher_model (Optional[str]): The identifier for the teacher model. If None,
                                       the function returns None.
        overrides (Optional[Dict[str, Any]]): A dictionary of settings to
                                               override the defaults.

    Returns:
        Optional[DistillationConfig]: A configuration object if a teacher model
                                      is provided, otherwise None.
    """
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
