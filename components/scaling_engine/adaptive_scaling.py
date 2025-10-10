"""Adaptive scaling utilities inspired by the Multi-Vibe partner swarm.

This module determines LoRA-style adapter parameters based on the base model,
desired capacity multiplier, and available hardware.  It centralises the logic
so both the CLI prototype and the TransformerLab plugin share the same rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class HardwareProfile:
    """Lightweight description of the runtime environment."""

    total_vram_gb: float
    gpu_count: int = 1
    cpu_mem_gb: float = 32.0


@dataclass
class ScalingTargets:
    """Desired capacity and context goals."""

    base_params_b: float  # billions of parameters
    target_multiplier: float = 1.0
    context_multiplier: float = 1.0


@dataclass
class LoRAConfigSuggestion:
    """Suggested configuration values for PEFT injection."""

    rank: int
    alpha: int
    dropout: float
    use_qlora: bool


def _normalise_multiplier(multiplier: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, multiplier))


def calculate_optimal_lora_rank(
    base_model_params: float,
    target_capacity: float,
    hardware_constraints: Dict[str, float],
) -> int:
    """Qwen's original heuristic packaged for backwards compatibility."""

    base_rank = min(32, max(8, int(base_model_params * 0.01)))
    adjustment_factor = min(1.5, target_capacity / base_model_params)
    hardware_factor = min(1.0, hardware_constraints.get("available_memory", 16.0) / 16.0)

    return max(4, int(base_rank * adjustment_factor * hardware_factor))


def suggest_lora_config(
    targets: ScalingTargets,
    hardware: HardwareProfile,
    prefer_low_memory: bool = False,
    dropout_override: Optional[float] = None,
) -> LoRAConfigSuggestion:
    """Generate a LoRA configuration suggestion.

    The heuristic expands on GLM's proposal by factoring in both parameter and
    context multipliers.  When context length grows aggressively we temper the
    rank to keep memory requirements reasonable.
    """

    # Effective multiplier blends parameter and context ambitions.
    capacity_multiplier = _normalise_multiplier(targets.target_multiplier, 1.0, 4.0)
    context_pressure = _normalise_multiplier(targets.context_multiplier, 1.0, 8.0)

    # Use billions-of-parameters metric to remain model-agnostic.
    base_params_b = max(0.5, targets.base_params_b)
    raw_rank = int(base_params_b * 16)  # 1B -> rank 16 baseline

    # Scale by desired capacity while penalising heavy context growth.
    scaled_rank = raw_rank * capacity_multiplier / (1 + 0.25 * (context_pressure - 1))

    # Hardware adjustments.
    vram_factor = _normalise_multiplier(hardware.total_vram_gb / 16.0, 0.25, 2.0)
    gpu_factor = _normalise_multiplier(hardware.gpu_count, 0.5, 4.0)
    adjusted_rank = int(scaled_rank * min(vram_factor, gpu_factor))

    # Soft limits: keep rank within practical bounds.
    rank = max(4, min(128, adjusted_rank))

    # Alpha typically scales 2–4× the rank; adapt based on VRAM headroom.
    alpha = max(rank * 2, int(rank * (1.5 + vram_factor)))

    # Dropout: raise when context multiplier is large to stabilise training.
    base_dropout = 0.05 if not prefer_low_memory else 0.1
    dropout = dropout_override if dropout_override is not None else min(
        0.25, base_dropout + 0.02 * (context_pressure - 1)
    )

    # Switch to QLoRA automatically on low-memory setups.
    use_qlora = prefer_low_memory or hardware.total_vram_gb < 20.0

    return LoRAConfigSuggestion(rank=rank, alpha=alpha, dropout=dropout, use_qlora=use_qlora)


__all__ = [
    "HardwareProfile",
    "ScalingTargets",
    "LoRAConfigSuggestion",
    "calculate_optimal_lora_rank",
    "suggest_lora_config",
]
