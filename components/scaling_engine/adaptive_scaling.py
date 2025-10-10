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
    """A lightweight description of the runtime environment.

    Attributes:
        total_vram_gb (float): The total available VRAM across all GPUs in GB.
        gpu_count (int): The number of GPUs available.
        cpu_mem_gb (float): The total available CPU memory in GB.
    """
    total_vram_gb: float
    gpu_count: int = 1
    cpu_mem_gb: float = 32.0


@dataclass
class ScalingTargets:
    """Specifies the desired capacity and context goals for scaling.

    Attributes:
        base_params_b (float): The number of parameters in the base model, in billions.
        target_multiplier (float): The desired increase in model capacity (e.g., 2.0 for 2x).
        context_multiplier (float): The desired increase in context length (e.g., 2.0 for 2x).
    """
    base_params_b: float
    target_multiplier: float = 1.0
    context_multiplier: float = 1.0


@dataclass
class LoRAConfigSuggestion:
    """A suggested LoRA (Low-Rank Adaptation) configuration.

    This data class holds the parameters for PEFT (Parameter-Efficient Fine-Tuning)
    injection, including rank, alpha, dropout, and whether to use QLoRA.

    Attributes:
        rank (int): The rank of the LoRA matrices.
        alpha (int): The scaling factor for the LoRA adaptation.
        dropout (float): The dropout rate to apply to the LoRA layers.
        use_qlora (bool): Whether to use QLoRA for 4-bit quantization.
    """
    rank: int
    alpha: int
    dropout: float
    use_qlora: bool


def _normalise_multiplier(multiplier: float, lower: float, upper: float) -> float:
    """Clamps a multiplier value within a specified range.

    Args:
        multiplier (float): The value to clamp.
        lower (float): The minimum allowed value.
        upper (float): The maximum allowed value.

    Returns:
        float: The clamped value.
    """
    return max(lower, min(upper, multiplier))


def calculate_optimal_lora_rank(
    base_model_params: float,
    target_capacity: float,
    hardware_constraints: Dict[str, float],
) -> int:
    """Calculates an optimal LoRA rank based on a heuristic.

    This function provides backward compatibility with Qwen's original heuristic
    for determining LoRA rank.

    Args:
        base_model_params (float): The base model's parameters in billions.
        target_capacity (float): The desired target capacity.
        hardware_constraints (Dict[str, float]): A dictionary with hardware info,
                                                 like 'available_memory'.

    Returns:
        int: The calculated optimal LoRA rank.
    """
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
    """Generates a LoRA configuration suggestion based on targets and hardware.

    The heuristic expands on GLM's proposal by factoring in both parameter and
    context multipliers. When context length grows aggressively, it tempers the
    rank to keep memory requirements reasonable.

    Args:
        targets (ScalingTargets): The desired scaling goals.
        hardware (HardwareProfile): The detected hardware profile.
        prefer_low_memory (bool): If True, prioritizes memory-saving options.
        dropout_override (Optional[float]): An optional value to override the
                                             calculated dropout.

    Returns:
        LoRAConfigSuggestion: The suggested LoRA configuration.
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
