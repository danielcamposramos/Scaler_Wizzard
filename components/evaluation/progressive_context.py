"""Progressive context expansion utilities.

Implements GLM's recommendation to increase context length step-by-step,
running guardrail evaluations between each jump.  The module is pure-Python so
it can be reused in both CLI and plugin runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


@dataclass
class ContextStep:
    """Represents a single step in the progressive context expansion schedule.

    Attributes:
        length (int): The context length for this step.
        probes (Sequence[str]): A sequence of evaluation probe names to run at this step.
        notes (str): Optional notes for this step, e.g., 'base context checkpoint'.
    """
    length: int
    probes: Sequence[str] = field(default_factory=lambda: ["synthetic_longform"])
    notes: str = ""


def build_context_schedule(
    base_context: int,
    target_context: int,
    explicit_schedule: Iterable[int] | None = None,
) -> List[ContextStep]:
    """Computes a monotonic schedule of context lengths for progressive expansion.

    This function generates a series of `ContextStep` objects, starting from a
    `base_context` and moving towards a `target_context`. The schedule can either
    be explicitly provided or generated automatically with exponential growth.

    Args:
        base_context (int): The starting context length.
        target_context (int): The final desired context length.
        explicit_schedule (Iterable[int] | None): An optional, ordered list of
                                                  context lengths to use. If not
                                                  provided, the schedule grows
                                                  exponentially.

    Returns:
        List[ContextStep]: A list of context steps, including evaluation probes.

    Raises:
        ValueError: If the `explicit_schedule` contains values below the
                    `base_context`.
    """
    if explicit_schedule:
        ordered = sorted(set(explicit_schedule))
        if ordered[0] < base_context:
            raise ValueError("Explicit schedule cannot include lengths below base context.")
        if ordered[-1] != target_context:
            ordered.append(target_context)
        lengths = ordered
    else:
        lengths: List[int] = []
        current = max(base_context, 1024)
        while current < target_context:
            lengths.append(current)
            current *= 2
        if not lengths or lengths[-1] < target_context:
             lengths.append(target_context)

    steps = [ContextStep(length=base_context, notes="base context checkpoint")]
    for length in lengths:
        if length <= base_context:
            continue
        probes = ["synthetic_longform", "retrieval_qa"]
        if length >= 32768:
            probes.append("code_generation")
        steps.append(ContextStep(length=length, probes=probes))

    return steps


def should_halt(
    scores: List[float],
    threshold: float,
    window: int = 3,
) -> bool:
    """Determines if progressive expansion should halt due to score degradation.

    This function checks if the moving average of the last `window` scores has
    dropped below a specified `threshold`. It's used as a circuit breaker
    to stop context expansion when model performance degrades.

    Args:
        scores (List[float]): A list of evaluation scores, where higher is better.
        threshold (float): The minimum acceptable score for the moving average.
        window (int): The number of recent scores to consider for the moving average.

    Returns:
        bool: True if the expansion should be halted, False otherwise.
    """
    if len(scores) < window:
        return False
    recent = scores[-window:]
    moving_avg = sum(recent) / window
    return moving_avg < threshold


__all__ = ["ContextStep", "build_context_schedule", "should_halt"]
