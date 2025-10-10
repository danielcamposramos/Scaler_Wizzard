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
    length: int
    probes: Sequence[str] = field(default_factory=lambda: ["synthetic_longform"])
    notes: str = ""


def build_context_schedule(
    base_context: int,
    target_context: int,
    explicit_schedule: Iterable[int] | None = None,
) -> List[ContextStep]:
    """Compute a monotonic schedule from base to target context.

    If `explicit_schedule` is provided it is validated and wrapped as steps.
    Otherwise the schedule grows exponentially up to the target.
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
            current = min(target_context, current * 2)
            lengths.append(current)
        if not lengths or lengths[-1] != target_context:
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
    """Determine if progressive expansion should stop based on score degradation.

    Scores are assumed to be higher-is-better (e.g., accuracy).  When the moving
    average over `window` drops below `threshold`, we signal a halt.
    """

    if len(scores) < window:
        return False
    recent = scores[-window:]
    moving_avg = sum(recent) / window
    return moving_avg < threshold


__all__ = ["ContextStep", "build_context_schedule", "should_halt"]
