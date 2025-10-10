"""Training circuit breaker to enforce safety envelopes.

Implements Kimi's eagle-vision insight: automatically halt or warn when model
quality drops below acceptable bounds.  Integrates with telemetry so Daniel can
monitor the process from the cockpit dashboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CBConfig:
    window_steps: int = 10
    max_perplexity_delta: float = 0.15
    min_accuracy: float = 0.5
    hard_stop: bool = True
    telegram_path: Optional[Path] = None


@dataclass
class MetricSnapshot:
    step: int
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None


class CircuitBreaker:
    def __init__(self, config: CBConfig):
        if config.window_steps < 2:
            raise ValueError("window_steps must be >= 2")
        self.config = config
        self.history: List[MetricSnapshot] = []

    def register(self, step: int, perplexity: Optional[float], accuracy: Optional[float]) -> Dict[str, str]:
        """Record a metric snapshot and return cockpit telegram."""

        snapshot = MetricSnapshot(step=step, perplexity=perplexity, accuracy=accuracy)
        self.history.append(snapshot)
        if len(self.history) > self.config.window_steps:
            self.history.pop(0)

        action = "continue"
        reasons: List[str] = []

        # Perplexity slope check.
        ppl_slope = self._perplexity_slope()
        if ppl_slope is not None and ppl_slope > self.config.max_perplexity_delta:
            action = "stop" if self.config.hard_stop else "warn"
            reasons.append(f"perplexity rising {ppl_slope:.3f}/step")

        # Accuracy floor check.
        if accuracy is not None and accuracy < self.config.min_accuracy:
            action = "stop" if self.config.hard_stop else "warn"
            reasons.append(f"accuracy {accuracy:.2f} below floor {self.config.min_accuracy:.2f}")

        telegram = {
            "action": action,
            "reason": "; ".join(reasons),
            "step": step,
        }
        self._emit_telegram(telegram)
        return telegram

    def _perplexity_slope(self) -> Optional[float]:
        recent = [snap.perplexity for snap in self.history if snap.perplexity is not None]
        if len(recent) < self.config.window_steps:
            return None
        delta = recent[-1] - recent[0]
        return delta / self.config.window_steps

    def _emit_telegram(self, telegram: Dict[str, str]) -> None:
        message = json.dumps(telegram)
        print(message, flush=True)
        if self.config.telegram_path:
            self.config.telegram_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.telegram_path.write_text(message)


__all__ = ["CBConfig", "CircuitBreaker", "MetricSnapshot"]
