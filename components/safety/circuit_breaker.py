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
    """Configuration for the CircuitBreaker.

    Attributes:
        window_steps (int): The number of steps to consider for metric trends.
        max_perplexity_delta (float): The maximum allowable increase in perplexity
                                      over the window.
        min_accuracy (float): The minimum acceptable accuracy.
        hard_stop (bool): If True, triggers a 'stop' action. Otherwise, 'warn'.
        telegram_path (Optional[Path]): If provided, writes the status message
                                         to this file.
    """
    window_steps: int = 10
    max_perplexity_delta: float = 0.15
    min_accuracy: float = 0.5
    hard_stop: bool = True
    telegram_path: Optional[Path] = None


@dataclass
class MetricSnapshot:
    """A snapshot of model metrics at a specific training step.

    Attributes:
        step (int): The training step number.
        perplexity (Optional[float]): The model's perplexity at this step.
        accuracy (Optional[float]): The model's accuracy at this step.
    """
    step: int
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None


class CircuitBreaker:
    """Monitors training metrics and decides whether to continue, warn, or stop.

    This class tracks perplexity and accuracy over a sliding window of training
    steps. It checks for two conditions:
    1. A rapid increase in perplexity.
    2. Accuracy dropping below a predefined threshold.
    """
    def __init__(self, config: CBConfig):
        """Initializes the CircuitBreaker with a given configuration.

        Args:
            config (CBConfig): The configuration object for the circuit breaker.

        Raises:
            ValueError: If `window_steps` is less than 2.
        """
        if config.window_steps < 2:
            raise ValueError("window_steps must be >= 2")
        self.config = config
        self.history: List[MetricSnapshot] = []

    def register(self, step: int, perplexity: Optional[float], accuracy: Optional[float]) -> Dict[str, str]:
        """Records a new metric snapshot, evaluates the safety conditions, and returns a status.

        This method is the main entry point for the circuit breaker. It updates
        its history, checks the rules, and emits a "telegram" (a status message)
        to stdout and optionally to a file.

        Args:
            step (int): The current training step.
            perplexity (Optional[float]): The latest perplexity score.
            accuracy (Optional[float]): The latest accuracy score.

        Returns:
            Dict[str, str]: A dictionary (telegram) containing the action to take
                            ('continue', 'warn', or 'stop'), the reason, and the step.
        """
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
            "step": str(step),
        }
        self._emit_telegram(telegram)
        return telegram

    def _perplexity_slope(self) -> Optional[float]:
        """Calculates the slope of perplexity over the current history window.

        The slope is computed as the change in perplexity between the first and
        last data points in the window, normalized by the number of steps.

        Returns:
            Optional[float]: The calculated slope, or None if there is not
                             enough data.
        """
        recent = [snap.perplexity for snap in self.history if snap.perplexity is not None]
        if len(recent) < self.config.window_steps:
            return None
        delta = recent[-1] - recent[0]
        return delta / self.config.window_steps

    def _emit_telegram(self, telegram: Dict[str, str]) -> None:
        """Emits the status telegram to stdout and an optional file.

        Args:
            telegram (Dict[str, str]): The status message to emit.
        """
        message = json.dumps(telegram)
        print(message, flush=True)
        if self.config.telegram_path:
            self.config.telegram_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.telegram_path.write_text(message)


__all__ = ["CBConfig", "CircuitBreaker", "MetricSnapshot"]
