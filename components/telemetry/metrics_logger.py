"""Lightweight telemetry sink for Scaler Wizard."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TelemetryConfig:
    """Configuration for the MetricsLogger.

    Attributes:
        metrics_path (Path): The file path where metrics will be logged.
        flush_every (int): The number of log entries to buffer before writing
                           to the file.
    """
    metrics_path: Path
    flush_every: int = 1


class MetricsLogger:
    """A simple logger for recording telemetry data to a file.

    This class buffers metric events and flushes them to a JSONL file
    periodically.
    """
    def __init__(self, config: TelemetryConfig):
        """Initializes the MetricsLogger.

        Args:
            config (TelemetryConfig): The configuration for the logger.
        """
        self.config = config
        self._buffer = []
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metric: str, value: float, step: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Logs a single metric event.

        The event is added to an in-memory buffer. If the buffer size reaches
        the `flush_every` threshold, the buffer is written to the log file.

        Args:
            metric (str): The name of the metric (e.g., 'loss', 'accuracy').
            value (float): The value of the metric.
            step (Optional[int]): The training step associated with the metric.
            extra (Optional[Dict[str, Any]]): Any additional key-value pairs
                                               to include in the log entry.
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metric": metric,
            "value": value,
        }
        if step is not None:
            event["step"] = step
        if extra:
            event.update(extra)
        self._buffer.append(event)

        if len(self._buffer) >= self.config.flush_every:
            self.flush()

    def flush(self) -> None:
        """Writes all buffered metric events to the log file.

        The buffer is cleared after the events are written.
        """
        if not self._buffer:
            return
        with self.config.metrics_path.open("a", encoding="utf-8") as handle:
            for event in self._buffer:
                handle.write(json.dumps(event) + "\n")
        self._buffer.clear()

    def snapshot(self) -> Dict[str, Any]:
        """Returns a snapshot of the current logger configuration.

        Returns:
            Dict[str, Any]: A dictionary representation of the logger's config.
        """
        return asdict(self.config)
