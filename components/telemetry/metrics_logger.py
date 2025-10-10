"""Lightweight telemetry sink for Scaler Wizard."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TelemetryConfig:
    metrics_path: Path
    flush_every: int = 1


class MetricsLogger:
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self._buffer = []
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metric: str, value: float, step: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> None:
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
        if not self._buffer:
            return
        with self.config.metrics_path.open("a", encoding="utf-8") as handle:
            for event in self._buffer:
                handle.write(json.dumps(event) + "\n")
        self._buffer.clear()

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self.config)
