"""Checkpoint cache utilities to reduce rollback latency."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class CacheEntry:
    phase: int
    checkpoint_path: Path
    metadata: Dict[str, str] = field(default_factory=dict)


class CheckpointCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[int, CacheEntry] = {}

    def put(self, phase: int, checkpoint: Path, metadata: Optional[Dict[str, str]] = None) -> CacheEntry:
        target = self.cache_dir / f"phase_{phase}"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(checkpoint, target)
        entry = CacheEntry(phase=phase, checkpoint_path=target, metadata=metadata or {})
        self._entries[phase] = entry
        return entry

    def get(self, phase: int) -> Optional[CacheEntry]:
        if phase in self._entries:
            return self._entries[phase]
        target = self.cache_dir / f"phase_{phase}"
        if target.exists():
            entry = CacheEntry(phase=phase, checkpoint_path=target)
            self._entries[phase] = entry
            return entry
        return None

    def latest(self) -> Optional[CacheEntry]:
        if not self._entries:
            return None
        phase = max(self._entries)
        return self._entries[phase]


__all__ = ["CheckpointCache", "CacheEntry"]
