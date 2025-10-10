"""Checkpoint cache utilities to reduce rollback latency."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class CacheEntry:
    """Represents a single entry in the checkpoint cache.

    Attributes:
        phase (int): The phase number of the scaling process.
        checkpoint_path (Path): The path to the cached checkpoint directory.
        metadata (Dict[str, str]): Optional metadata associated with the checkpoint.
    """
    phase: int
    checkpoint_path: Path
    metadata: Dict[str, str] = field(default_factory=dict)


class CheckpointCache:
    """Manages a cache of scaling checkpoints to speed up rollbacks.

    This class provides a simple key-value store for checkpoint directories,
    keyed by the scaling phase number.
    """
    def __init__(self, cache_dir: Path):
        """Initializes the CheckpointCache.

        Args:
            cache_dir (Path): The directory where checkpoints will be stored.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[int, CacheEntry] = {}

    def put(self, phase: int, checkpoint: Path, metadata: Optional[Dict[str, str]] = None) -> CacheEntry:
        """Adds or updates a checkpoint in the cache.

        If a checkpoint for the given phase already exists, it will be overwritten.

        Args:
            phase (int): The scaling phase number to associate with the checkpoint.
            checkpoint (Path): The path to the checkpoint directory to be cached.
            metadata (Optional[Dict[str, str]]): Optional metadata to store with the checkpoint.

        Returns:
            CacheEntry: The newly created cache entry.
        """
        target = self.cache_dir / f"phase_{phase}"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(checkpoint, target)
        entry = CacheEntry(phase=phase, checkpoint_path=target, metadata=metadata or {})
        self._entries[phase] = entry
        return entry

    def get(self, phase: int) -> Optional[CacheEntry]:
        """Retrieves a checkpoint from the cache.

        First, it checks the in-memory index. If not found, it checks the
        filesystem to allow for persistence across sessions.

        Args:
            phase (int): The scaling phase number of the checkpoint to retrieve.

        Returns:
            Optional[CacheEntry]: The cache entry if found, otherwise None.
        """
        if phase in self._entries:
            return self._entries[phase]
        target = self.cache_dir / f"phase_{phase}"
        if target.exists():
            entry = CacheEntry(phase=phase, checkpoint_path=target)
            self._entries[phase] = entry
            return entry
        return None

    def latest(self) -> Optional[CacheEntry]:
        """Retrieves the latest checkpoint from the cache based on the phase number.

        If the in-memory cache is empty, it first populates it from the filesystem.

        Returns:
            Optional[CacheEntry]: The cache entry with the highest phase number,
                                  or None if the cache is empty.
        """
        # Ensure cache is populated from disk if it's empty in memory
        if not self._entries:
            phases_on_disk = [int(p.name.split('_')[1]) for p in self.cache_dir.glob('phase_*') if p.is_dir()]
            for phase in phases_on_disk:
                self.get(phase)

        if not self._entries:
            return None

        latest_phase = max(self._entries.keys())
        return self._entries[latest_phase]


__all__ = ["CheckpointCache", "CacheEntry"]
