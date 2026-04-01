"""Multi-tier, adaptive context window management for TRMC models.

This module implements a memory management system that tiers context across
GPU VRAM, CPU RAM, and File-based storage (Disk), inspired by Ollama's
resource management and Clawdbot's efficient retrieval.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch


class AdaptiveContextManager:
    """Manages multi-tier context storage for large-scale reasoning.

    Tiers:
    1. VRAM: GPU memory (Fastest, limited capacity)
    2. RAM: System memory (Moderate speed, high capacity)
    3. DISK: File-based storage (Slowest, near-infinite capacity)

    Attributes:
        vram_threshold_gb (float): Target VRAM usage threshold.
        ram_threshold_gb (float): Target RAM usage threshold.
        storage_path (Path): Path to store disk-based context.
    """

    def __init__(
        self,
        vram_threshold_gb: float = 6.0,  # Optimized for 8GB GPU
        ram_threshold_gb: float = 24.0, # Optimized for 32GB RAM
        storage_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initializes the AdaptiveContextManager.

        Args:
            vram_threshold_gb (float): VRAM threshold in GB.
            ram_threshold_gb (float): RAM threshold in GB.
            storage_path (Optional): Directory for disk context.
        """
        self.vram_threshold_gb = vram_threshold_gb
        self.ram_threshold_gb = ram_threshold_gb

        if storage_path is None:
            self.storage_path = Path(tempfile.mkdtemp(prefix="trmc_context_"))
        else:
            self.storage_path = Path(storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)

        self.context_registry: Dict[str, Dict[str, Union[torch.Tensor, str]]] = {}

    def _get_current_vram_gb(self) -> float:
        """Heuristic for current VRAM usage of the process."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0

    def _get_current_ram_gb(self) -> float:
        """Heuristic for current system RAM usage."""
        import psutil
        return psutil.virtual_memory().used / (1024**3)

    def store_context(self, key: str, latent_state: torch.Tensor) -> str:
        """Stores a latent state in the most appropriate tier.

        Args:
            key (str): Unique identifier for the context chunk.
            latent_state (torch.Tensor): The tensor to store.

        Returns:
            str: The tier where the context was stored ('vram', 'ram', or 'disk').
        """
        vram_usage = self._get_current_vram_gb()

        # 1. Try VRAM
        if vram_usage < self.vram_threshold_gb:
            self.context_registry[key] = {"tier": "vram", "data": latent_state.detach().cuda()}
            return "vram"

        # 2. Try RAM
        ram_usage = self._get_current_ram_gb()
        if ram_usage < self.ram_threshold_gb:
            self.context_registry[key] = {"tier": "ram", "data": latent_state.detach().cpu()}
            return "ram"

        # 3. Fallback to Disk
        disk_file = self.storage_path / f"{key}.pt"
        torch.save(latent_state.detach().cpu(), disk_file)
        self.context_registry[key] = {"tier": "disk", "data": str(disk_file)}
        return "disk"

    def retrieve_context(self, key: str) -> Optional[torch.Tensor]:
        """Retrieves a stored latent state.

        Args:
            key (str): Identifier for the context chunk.

        Returns:
            Optional[torch.Tensor]: The retrieved tensor, moved to CPU if from disk/RAM.
        """
        if key not in self.context_registry:
            return None

        entry = self.context_registry[key]
        tier = entry["tier"]

        if tier == "vram" or tier == "ram":
            return entry["data"]

        if tier == "disk":
            return torch.load(entry["data"])

        return None

    def cleanup(self) -> None:
        """Removes temporary disk storage."""
        if self.storage_path.exists() and "trmc_context_" in str(self.storage_path):
            shutil.rmtree(self.storage_path)

    def __del__(self) -> None:
        self.cleanup()


def adaptive_memory_offload(model: torch.nn.Module) -> None:
    """Applies weight offloading strategies to the model for massive context support.

    This is a placeholder for more advanced integration with deepspeed/accelerate
    style offloading tailored for recursive models.
    """
    pass


if __name__ == "__main__":
    manager = AdaptiveContextManager(vram_threshold_gb=1.0, ram_threshold_gb=2.0)
    dummy_latent = torch.randn(1, 1024, 128)

    tier = manager.store_context("chunk_1", dummy_latent)
    print(f"Stored in: {tier}")

    retrieved = manager.retrieve_context("chunk_1")
    print(f"Retrieved shape: {retrieved.shape}")

    manager.cleanup()
