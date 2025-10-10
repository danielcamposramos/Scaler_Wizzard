"""Hardware-aware scaling profile recommender.

Extends the template system with dynamic recommendations based on detected
resources, inspired by the final Step 1 chain round.  The module is defensive
and falls back to conservative defaults when optional dependencies are absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:  # Optional dependency
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


TEMPLATES_PATH = Path(__file__).resolve().parents[1] / "templates" / "scaling_profiles.yaml"


@dataclass
class HardwareProfile:
    gpu_name: str
    vram_total: float
    vram_free: float
    gpu_count: int
    ram_total: float
    ram_free: float

    @property
    def to_dict(self) -> Dict[str, float]:
        return {
            "gpu_name": self.gpu_name,
            "vram_total": self.vram_total,
            "vram_free": self.vram_free,
            "gpu_count": self.gpu_count,
            "ram_total": self.ram_total,
            "ram_free": self.ram_free,
        }


class ProfileRecommender:
    def __init__(self, templates_path: Path = TEMPLATES_PATH):
        self.templates_path = templates_path
        self.profiles: Dict[str, Dict[str, float]] = {}
        self.hardware: Optional[HardwareProfile] = None
        self.load_profiles()

    def load_profiles(self) -> None:
        if not self.templates_path.exists():
            raise FileNotFoundError(f"Scaling templates not found: {self.templates_path}")
        if yaml is None:
            raise RuntimeError("PyYAML is required to load scaling profiles")
        with self.templates_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        self.profiles = data.get("templates", {})

    def detect_hardware(self) -> HardwareProfile:
        gpu_name = "Unknown"
        vram_total = 8.0
        vram_free = 4.0
        gpu_count = 1
        ram_total = 16.0
        ram_free = 8.0

        try:
            import GPUtil  # type: ignore

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_name = gpu.name
                vram_total = float(getattr(gpu, "memoryTotal", vram_total))
                vram_free = float(getattr(gpu, "memoryFree", vram_free))
                gpu_count = len(gpus)
        except Exception:
            pass

        try:
            import psutil  # type: ignore

            vm = psutil.virtual_memory()
            ram_total = vm.total / (1024**3)
            ram_free = vm.available / (1024**3)
        except Exception:
            pass

        profile = HardwareProfile(
            gpu_name=gpu_name,
            vram_total=vram_total,
            vram_free=vram_free,
            gpu_count=gpu_count,
            ram_total=ram_total,
            ram_free=ram_free,
        )
        self.hardware = profile
        return profile

    def recommend_profile(self, use_case: str, model_size: str = "small") -> Dict[str, float]:
        if not self.hardware:
            self.detect_hardware()
        assert self.hardware

        vram = self.hardware.vram_free
        order = self._profile_priority(use_case, model_size)
        for key in order:
            template = self.profiles.get(key)
            if not template:
                continue
            min_vram = float(template.get("min_vram_gb", 0))
            if vram >= min_vram:
                return template
        # fallback to first available profile
        for template in self.profiles.values():
            return template
        return {}

    def _profile_priority(self, use_case: str, model_size: str) -> list[str]:
        mapping = {
            "long_context": ["max_context", "balanced_scaling", "budget_context"],
            "reasoning": ["max_parameters", "reasoning_boost", "balanced_scaling"],
            "creative": ["creative_fusion", "balanced_scaling"],
            "default": ["balanced_scaling", "budget_context"],
        }
        priority = list(mapping.get(use_case, mapping["default"]))
        if model_size == "large":
            priority.insert(0, "max_parameters")
        # Remove duplicates while preserving order.
        seen = set()
        ordered: list[str] = []
        for item in priority:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def get_profile_adjustments(self, profile_name: str) -> Dict[str, float]:
        if profile_name not in self.profiles:
            return {}
        if not self.hardware:
            self.detect_hardware()
        assert self.hardware

        profile = dict(self.profiles[profile_name])
        required = float(profile.get("min_vram_gb", 0))
        if self.hardware.vram_free < required:
            profile["lora_rank"] = max(4, int(profile.get("lora_rank", 8) / 2))
            profile["context_multiplier"] = max(2, int(profile.get("context_multiplier", 2) / 2))
        return profile

    def snapshot(self) -> str:
        payload = {
            "hardware": self.hardware.to_dict if self.hardware else None,
            "profiles": self.profiles,
        }
        return json.dumps(payload, indent=2)


__all__ = [
    "ProfileRecommender",
    "HardwareProfile",
]
