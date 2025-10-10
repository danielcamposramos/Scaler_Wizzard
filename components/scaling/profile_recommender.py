"""Hardware-aware scaling profile recommender.

Extends the template system with dynamic recommendations based on detected
resources, inspired by the final Step 1 chain round.  The module is defensive
and falls back to conservative defaults when optional dependencies are absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

try:  # Optional dependency
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


TEMPLATES_PATH = Path(__file__).resolve().parents[1] / "templates" / "scaling_profiles.yaml"


@dataclass
class HardwareProfile:
    """A data class to store detected hardware specifications.

    Attributes:
        gpu_name (str): The name of the detected GPU.
        vram_total (float): Total GPU VRAM in gigabytes.
        vram_free (float): Free GPU VRAM in gigabytes.
        gpu_count (int): The number of detected GPUs.
        ram_total (float): Total system RAM in gigabytes.
        ram_free (float): Free system RAM in gigabytes.
    """
    gpu_name: str
    vram_total: float
    vram_free: float
    gpu_count: int
    ram_total: float
    ram_free: float

    @property
    def to_dict(self) -> Dict[str, Any]:
        """Converts the hardware profile to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the hardware profile.
        """
        return {
            "gpu_name": self.gpu_name,
            "vram_total": self.vram_total,
            "vram_free": self.vram_free,
            "gpu_count": self.gpu_count,
            "ram_total": self.ram_total,
            "ram_free": self.ram_free,
        }


class ProfileRecommender:
    """Recommends and adjusts scaling profiles based on hardware and use case.

    This class loads scaling profile templates, detects available hardware,
    and provides recommendations for scaling configurations. It can also
    dynamically adjust profiles if hardware resources are constrained.
    """
    def __init__(self, templates_path: Path = TEMPLATES_PATH):
        """Initializes the ProfileRecommender.

        Args:
            templates_path (Path): The path to the YAML file containing scaling
                                   profile templates.
        """
        self.templates_path = templates_path
        self.profiles: Dict[str, Dict[str, float]] = {}
        self.hardware: Optional[HardwareProfile] = None
        self.load_profiles()

    def load_profiles(self) -> None:
        """Loads scaling profiles from the specified YAML file.

        Raises:
            FileNotFoundError: If the templates file does not exist.
            RuntimeError: If the PyYAML library is not installed.
        """
        if not self.templates_path.exists():
            raise FileNotFoundError(f"Scaling templates not found: {self.templates_path}")
        if yaml is None:
            raise RuntimeError("PyYAML is required to load scaling profiles")
        with self.templates_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        self.profiles = data.get("templates", {})

    def detect_hardware(self) -> HardwareProfile:
        """Detects available hardware resources (GPU and RAM).

        This method attempts to use `GPUtil` and `psutil` to get hardware
        information. If these libraries are not available, it falls back to
        conservative default values.

        Returns:
            HardwareProfile: A data object containing the detected hardware specs.
        """
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
                vram_total = float(getattr(gpu, "memoryTotal", vram_total)) / 1024
                vram_free = float(getattr(gpu, "memoryFree", vram_free)) / 1024
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
        """Recommends a scaling profile based on use case and available VRAM.

        It selects the best profile from the loaded templates that fits within
        the available free VRAM.

        Args:
            use_case (str): The intended use case (e.g., 'long_context', 'reasoning').
            model_size (str): The size of the model ('small' or 'large').

        Returns:
            Dict[str, float]: The recommended scaling profile as a dictionary of settings.
                              Returns an empty dictionary if no suitable profile is found.
        """
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
        if self.profiles:
            return next(iter(self.profiles.values()))
        return {}

    def _profile_priority(self, use_case: str, model_size: str) -> list[str]:
        """Determines the order of profiles to check based on use case and model size.

        Args:
            use_case (str): The primary use case.
            model_size (str): The model size.

        Returns:
            list[str]: A list of profile names in order of priority.
        """
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
        """Adjusts a profile's parameters if hardware resources are constrained.

        If the available VRAM is less than the profile's `min_vram_gb`, it reduces
        the LoRA rank and context multiplier to conserve memory.

        Args:
            profile_name (str): The name of the profile to adjust.

        Returns:
            Dict[str, float]: The adjusted profile settings. Returns an empty
                              dictionary if the profile name is not found.
        """
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
        """Creates a JSON snapshot of the current hardware and loaded profiles.

        Returns:
            str: A JSON string representing the current state.
        """
        payload: Dict[str, Any] = {
            "hardware": self.hardware.to_dict if self.hardware else None,
            "profiles": self.profiles,
        }
        return json.dumps(payload, indent=2)


__all__ = [
    "ProfileRecommender",
    "HardwareProfile",
]
