"""Cognitive-load throttle for cockpit alerts.

Maintains priority ordering (voice > visual > telegram) so only one modality is
active at a time.  Designed for integration with the cockpit dashboard event
loop.  Inspired by Qwen-VL30B's UX directives.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class Modality(Enum):
    VOICE = auto()
    VISUAL = auto()
    TELEGRAM = auto()


PRIORITY = [Modality.VOICE, Modality.VISUAL, Modality.TELEGRAM]


@dataclass
class Alert:
    modality: Modality
    message: str


class CognitiveLoadThrottle:
    def __init__(self) -> None:
        self.active: Optional[Modality] = None

    def request(self, alert: Alert) -> Tuple[bool, Optional[Modality]]:
        """Request activation for a modality.

        Returns (accepted, preempted_modality).  Only modalities with higher
        priority than the current active channel can pre-empt.
        """

        if self.active is None:
            self.active = alert.modality
            return True, None

        current_index = PRIORITY.index(self.active)
        new_index = PRIORITY.index(alert.modality)

        if new_index < current_index:
            previous = self.active
            self.active = alert.modality
            return True, previous

        return False, None

    def release(self, modality: Modality) -> None:
        if self.active == modality:
            self.active = None
