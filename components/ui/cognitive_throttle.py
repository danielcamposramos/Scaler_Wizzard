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
    """Enumeration for the different types of alert modalities."""
    VOICE = auto()
    VISUAL = auto()
    TELEGRAM = auto()


PRIORITY = [Modality.VOICE, Modality.VISUAL, Modality.TELEGRAM]
"""Defines the priority order of modalities, from highest to lowest."""


@dataclass
class Alert:
    """Represents an alert to be sent through a specific modality.

    Attributes:
        modality (Modality): The communication channel for the alert.
        message (str): The content of the alert message.
    """
    modality: Modality
    message: str


class CognitiveLoadThrottle:
    """Manages alert modalities to prevent overwhelming the user.

    This class ensures that only one alert modality is active at any given time,
    based on a predefined priority list. A higher-priority modality can preempt
    a lower-priority one.
    """
    def __init__(self) -> None:
        """Initializes the CognitiveLoadThrottle."""
        self.active: Optional[Modality] = None

    def request(self, alert: Alert) -> Tuple[bool, Optional[Modality]]:
        """Requests activation for an alert's modality.

        The request is granted if no other modality is active, or if the new
        alert's modality has a higher priority than the currently active one.

        Args:
            alert (Alert): The alert requesting activation.

        Returns:
            Tuple[bool, Optional[Modality]]: A tuple containing:
                - A boolean indicating if the request was accepted.
                - The modality that was preempted, if any.
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
        """Releases the lock on a modality.

        If the specified modality is the currently active one, the throttle is
        reset, allowing other modalities to become active.

        Args:
            modality (Modality): The modality to release.
        """
        if self.active == modality:
            self.active = None
