"""Continuous steering control.

The project boundary: `ControlCommand` is what a robot layer would consume.
Nothing here knows about motors or gaits.
"""

from eegbot.control.controller import ContinuousController, ControlCommand
from eegbot.control.law import center, dead_zone, ema_alpha, slew_limit

__all__ = [
    "ContinuousController",
    "ControlCommand",
    "center",
    "dead_zone",
    "ema_alpha",
    "slew_limit",
]
