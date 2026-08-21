"""Recording sessions: scheduling, cues, markers, and storage.

Everything here runs headless, so the full protocol can be rehearsed before
PsychoPy is installed or the DSI-7 arrives. `recording` is imported lazily
because it needs MNE.
"""

from typing import Any

from eegbot.session.cues import Display, HeadlessDisplay
from eegbot.session.markers import MarkerSink, MemoryMarkerSink
from eegbot.session.protocol import Trial, build_schedule, estimate_duration_s
from eegbot.session.runner import SessionLog, SessionRunner

__all__ = [
    "Display",
    "HeadlessDisplay",
    "MarkerSink",
    "MemoryMarkerSink",
    "SessionLog",
    "SessionRunner",
    "SessionMetadata",
    "Trial",
    "build_schedule",
    "estimate_duration_s",
    "save_session",
]

_LAZY = {
    "SessionMetadata": "eegbot.session.recording",
    "save_session": "eegbot.session.recording",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
