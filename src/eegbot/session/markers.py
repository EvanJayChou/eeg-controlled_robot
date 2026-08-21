"""Event markers.

Markers are the join between what the screen showed and what the amplifier
recorded. Get them wrong and every epoch is misaligned, which looks exactly like
"motor imagery is hard" rather than like a bug.

Three sinks behind one interface:

* `MemoryMarkerSink` -- records to a list. Used by tests and by session dry runs,
  so the whole runner is exercisable with no LSL installed.
* `LSLMarkerSink` -- publishes an LSL `Markers` outlet for a recorder to capture
  alongside EEG. The normal path.
* `TeeMarkerSink` -- both, so a live session still keeps a local copy. Cheap
  insurance against an LSL recorder that was not actually running.

The DSI-7's 4-bit hardware `TRG` input is more precise than LSL and would remove
software-timing jitter entirely. It is not implemented yet; see README.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Marker:
    label: str
    timestamp: float


@runtime_checkable
class MarkerSink(Protocol):
    def push(self, label: str, timestamp: float | None = None) -> None: ...
    def close(self) -> None: ...


@dataclass
class MemoryMarkerSink:
    """Collects markers in memory. Default for dry runs and tests."""

    markers: list[Marker] = field(default_factory=list)

    def push(self, label: str, timestamp: float | None = None) -> None:
        self.markers.append(Marker(label, timestamp if timestamp is not None else time.time()))

    def close(self) -> None:
        pass

    def labels(self) -> list[str]:
        return [m.label for m in self.markers]


class LSLMarkerSink:
    """Publishes markers on an LSL outlet.

    `pylsl` is imported lazily so that analysis-only environments -- which is
    most of the team most of the time -- need no LSL install.
    """

    def __init__(self, name: str = "eegbot-markers", source_id: str = "eegbot") -> None:
        from pylsl import StreamInfo, StreamOutlet

        info = StreamInfo(
            name=name,
            type="Markers",
            channel_count=1,
            nominal_srate=0,  # irregular rate: markers are events, not a signal
            channel_format="string",
            source_id=source_id,
        )
        self._outlet = StreamOutlet(info)

    def push(self, label: str, timestamp: float | None = None) -> None:
        if timestamp is None:
            self._outlet.push_sample([label])
        else:
            self._outlet.push_sample([label], timestamp)

    def close(self) -> None:
        self._outlet = None


@dataclass
class TeeMarkerSink:
    """Fans out to several sinks.

    Use this live: publish over LSL *and* keep a local copy, so a session is not
    lost to a recorder that quietly was not listening.
    """

    sinks: list[MarkerSink]

    def push(self, label: str, timestamp: float | None = None) -> None:
        for sink in self.sinks:
            sink.push(label, timestamp)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


# === Marker vocabulary ===

FIXATION = "fixation"
CUE_LEFT = "cue_left"
CUE_RIGHT = "cue_right"
IMAGERY_START = "mi_start"
IMAGERY_END = "mi_end"
REST = "rest"
RUN_START = "run_start"
RUN_END = "run_end"
ALPHA_EYES_OPEN = "alpha_eyes_open"
ALPHA_EYES_CLOSED = "alpha_eyes_closed"
IDLE_BLOCK = "idle_block"
BLOCK_END = "block_end"


def cue_marker(label_name: str) -> str:
    return {"left_hand": CUE_LEFT, "right_hand": CUE_RIGHT}[label_name]
