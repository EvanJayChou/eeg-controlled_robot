"""Live and replayed EEG streams, and the online decoding loop."""

from eegbot.stream.loop import OnlineDecoder, RingBuffer, Update
from eegbot.stream.replay import ArrayReplaySource, FileReplaySource
from eegbot.stream.source import EEGSource
from eegbot.stream.synthetic import synthetic_mi_recording

__all__ = [
    "ArrayReplaySource",
    "EEGSource",
    "FileReplaySource",
    "OnlineDecoder",
    "RingBuffer",
    "Update",
    "synthetic_mi_recording",
]
