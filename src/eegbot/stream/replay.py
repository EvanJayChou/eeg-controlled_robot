"""Replay sources -- how the project runs with no headset.

Deliberately built on plain numpy rather than LSL. Replay is the default path
for development, tests, and CI, and making that path depend on `liblsl` (a
native library with its own Windows install story) would mean a teammate whose
LSL install is broken cannot run the test suite either. `LSLSource` exists
separately for when real hardware, or a genuine LSL round-trip, is what you want
to exercise.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from eegbot.constants import DSI7_CHANNELS, DSI7_SFREQ


class ArrayReplaySource:
    """Replay an in-memory ``(n_channels, n_times)`` array.

    Parameters
    ----------
    realtime
        When True, sleep so reads arrive at wall-clock pace -- use this to check
        the loop keeps up with 300 Hz. When False (default), data is served as
        fast as requested, which is what tests want.
    """

    def __init__(
        self,
        data: np.ndarray,
        sfreq: float = DSI7_SFREQ,
        channels: tuple[str, ...] = DSI7_CHANNELS,
        *,
        realtime: bool = False,
    ) -> None:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"expected (n_channels, n_times), got {data.shape}")
        if data.shape[0] != len(channels):
            raise ValueError(
                f"data has {data.shape[0]} channels but {len(channels)} names were given"
            )
        self.data = data
        self.sfreq = sfreq
        self.channels = channels
        self.realtime = realtime
        self._position = 0
        self._started = False

    def start(self) -> None:
        self._position = 0
        self._started = True

    def read(self, n_samples: int) -> np.ndarray | None:
        if not self._started:
            raise RuntimeError("call start() before read()")
        if self._position >= self.data.shape[-1]:
            return None
        stop = min(self._position + n_samples, self.data.shape[-1])
        chunk = self.data[:, self._position : stop]
        self._position = stop
        if self.realtime:
            time.sleep(chunk.shape[-1] / self.sfreq)
        return chunk

    def stop(self) -> None:
        self._started = False

    @property
    def exhausted(self) -> bool:
        return self._position >= self.data.shape[-1]


class FileReplaySource(ArrayReplaySource):
    """Replay a recording from disk (FIF or EDF).

    Channels are picked and reordered to canonical DSI-7 order on load, so a
    file saved with a different ordering cannot silently feed the decoder
    scrambled inputs.
    """

    def __init__(
        self,
        path: str | Path,
        channels: tuple[str, ...] = DSI7_CHANNELS,
        *,
        realtime: bool = False,
    ) -> None:
        import mne

        from eegbot.sigproc.montage import pick_canonical

        path = Path(path)
        if path.suffix == ".edf":
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        else:
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)

        raw = pick_canonical(raw, channels)
        super().__init__(
            raw.get_data(),
            sfreq=float(raw.info["sfreq"]),
            channels=channels,
            realtime=realtime,
        )
        self.path = path
