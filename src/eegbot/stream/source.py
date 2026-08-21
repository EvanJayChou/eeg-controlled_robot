"""EEG source interface.

One protocol, several implementations, so the online loop is identical whether
it is fed by a real headset, a replayed recording, or synthetic data. The
headset is on order; everything downstream of this interface is nonetheless
fully exercisable today, which is the point.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EEGSource(Protocol):
    """A stream of DSI-7-shaped samples.

    Implementations yield ``(n_channels, n_samples)`` chunks in **volts**, with
    channels in canonical order.
    """

    sfreq: float
    channels: tuple[str, ...]

    def start(self) -> None:
        """Open the stream. Idempotent."""
        ...

    def read(self, n_samples: int) -> np.ndarray | None:
        """Return up to `n_samples` of data, or None when exhausted.

        A live source blocks until enough samples arrive; a finite source
        returns None at the end of its data.
        """
        ...

    def stop(self) -> None:
        """Close the stream. Idempotent."""
        ...
