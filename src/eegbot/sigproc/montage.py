"""DSI-7 montage handling.

Canonical channel order is enforced here and nowhere else. Every array with a
channel axis in this codebase is ordered as `constants.DSI7_CHANNELS`; a
covariance-based decoder is order-sensitive, so a silent reordering between
training and inference would be a genuine (and very hard to spot) bug.
"""

from __future__ import annotations

import mne
import numpy as np

from eegbot.constants import DSI7_CHANNELS, DSI7_SFREQ


def dsi7_info(sfreq: float = DSI7_SFREQ, channels: tuple[str, ...] = DSI7_CHANNELS) -> mne.Info:
    """Build an MNE Info for the DSI-7 with standard 10-20 electrode positions."""
    info = mne.create_info(ch_names=list(channels), sfreq=sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    return info


def pick_canonical(
    raw: mne.io.BaseRaw,
    channels: tuple[str, ...] = DSI7_CHANNELS,
) -> mne.io.BaseRaw:
    """Reduce `raw` to `channels`, in canonical order.

    Raises if any expected channel is absent, rather than silently returning a
    narrower montage that would then mismatch the model's expected input.
    """
    missing = [ch for ch in channels if ch not in raw.ch_names]
    if missing:
        raise ValueError(
            f"channels missing from recording: {missing}. Present: {raw.ch_names}"
        )
    return raw.copy().pick(list(channels)).reorder_channels(list(channels))


def as_canonical_array(raw: mne.io.BaseRaw, channels: tuple[str, ...] = DSI7_CHANNELS) -> np.ndarray:
    """Extract ``(n_channels, n_times)`` in canonical order, in volts."""
    return pick_canonical(raw, channels).get_data()
