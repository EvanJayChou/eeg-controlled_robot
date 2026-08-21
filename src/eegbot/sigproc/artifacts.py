"""Amplitude-based artifact screening.

Deliberately simple. With 7 channels there is no room for ICA -- you cannot
separate 7 mixed sources into meaningful components, and attempting it on this
montage would remove signal along with the artifact. Peak-to-peak rejection and
flat-channel detection catch the failures that actually matter on a dry headset:
electrode pop, cable movement, and a sensor that has lost contact entirely.

All arrays are in **volts**; thresholds in `PreprocessSpec` are in microvolts
and are converted here.
"""

from __future__ import annotations

import numpy as np

from eegbot.constants import MICROVOLTS_PER_VOLT
from eegbot.sigproc.spec import PreprocessSpec


def peak_to_peak(x: np.ndarray) -> np.ndarray:
    """Per-channel peak-to-peak amplitude over the last axis."""
    return np.ptp(x, axis=-1)


def reject_mask(spec: PreprocessSpec, epochs: np.ndarray) -> np.ndarray:
    """Boolean mask of epochs to **keep**.

    Parameters
    ----------
    epochs
        ``(n_epochs, n_channels, n_times)`` in volts.

    Returns
    -------
    ``(n_epochs,)`` boolean array, True where the epoch passes screening.
    """
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 3:
        raise ValueError(f"expected (n_epochs, n_channels, n_times), got {epochs.shape}")

    keep = np.ones(epochs.shape[0], dtype=bool)
    if spec.reject_p2p_uv is None:
        return keep

    threshold_v = spec.reject_p2p_uv / MICROVOLTS_PER_VOLT
    ptp = peak_to_peak(epochs)  # (n_epochs, n_channels)
    keep &= ~np.any(ptp > threshold_v, axis=1)
    return keep


def flat_channels(spec: PreprocessSpec, x: np.ndarray) -> list[str]:
    """Names of channels that look disconnected.

    Run this on the alpha bookend blocks during a session. A flat channel is a
    session-stopping problem and it is far cheaper to catch it while the subject
    is still wearing the headset.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (n_channels, n_times), got {x.shape}")

    threshold_v = spec.flat_p2p_uv / MICROVOLTS_PER_VOLT
    ptp = peak_to_peak(x)
    return [name for name, amp in zip(spec.channels, ptp, strict=True) if amp < threshold_v]
