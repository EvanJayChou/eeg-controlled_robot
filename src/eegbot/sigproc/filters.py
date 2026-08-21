"""Filter design and application.

The anti-skew invariant: **the offline and online paths share one filter
design.** `design_sos` is called by both; the only difference is direction of
application -- `sosfiltfilt` (zero-phase, offline) vs `sosfilt` (causal,
online). Any change to the response therefore lands on both paths at once.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from eegbot.sigproc.spec import PreprocessSpec

# === Design ===


def design_sos(spec: PreprocessSpec) -> np.ndarray:
    """Build the cascaded second-order sections for `spec`.

    Notch (if any) is stacked ahead of the bandpass into a single SOS array, so
    both stages share one filter state in the streaming case.

    Returns an array of shape ``(n_sections, 6)``.
    """
    sections: list[np.ndarray] = []

    if spec.notch is not None:
        sections.append(
            signal.butter(
                spec.filter_order,
                spec.notch,
                btype="bandstop",
                fs=spec.sfreq,
                output="sos",
            )
        )

    sections.append(
        signal.butter(
            spec.filter_order,
            spec.band,
            btype="bandpass",
            fs=spec.sfreq,
            output="sos",
        )
    )

    return np.vstack(sections)


def warmup_samples(spec: PreprocessSpec) -> int:
    """Samples the causal filter needs before its output is trustworthy.

    Driven by the high-pass corner: a 4 Hz corner settles in roughly a second.
    The online loop discards this much data after connecting.
    """
    low = spec.band[0]
    return int(np.ceil(3.0 * spec.sfreq / max(low, 0.1)))


# === Application ===


def filter_zero_phase(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Offline, non-causal filtering. Uses future samples -- never online."""
    padlen = 3 * (sos.shape[0] * 2)
    if x.shape[-1] <= padlen:
        raise ValueError(
            f"signal length {x.shape[-1]} too short for zero-phase filtering "
            f"(needs > {padlen} samples)"
        )
    return signal.sosfiltfilt(sos, x, axis=-1)


class CausalFilter:
    """Stateful causal filter for streaming chunks.

    Holds `sosfilt` state across calls so that chunk boundaries are invisible:
    filtering a signal in one call and in many small calls gives identical
    output. That equivalence is asserted in the test suite, because a stateless
    per-chunk filter is a subtle and very common online-BCI bug.
    """

    def __init__(self, spec: PreprocessSpec, n_channels: int | None = None) -> None:
        self.spec = spec
        self.sos = design_sos(spec)
        self.n_channels = n_channels if n_channels is not None else spec.n_channels
        self._zi: np.ndarray | None = None

    def reset(self, x0: np.ndarray | None = None) -> None:
        """Reset filter state.

        If `x0` (one sample per channel) is given, initialise to the steady
        state for that DC level instead of zeros. Zeros would inject a large
        startup transient that takes seconds to decay at a 4 Hz corner -- during
        which the decoder would see garbage.
        """
        zi_unit = signal.sosfilt_zi(self.sos)  # (n_sections, 2)
        if x0 is None:
            self._zi = np.zeros((self.sos.shape[0], self.n_channels, 2))
        else:
            x0 = np.asarray(x0, dtype=float).reshape(self.n_channels)
            self._zi = zi_unit[:, None, :] * x0[None, :, None]

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """Filter a ``(n_channels, n_times)`` chunk, advancing internal state."""
        chunk = np.asarray(chunk, dtype=float)
        if chunk.ndim != 2 or chunk.shape[0] != self.n_channels:
            raise ValueError(
                f"expected chunk of shape ({self.n_channels}, n_times), got {chunk.shape}"
            )
        if self._zi is None:
            self.reset(x0=chunk[:, 0])
        out, self._zi = signal.sosfilt(self.sos, chunk, axis=-1, zi=self._zi)
        return out


def filter_causal(spec: PreprocessSpec, x: np.ndarray) -> np.ndarray:
    """One-shot causal filtering of ``(n_channels, n_times)``.

    Convenience wrapper for offline simulation of the online path -- used by the
    train/serve skew check.
    """
    filt = CausalFilter(spec, n_channels=x.shape[0])
    filt.reset(x0=x[:, 0])
    return filt(x)
