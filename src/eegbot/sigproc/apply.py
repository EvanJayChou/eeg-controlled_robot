"""The single preprocessing entry point.

## Why the offline/online split lands where it does

Two distinct stages, and conflating them is how train/serve skew gets in:

**Harmonization** -- re-referencing, channel selection, resampling. This exists
only to make a *foreign* dataset look like DSI-7 output. It has no online
counterpart because the headset already delivers 7 canonical channels at 300 Hz
against ear clips. It lives in `eegbot.datasets`, not here.

**Preprocessing** -- notch, bandpass, artifact screening. This runs on every
sample the decoder ever sees, offline and online alike, and therefore lives
here behind one function.

## Order of operations

Filter continuous data *before* cutting it into crops, on both paths. Filtering
each 1-second crop independently would give every crop its own edge transient
and is not what the online loop does. `preprocess` refuses causal filtering of
pre-cut epochs for exactly this reason -- it turns a silent distribution shift
into an exception.
"""

from __future__ import annotations

import numpy as np

from eegbot.sigproc.filters import design_sos, filter_causal, filter_zero_phase
from eegbot.sigproc.spec import PreprocessSpec


def preprocess(
    spec: PreprocessSpec,
    x: np.ndarray,
    *,
    causal: bool = False,
) -> np.ndarray:
    """Apply `spec`'s notch and bandpass to `x`.

    Parameters
    ----------
    spec
        The shared preprocessing specification.
    x
        Continuous data as ``(n_channels, n_times)``, in volts. A 3-D array of
        pre-cut epochs is accepted only when ``causal=False``.
    causal
        ``False`` (default) applies zero-phase filtering, valid offline only.
        ``True`` applies the causal filter, matching the online path.

    Returns
    -------
    Filtered array with the same shape as `x`.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 3:
        if causal:
            raise ValueError(
                "causal=True on pre-cut epochs does not match the online path, which "
                "filters continuously and then windows. Filter the continuous signal "
                "first, then crop -- see eegbot.datasets.crops."
            )
        sos = design_sos(spec)
        return filter_zero_phase(sos, x)

    if x.ndim != 2:
        raise ValueError(f"expected (n_channels, n_times) or (n_epochs, n_channels, n_times), got {x.shape}")

    if x.shape[0] != spec.n_channels:
        raise ValueError(
            f"expected {spec.n_channels} channels ({', '.join(spec.channels)}), "
            f"got {x.shape[0]}"
        )

    if causal:
        return filter_causal(spec, x)
    return filter_zero_phase(design_sos(spec), x)
