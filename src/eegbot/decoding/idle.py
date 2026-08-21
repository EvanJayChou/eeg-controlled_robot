"""Idle detection -- the no-control state.

## Why the dead zone is not enough on its own

The original design assumed that a 2-class left/right decoder would sit near
p = 0.5 when the subject is resting, so a dead zone around the middle would
double as the idle state. Running the decoder over a full continuous recording
shows that assumption is false.

A discriminatively trained 2-class model has no representation of "neither". It
is optimised only to separate left from right, so a resting window -- which
resembles neither -- still gets pushed to a confident output, often past 0.95.
Widening the dead zone does not rescue it: at a threshold of 0.875 the decoder
was still active for 16% of rest time, by which point the subject can barely
issue a deliberate command either.

This is the **no-control state problem**, and it is well known in asynchronous
BCI. The standard fix is a second, orthogonal question:

    steering decoder:  "left or right?"        -- trained on imagery only
    idle gate:         "imagining, or not?"    -- trained on imagery vs rest

They are different discriminations and need different models. Rest differs from
imagery mainly in *overall* mu/beta power (both hemispheres desynchronize during
imagery), whereas left differs from right in the *lateralization* of that
power. A single 2-class model cannot express both.

The gate consumes the 60-second idle block the session runner already records,
so it costs no extra data collection.
"""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from eegbot.datasets.crops import sliding_windows
from eegbot.decoding.align import TraceNormalize
from eegbot.sigproc.spec import PreprocessSpec

#: Label for windows during which the subject was imagining.
ENGAGED = 1
#: Label for windows during which the subject was at rest.
RESTING = 0


def build_gate(spec: PreprocessSpec) -> Pipeline:
    """Rest-vs-imagery classifier.

    Note the deliberate absence of `TraceNormalize` compared with the steering
    pipeline. Unit-trace normalization discards total power, which is precisely
    the feature separating rest from imagery -- normalizing it away would leave
    the gate nothing to work with. The steering decoder wants it gone (amplitude
    varies with electrode contact); the gate needs it kept. Same reason they
    cannot be one model.
    """
    return Pipeline(
        [
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
        ]
    )


def build_gate_trace_normalized(spec: PreprocessSpec) -> Pipeline:
    """Variant that does normalize trace.

    Worth benchmarking on real data: if session-to-session amplitude drift on the
    dry headset turns out to dominate the rest/imagery power difference, the
    normalized version may generalize better across sessions even though it is
    weaker within one.
    """
    return Pipeline(
        [
            ("cov", Covariances(estimator="oas")),
            ("trace", TraceNormalize()),
            ("ts", TangentSpace(metric="riemann")),
            ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
        ]
    )


def windows_from_mask(
    spec: PreprocessSpec,
    data: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Slide windows over `data`, keeping those lying entirely inside `mask`.

    Parameters
    ----------
    data
        ``(n_channels, n_times)``, already filtered.
    mask
        ``(n_times,)`` boolean. A window is kept only if every sample it covers
        is True, so windows straddling a rest/imagery boundary -- which belong to
        neither class -- are dropped rather than mislabelled.
    """
    data = np.asarray(data, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[-1] != data.shape[-1]:
        raise ValueError(f"mask length {mask.shape[-1]} != data length {data.shape[-1]}")

    windows = sliding_windows(data, spec.window_samples, spec.hop_samples)
    mask_windows = sliding_windows(mask[None, :].astype(float), spec.window_samples, spec.hop_samples)
    keep = mask_windows[:, 0, :].all(axis=-1)
    return windows[keep]


def build_gate_dataset(
    spec: PreprocessSpec,
    data: np.ndarray,
    events: np.ndarray,
    *,
    imagery_s: float = 4.0,
    rest_guard_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble rest-vs-imagery training data from a continuous recording.

    Parameters
    ----------
    rest_guard_s
        Seconds excluded on either side of each imagery period before calling
        the remainder "rest". Post-imagery beta rebound and pre-cue anticipation
        are neither clean rest nor clean imagery, and labelling them as rest
        teaches the gate to fire on them.
    """
    data = np.asarray(data, dtype=float)
    n_times = data.shape[-1]

    imagery_len = int(round(imagery_s * spec.sfreq))
    guard = int(round(rest_guard_s * spec.sfreq))

    engaged = np.zeros(n_times, dtype=bool)
    excluded = np.zeros(n_times, dtype=bool)
    for onset in events[:, 0]:
        start, stop = int(onset), int(onset) + imagery_len
        engaged[max(start, 0) : min(stop, n_times)] = True
        excluded[max(start - guard, 0) : min(stop + guard, n_times)] = True

    resting = ~excluded

    engaged_windows = windows_from_mask(spec, data, engaged)
    resting_windows = windows_from_mask(spec, data, resting)

    if len(engaged_windows) == 0 or len(resting_windows) == 0:
        raise ValueError(
            f"need both classes to train the gate; got {len(engaged_windows)} engaged "
            f"and {len(resting_windows)} resting windows"
        )

    X = np.concatenate([engaged_windows, resting_windows], axis=0)
    y = np.concatenate(
        [
            np.full(len(engaged_windows), ENGAGED),
            np.full(len(resting_windows), RESTING),
        ]
    )
    return X, y


class IdleGate:
    """Wraps a fitted rest/imagery model and exposes an engagement probability."""

    def __init__(self, spec: PreprocessSpec, model: Pipeline | None = None) -> None:
        self.spec = spec
        self.model = model if model is not None else build_gate(spec)

    def fit(self, X: np.ndarray, y: np.ndarray) -> IdleGate:
        self.model.fit(X, y)
        self._engaged_index = list(self.model.classes_).index(ENGAGED)
        return self

    def engagement(self, X: np.ndarray) -> np.ndarray:
        """p(engaged) for each window."""
        return self.model.predict_proba(X)[:, self._engaged_index]

    def engagement_one(self, window: np.ndarray) -> float:
        """p(engaged) for a single ``(n_channels, n_times)`` window."""
        return float(self.engagement(window[None, ...])[0])
