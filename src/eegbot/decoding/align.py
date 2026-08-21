"""Riemannian alignment: the fix for session and subject shift.

## Why this is in the baseline and not in "future work"

Dry electrodes drift. Impedance changes as the subject's scalp warms, the
headset settles a few millimetres over an hour, and tomorrow's placement is
never quite today's. All of that moves the *whole cloud* of covariance matrices
to a new location on the SPD manifold while leaving the class structure inside
it largely intact.

Recentering exploits exactly that: whiten every session by its own geometric
mean, so each session's cloud lands at the identity and the classifier sees
comparable geometry regardless of where the raw data sat.

    C_aligned = G^{-1/2} C G^{-1/2}    where G is the session's Riemannian mean

The property that makes this practical is that **G needs no labels**. It is
computed from unlabelled data, so a new session can be aligned from the first
thirty seconds of recording before the subject has performed a single cued
trial. That is why `adapt` exists alongside `fit`: `fit` learns from training
data, `adapt` re-homes the transform onto a new session at inference time.

Ablate it with the `riemann_ts_lr_noalign` pipeline to see what it is worth on
any given dataset.
"""

from __future__ import annotations

import numpy as np
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_logeuclid, mean_riemann
from sklearn.base import BaseEstimator, TransformerMixin

#: Dispatched explicitly rather than via the generic `mean_covariance`, which
#: pyriemann has deprecated, and so that an unsupported metric name fails at
#: construction instead of deep inside a fit.
_MEANS = {
    "riemann": mean_riemann,
    "logeuclid": mean_logeuclid,
}


class TraceNormalize(BaseEstimator, TransformerMixin):
    """Scale each covariance matrix to unit trace.

    Removes per-trial and per-session amplitude differences, which on a dry
    headset can swamp the spatial structure we actually want. Total power varies
    with electrode contact; the *ratio* between C3 and C4 is the signal.
    """

    def fit(self, X: np.ndarray, y=None) -> TraceNormalize:  # noqa: ARG002
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        traces = np.trace(X, axis1=-2, axis2=-1)[..., None, None]
        return X / np.maximum(traces, 1e-20)


class RiemannRecenter(BaseEstimator, TransformerMixin):
    """Whiten covariances by a reference Riemannian mean.

    Parameters
    ----------
    metric
        Mean to use. ``"riemann"`` is the geometric mean (correct but iterative);
        ``"logeuclid"`` is a good deal faster and usually close enough if fitting
        becomes a bottleneck.
    """

    def __init__(self, metric: str = "riemann") -> None:
        if metric not in _MEANS:
            raise ValueError(f"unknown metric {metric!r}; available: {sorted(_MEANS)}")
        self.metric = metric

    def fit(self, X: np.ndarray, y=None) -> RiemannRecenter:  # noqa: ARG002
        self._set_reference(X)
        return self

    def adapt(self, X: np.ndarray) -> RiemannRecenter:
        """Re-home the transform onto a new session, without labels.

        Call with a buffer of covariances from the current session -- resting
        data is fine -- before decoding. This is the whole point of the module:
        the alignment target follows the session, while the classifier trained
        on top of it stays fixed.
        """
        self._set_reference(X)
        return self

    def _set_reference(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 3 or X.shape[-1] != X.shape[-2]:
            raise ValueError(f"expected (n_matrices, n_channels, n_channels), got {X.shape}")
        self.reference_ = _MEANS[self.metric](X)
        self.whitener_ = invsqrtm(self.reference_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "whitener_"):
            raise RuntimeError("call fit or adapt before transform")
        X = np.asarray(X, dtype=float)
        return self.whitener_ @ X @ self.whitener_
