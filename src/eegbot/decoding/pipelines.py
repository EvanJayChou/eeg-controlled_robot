"""Decoder registry.

Training and evaluation code asks for a pipeline by name and never imports a
specific model, so swapping decoders is a config change.

## Why Riemannian first

Seven channels give a 7x7 covariance -- 28 free parameters. That is a small
enough feature space to fit reliably on a few hundred trials, which is all a
student BCI session will ever produce. Tangent-space mapping turns those SPD
matrices into ordinary Euclidean vectors, after which plain logistic regression
is not only sufficient but desirable: it trains in seconds and, importantly for
us, produces **calibrated probabilities**. The steering dial consumes
`predict_proba` directly, so a decoder that only emits hard labels would be
useless no matter how accurate.

CSP is kept as a comparison baseline rather than the primary. With two genuine
sensorimotor channels out of seven there is very little spatial structure for it
to find, and it has no probabilistic output of its own.

Deep models are deliberately absent. An untuned EEGNet on 120 trials will lose
to this and teach us nothing; revisit once the classical baseline is beaten on
real DSI-7 data.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy import signal as sp_signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from eegbot.decoding.align import RiemannRecenter, TraceNormalize
from eegbot.sigproc.spec import PreprocessSpec

#: Filter-bank edges for the CSP baselines, spanning mu and beta.
DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 30.0),
)


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Per-band CSP with log-variance features, concatenated.

    Implemented here rather than assembled from a `FeatureUnion` because each
    band needs its own temporal filter applied to the raw epochs before CSP
    sees them, which a union over a shared input cannot express.
    """

    def __init__(
        self,
        sfreq: float,
        bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS,
        n_components: int = 2,
    ) -> None:
        self.sfreq = sfreq
        self.bands = bands
        self.n_components = n_components

    def _filter(self, X: np.ndarray, band: tuple[float, float]) -> np.ndarray:
        sos = sp_signal.butter(4, band, btype="bandpass", fs=self.sfreq, output="sos")
        return sp_signal.sosfiltfilt(sos, X, axis=-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FilterBankCSP:
        from mne.decoding import CSP

        self.csps_ = []
        for band in self.bands:
            csp = CSP(n_components=self.n_components, log=True, norm_trace=False)
            csp.fit(self._filter(X, band), y)
            self.csps_.append(csp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        feats = [
            csp.transform(self._filter(X, band))
            for csp, band in zip(self.csps_, self.bands, strict=True)
        ]
        return np.concatenate(feats, axis=1)


# === Registry ===


def _riemann_ts_lr(spec: PreprocessSpec, *, align: bool = True) -> Pipeline:
    steps: list[tuple[str, object]] = [
        # OAS shrinkage rather than the empirical estimator: at a 1-second
        # window there are only ~300 samples per 7x7 covariance, and shrinkage
        # keeps the matrix well-conditioned enough for the matrix logarithm.
        ("cov", Covariances(estimator="oas")),
        ("trace", TraceNormalize()),
    ]
    if align:
        steps.append(("align", RiemannRecenter(metric="riemann")))
    steps += [
        ("ts", TangentSpace(metric="riemann")),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    ]
    return Pipeline(steps)


def _fbcsp_lda(spec: PreprocessSpec) -> Pipeline:
    return Pipeline(
        [
            ("fbcsp", FilterBankCSP(sfreq=spec.sfreq)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def _csp_lda(spec: PreprocessSpec) -> Pipeline:
    from mne.decoding import CSP

    return Pipeline(
        [
            ("csp", CSP(n_components=4, log=True, norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


REGISTRY: dict[str, Callable[[PreprocessSpec], Pipeline]] = {
    "riemann_ts_lr": lambda spec: _riemann_ts_lr(spec, align=True),
    "riemann_ts_lr_noalign": lambda spec: _riemann_ts_lr(spec, align=False),
    "fbcsp_lda": _fbcsp_lda,
    "csp_lda": _csp_lda,
}


def build(name: str, spec: PreprocessSpec) -> Pipeline:
    """Instantiate a decoder by name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown decoder {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](spec)


def available() -> list[str]:
    return sorted(REGISTRY)
