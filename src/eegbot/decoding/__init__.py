"""Decoders and their per-subject calibration.

`calibrate` is re-exported eagerly because it is pure numpy. The pipeline and
alignment modules depend on pyriemann and are imported lazily, so that control
and evaluation code stays importable in an environment without it.
"""

from typing import Any

from eegbot.decoding.calibrate import Calibration, calibrate_from_idle

__all__ = [
    "Calibration",
    "RiemannRecenter",
    "TraceNormalize",
    "available",
    "build",
    "calibrate_from_idle",
]

_LAZY = {
    "RiemannRecenter": "eegbot.decoding.align",
    "TraceNormalize": "eegbot.decoding.align",
    "available": "eegbot.decoding.pipelines",
    "build": "eegbot.decoding.pipelines",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
