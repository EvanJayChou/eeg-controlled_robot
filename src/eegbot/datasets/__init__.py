"""Datasets, normalized to a common `EpochSet`.

Harmonization (re-referencing, channel selection, resampling) lives here rather
than in `sigproc` because it exists only to make foreign recordings resemble
DSI-7 output -- it has no online counterpart.
"""

from eegbot.datasets.base import EpochSet, concat
from eegbot.datasets.crops import crop_trials, sliding_windows

__all__ = ["EpochSet", "concat", "crop_trials", "sliding_windows"]
