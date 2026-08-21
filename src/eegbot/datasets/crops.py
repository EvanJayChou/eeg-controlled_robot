"""Cropped training: slice trials into the windows the decoder will actually see.

## Why this exists

The online loop classifies a 1-second sliding window every 100 ms. If the model
were trained on 3-second trial epochs instead, training and deployment would see
different input distributions -- different signal length, different variance,
different covariance conditioning -- and the resulting degradation is invisible
offline. Your benchmark stays healthy while the live decoder quietly gets worse.

So we train on exactly what we deploy on. Each trial's imagery period is sliced
into overlapping `spec.window_s` crops at `spec.hop_s`, and the model sees only
those. A useful side effect at 120 trials/session: roughly 20x more training
rows, which matters a great deal when trials are this scarce.

## The trap this module is careful about

Overlapping crops from one trial are near-duplicates. Letting them fall on both
sides of a CV split leaks the test set into training and inflates accuracy to
the point of meaninglessness. Every crop therefore carries its source trial id
in `EpochSet.groups`, and evaluation splits on that.
"""

from __future__ import annotations

import numpy as np

from eegbot.datasets.base import EpochSet
from eegbot.sigproc.spec import PreprocessSpec


def sliding_windows(x: np.ndarray, window: int, hop: int) -> np.ndarray:
    """Cut ``(n_channels, n_times)`` into ``(n_windows, n_channels, window)``.

    Uses a strided view internally, so this is cheap even for long recordings;
    the result is copied so downstream in-place ops stay safe.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (n_channels, n_times), got {x.shape}")
    n_times = x.shape[-1]
    if n_times < window:
        raise ValueError(f"signal of {n_times} samples is shorter than window {window}")

    n_windows = 1 + (n_times - window) // hop
    strides = (x.strides[-1] * hop, x.strides[0], x.strides[-1])
    view = np.lib.stride_tricks.as_strided(
        x, shape=(n_windows, x.shape[0], window), strides=strides, writeable=False
    )
    return view.copy()


def crop_trials(
    spec: PreprocessSpec,
    trials: np.ndarray,
    y: np.ndarray,
    *,
    subjects: np.ndarray | str | int = 0,
    sessions: np.ndarray | str | int = 0,
    trial_ids: np.ndarray | None = None,
) -> EpochSet:
    """Expand trial epochs into overlapping decoding windows.

    Parameters
    ----------
    trials
        ``(n_trials, n_channels, n_trial_samples)``, already filtered. Filtering
        must happen on the continuous signal *before* cropping -- see
        `eegbot.sigproc.apply`.
    y
        ``(n_trials,)`` labels; each crop inherits its trial's label.
    trial_ids
        Optional explicit ids. Defaults to positional index. Every crop of a
        trial carries that trial's id in `EpochSet.groups`.
    """
    trials = np.asarray(trials, dtype=float)
    if trials.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_times), got {trials.shape}")

    y = np.asarray(y)
    if y.shape != (trials.shape[0],):
        raise ValueError(f"y must have shape ({trials.shape[0]},), got {y.shape}")

    n_trials = trials.shape[0]
    if trial_ids is None:
        trial_ids = np.arange(n_trials)
    trial_ids = np.asarray(trial_ids)

    subjects_arr = _broadcast(subjects, n_trials, "subjects")
    sessions_arr = _broadcast(sessions, n_trials, "sessions")

    window, hop = spec.window_samples, spec.hop_samples

    crops_list, y_list, groups_list, subj_list, sess_list = [], [], [], [], []
    for i in range(n_trials):
        crops = sliding_windows(trials[i], window, hop)
        n = crops.shape[0]
        crops_list.append(crops)
        y_list.append(np.full(n, y[i]))
        groups_list.append(np.full(n, trial_ids[i]))
        subj_list.append(np.full(n, subjects_arr[i]))
        sess_list.append(np.full(n, sessions_arr[i]))

    return EpochSet(
        X=np.concatenate(crops_list, axis=0),
        y=np.concatenate(y_list, axis=0),
        groups=np.concatenate(groups_list, axis=0),
        subjects=np.concatenate(subj_list, axis=0),
        sessions=np.concatenate(sess_list, axis=0),
        spec=spec,
        meta={"n_source_trials": n_trials, "crops_per_trial": spec.crops_per_trial},
    )


def _broadcast(value: np.ndarray | str | int, n: int, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {value.shape}")
        return value
    return np.full(n, value)
