"""Cross-validation protocols.

## The leak this module exists to prevent

Cropped training turns one trial into ~20 heavily overlapping windows. Two crops
100 ms apart are near-duplicates. If some land in the training fold and others in
the test fold, the model is effectively tested on its training data and accuracy
climbs toward 95% -- a number that means nothing and that looks, from the outside,
like a breakthrough.

Every splitter here is therefore **group-aware on `EpochSet.groups`**, which
carries the source trial id. `assert_no_group_leak` makes the guarantee
checkable, and the test suite calls it on every protocol.

## The three protocols and what each answers

* **within_session** -- "can we decode this at all?" Optimistic; the easiest
  number and the least predictive of real use.
* **cross_session** -- "does yesterday's calibration still work today?" Train on
  session 1, test on session 2. This is the number that decides whether a rider
  has to recalibrate every time they put the headset on, and it is the one to
  quote when reporting progress.
* **loso** -- "could a new person use a model trained on others?" Leave one
  subject out; the ceiling for zero-calibration deployment.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from eegbot.datasets.base import EpochSet


@dataclass(frozen=True)
class Split:
    """One train/test division, with a label for reporting."""

    name: str
    train: np.ndarray
    test: np.ndarray


def assert_no_group_leak(epochs: EpochSet, split: Split) -> None:
    """Raise if any source trial appears on both sides of `split`.

    Cheap enough to run on every split in production, not just in tests.
    """
    shared = np.intersect1d(epochs.groups[split.train], epochs.groups[split.test])
    if shared.size:
        raise AssertionError(
            f"{split.name}: {shared.size} trial(s) appear in both train and test "
            f"(e.g. {shared[:5].tolist()}). Crops of one trial must never be split -- "
            f"this inflates accuracy and invalidates the result."
        )


def within_session(epochs: EpochSet, n_splits: int = 5, seed: int = 0) -> Iterator[Split]:
    """Stratified group k-fold within each session separately."""
    for session in np.unique(epochs.sessions):
        mask = epochs.sessions == session
        idx = np.flatnonzero(mask)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(
            cv.split(idx, epochs.y[idx], groups=epochs.groups[idx])
        ):
            yield Split(f"{session}/fold{fold}", idx[tr], idx[te])


def cross_session(epochs: EpochSet) -> Iterator[Split]:
    """Train on one session, test on each other session.

    No grouping needed -- sessions are disjoint sets of trials by construction --
    but the check runs anyway, because a harmonization bug that duplicated a run
    across sessions would otherwise pass silently.
    """
    sessions = np.unique(epochs.sessions)
    if len(sessions) < 2:
        return
    for train_session in sessions:
        train = np.flatnonzero(epochs.sessions == train_session)
        test = np.flatnonzero(epochs.sessions != train_session)
        yield Split(f"train:{train_session}", train, test)


def loso(epochs: EpochSet) -> Iterator[Split]:
    """Leave one subject out."""
    subjects = np.unique(epochs.subjects)
    if len(subjects) < 2:
        return
    for held_out in subjects:
        test = np.flatnonzero(epochs.subjects == held_out)
        train = np.flatnonzero(epochs.subjects != held_out)
        yield Split(f"holdout:{held_out}", train, test)


PROTOCOLS = {
    "within_session": within_session,
    "cross_session": cross_session,
    "loso": loso,
}


def get_protocol(name: str):
    if name not in PROTOCOLS:
        raise KeyError(f"unknown protocol {name!r}; available: {sorted(PROTOCOLS)}")
    return PROTOCOLS[name]
