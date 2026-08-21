"""Crop construction and the leak it makes possible.

The leak test is the most important one in the suite. Overlapping crops from a
single trial are near-duplicates; if they land on both sides of a CV split, the
reported accuracy is fiction. Everything here exists to make that impossible to
introduce silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from eegbot.constants import EVENT_IDS
from eegbot.datasets.base import EpochSet, concat
from eegbot.datasets.crops import crop_trials, sliding_windows
from eegbot.evaluation.protocols import (
    assert_no_group_leak,
    cross_session,
    loso,
    within_session,
)
from eegbot.sigproc.spec import PreprocessSpec

SPEC = PreprocessSpec()


def make_trials(n_trials=24, seed=0):
    rng = np.random.default_rng(seed)
    trials = rng.standard_normal((n_trials, SPEC.n_channels, SPEC.trial_samples)) * 1e-5
    labels = np.array([list(EVENT_IDS.values())[i % 2] for i in range(n_trials)])
    return trials, labels


# === Windowing ===


def test_sliding_windows_shape_and_content():
    x = np.arange(2 * 100, dtype=float).reshape(2, 100)
    windows = sliding_windows(x, window=10, hop=5)
    assert windows.shape == (19, 2, 10)
    np.testing.assert_array_equal(windows[0], x[:, 0:10])
    np.testing.assert_array_equal(windows[1], x[:, 5:15])


def test_sliding_windows_rejects_short_signal():
    with pytest.raises(ValueError, match="shorter than window"):
        sliding_windows(np.zeros((2, 5)), window=10, hop=5)


def test_crops_per_trial_matches_the_spec():
    trials, labels = make_trials()
    epochs = crop_trials(SPEC, trials, labels)
    assert len(epochs) == len(trials) * SPEC.crops_per_trial
    assert epochs.X.shape[-1] == SPEC.window_samples


def test_every_crop_inherits_its_trial_label_and_group():
    trials, labels = make_trials(n_trials=4)
    epochs = crop_trials(SPEC, trials, labels)
    for trial_index in range(4):
        mask = epochs.groups == trial_index
        assert mask.sum() == SPEC.crops_per_trial
        assert set(np.unique(epochs.y[mask])) == {labels[trial_index]}


def test_n_trials_reports_source_trials_not_crops():
    """The real sample size. Confusing the two badly overstates statistical power."""
    trials, labels = make_trials(n_trials=10)
    epochs = crop_trials(SPEC, trials, labels)
    assert epochs.n_trials == 10
    assert len(epochs) > 10


# === Leakage ===


@pytest.mark.parametrize("protocol", [within_session, cross_session, loso])
def test_no_protocol_splits_a_trial(protocol):
    sets = []
    for subject in range(3):
        for session in range(2):
            trials, labels = make_trials(n_trials=12, seed=subject * 10 + session)
            sets.append(
                crop_trials(
                    SPEC,
                    trials,
                    labels,
                    subjects=f"sub-{subject}",
                    sessions=f"ses-{session}",
                )
            )
    epochs = concat(sets)

    splits = list(protocol(epochs))
    assert splits, "protocol produced no splits"
    for split in splits:
        assert_no_group_leak(epochs, split)


def test_leak_detector_actually_detects_a_leak():
    """Guard against a check that silently passes everything."""
    from eegbot.evaluation.protocols import Split

    trials, labels = make_trials(n_trials=8)
    epochs = crop_trials(SPEC, trials, labels)

    n = len(epochs)
    naive = Split("naive", np.arange(0, n, 2), np.arange(1, n, 2))

    with pytest.raises(AssertionError, match="appear in both train and test"):
        assert_no_group_leak(epochs, naive)


def test_concat_keeps_trial_ids_unique_across_subjects():
    """Without offsetting, trial 0 of two subjects would look like one trial."""
    a = crop_trials(SPEC, *make_trials(n_trials=6, seed=1), subjects="sub-a")
    b = crop_trials(SPEC, *make_trials(n_trials=6, seed=2), subjects="sub-b")
    merged = concat([a, b])
    assert merged.n_trials == 12
    assert np.intersect1d(merged.groups[: len(a)], merged.groups[len(a) :]).size == 0


def test_concat_rejects_mismatched_specs():
    from dataclasses import replace

    a = crop_trials(SPEC, *make_trials(n_trials=4))
    other = replace(SPEC, window_s=0.5)
    b = crop_trials(other, *make_trials(n_trials=4))
    with pytest.raises(ValueError, match="different PreprocessSpec"):
        concat([a, b])


# === EpochSet validation ===


def test_epochset_rejects_channel_count_mismatch():
    with pytest.raises(ValueError, match="channels but spec expects"):
        EpochSet(
            X=np.zeros((5, 3, SPEC.window_samples)),
            y=np.zeros(5),
            groups=np.zeros(5),
            subjects=np.zeros(5),
            sessions=np.zeros(5),
            spec=SPEC,
        )


def test_epochset_rejects_misaligned_metadata():
    with pytest.raises(ValueError, match="to match X"):
        EpochSet(
            X=np.zeros((5, SPEC.n_channels, SPEC.window_samples)),
            y=np.zeros(4),
            groups=np.zeros(5),
            subjects=np.zeros(5),
            sessions=np.zeros(5),
            spec=SPEC,
        )
