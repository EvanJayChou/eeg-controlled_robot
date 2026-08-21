"""Session scheduling and the runner.

The runner is exercised headless end to end, which is the point: the protocol
gets debugged before a human subject is ever sitting under the headset.
"""

from __future__ import annotations

import numpy as np
import pytest

from eegbot.config import ParadigmConfig
from eegbot.constants import EVENT_IDS, MICROVOLTS_PER_VOLT
from eegbot.session import markers as mk
from eegbot.session.cues import HeadlessDisplay
from eegbot.session.markers import MemoryMarkerSink, TeeMarkerSink
from eegbot.session.protocol import (
    _max_streak,
    balanced_labels,
    build_schedule,
    estimate_duration_s,
)
from eegbot.session.recording import microvolts_to_volts, volts_to_microvolts
from eegbot.session.runner import SessionRunner

CONFIG = ParadigmConfig()


# === Units ===


def test_microvolt_volt_roundtrip():
    """The silent 1e6 bug.

    Catastrophic and invisible: most of the pipeline is scale-invariant, so a
    misplaced conversion still produces plausible accuracy while every amplitude
    threshold in the project is off by a million.
    """
    values = np.array([0.0, 1.0, -35.5, 150.0, 1e4])
    np.testing.assert_allclose(volts_to_microvolts(microvolts_to_volts(values)), values)


def test_conversion_direction_is_not_inverted():
    assert microvolts_to_volts(1.0) == pytest.approx(1e-6)
    assert volts_to_microvolts(1e-6) == pytest.approx(1.0)
    assert MICROVOLTS_PER_VOLT == 1e6


def test_realistic_eeg_lands_in_a_sane_volt_range():
    assert 1e-6 < microvolts_to_volts(30.0) < 1e-4


# === Scheduling ===


def test_labels_are_balanced_within_every_run():
    """Imbalance in a calibration run becomes a permanently off-center dial."""
    schedule = build_schedule(CONFIG, seed=0)
    for run in range(CONFIG.n_runs):
        labels = [t.label for t in schedule if t.run == run]
        assert len(labels) == CONFIG.trials_per_run
        counts = {code: labels.count(code) for code in EVENT_IDS.values()}
        assert len(set(counts.values())) == 1, f"run {run} is imbalanced: {counts}"


@pytest.mark.parametrize("seed", range(10))
def test_no_long_same_class_streaks(seed):
    """Subjects notice streaks and start predicting the cue."""
    schedule = build_schedule(CONFIG, seed=seed)
    for run in range(CONFIG.n_runs):
        labels = np.array([t.label for t in schedule if t.run == run])
        assert _max_streak(labels) <= CONFIG.max_consecutive_same


def test_rest_intervals_are_jittered():
    """A fixed ITI lets subjects entrain, producing anticipatory activity."""
    schedule = build_schedule(CONFIG, seed=0)
    rests = np.array([t.rest_s for t in schedule])
    lo, hi = CONFIG.rest_range_s
    assert rests.min() >= lo and rests.max() <= hi
    assert rests.std() > 0.1


def test_schedule_size_and_indexing():
    schedule = build_schedule(CONFIG, seed=0)
    assert len(schedule) == CONFIG.n_trials
    assert [t.index for t in schedule] == list(range(CONFIG.n_trials))


def test_balanced_labels_rejects_odd_counts():
    with pytest.raises(ValueError, match="must be even"):
        balanced_labels(7, 3, np.random.default_rng(0))


def test_impossible_streak_constraint_fails_loudly():
    """Better a clear error than a silent infinite loop."""
    with pytest.raises(RuntimeError, match="could not generate"):
        balanced_labels(20, 1, np.random.default_rng(0), max_attempts=50)


def test_paradigm_rejects_odd_trials_per_run():
    with pytest.raises(ValueError, match="must be even"):
        ParadigmConfig(trials_per_run=15)


def test_duration_estimate_is_plausible():
    minutes = estimate_duration_s(CONFIG) / 60
    assert 20 < minutes < 60


# === Markers ===


def test_memory_sink_records_in_order():
    sink = MemoryMarkerSink()
    sink.push("a")
    sink.push("b")
    assert sink.labels() == ["a", "b"]


def test_tee_sink_fans_out():
    """Live sessions keep a local copy in case the LSL recorder was not running."""
    a, b = MemoryMarkerSink(), MemoryMarkerSink()
    tee = TeeMarkerSink([a, b])
    tee.push("x")
    assert a.labels() == b.labels() == ["x"]


def test_cue_marker_maps_class_names():
    assert mk.cue_marker("left_hand") == mk.CUE_LEFT
    assert mk.cue_marker("right_hand") == mk.CUE_RIGHT


# === Full dry run ===


def small_config(**kwargs):
    defaults = dict(
        n_runs=2, trials_per_run=4, alpha_block_s=1.0, idle_block_s=1.0, instructions="test"
    )
    defaults.update(kwargs)
    return ParadigmConfig(**defaults)


def test_full_session_dry_run_emits_expected_markers():
    config = small_config()
    sink = MemoryMarkerSink()
    runner = SessionRunner(config, HeadlessDisplay(speed=0), sink, seed=0)

    log = runner.run()

    labels = sink.labels()
    assert log.completed_runs == 2
    assert log.completed_trials == 8

    assert labels.count(mk.RUN_START) == 2
    assert labels.count(mk.IMAGERY_START) == 8
    assert labels.count(mk.IMAGERY_END) == 8
    assert labels.count(mk.CUE_LEFT) + labels.count(mk.CUE_RIGHT) == 8
    # Alpha bookends at both ends, eyes-open and eyes-closed each time.
    assert labels.count(mk.ALPHA_EYES_OPEN) == 2
    assert labels.count(mk.ALPHA_EYES_CLOSED) == 2
    assert labels.count(mk.IDLE_BLOCK) == 1


def test_marker_order_within_a_trial():
    config = small_config(n_runs=1, trials_per_run=2)
    sink = MemoryMarkerSink()
    SessionRunner(config, HeadlessDisplay(speed=0), sink, seed=0).run()

    labels = sink.labels()
    start = labels.index(mk.RUN_START)
    trial = labels[start + 1 : start + 6]
    assert trial[0] == mk.FIXATION
    assert trial[1] in (mk.CUE_LEFT, mk.CUE_RIGHT)
    assert trial[2] == mk.IMAGERY_START
    assert trial[3] == mk.IMAGERY_END
    assert trial[4] == mk.REST


def test_idle_block_precedes_the_first_run():
    """The dead zone is fitted from idle data, so it must be collected first."""
    sink = MemoryMarkerSink()
    SessionRunner(small_config(), HeadlessDisplay(speed=0), sink, seed=0).run()
    labels = sink.labels()
    assert labels.index(mk.IDLE_BLOCK) < labels.index(mk.RUN_START)


def test_impedance_is_logged_before_every_run():
    config = small_config()
    probe_calls = []

    def probe():
        probe_calls.append(1)
        return {"C3": 8.0, "C4": 45.0}  # C4 deliberately bad

    runner = SessionRunner(
        config, HeadlessDisplay(speed=0), MemoryMarkerSink(), impedance_probe=probe
    )
    log = runner.run()

    assert len(probe_calls) == config.n_runs
    assert set(log.impedances) == {"run-0", "run-1"}
    assert log.impedances["run-0"]["C4"] == 45.0


def test_display_shows_arrows_matching_the_schedule():
    config = small_config(n_runs=1, trials_per_run=4)
    display = HeadlessDisplay(speed=0)
    SessionRunner(config, display, MemoryMarkerSink(), seed=0).run()

    arrows = [value for kind, value in display.shown if kind == "arrow"]
    assert len(arrows) == 4
    assert set(arrows) <= {"left", "right"}
    assert arrows.count("left") == arrows.count("right") == 2
