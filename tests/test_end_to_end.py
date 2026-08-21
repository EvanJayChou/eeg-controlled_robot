"""End-to-end decoding on synthetic data.

Requires pyriemann. These are the tests that would catch a broken decoder, a
broken online loop, or -- most importantly -- a train/serve mismatch between the
two.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyriemann")

from eegbot.config import ControlConfig  # noqa: E402
from eegbot.constants import EVENT_IDS, POSITIVE_CLASS  # noqa: E402
from eegbot.control.controller import ContinuousController  # noqa: E402
from eegbot.datasets.crops import crop_trials  # noqa: E402
from eegbot.decoding.pipelines import available, build  # noqa: E402
from eegbot.evaluation.metrics import results_frame, score_split  # noqa: E402
from eegbot.evaluation.protocols import within_session  # noqa: E402
from eegbot.sigproc.apply import preprocess  # noqa: E402
from eegbot.sigproc.spec import PreprocessSpec  # noqa: E402
from eegbot.stream.loop import OnlineDecoder  # noqa: E402
from eegbot.stream.replay import ArrayReplaySource  # noqa: E402
from eegbot.stream.synthetic import synthetic_mi_recording  # noqa: E402

SPEC = PreprocessSpec()


def build_epochs(n_trials=60, seed=0, erd_depth=0.45, causal=False):
    data, events, labels = synthetic_mi_recording(
        SPEC, n_trials=n_trials, seed=seed, erd_depth=erd_depth
    )
    filtered = preprocess(SPEC, data, causal=causal)
    start_off = int(round(SPEC.trial_window[0] * SPEC.sfreq))
    trials = np.stack(
        [
            filtered[:, int(o) + start_off : int(o) + start_off + SPEC.trial_samples]
            for o in events[:, 0]
        ]
    )
    epochs = crop_trials(SPEC, trials, labels, subjects="synthetic", sessions="ses-0")
    return epochs, data


def positive_index(model):
    return list(model.classes_).index(EVENT_IDS[POSITIVE_CLASS])


# === The M1 gate ===


def test_pipeline_recovers_a_planted_erd():
    """If we cannot decode a signal we injected ourselves, the bug is ours."""
    epochs, _ = build_epochs()
    model = build("riemann_ts_lr", SPEC)
    results = [score_split(model, epochs, s) for s in within_session(epochs, n_splits=5)]
    df = results_frame(results)
    assert df["trial_acc"].mean() > 0.90, f"got {df['trial_acc'].mean():.3f}"


def test_weak_effect_degrades_but_stays_above_chance():
    """A poor subject should degrade gracefully, not break the pipeline."""
    epochs, _ = build_epochs(erd_depth=0.12, n_trials=80)
    model = build("riemann_ts_lr", SPEC)
    results = [score_split(model, epochs, s) for s in within_session(epochs, n_splits=5)]
    accuracy = results_frame(results)["crop_acc"].mean()
    assert 0.45 < accuracy < 0.95


def test_no_effect_gives_chance_performance():
    """The most important negative control.

    With zero planted effect the decoder must land at chance. Anything well
    above 0.5 here would mean the evaluation itself leaks.
    """
    epochs, _ = build_epochs(erd_depth=0.0, n_trials=80)
    model = build("riemann_ts_lr", SPEC)
    results = [score_split(model, epochs, s) for s in within_session(epochs, n_splits=5)]
    accuracy = results_frame(results)["crop_acc"].mean()
    assert 0.38 < accuracy < 0.62, f"chance-level data decoded at {accuracy:.3f}"


@pytest.mark.parametrize("name", ["riemann_ts_lr", "riemann_ts_lr_noalign", "csp_lda"])
def test_registered_decoders_all_run(name):
    epochs, _ = build_epochs(n_trials=40)
    model = build(name, SPEC)
    result = score_split(model, epochs, next(within_session(epochs, n_splits=3)))
    assert 0.0 <= result.crop_accuracy <= 1.0
    assert 0.0 <= result.auc <= 1.0


def test_registry_lists_expected_decoders():
    assert "riemann_ts_lr" in available()
    assert "fbcsp_lda" in available()


def test_unknown_decoder_fails_clearly():
    with pytest.raises(KeyError, match="unknown decoder"):
        build("does_not_exist", SPEC)


# === Train/serve skew ===


def test_offline_and_causal_paths_agree():
    """The skew check.

    Train on zero-phase-filtered crops, then score the same trials filtered
    causally as the online loop would. A large gap means offline numbers do not
    predict online behaviour, which is the failure this architecture exists to
    prevent.
    """
    offline_epochs, _ = build_epochs(n_trials=60, causal=False)
    causal_epochs, _ = build_epochs(n_trials=60, causal=True)

    model = build("riemann_ts_lr", SPEC)
    model.fit(offline_epochs.X, offline_epochs.y)

    offline_accuracy = model.score(offline_epochs.X, offline_epochs.y)
    causal_accuracy = model.score(causal_epochs.X, causal_epochs.y)

    assert causal_accuracy > 0.65, f"causal path collapsed to {causal_accuracy:.3f}"
    assert offline_accuracy - causal_accuracy < 0.20, (
        f"train/serve gap {offline_accuracy - causal_accuracy:.3f} is too large; "
        f"the offline numbers would not predict online behaviour"
    )


# === Online loop ===


def test_online_loop_produces_steady_updates():
    epochs, data = build_epochs(n_trials=30)
    model = build("riemann_ts_lr", SPEC)
    model.fit(epochs.X, epochs.y)

    controller = ContinuousController(ControlConfig(), SPEC.hop_s)
    decoder = OnlineDecoder(
        SPEC, model, controller, positive_index=positive_index(model), align_seconds=0.0
    )
    updates = decoder.run(ArrayReplaySource(data), max_updates=50)

    assert len(updates) == 50
    assert all(0.0 <= u.p_right <= 1.0 for u in updates)
    assert all(-1.0 <= u.command.command <= 1.0 for u in updates)
    np.testing.assert_allclose(
        [u.time_s for u in updates], np.arange(50) * SPEC.hop_s, atol=1e-9
    )


def test_online_loop_steers_correctly_during_imagery():
    """The payoff test: the dial should lean the right way during real trials."""
    epochs, data = build_epochs(n_trials=40, erd_depth=0.6)
    model = build("riemann_ts_lr", SPEC)
    model.fit(epochs.X, epochs.y)

    _, events, labels = synthetic_mi_recording(SPEC, n_trials=40, seed=0, erd_depth=0.6)

    controller = ContinuousController(ControlConfig(), SPEC.hop_s)
    decoder = OnlineDecoder(
        SPEC, model, controller, positive_index=positive_index(model), align_seconds=0.0
    )
    updates = decoder.run(ArrayReplaySource(data))

    from eegbot.sigproc.filters import warmup_samples

    offset = warmup_samples(SPEC)
    right_intent, left_intent = [], []
    for update in updates:
        sample = offset + int(update.time_s * SPEC.sfreq)
        matching = events[(events[:, 0] <= sample) & (sample < events[:, 0] + 4 * SPEC.sfreq)]
        if matching.size == 0:
            continue
        if matching[0, 2] == EVENT_IDS["right_hand"]:
            right_intent.append(update.command.intent)
        else:
            left_intent.append(update.command.intent)

    assert right_intent and left_intent
    assert np.mean(right_intent) > np.mean(left_intent)


def test_online_loop_stops_when_source_is_exhausted():
    epochs, data = build_epochs(n_trials=5)
    model = build("riemann_ts_lr", SPEC)
    model.fit(epochs.X, epochs.y)

    controller = ContinuousController(ControlConfig(), SPEC.hop_s)
    decoder = OnlineDecoder(
        SPEC, model, controller, positive_index=positive_index(model), align_seconds=0.0
    )
    updates = decoder.run(ArrayReplaySource(data), max_updates=100_000)
    assert 0 < len(updates) < 100_000


# === Alignment ===


def test_alignment_adapts_without_labels():
    """Recentering must work from unlabelled data -- that is its whole value."""
    from pyriemann.estimation import Covariances

    from eegbot.decoding.align import RiemannRecenter

    epochs, _ = build_epochs(n_trials=30)
    covariances = Covariances(estimator="oas").fit_transform(epochs.X)

    recenter = RiemannRecenter().fit(covariances)
    first = recenter.reference_.copy()

    shifted = covariances * 4.0
    recenter.adapt(shifted)  # no labels passed

    # atol=0 is essential here. Data is in volts, so these covariances are
    # ~1e-10 -- far below np.allclose's default atol of 1e-8, which would make
    # the assertion pass no matter what the values were. Any volt-scale array
    # comparison in this project needs an explicit tolerance.
    assert not np.allclose(first, recenter.reference_, rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(recenter.reference_, first * 4.0, rtol=1e-6)

    # After recentering, a scaled session should map to nearly the same place.
    np.testing.assert_allclose(
        recenter.transform(shifted).mean(axis=0),
        RiemannRecenter().fit(covariances).transform(covariances).mean(axis=0),
        atol=0.05,
    )
