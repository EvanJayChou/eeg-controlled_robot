"""The M1 gate: can the pipeline recover a signal we planted ourselves?

If these fail, the bug is in our code and there is no point looking at real EEG.
The check is deliberately made without any classifier -- it measures band power
directly, so a failure points at the signal path rather than at the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from eegbot.constants import EVENT_IDS
from eegbot.datasets.crops import crop_trials
from eegbot.sigproc.apply import preprocess
from eegbot.sigproc.spec import PreprocessSpec
from eegbot.stream.synthetic import synthetic_mi_recording

SPEC = PreprocessSpec()


def mu_power(x: np.ndarray, sfreq: float, band=(8.0, 13.0)) -> np.ndarray:
    """Mean power in `band` over the last axis, per channel."""
    spectrum = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], 1 / sfreq)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return spectrum[..., mask].mean(axis=-1)


def build_trials(spec=SPEC, **kwargs):
    data, events, labels = synthetic_mi_recording(spec, **kwargs)
    filtered = preprocess(spec, data, causal=False)

    tmin, _ = spec.trial_window
    start_off = int(round(tmin * spec.sfreq))
    trials = np.stack(
        [
            filtered[:, int(onset) + start_off : int(onset) + start_off + spec.trial_samples]
            for onset in events[:, 0]
        ]
    )
    return trials, labels


def test_contralateral_erd_is_present():
    """Right-hand imagery must suppress mu at C3, left-hand at C4.

    This is the physiological claim the whole project rests on. If the planted
    version does not survive the pipeline, harmonization or filtering is broken.
    """
    trials, labels = build_trials(n_trials=60, seed=0)
    c3 = SPEC.channels.index("C3")
    c4 = SPEC.channels.index("C4")

    power = mu_power(trials, SPEC.sfreq)
    right = labels == EVENT_IDS["right_hand"]
    left = labels == EVENT_IDS["left_hand"]

    # C3 (left hemisphere) desynchronizes for the RIGHT hand.
    assert power[right, c3].mean() < power[left, c3].mean()
    # C4 (right hemisphere) desynchronizes for the LEFT hand.
    assert power[left, c4].mean() < power[right, c4].mean()


def test_erd_is_lateralized_not_global():
    """A global power drop would decode just as well but mean nothing.

    Guards against a generator (or a preprocessing bug) that dims everything at
    once -- the effect must be specific to the contralateral channel.
    """
    trials, labels = build_trials(n_trials=60, seed=1)
    power = mu_power(trials, SPEC.sfreq)
    right = labels == EVENT_IDS["right_hand"]
    left = labels == EVENT_IDS["left_hand"]

    parietal = SPEC.channels.index("Pz")
    ratio = power[right, parietal].mean() / power[left, parietal].mean()
    assert 0.85 < ratio < 1.15, "non-motor channel should not differ between classes"


def test_erd_depth_controls_effect_size():
    """A weaker planted effect must produce a smaller measured effect."""
    c3 = SPEC.channels.index("C3")

    def lateralization(depth):
        trials, labels = build_trials(n_trials=60, seed=2, erd_depth=depth)
        power = mu_power(trials, SPEC.sfreq)
        right = labels == EVENT_IDS["right_hand"]
        left = labels == EVENT_IDS["left_hand"]
        return power[left, c3].mean() / power[right, c3].mean()

    assert lateralization(0.5) > lateralization(0.15) > 0.95


def test_line_noise_is_present_before_filtering_and_gone_after():
    data, _, _ = synthetic_mi_recording(SPEC, n_trials=10)
    filtered = preprocess(SPEC, data, causal=False)

    def power_at(sig, freq):
        spectrum = np.abs(np.fft.rfft(sig[0]))
        freqs = np.fft.rfftfreq(sig.shape[-1], 1 / SPEC.sfreq)
        return spectrum[np.argmin(np.abs(freqs - freq))]

    assert power_at(filtered, 60.0) < 0.05 * power_at(data, 60.0)


def test_synthetic_output_shapes_and_balance():
    data, events, labels = synthetic_mi_recording(SPEC, n_trials=40)
    assert data.shape[0] == SPEC.n_channels
    assert events.shape == (40, 3)
    assert len(labels) == 40
    assert abs(np.sum(labels == EVENT_IDS["right_hand"]) - 20) <= 0


def test_amplitudes_are_physiologically_plausible():
    """Volts, not microvolts. Catches a units error at the source."""
    data, _, _ = synthetic_mi_recording(SPEC, n_trials=10)
    rms_uv = np.sqrt(np.mean(data**2)) * 1e6
    assert 5.0 < rms_uv < 200.0


def test_crops_preserve_the_erd():
    """Cropping to 1 s windows must not destroy the effect we decode."""
    trials, labels = build_trials(n_trials=40, seed=3)
    epochs = crop_trials(SPEC, trials, labels)
    c3 = SPEC.channels.index("C3")

    power = mu_power(epochs.X, SPEC.sfreq)
    right = epochs.y == EVENT_IDS["right_hand"]
    left = epochs.y == EVENT_IDS["left_hand"]
    assert power[right, c3].mean() < power[left, c3].mean()


@pytest.mark.parametrize("causal", [False, True])
def test_preprocess_preserves_erd_on_both_paths(causal):
    """Train/serve skew check at the signal level.

    The causal (online) and zero-phase (offline) paths must both retain the
    effect. If only one does, offline results would not predict online
    behaviour at all.
    """
    data, events, labels = synthetic_mi_recording(SPEC, n_trials=50, seed=4)
    filtered = preprocess(SPEC, data, causal=causal)

    tmin, _ = SPEC.trial_window
    start_off = int(round(tmin * SPEC.sfreq))
    trials = np.stack(
        [
            filtered[:, int(o) + start_off : int(o) + start_off + SPEC.trial_samples]
            for o in events[:, 0]
        ]
    )

    c3 = SPEC.channels.index("C3")
    power = mu_power(trials, SPEC.sfreq)
    right = labels == EVENT_IDS["right_hand"]
    left = labels == EVENT_IDS["left_hand"]
    assert power[right, c3].mean() < power[left, c3].mean()
