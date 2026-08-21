"""Synthetic DSI-7 recordings with a planted mu-band ERD.

This is the M1 gate. If the pipeline cannot recover a signal we injected
ourselves, the bug is in our code and there is no point looking at real EEG yet.
It also lets the whole project be developed and tested with no headset, no
downloads, and no network.

The generator models the physiology we actually intend to decode:

* 1/f background, the dominant feature of real EEG
* a mu rhythm around 10 Hz present at all sensors
* **event-related desynchronization** -- mu power *drops* over the hemisphere
  contralateral to the imagined hand (right hand -> C3, left hand -> C4)
* volume conduction, so channels are correlated rather than independent
* 60 Hz line noise, to give the notch filter something to remove

Amplitudes are in volts and roughly scaled to real scalp EEG (tens of
microvolts), so artifact thresholds behave sensibly against this data.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from eegbot.constants import EVENT_IDS, LINE_FREQ_HZ
from eegbot.sigproc.spec import PreprocessSpec

UV = 1e-6

#: Which channel desynchronizes for each imagined hand. Motor cortex is
#: contralateral: imagining the right hand suppresses mu over the left
#: hemisphere (C3), and vice versa.
ERD_CHANNEL = {"right_hand": "C3", "left_hand": "C4"}


def pink_noise(n_channels: int, n_times: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise via spectral shaping of white noise."""
    white = rng.standard_normal((n_channels, n_times))
    spectrum = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(n_times)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = np.fft.irfft(spectrum * scale, n=n_times, axis=-1)
    return shaped / (np.std(shaped, axis=-1, keepdims=True) + 1e-12)


def _narrowband(n_times: int, sfreq: float, band: tuple[float, float], rng) -> np.ndarray:
    """Band-limited noise -- a rhythm with a realistic, non-sinusoidal envelope."""
    sos = sp_signal.butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    x = sp_signal.sosfiltfilt(sos, rng.standard_normal(n_times))
    return x / (np.std(x) + 1e-12)


def synthetic_mi_recording(
    spec: PreprocessSpec,
    *,
    n_trials: int = 40,
    seed: int = 0,
    erd_depth: float = 0.45,
    mu_amplitude_uv: float = 12.0,
    background_uv: float = 18.0,
    line_amplitude_uv: float = 4.0,
    trial_period_s: float = 9.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a continuous recording with labelled MI trials.

    Parameters
    ----------
    erd_depth
        Fractional mu-power reduction on the contralateral channel during
        imagery. 0.45 is a strong but not unrealistic effect; drop it toward
        0.15 to simulate a poor subject and check the pipeline degrades
        gracefully rather than breaking.

    Returns
    -------
    ``(data, events, labels)`` with `data` shaped ``(n_channels, n_times)`` in
    volts, `events` an MNE-style ``(n_trials, 3)`` array whose first column is
    the cue-onset sample, and `labels` using `constants.EVENT_IDS`.
    """
    rng = np.random.default_rng(seed)
    sfreq = spec.sfreq
    n_channels = spec.n_channels
    period = int(round(trial_period_s * sfreq))
    n_times = period * n_trials

    channel_index = {name: i for i, name in enumerate(spec.channels)}

    # === Background: 1/f plus a shared component for volume conduction ===
    data = background_uv * UV * pink_noise(n_channels, n_times, rng)
    shared = pink_noise(1, n_times, rng)
    data += 0.4 * background_uv * UV * shared

    # === Mu rhythm, one independent generator per channel ===
    mu = np.stack([_narrowband(n_times, sfreq, (8.0, 13.0), rng) for _ in range(n_channels)])

    # === Envelope: 1.0 everywhere, dipping during contralateral imagery ===
    envelope = np.ones((n_channels, n_times))

    cue_offset = int(round(3.0 * sfreq))  # fixation 2 s + cue 1 s
    imagery_len = int(round(4.0 * sfreq))

    class_names = list(ERD_CHANNEL)
    labels = np.array(
        [EVENT_IDS[class_names[i % 2]] for i in range(n_trials)], dtype=int
    )
    rng.shuffle(labels)
    name_by_code = {v: k for k, v in EVENT_IDS.items()}

    events = np.zeros((n_trials, 3), dtype=int)
    for i in range(n_trials):
        onset = i * period + cue_offset
        events[i] = (onset, 0, labels[i])

        ch = channel_index[ERD_CHANNEL[name_by_code[labels[i]]]]
        start, stop = onset, onset + imagery_len
        # Smooth ramp in/out; an instantaneous step would be a spectral artifact
        # the classifier could latch onto instead of the power change.
        ramp = int(round(0.3 * sfreq))
        window = np.ones(stop - start)
        window[:ramp] = np.linspace(0, 1, ramp)
        window[-ramp:] = np.linspace(1, 0, ramp)
        envelope[ch, start:stop] -= erd_depth * window

    data += mu_amplitude_uv * UV * mu * envelope

    # === Line noise, so the notch has work to do ===
    t = np.arange(n_times) / sfreq
    data += line_amplitude_uv * UV * np.sin(2 * np.pi * LINE_FREQ_HZ * t)[None, :]

    return data, events, labels
