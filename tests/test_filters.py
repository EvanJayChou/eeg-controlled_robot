"""Filter behaviour -- especially the properties the online path depends on."""

from __future__ import annotations

import numpy as np
import pytest

from eegbot.sigproc.filters import (
    CausalFilter,
    design_sos,
    filter_causal,
    filter_zero_phase,
    warmup_samples,
)
from eegbot.sigproc.spec import PreprocessSpec

SPEC = PreprocessSpec()


def test_causal_filter_uses_no_future_samples():
    """The defining property of the online path.

    Feed an impulse late in an otherwise-zero signal; every output sample before
    it must be exactly zero. `filtfilt` fails this by construction, which is why
    it can never be used online.
    """
    n = 3000
    impulse_at = 2000
    x = np.zeros((SPEC.n_channels, n))
    x[:, impulse_at] = 1.0

    out = filter_causal(SPEC, x)

    assert np.allclose(out[:, :impulse_at], 0.0, atol=1e-12)
    assert np.any(np.abs(out[:, impulse_at:]) > 1e-9)


def test_zero_phase_filter_does_use_future_samples():
    """Documents the asymmetry, so nobody 'fixes' the online path to match."""
    n = 3000
    impulse_at = 2000
    x = np.zeros((SPEC.n_channels, n))
    x[:, impulse_at] = 1.0

    out = filter_zero_phase(design_sos(SPEC), x)

    assert np.any(np.abs(out[:, :impulse_at]) > 1e-9)


def test_chunked_filtering_matches_one_shot():
    """Filter state must survive chunk boundaries.

    A stateless per-chunk filter would inject an edge transient every 100 ms --
    ten times a second, forever. This is the single most common online-BCI bug.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((SPEC.n_channels, 4000))

    one_shot = filter_causal(SPEC, x)

    chunked_filter = CausalFilter(SPEC)
    chunked_filter.reset(x0=x[:, 0])
    hop = SPEC.hop_samples
    chunked = np.concatenate(
        [chunked_filter(x[:, i : i + hop]) for i in range(0, x.shape[-1], hop)], axis=-1
    )

    np.testing.assert_allclose(one_shot, chunked, atol=1e-10)


def test_uneven_chunks_also_match():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((SPEC.n_channels, 2000))

    one_shot = filter_causal(SPEC, x)

    filt = CausalFilter(SPEC)
    filt.reset(x0=x[:, 0])
    pieces, position = [], 0
    for size in [7, 128, 1, 400, 64, 999]:
        if position >= x.shape[-1]:
            break
        pieces.append(filt(x[:, position : position + size]))
        position += size
    if position < x.shape[-1]:
        pieces.append(filt(x[:, position:]))

    np.testing.assert_allclose(one_shot[:, :position], np.concatenate(pieces, axis=-1)[:, :position], atol=1e-10)


def test_steady_state_init_suppresses_startup_transient():
    """Zero-initialised state on a DC-offset signal rings for seconds."""
    offset = 50e-6
    x = np.full((SPEC.n_channels, 2000), offset)

    warm = CausalFilter(SPEC)
    warm.reset(x0=x[:, 0])
    warm_out = warm(x)

    cold = CausalFilter(SPEC)
    cold.reset(x0=None)
    cold_out = cold(x)

    early = slice(0, 200)
    assert np.max(np.abs(warm_out[:, early])) < np.max(np.abs(cold_out[:, early]))


def test_notch_attenuates_line_noise():
    sfreq = SPEC.sfreq
    t = np.arange(int(5 * sfreq)) / sfreq
    line = np.sin(2 * np.pi * 60.0 * t)
    mu = np.sin(2 * np.pi * 10.0 * t)
    x = np.tile(line + mu, (SPEC.n_channels, 1))

    out = filter_zero_phase(design_sos(SPEC), x)

    def power_at(sig, freq):
        spectrum = np.abs(np.fft.rfft(sig[0]))
        freqs = np.fft.rfftfreq(sig.shape[-1], 1 / sfreq)
        return spectrum[np.argmin(np.abs(freqs - freq))]

    assert power_at(out, 60.0) < 0.05 * power_at(x, 60.0)
    assert power_at(out, 10.0) > 0.5 * power_at(x, 10.0)


def test_bandpass_rejects_slow_drift():
    """Dry electrodes drift; the 4 Hz corner must remove it."""
    sfreq = SPEC.sfreq
    t = np.arange(int(10 * sfreq)) / sfreq
    drift = 100e-6 * np.sin(2 * np.pi * 0.2 * t)
    x = np.tile(drift, (SPEC.n_channels, 1))

    out = filter_zero_phase(design_sos(SPEC), x)
    interior = out[:, int(sfreq) : -int(sfreq)]  # ignore edge effects
    assert np.max(np.abs(interior)) < 0.05 * np.max(np.abs(x))


def test_warmup_is_long_enough_to_settle():
    n = warmup_samples(SPEC)
    assert n > SPEC.sfreq / 2
    assert n < 10 * SPEC.sfreq


def test_wrong_channel_count_is_rejected():
    filt = CausalFilter(SPEC)
    with pytest.raises(ValueError, match="expected chunk of shape"):
        filt(np.zeros((3, 100)))
