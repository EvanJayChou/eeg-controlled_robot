"""The preprocessing specification.

One dataclass describes every filtering and windowing choice in the project.
Offline analysis and the online loop both consume the *same* instance, which is
the structural reason the two paths cannot drift apart. See `apply.py` for the
offline/online split and why it lands where it does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from eegbot.constants import DSI7_CHANNELS, DSI7_SFREQ, NOTCH_BAND_HZ


@dataclass(frozen=True)
class PreprocessSpec:
    """Filtering + windowing parameters shared by the offline and online paths.

    Frozen because a spec that mutates mid-analysis is a debugging nightmare;
    build a new one with `dataclasses.replace` instead.
    """

    #: Canonical channel order. Arrays are always ordered like this.
    channels: tuple[str, ...] = DSI7_CHANNELS

    #: Sampling rate of the data reaching `apply`. Harmonized datasets are
    #: resampled to this before preprocessing.
    sfreq: float = DSI7_SFREQ

    #: Mains notch as (low, high) of a band-stop, or None to skip.
    notch: tuple[float, float] | None = NOTCH_BAND_HZ

    #: Bandpass. Wider than mu/beta so a filter bank has room to work later.
    band: tuple[float, float] = (4.0, 40.0)

    #: Butterworth order, used identically for the causal and zero-phase paths
    #: so the two differ only in direction of application.
    filter_order: int = 4

    #: Decoding window length in seconds. Training uses crops of exactly this
    #: length so the training distribution matches deployment.
    window_s: float = 1.0

    #: Crop / inference hop in seconds. 0.1 s gives 10 Hz control updates.
    hop_s: float = 0.1

    #: Trial window relative to cue onset, in seconds. Starts at 0.5 to exclude
    #: the cue-onset visual evoked response, which is otherwise a confound the
    #: classifier will happily exploit.
    trial_window: tuple[float, float] = (0.5, 3.5)

    #: Peak-to-peak amplitude rejection threshold in microvolts, or None.
    reject_p2p_uv: float | None = 150.0

    #: Flat-channel threshold in microvolts peak-to-peak.
    flat_p2p_uv: float = 0.5

    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        lo, hi = self.band
        nyquist = self.sfreq / 2.0
        if not 0 < lo < hi:
            raise ValueError(f"band must satisfy 0 < low < high, got {self.band}")
        if hi >= nyquist:
            raise ValueError(f"band high {hi} Hz must be below Nyquist {nyquist} Hz")
        if self.notch is not None:
            nlo, nhi = self.notch
            if not 0 < nlo < nhi:
                raise ValueError(f"notch must satisfy 0 < low < high, got {self.notch}")
            if nhi >= nyquist:
                raise ValueError(f"notch high {nhi} Hz must be below Nyquist {nyquist} Hz")
        if self.window_s <= 0 or self.hop_s <= 0:
            raise ValueError("window_s and hop_s must be positive")
        if self.hop_s > self.window_s:
            raise ValueError("hop_s > window_s would skip samples between crops")
        t0, t1 = self.trial_window
        if t1 - t0 < self.window_s:
            raise ValueError(
                f"trial_window {self.trial_window} is shorter than window_s "
                f"{self.window_s}; no crops could be extracted"
            )

    # === Derived quantities ===

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def window_samples(self) -> int:
        return int(round(self.window_s * self.sfreq))

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_s * self.sfreq))

    @property
    def update_rate_hz(self) -> float:
        """Control-loop update rate implied by the hop."""
        return 1.0 / self.hop_s

    @property
    def trial_samples(self) -> int:
        t0, t1 = self.trial_window
        return int(round((t1 - t0) * self.sfreq))

    @property
    def crops_per_trial(self) -> int:
        return 1 + (self.trial_samples - self.window_samples) // self.hop_samples

    # === Serialization ===

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # YAML round-trips lists, not tuples; normalize on the way out.
        for key in ("channels", "notch", "band", "trial_window"):
            if d[key] is not None:
                d[key] = list(d[key])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreprocessSpec:
        d = dict(d)
        if "channels" in d:
            d["channels"] = tuple(d["channels"])
        for key in ("notch", "band", "trial_window"):
            if d.get(key) is not None:
                d[key] = tuple(d[key])
        return cls(**d)
