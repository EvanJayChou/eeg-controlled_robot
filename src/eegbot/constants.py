"""Hardware and paradigm constants.

Single source of truth for anything that describes the DSI-7 itself or the
motor-imagery paradigm. Nothing here should be duplicated as a literal
elsewhere in the codebase.
"""

from __future__ import annotations

# === DSI-7 hardware ===

#: Default DSI-7 electrode montage, in canonical order. Every array in this
#: codebase with a channel axis is ordered like this. The team ordered the
#: default configuration, so this is fixed for the project.
DSI7_CHANNELS: tuple[str, ...] = ("F3", "F4", "C3", "C4", "P3", "Pz", "P4")

#: Of the seven, only these sit over sensorimotor cortex. The whole decoding
#: strategy is shaped by the fact that this tuple has two elements.
SENSORIMOTOR_CHANNELS: tuple[str, ...] = ("C3", "C4")

#: Native sampling rate. A 600 Hz upgrade exists but was not ordered.
DSI7_SFREQ: float = 300.0

#: Fz carries the common-mode follower. It is a driven ground, not a
#: reference, and is not available as a data channel.
DSI7_COMMON_MODE_SITE: str = "Fz"

#: The DSI-7 references to ear clips. Lee2019 is nose-referenced, so
#: harmonization re-references it to linked mastoids as the closest proxy.
DSI7_REFERENCE: tuple[str, ...] = ("A1", "A2")

#: Analog bandwidth of the amplifier (Hz). Nyquist at 300 Hz is 150 Hz, so the
#: hardware low-pass and the sampling rate are well matched.
DSI7_BANDWIDTH_HZ: tuple[float, float] = (0.003, 150.0)

#: DSI-Streamer emits microvolts; MNE works in volts. The conversion lives in
#: exactly one place (`eegbot.session.recording`) and is asserted by a test.
MICROVOLTS_PER_VOLT: float = 1e6


# === Lee2019 harmonization ===

#: Mastoid channels in the Lee2019 62-channel montage, used as the linked-ear
#: reference proxy. These must still be present when re-referencing, which is
#: why re-referencing happens before channel selection.
LEE2019_MASTOIDS: tuple[str, ...] = ("TP9", "TP10")

#: Lee2019 ships four EMG channels. They are excluded from features but kept
#: aside to flag muscle-contaminated trials -- see `datasets.lee2019`.
LEE2019_EMG_CHANNELS: tuple[str, ...] = ("EMG1", "EMG2", "EMG3", "EMG4")

LEE2019_SFREQ: float = 1000.0


# === Paradigm ===

#: Class labels. Integer codes match MOABB's Lee2019 event ids so that our own
#: recordings and the benchmark dataset can share evaluation code unchanged.
EVENT_IDS: dict[str, int] = {"right_hand": 1, "left_hand": 2}

CLASS_NAMES: tuple[str, ...] = ("right_hand", "left_hand")

#: Positive class for the steering dial. p(POSITIVE_CLASS) near 1 means "turn
#: right", near 0 means "turn left", and near 0.5 falls in the dead zone.
POSITIVE_CLASS: str = "right_hand"

#: Mains frequency (US). Notch band is deliberately wide enough to catch drift.
LINE_FREQ_HZ: float = 60.0
NOTCH_BAND_HZ: tuple[float, float] = (57.0, 63.0)
