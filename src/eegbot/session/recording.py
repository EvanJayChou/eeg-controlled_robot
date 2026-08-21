"""Reading and writing session recordings.

## The one place units are converted

DSI-Streamer reports microvolts. MNE stores volts. Everything inside this
package is in volts. The conversion happens in `microvolts_to_volts` and
`volts_to_microvolts` and **nowhere else** -- a stray 1e6 elsewhere would scale
every amplitude threshold, artifact rejection bound, and covariance in the
project by a factor of a million, and would still produce plausible-looking
classification accuracy because most of the pipeline is scale-invariant. That
combination -- catastrophic and invisible -- is why it gets its own function and
its own round-trip test.

## The sidecar

Every recording is written with a JSON sidecar. The impedance log and the notes
field are not bureaucracy: when a session turns out to be unusable, they are the
only way to find out why, and by then the subject is long gone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eegbot.constants import DSI7_CHANNELS, DSI7_SFREQ, MICROVOLTS_PER_VOLT


def microvolts_to_volts(x: np.ndarray | float) -> np.ndarray | float:
    """Convert amplifier units to MNE units."""
    return np.asarray(x, dtype=float) / MICROVOLTS_PER_VOLT


def volts_to_microvolts(x: np.ndarray | float) -> np.ndarray | float:
    """Convert MNE units back to amplifier units, for display and thresholds."""
    return np.asarray(x, dtype=float) * MICROVOLTS_PER_VOLT


@dataclass
class SessionMetadata:
    """Everything about a session that is not the signal itself."""

    subject: str
    session: str
    date: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    channels: tuple[str, ...] = DSI7_CHANNELS
    sfreq: float = DSI7_SFREQ
    reference: str = "linked_ears"
    cue_script_version: str = "v1"

    #: Impedance in kOhm per channel, recorded before each run:
    #: ``{"run-0": {"C3": 12.4, ...}, ...}``. Logged rather than merely checked,
    #: so a bad run can be traced to bad contact after the fact.
    impedances: dict[str, dict[str, float]] = field(default_factory=dict)

    #: Free text from the operator. Drowsiness, movement, an interruption, a
    #: subject who said they found it hard. Pays for itself the first time a
    #: session looks strange in analysis.
    notes: str = ""

    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["channels"] = list(d["channels"])
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> SessionMetadata:
        d = json.loads(text)
        d["channels"] = tuple(d["channels"])
        return cls(**d)


def session_dir(root: str | Path, subject: str, session: str) -> Path:
    return Path(root) / "raw" / subject / session


def save_session(
    root: str | Path,
    metadata: SessionMetadata,
    data_uv: np.ndarray,
    events: np.ndarray | None = None,
    event_names: dict[int, str] | None = None,
) -> Path:
    """Write a session to ``<root>/raw/<subject>/<session>/``.

    Parameters
    ----------
    data_uv
        ``(n_channels, n_times)`` **in microvolts**, as the amplifier reports
        it. Converted to volts here, once.
    """
    import mne

    from eegbot.sigproc.montage import dsi7_info

    data_uv = np.asarray(data_uv, dtype=float)
    if data_uv.shape[0] != len(metadata.channels):
        raise ValueError(
            f"data has {data_uv.shape[0]} channels but metadata lists "
            f"{len(metadata.channels)}"
        )

    out_dir = session_dir(root, metadata.subject, metadata.session)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = dsi7_info(metadata.sfreq, metadata.channels)
    raw = mne.io.RawArray(microvolts_to_volts(data_uv), info, verbose=False)

    if events is not None and len(events):
        names = event_names or {}
        onsets = events[:, 0] / metadata.sfreq
        descriptions = [names.get(int(code), str(code)) for code in events[:, 2]]
        raw.set_annotations(
            mne.Annotations(onset=onsets, duration=0.0, description=descriptions)
        )

    fif_path = out_dir / "eeg_raw.fif"
    raw.save(fif_path, overwrite=True, verbose=False)
    (out_dir / "session.json").write_text(metadata.to_json(), encoding="utf-8")
    return fif_path


def load_session(path: str | Path) -> tuple[Any, SessionMetadata]:
    """Load a session directory or FIF path, returning ``(raw, metadata)``."""
    import mne

    path = Path(path)
    directory = path if path.is_dir() else path.parent
    fif_path = path if path.is_file() else directory / "eeg_raw.fif"

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    metadata = SessionMetadata.from_json(
        (directory / "session.json").read_text(encoding="utf-8")
    )
    return raw, metadata
