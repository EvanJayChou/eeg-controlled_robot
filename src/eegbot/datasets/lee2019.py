"""Lee2019 (MOABB) harmonized to look like DSI-7 output.

## What this dataset is and is not for

Lee2019 is 54 subjects x 2 sessions of left/right hand grasping imagery, 62
channels at 1000 Hz on a wet-gel BrainAmp, nose-referenced. We use it to build
and test the pipeline before hardware arrives, and to establish a realistic
accuracy ceiling for 7-channel motor imagery.

It is **not** a source of weights that will work on the DSI-7. Wet gel versus
dry contact, 62 channels versus 7, laboratory impedances versus whatever a
student headset achieves -- a model trained here will not transfer without
per-subject calibration on the real device. Treating it otherwise is the most
likely way for this project to produce numbers that mean nothing.

## Harmonization order (load-bearing)

1. Extract events while still at 1000 Hz.
2. Set the EMG channels aside -- they are the confound detector, not features.
3. Re-reference nose -> linked mastoids. **Before** channel selection, because
   TP9/TP10 must still exist.
4. Pick the 7 DSI-7 channels in canonical order.
5. Resample to 300 Hz, passing events through so markers are re-indexed rather
   than jittered.

Filtering is deliberately *not* here. It belongs to the shared `PreprocessSpec`
and runs at 300 Hz on the harmonized signal, exactly as it will online.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np

from eegbot.constants import EVENT_IDS, LEE2019_EMG_CHANNELS, LEE2019_MASTOIDS
from eegbot.datasets.base import EpochSet, concat
from eegbot.datasets.crops import crop_trials
from eegbot.sigproc.apply import preprocess
from eegbot.sigproc.montage import pick_canonical
from eegbot.sigproc.reference import rereference_to_mastoids
from eegbot.sigproc.spec import PreprocessSpec

log = logging.getLogger(__name__)

#: Default development subset. Downloading all 54 subjects is tens of GB and
#: hours of transfer; ten is plenty to iterate on and already shows the
#: bimodal literacy split.
DEV_SUBJECTS: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21, 34, 44, 52)


# === Event extraction ===


def extract_events(raw: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(events, labels)`` where labels use `constants.EVENT_IDS`.

    MOABB has shipped events as annotations and as a stim channel at different
    versions, so both are handled. Labels are resolved by *name*, never by the
    integer code -- `events_from_annotations` assigns codes alphabetically, which
    would silently swap left and right relative to our convention.
    """
    events: np.ndarray | None = None
    name_by_code: dict[int, str] = {}

    if raw.annotations is not None and len(raw.annotations) > 0:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        name_by_code = {code: name for name, code in event_id.items()}
    else:
        stim_picks = mne.pick_types(raw.info, stim=True)
        if len(stim_picks) == 0:
            raise ValueError("no annotations and no stim channel; cannot recover events")
        events = mne.find_events(raw, shortest_event=0, verbose=False)
        # MOABB's stim codes already follow its own event_id mapping.
        name_by_code = {v: k for k, v in EVENT_IDS.items()}

    keep = [i for i, code in enumerate(events[:, 2]) if name_by_code.get(code) in EVENT_IDS]
    if not keep:
        raise ValueError(
            f"no left/right hand events found; saw codes {sorted(set(events[:, 2]))} "
            f"mapping to {sorted(set(name_by_code.values()))}"
        )

    events = events[keep]
    labels = np.array([EVENT_IDS[name_by_code[code]] for code in events[:, 2]], dtype=int)
    return events, labels


# === Harmonization ===


def harmonize_raw(
    raw: mne.io.BaseRaw,
    spec: PreprocessSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one Lee2019 run into DSI-7-shaped data.

    Returns ``(data, events, labels)`` where `data` is
    ``(7, n_times)`` in volts at ``spec.sfreq``, unfiltered.
    """
    raw = raw.copy().load_data(verbose=False)
    events, labels = extract_events(raw)

    # Drop everything that is not EEG or a mastoid before re-referencing, so
    # stim/EMG channels cannot contaminate the reference computation.
    keep = [ch for ch in raw.ch_names if ch not in LEE2019_EMG_CHANNELS]
    raw = raw.pick(keep, verbose=False)
    raw = raw.pick_types(eeg=True, verbose=False)

    raw = rereference_to_mastoids(raw, LEE2019_MASTOIDS)
    raw = pick_canonical(raw, spec.channels)

    if not np.isclose(raw.info["sfreq"], spec.sfreq):
        raw, events = raw.resample(spec.sfreq, events=events, verbose=False)

    data = raw.get_data()
    # `events` samples are absolute; make them relative to the data array.
    events = events.copy()
    events[:, 0] -= raw.first_samp
    return data, events, labels


def epoch_array(
    data: np.ndarray,
    events: np.ndarray,
    labels: np.ndarray,
    spec: PreprocessSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut `spec.trial_window` around each event.

    Returns ``(trials, labels, kept_index)``. Events too close to a recording
    boundary are dropped, and `kept_index` says which survived so callers can
    keep parallel arrays (like EMG flags) aligned.
    """
    tmin, tmax = spec.trial_window
    start_off = int(round(tmin * spec.sfreq))
    n_samples = spec.trial_samples

    trials, kept = [], []
    for i, onset in enumerate(events[:, 0]):
        start = int(onset) + start_off
        stop = start + n_samples
        if start < 0 or stop > data.shape[-1]:
            log.warning("dropping trial %d: window [%d, %d) outside recording", i, start, stop)
            continue
        trials.append(data[:, start:stop])
        kept.append(i)

    if not trials:
        raise ValueError("no trials survived epoching; check event alignment")

    kept_index = np.array(kept, dtype=int)
    return np.stack(trials), labels[kept_index], kept_index


# === EMG confound detection ===


def emg_contaminated_trials(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    spec: PreprocessSpec,
    *,
    mad_threshold: float = 3.0,
) -> np.ndarray:
    """Flag trials with unusual EMG activity during the imagery window.

    Lee2019 records four EMG channels. We never decode from them, but they let
    us ask a question that is otherwise unanswerable: *is the classifier reading
    cortex, or muscle?* Re-run the benchmark with these trials excluded; if
    accuracy collapses, it was reading muscle.

    Returns a ``(n_events,)`` boolean array, True where contaminated. Returns
    all-False if the recording has no EMG channels.
    """
    present = [ch for ch in LEE2019_EMG_CHANNELS if ch in raw.ch_names]
    if not present:
        return np.zeros(len(events), dtype=bool)

    emg = raw.copy().load_data(verbose=False).pick(present, verbose=False)
    sfreq = emg.info["sfreq"]
    # EMG lives well above the EEG band; 20 Hz high-pass isolates muscle.
    emg = emg.filter(l_freq=20.0, h_freq=min(200.0, sfreq / 2 - 1), verbose=False)
    x = emg.get_data()

    tmin, tmax = spec.trial_window
    start_off = int(round(tmin * sfreq))
    n_samples = int(round((tmax - tmin) * sfreq))

    rms = np.full(len(events), np.nan)
    for i, onset in enumerate(events[:, 0]):
        start = int(onset) - raw.first_samp + start_off
        stop = start + n_samples
        if start < 0 or stop > x.shape[-1]:
            continue
        rms[i] = float(np.sqrt(np.mean(x[:, start:stop] ** 2)))

    valid = ~np.isnan(rms)
    if valid.sum() < 3:
        return np.zeros(len(events), dtype=bool)

    median = np.median(rms[valid])
    mad = np.median(np.abs(rms[valid] - median)) or 1e-12
    flagged = np.zeros(len(events), dtype=bool)
    flagged[valid] = rms[valid] > median + mad_threshold * mad
    return flagged


# === Public loader ===


def load_subject(
    subject: int,
    spec: PreprocessSpec,
    *,
    sessions: tuple[int, ...] = (1, 2),
    emg_check: bool = True,
) -> EpochSet:
    """Download (if needed), harmonize, filter and crop one subject.

    The returned `EpochSet` holds 1-second decoding windows, with
    ``meta["emg_contaminated"]`` giving a per-window flag for the Risk-3 check.
    """
    from moabb.datasets import Lee2019_MI  # imported lazily; heavy and optional

    dataset = Lee2019_MI(train_run=True, test_run=True, sessions=list(sessions))
    data = dataset.get_data(subjects=[subject])

    per_run: list[EpochSet] = []
    emg_flags: list[np.ndarray] = []

    for session_name, runs in data[subject].items():
        for run_name, raw in runs.items():
            if "Rest" in run_name:  # resting-state runs carry no MI events
                continue
            try:
                arr, events, labels = harmonize_raw(raw, spec)
            except ValueError as exc:
                log.warning("skipping %s/%s: %s", session_name, run_name, exc)
                continue

            flags = (
                emg_contaminated_trials(raw, events, spec)
                if emg_check
                else np.zeros(len(events), dtype=bool)
            )

            filtered = preprocess(spec, arr, causal=False)
            trials, trial_labels, kept = epoch_array(filtered, events, labels, spec)

            epochs = crop_trials(
                spec,
                trials,
                trial_labels,
                subjects=f"sub-{subject:02d}",
                sessions=f"{session_name}",
            )
            per_run.append(epochs)
            emg_flags.append(np.repeat(flags[kept], spec.crops_per_trial))

    if not per_run:
        raise RuntimeError(f"no usable runs for subject {subject}")

    out = concat(per_run)
    out.meta["emg_contaminated"] = np.concatenate(emg_flags)
    out.meta["subject"] = subject
    return out


def cache_path(root: Path, subject: int) -> Path:
    return Path(root) / "lee2019" / f"sub-{subject:02d}_crops.npz"


def save_cache(path: Path, epochs: EpochSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=epochs.X.astype(np.float32),  # float32 halves the file; well under EEG noise floor
        y=epochs.y,
        groups=epochs.groups,
        subjects=epochs.subjects,
        sessions=epochs.sessions,
        emg_contaminated=epochs.meta.get(
            "emg_contaminated", np.zeros(len(epochs), dtype=bool)
        ),
    )


def load_cache(path: Path, spec: PreprocessSpec) -> EpochSet:
    with np.load(path, allow_pickle=False) as z:
        out = EpochSet(
            X=z["X"].astype(float),
            y=z["y"],
            groups=z["groups"],
            subjects=z["subjects"],
            sessions=z["sessions"],
            spec=spec,
        )
        out.meta["emg_contaminated"] = z["emg_contaminated"]
    return out
