"""Command-line entry points.

    eegbot demo             end-to-end on synthetic data -- no downloads, no headset
    eegbot prepare-lee2019  download and harmonize the benchmark dataset
    eegbot evaluate         cross-validate a decoder, write a per-subject report
    eegbot record           run a recording session (or rehearse one headless)
    eegbot online           live decoding, printing turn rate

`demo` is the one to run first. It exercises every stage the real pipeline uses
-- shared preprocessing, cropping, group-aware CV, the decoder, calibration, the
control law, and the online loop -- against a signal whose ground truth we
planted, so a failure points at our code rather than at the EEG.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from eegbot.config import (
    load_control_config,
    load_paradigm_config,
    load_preprocess_spec,
)

log = logging.getLogger("eegbot")

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_ROOT = Path("outputs")


def _run_dir(root: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"{name}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _require(module: str, extra: str) -> None:
    from importlib.util import find_spec

    if find_spec(module) is None:
        raise SystemExit(
            f"'{module}' is required for this command but is not installed.\n"
            f"  conda activate eegbot && pip install -e \".[{extra}]\""
        )


# === demo ===


def cmd_demo(args: argparse.Namespace) -> int:
    """Full pipeline on synthetic data with a planted ERD."""
    _require("pyriemann", "dev")

    from eegbot.datasets.crops import crop_trials
    from eegbot.decoding.calibrate import calibrate_from_idle
    from eegbot.decoding.pipelines import build
    from eegbot.evaluation.control_sim import simulate_control
    from eegbot.evaluation.metrics import report, results_frame, score_split
    from eegbot.evaluation.protocols import within_session
    from eegbot.sigproc.apply import preprocess
    from eegbot.stream.synthetic import synthetic_mi_recording

    spec = load_preprocess_spec(args.preprocess)
    control = load_control_config(args.control)

    print(f"spec: {spec.n_channels}ch @ {spec.sfreq:g} Hz, "
          f"{spec.window_s:g}s windows / {spec.hop_s:g}s hop "
          f"({spec.update_rate_hz:g} Hz control), {spec.crops_per_trial} crops/trial")

    # --- generate and preprocess exactly as the real path does ---
    data, events, labels = synthetic_mi_recording(
        spec, n_trials=args.trials, seed=args.seed, erd_depth=args.erd_depth
    )
    filtered = preprocess(spec, data, causal=False)

    start_off = int(round(spec.trial_window[0] * spec.sfreq))
    trials = np.stack(
        [
            filtered[:, int(o) + start_off : int(o) + start_off + spec.trial_samples]
            for o in events[:, 0]
        ]
    )
    epochs = crop_trials(spec, trials, labels, subjects="synthetic", sessions="ses-0")
    print(f"\n{epochs.summary()}")

    # --- decode ---
    model = build(args.decoder, spec)
    results = [
        score_split(model, epochs, split) for split in within_session(epochs, n_splits=5)
    ]
    df = results_frame(results)
    print("\n" + report(df, decoder=args.decoder, protocol="within_session"))
    print(
        "NOTE: this is synthetic data with a strong planted ERD, so high accuracy is\n"
        "expected here and is the M1 gate passing -- not a warning sign. The >85%\n"
        "caution above applies to real recordings.\n"
    )

    # --- run the decoder over the CONTINUOUS recording ---
    # Drivability has to be measured on a real timeline, with genuine rest
    # periods between trials. Replaying crops in trial order would butt
    # opposite-class trials against each other with no rest in between, and the
    # resulting numbers would describe nothing.
    from eegbot.constants import EVENT_IDS, POSITIVE_CLASS
    from eegbot.control.controller import ContinuousController
    from eegbot.stream.loop import OnlineDecoder
    from eegbot.stream.replay import ArrayReplaySource

    model.fit(epochs.X, epochs.y)
    positive_index = list(model.classes_).index(EVENT_IDS[POSITIVE_CLASS])

    passthrough = ContinuousController(control, spec.hop_s, bias=0.0, threshold=0.0)
    scanner = OnlineDecoder(
        spec, model, passthrough, positive_index=positive_index, align_seconds=0.0
    )
    updates = scanner.run(ArrayReplaySource(data))

    # Intent timeline: +1/-1 during each imagery window, 0 everywhere else.
    imagery_len = int(round(4.0 * spec.sfreq))
    intents = np.zeros(len(updates), dtype=int)
    probs = np.array([u.p_right for u in updates])
    for i, update in enumerate(updates):
        sample = update.center_sample
        hits = events[(events[:, 0] <= sample) & (sample < events[:, 0] + imagery_len)]
        if hits.size:
            intents[i] = 1 if hits[0, 2] == EVENT_IDS[POSITIVE_CLASS] else -1

    print(
        f"online scan: {len(updates)} updates at {spec.update_rate_hz:g} Hz "
        f"({len(updates) * spec.hop_s:.0f}s), "
        f"{int(np.sum(intents != 0))} during imagery, {int(np.sum(intents == 0))} at rest"
    )

    # --- idle gate ---
    # A 2-class decoder cannot represent "neither", so it stays confident at
    # rest and no dead-zone width produces a usable idle state. The gate asks
    # the orthogonal question -- imagining or not? -- and suppresses commands
    # when the answer is no. See eegbot.decoding.idle.
    from eegbot.decoding.idle import IdleGate, build_gate_dataset

    gate_X, gate_y = build_gate_dataset(spec, filtered, events)
    gate = IdleGate(spec).fit(gate_X, gate_y)

    windows = np.stack(
        [filtered[:, u.end_sample - spec.window_samples : u.end_sample] for u in updates]
    )
    engagement = gate.engagement(windows)

    # --- calibrate, with and without the gate ---
    # Calibrating without the gate and deploying with it would leave the dead
    # zone far wider than needed, costing the subject most of their ability to
    # issue a command. The two must be fitted together.
    rest = intents == 0
    ungated_cal = calibrate_from_idle(probs[rest], control, spec.hop_s)
    gated_cal = calibrate_from_idle(
        probs[rest], control, spec.hop_s, idle_engagement=engagement[rest]
    )
    print(f"calibration, no gate:   {ungated_cal.summary()}")
    print(f"calibration, with gate: {gated_cal.summary()}")

    without_gate = simulate_control(
        probs, intents, control, spec.hop_s,
        bias=ungated_cal.bias, threshold=ungated_cal.threshold,
    )
    with_gate = simulate_control(
        probs, intents, control, spec.hop_s,
        bias=gated_cal.bias, threshold=gated_cal.threshold, engagement=engagement,
    )

    print("\ndrivability")
    print(f"  without idle gate: {without_gate.summary()}")
    print(f"  with idle gate:    {with_gate.summary()}")

    if args.output:
        out_dir = _run_dir(DEFAULT_OUTPUT_ROOT, "demo")
        (out_dir / "report.md").write_text(
            report(df, decoder=args.decoder, protocol="within_session"), encoding="utf-8"
        )
        df.to_csv(out_dir / "splits.csv", index=False)
        print(f"\nwrote {out_dir}")
    return 0


# === prepare-lee2019 ===


def cmd_prepare_lee2019(args: argparse.Namespace) -> int:
    _require("moabb", "datasets")
    _require("mne", "dev")

    from eegbot.datasets.lee2019 import DEV_SUBJECTS, cache_path, load_subject, save_cache

    spec = load_preprocess_spec(args.preprocess)
    subjects = args.subjects or list(DEV_SUBJECTS)
    root = Path(args.data_root) / "processed"

    for subject in subjects:
        path = cache_path(root, subject)
        if path.exists() and not args.force:
            print(f"sub-{subject:02d}: cached, skipping")
            continue
        print(f"sub-{subject:02d}: downloading and harmonizing ...")
        try:
            epochs = load_subject(subject, spec, emg_check=not args.no_emg_check)
        except Exception as exc:  # noqa: BLE001 - one bad subject must not stop the batch
            log.error("sub-%02d failed: %s", subject, exc)
            continue
        save_cache(path, epochs)
        print(f"  {epochs.summary()}")
        print(f"  -> {path}")
    return 0


# === evaluate ===


def cmd_evaluate(args: argparse.Namespace) -> int:
    _require("pyriemann", "dev")

    from eegbot.datasets.base import concat
    from eegbot.datasets.lee2019 import DEV_SUBJECTS, cache_path, load_cache
    from eegbot.decoding.pipelines import build
    from eegbot.evaluation.metrics import report, results_frame, score_split
    from eegbot.evaluation.protocols import get_protocol

    spec = load_preprocess_spec(args.preprocess)
    root = Path(args.data_root) / "processed"
    subjects = args.subjects or list(DEV_SUBJECTS)

    sets = []
    for subject in subjects:
        path = cache_path(root, subject)
        if not path.exists():
            log.warning("sub-%02d not cached; run `eegbot prepare-lee2019` first", subject)
            continue
        sets.append(load_cache(path, spec))

    if not sets:
        raise SystemExit("no cached subjects found; run `eegbot prepare-lee2019` first")

    epochs = concat(sets)
    print(epochs.summary())

    model = build(args.decoder, spec)
    protocol = get_protocol(args.protocol)

    results = [score_split(model, epochs, split) for split in protocol(epochs)]
    df = results_frame(results)
    text = report(df, decoder=args.decoder, protocol=args.protocol)
    print("\n" + text)

    out_dir = _run_dir(DEFAULT_OUTPUT_ROOT, f"eval_{args.decoder}_{args.protocol}")
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    df.to_csv(out_dir / "splits.csv", index=False)
    print(f"wrote {out_dir}")
    return 0


# === record ===


def cmd_record(args: argparse.Namespace) -> int:
    from eegbot.session.cues import HeadlessDisplay
    from eegbot.session.markers import MemoryMarkerSink
    from eegbot.session.protocol import estimate_duration_s
    from eegbot.session.runner import SessionRunner

    config = load_paradigm_config(args.paradigm)
    print(f"estimated duration: {estimate_duration_s(config) / 60:.0f} min")

    if args.dry_run:
        display = HeadlessDisplay(speed=args.speed, verbose=args.verbose)
        sink = MemoryMarkerSink()
    else:
        _require("psychopy", "session")
        from eegbot.session.cues import PsychoPyDisplay
        from eegbot.session.markers import LSLMarkerSink, TeeMarkerSink

        _require("pylsl", "stream")
        display = PsychoPyDisplay(fullscreen=not args.windowed)
        sink = TeeMarkerSink([LSLMarkerSink(), MemoryMarkerSink()])

    runner = SessionRunner(config, display, sink, seed=args.seed)
    session_log = runner.run()

    print(
        f"completed {session_log.completed_trials} trials "
        f"across {session_log.completed_runs} runs"
    )
    if args.dry_run:
        memory = sink if isinstance(sink, MemoryMarkerSink) else None
        if memory is not None:
            print(f"emitted {len(memory.markers)} markers")
    return 0


# === online ===


def cmd_online(args: argparse.Namespace) -> int:
    _require("pyriemann", "dev")

    import joblib

    from eegbot.control.controller import ContinuousController
    from eegbot.stream.loop import OnlineDecoder
    from eegbot.stream.replay import ArrayReplaySource, FileReplaySource
    from eegbot.stream.synthetic import synthetic_mi_recording

    spec = load_preprocess_spec(args.preprocess)
    control = load_control_config(args.control)

    if args.model:
        model = joblib.load(args.model)
    else:
        raise SystemExit("--model is required (train one with `eegbot demo --save-model`)")

    if args.replay:
        source = FileReplaySource(args.replay, spec.channels, realtime=args.realtime)
    else:
        print("no --replay given; streaming synthetic data")
        data, _, _ = synthetic_mi_recording(spec, n_trials=20)
        source = ArrayReplaySource(data, spec.sfreq, spec.channels, realtime=args.realtime)

    controller = ContinuousController(control, spec.hop_s, bias=args.bias, threshold=args.dead_zone)
    decoder = OnlineDecoder(spec, model, controller, align_seconds=args.align_seconds)

    def show(update) -> None:
        bar_width = 30
        position = int((update.command.command + 1) / 2 * bar_width)
        bar = "".join(
            "|" if i == bar_width // 2 else ("#" if i == position else "-")
            for i in range(bar_width + 1)
        )
        state = "idle" if update.command.is_idle else f"{update.command.turn_rate:+.2f}"
        print(f"{update.time_s:7.1f}s  p={update.p_right:.3f}  [{bar}]  {state}")

    decoder.run(source, max_updates=args.max_updates, on_update=show)
    return 0


# === check-signal ===


def cmd_check_signal(args: argparse.Namespace) -> int:
    """Alpha eyes-open/closed check -- the go/no-go on electrode contact."""
    _require("mne", "dev")

    from eegbot.session.recording import load_session
    from eegbot.sigproc.artifacts import flat_channels

    spec = load_preprocess_spec(args.preprocess)
    raw, metadata = load_session(args.session)
    print(f"subject {metadata.subject} / session {metadata.session}")

    from eegbot.sigproc.montage import as_canonical_array

    data = as_canonical_array(raw, spec.channels)
    flats = flat_channels(spec, data)
    if flats:
        print(f"  FLAT CHANNELS: {flats} -- check electrode contact")
    else:
        print("  no flat channels")
    return 0


# === parser ===


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eegbot", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--preprocess", default="preprocess/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="end-to-end on synthetic data")
    demo.add_argument("--decoder", default="riemann_ts_lr")
    demo.add_argument("--control", default="control/default.yaml")
    demo.add_argument("--trials", type=int, default=60)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument(
        "--erd-depth",
        type=float,
        default=0.45,
        help="planted effect size; lower simulates a poorer subject",
    )
    demo.add_argument("--online-updates", type=int, default=100)
    demo.add_argument("--output", action="store_true", help="write a report to outputs/")
    demo.set_defaults(func=cmd_demo)

    prep = sub.add_parser("prepare-lee2019", help="download and harmonize Lee2019")
    prep.add_argument("--subjects", type=int, nargs="*")
    prep.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    prep.add_argument("--force", action="store_true")
    prep.add_argument("--no-emg-check", action="store_true")
    prep.set_defaults(func=cmd_prepare_lee2019)

    ev = sub.add_parser("evaluate", help="cross-validate a decoder")
    ev.add_argument("--decoder", default="riemann_ts_lr")
    ev.add_argument(
        "--protocol", default="within_session",
        choices=["within_session", "cross_session", "loso"],
    )
    ev.add_argument("--subjects", type=int, nargs="*")
    ev.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ev.set_defaults(func=cmd_evaluate)

    rec = sub.add_parser("record", help="run or rehearse a recording session")
    rec.add_argument("--paradigm", default="paradigm/mi_2class.yaml")
    rec.add_argument("--dry-run", action="store_true", help="headless rehearsal, no PsychoPy")
    rec.add_argument("--speed", type=float, default=0.0, help="0 = instant, 1 = real time")
    rec.add_argument("--windowed", action="store_true")
    rec.add_argument("--seed", type=int, default=0)
    rec.set_defaults(func=cmd_record)

    on = sub.add_parser("online", help="live decoding")
    on.add_argument("--model", help="joblib-saved fitted pipeline")
    on.add_argument("--control", default="control/default.yaml")
    on.add_argument("--replay", help="FIF/EDF file to replay instead of live data")
    on.add_argument("--realtime", action="store_true", help="pace at wall clock")
    on.add_argument("--bias", type=float, default=0.0)
    on.add_argument("--dead-zone", type=float, default=None)
    on.add_argument("--align-seconds", type=float, default=30.0)
    on.add_argument("--max-updates", type=int, default=200)
    on.set_defaults(func=cmd_online)

    chk = sub.add_parser("check-signal", help="electrode-quality check on a recording")
    chk.add_argument("session", help="session directory or FIF path")
    chk.set_defaults(func=cmd_check_signal)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
