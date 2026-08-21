# EEG-Controlled Pet Robot

Decoding motor imagery from a **Wearable Sensing DSI-7** into a continuous
steering signal for a robot.

The DSI-7 is on order. Everything in this repo runs today without it.

---

## Why motor imagery, and not SSVEP

SSVEP is the easier paradigm and scales to more classes, but it lives in visual
cortex and needs occipital electrodes (O1/O2/Oz). The DSI-7's default montage is

```
F3  F4   C3  C4   P3  Pz  P4        common-mode follower at Fz
                                     ear-clip references, 300 Hz
```

No occipital coverage, so SSVEP is out. That leaves **2-class left/right hand
motor imagery**.

The constraint that shapes everything downstream: **only C3 and C4 sit over
sensorimotor cortex.** Two usable channels out of seven means no surface
Laplacian, very little for CSP to work with, and a hard ceiling on how many
classes are separable. Expect **65–75% within-session accuracy**, and expect
roughly a third of subjects to sit near chance — motor-imagery ability is
genuinely bimodal and no amount of modelling fixes it.

## Two classes is not two commands

The classifier output is treated as a **dial, not a switch**. `p(right_hand)` is
centered to [-1, 1], smoothed, and passed through a dead zone:

```
s = 2p - 1 - bias        →  ema(τ≈400ms)  →  dead zone θ  →  slew limit  →  turn rate
```

Both `bias` and `θ` are fitted per subject from a recorded idle block — neither
can be hardcoded. See [`control/law.py`](src/eegbot/control/law.py).

### The dead zone alone does not give you an idle state

This was the plan's assumption and **running the decoder over a full continuous
recording showed it to be wrong**, so it is worth stating plainly.

A 2-class model is trained only to separate left from right. It has no
representation of "neither", so a resting window — which resembles neither class
— still gets pushed to a confident output, frequently past 0.95. Widening the
dead zone does not rescue it: at θ = 0.875 the decoder was still active for 16%
of rest time, by which point the subject can barely issue a deliberate command
either.

This is the **no-control state problem** in asynchronous BCI. The fix is a
second, orthogonal model — an **idle gate** asking *"imagining, or not?"* rather
than *"left or right?"*:

```
steering decoder:  left vs right    — lateralization of mu power
idle gate:         imagery vs rest  — overall mu/beta power
```

Different discriminations, so they cannot be one model. Note they even want
opposite preprocessing: the steering decoder normalizes covariance trace to
survive electrode-contact drift, while the gate needs that same total power kept,
because it *is* the signal. See [`decoding/idle.py`](src/eegbot/decoding/idle.py).

The gate trains on the 60-second idle block the session runner already records,
so it costs no extra collection. Engagement is smoothed and applied with
hysteresis — start above 0.60, stop below 0.40 — because a single threshold on a
raw per-window probability chatters and produces bursts of brief false
activations.

On synthetic data the gate cuts idle-time activity roughly 3× (49% → 15%) with
wrong-direction rate dropping 3.4% → 0.9%. It is a large improvement, not a
complete solution; the real numbers have to come from real recordings.

---

## Quickstart

```bash
conda env create -f environment.yml
conda activate eegbot
pip install -e .

pytest                 # ~115 tests, no network, no hardware
eegbot demo            # full pipeline on synthetic data with a planted ERD
```

`eegbot demo` is the thing to run first. It exercises every stage the real
pipeline uses — shared preprocessing, cropping, group-aware CV, the Riemannian
decoder, calibration, the control law, and the online loop — against a signal
whose ground truth we planted. If it fails, the bug is ours, not the EEG's.

> **Windows note:** bare `python` on some machines resolves to the broken
> Microsoft Store stub. Run everything inside the activated conda env.

### Optional extras

Kept out of the core environment on purpose — both are fragile installs on
Windows, and neither is needed for analysis. A broken PsychoPy wheel should not
stop a teammate from running the test suite.

```bash
conda install -c conda-forge liblsl && pip install -e ".[stream]"   # live headset
pip install -e ".[session]"                                          # cue presentation
```

---

## Commands

| Command | What it does |
|---|---|
| `eegbot demo` | End-to-end on synthetic data. No downloads, no hardware. |
| `eegbot prepare-lee2019` | Download and harmonize the benchmark dataset to DSI-7 shape. |
| `eegbot evaluate --protocol cross_session` | Cross-validate a decoder, write a per-subject report. |
| `eegbot record --dry-run` | Rehearse a full recording session headless. |
| `eegbot online --model m.joblib --replay rec.fif` | Live decoding; prints turn rate. |
| `eegbot check-signal <session>` | Electrode-quality check on a recording. |

---

## Layout

```
configs/          YAML: preprocessing, paradigm, control, hardware
src/eegbot/
  sigproc/        THE shared core -- offline and online both go through it
  datasets/       Lee2019 harmonization, cropping, the EpochSet container
  decoding/       Riemannian pipelines, alignment, per-subject calibration
  control/        The steering law (pure) and the stateful controller
  evaluation/     CV protocols, scoring, and control simulation
  stream/         Sources (live/replay/synthetic) and the online loop
  session/        Trial scheduling, cues, markers, recording I/O
scripts/          Thin wrappers over the CLI
tests/            No network, no hardware
data/             gitignored
```

---

## Four invariants worth knowing before you change anything

**1. Offline and online share one preprocessing definition.**
`PreprocessSpec` → `preprocess(spec, x, causal=...)` is the only preprocessing
entry point. Offline uses zero-phase `filtfilt`; online uses a stateful causal
filter. They differ *only* in direction of application, and both are designed
from the same coefficients. Passing pre-cut epochs with `causal=True` raises,
because filtering each crop independently is not what the online loop does.

**2. Train on what you deploy on.**
The online loop classifies a 1-second window every 100 ms, so the model is
trained on 1-second crops at 100 ms hop — not on 3-second trial epochs. Training
on longer epochs than you deploy on produces a distribution mismatch that is
completely invisible in offline metrics.

**3. Never split a trial across CV folds.**
Crops from one trial are near-duplicates. Letting them land on both sides of a
split inflates accuracy toward 95% — a number that looks like a breakthrough and
means nothing. Every `EpochSet` carries source trial ids in `.groups`, every
protocol is group-aware, and `assert_no_group_leak` is called on every split.

**4. Microvolts convert to volts in exactly one function.**
[`session/recording.py`](src/eegbot/session/recording.py). A stray `1e6`
elsewhere would scale every threshold in the project by a million while still
producing plausible accuracy, because most of the pipeline is scale-invariant.
Catastrophic and invisible — hence one function and one round-trip test.

**If a benchmark comes back above 85%, treat it as a leak until proven
otherwise.** Check, in order: crops split across folds, a trial window that
includes cue onset, EMG contamination.

---

## Recording session runbook

A session is ~25 minutes of trials plus quality blocks. **Session time with a
human subject is the scarcest resource in this project** and a flawed session is
unrecoverable — you find out days later during analysis.

Rehearse first: `eegbot record --dry-run --speed 1` runs the real timings with
no headset.

### Before the subject arrives
- [ ] Charge the DSI-7. Confirm DSI-Streamer connects.
- [ ] `eegbot record --dry-run` to confirm markers flow.
- [ ] Have the impedance readout visible on a second screen.

### Order of operations
1. **Fit the headset.** Adjust until impedances settle. Log them.
2. **Alpha bookend (start)** — 60 s eyes open, 60 s eyes closed. With P3/Pz/P4
   you should see an obvious posterior alpha rise on eye closure. **If you do
   not, stop and refit.** This is the go/no-go on contact, and it is far cheaper
   to catch now than in analysis.
3. **Idle block** — 60 s relaxed, no imagery. This is what the dead zone and the
   centering bias are fitted from. Not optional.
4. **6 runs × 20 trials**, impedance logged before each, self-paced breaks
   between. Watch for drowsiness; MI degrades sharply in a tired subject.
5. **Alpha bookend (end)** — catches an electrode that drifted loose partway
   through, which is otherwise indistinguishable from a subject who did poorly.
6. **Write the notes field.** Drowsiness, movement, interruptions, "found it
   hard". It pays for itself the first time a session looks strange.

### Coaching the subject
The instruction script lives in
[`configs/paradigm/mi_2class.yaml`](configs/paradigm/mi_2class.yaml), versioned
so it is identical across sessions and operators. The essentials:

- **Kinesthetic, not visual.** "Feel your hand squeezing", never "picture a
  hand". Visual imagery decodes far worse.
- **No actual movement**, no jaw clenching, no eye movement during the arrow.
  Muscle activity looks like strong signal and teaches the system nothing.
- Blink freely during rest.

---

## Lee2019: what it is for, and what it is not for

[Lee2019](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_MI.html)
is 54 subjects × 2 sessions of left/right hand grasping imagery, 62 channels at
1000 Hz on a wet-gel BrainAmp, nose-referenced.

**For:** developing and testing the pipeline before hardware arrives, and
establishing a realistic 7-channel accuracy ceiling.

**Not for:** weights that will work on the DSI-7. Wet gel vs dry contact, 62
channels vs 7, lab impedances vs a student headset. Per-subject calibration on
the real device will always be required.

Harmonization order is load-bearing — see
[`datasets/lee2019.py`](src/eegbot/datasets/lee2019.py):

1. Extract events at 1000 Hz.
2. Set EMG channels aside (they are the confound detector, not features).
3. **Re-reference nose → linked mastoids, before dropping channels.** Nose-
   referenced data is `V_ch − V_nose`; subtracting `mean(TP9, TP10)` cancels the
   nose term exactly. Covariance decoders are reference-sensitive, so this is
   not cosmetic.
4. Pick the 7 DSI-7 channels in canonical order.
5. Resample to 300 Hz **passing events through**, so markers are re-indexed
   rather than jittered.

Trials are cut **0.5–3.5 s** post-cue. The 0.5 s offset excludes the cue-onset
visual evoked response — without it the classifier learns the arrow, not the
imagery, and reports excellent accuracy that vanishes online.

---

## Hardware notes

- **300 Hz** sampling (a 600 Hz upgrade exists; not ordered). Analog bandwidth
  0.003–150 Hz, so Nyquist and the hardware low-pass are well matched.
- **Fz is a driven ground**, not a reference — it is not available as a data
  channel and needs no counterpart transform when harmonizing.
- **Markers** currently go over LSL from the cue program. The DSI-7's 4-bit
  hardware `TRG` input would remove software-timing jitter entirely and is the
  better option if marker latency turns out to matter — not implemented yet.
- Only one program can hold the device at a time. Close DSI-Streamer before
  running `dsi2lsl`.

---

## Status

Done: signal-processing core, Lee2019 harmonization, cropping and leak-free CV,
Riemannian decoder with session alignment, control law and calibration,
drivability metrics, synthetic generator, replay sources, online loop, session
runner. ~115 tests, all offline.

Not done: **the robot.** The boundary is `ControlCommand` (turn rate +
throttle); wiring it to motors is deliberately a separate piece of work. Also
pending: validation against real DSI-7 data, hardware trigger support, and deep
models — the last only once the classical baseline is beaten on real recordings.
