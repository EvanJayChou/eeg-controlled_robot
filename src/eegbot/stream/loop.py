"""The online decoding loop.

    source -> ring buffer -> causal filter -> covariance -> p(right) -> controller

Three details here are the difference between a loop that works and one that
looks like it works:

**Causal filtering, with persistent state.** The offline path uses `filtfilt`,
which reads future samples. Online that is impossible. `CausalFilter` carries its
state across chunks so that filtering in many small reads gives bit-identical
output to filtering the whole signal at once -- a stateless per-chunk filter
would inject an edge transient every 100 ms.

**Warmup.** A 4 Hz high-pass takes about a second to settle. The loop discards
`warmup_samples` before emitting anything; without this the first few seconds of
every session are garbage the rider would experience as random turning.

**Alignment on unlabelled data.** If the decoder was trained elsewhere -- another
session, another subject -- its Riemannian reference belongs to that data. The
loop collects a short buffer at startup and calls `adapt`, re-homing the
alignment onto the current session before any command is emitted. No labels
required, which is why it can happen while the subject simply sits still.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from eegbot.control.controller import ContinuousController, ControlCommand
from eegbot.sigproc.filters import CausalFilter, warmup_samples
from eegbot.sigproc.spec import PreprocessSpec
from eegbot.stream.source import EEGSource

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Update:
    """One tick of the loop.

    `end_sample` is the index, in the source's own sample numbering, one past
    the last sample this decision saw. It is what lets an offline analysis line
    updates up against an event timeline -- without it, callers have to
    re-derive the warmup and buffer-fill offsets by hand and will get it wrong.
    """

    index: int
    time_s: float
    p_right: float
    command: ControlCommand
    end_sample: int
    window_samples: int

    @property
    def center_sample(self) -> int:
        """Middle of the window this decision was based on."""
        return self.end_sample - self.window_samples // 2


class RingBuffer:
    """Fixed-length rolling window of the most recent samples."""

    def __init__(self, n_channels: int, n_samples: int) -> None:
        self.buffer = np.zeros((n_channels, n_samples))
        self.n_samples = n_samples
        self._filled = 0

    def push(self, chunk: np.ndarray) -> None:
        n = chunk.shape[-1]
        if n >= self.n_samples:
            self.buffer[:] = chunk[:, -self.n_samples :]
        else:
            self.buffer[:, :-n] = self.buffer[:, n:]
            self.buffer[:, -n:] = chunk
        self._filled = min(self._filled + n, self.n_samples)

    @property
    def ready(self) -> bool:
        return self._filled >= self.n_samples

    def snapshot(self) -> np.ndarray:
        return self.buffer.copy()


class OnlineDecoder:
    """Drives a fitted model over a live or replayed source.

    Parameters
    ----------
    model
        A fitted pipeline exposing `predict_proba`. Must have been trained on
        crops of `spec.window_s`, or the input distribution will not match --
        see `eegbot.datasets.crops`.
    positive_index
        Column of `predict_proba` corresponding to `constants.POSITIVE_CLASS`.
    align_seconds
        Seconds of unlabelled data to collect for Riemannian re-alignment before
        emitting commands. Zero disables adaptation.
    """

    def __init__(
        self,
        spec: PreprocessSpec,
        model,
        controller: ContinuousController,
        *,
        positive_index: int = 0,
        align_seconds: float = 30.0,
        idle_gate=None,
    ) -> None:
        self.spec = spec
        self.model = model
        self.controller = controller
        self.positive_index = positive_index
        self.align_seconds = align_seconds
        self.idle_gate = idle_gate

    def run(
        self,
        source: EEGSource,
        *,
        max_updates: int | None = None,
        on_update: Callable[[Update], None] | None = None,
    ) -> list[Update]:
        """Run to exhaustion (or `max_updates`), returning every update."""
        return list(self.stream(source, max_updates=max_updates, on_update=on_update))

    def stream(
        self,
        source: EEGSource,
        *,
        max_updates: int | None = None,
        on_update: Callable[[Update], None] | None = None,
    ) -> Iterator[Update]:
        spec = self.spec
        filt = CausalFilter(spec, n_channels=spec.n_channels)
        buffer = RingBuffer(spec.n_channels, spec.window_samples)

        source.start()
        self.controller.reset()

        try:
            consumed = self._warmup(source, filt, buffer)
            log.debug("discarded %d warmup samples", consumed)
            consumed += self._align(source, filt, buffer)

            index = 0
            while max_updates is None or index < max_updates:
                chunk = source.read(spec.hop_samples)
                if chunk is None:
                    return
                consumed += chunk.shape[-1]
                buffer.push(filt(chunk))
                if not buffer.ready:
                    continue

                window = buffer.snapshot()
                p_right = self._predict(window)
                engagement = (
                    1.0 if self.idle_gate is None else self.idle_gate.engagement_one(window)
                )
                command = self.controller.update(p_right, engagement)
                update = Update(
                    index=index,
                    time_s=index * spec.hop_s,
                    p_right=p_right,
                    command=command,
                    end_sample=consumed,
                    window_samples=spec.window_samples,
                )
                if on_update is not None:
                    on_update(update)
                yield update
                index += 1
        finally:
            source.stop()

    # === Internals ===

    def _predict(self, window: np.ndarray) -> float:
        proba = self.model.predict_proba(window[None, ...])
        return float(proba[0, self.positive_index])

    def _warmup(self, source: EEGSource, filt: CausalFilter, buffer: RingBuffer) -> int:
        """Consume and discard filter settling time."""
        needed = warmup_samples(self.spec)
        consumed = 0
        while consumed < needed:
            chunk = source.read(min(self.spec.hop_samples * 10, needed - consumed))
            if chunk is None:
                break
            filt(chunk)
            consumed += chunk.shape[-1]
        return consumed

    def _align(self, source: EEGSource, filt: CausalFilter, buffer: RingBuffer) -> int:
        """Re-home Riemannian alignment onto this session, if the model has it.

        Returns the number of samples consumed, so the caller can keep
        `Update.end_sample` aligned with the source's numbering.
        """
        align_step = getattr(self.model, "named_steps", {}).get("align")
        if align_step is None or self.align_seconds <= 0:
            return 0

        spec = self.spec
        n_windows = max(int(self.align_seconds / spec.hop_s), 10)
        windows: list[np.ndarray] = []
        consumed = 0
        for _ in range(n_windows):
            chunk = source.read(spec.hop_samples)
            if chunk is None:
                break
            consumed += chunk.shape[-1]
            buffer.push(filt(chunk))
            if buffer.ready:
                windows.append(buffer.snapshot())

        if len(windows) < 10:
            log.warning(
                "only %d windows available for alignment; skipping adaptation", len(windows)
            )
            return consumed

        from eegbot.evaluation.metrics import adapt_alignment

        adapt_alignment(self.model, np.stack(windows))
        log.info("aligned to current session using %d unlabelled windows", len(windows))
        return consumed
