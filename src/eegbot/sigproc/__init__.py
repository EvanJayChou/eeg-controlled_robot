"""Shared signal-processing core.

Both the offline analysis path and the online decoding loop go through
`preprocess`, which is what keeps them from drifting apart. See `apply.py` for
the reasoning behind the harmonization/preprocessing split.

Only the numpy-level API is re-exported here. `montage` and `reference` operate
on MNE objects and are imported explicitly from their modules, so that the core
-- and anything that only needs filtering, cropping or control -- does not pull
in MNE at import time.
"""

from eegbot.sigproc.apply import preprocess
from eegbot.sigproc.artifacts import flat_channels, peak_to_peak, reject_mask
from eegbot.sigproc.filters import CausalFilter, design_sos, warmup_samples
from eegbot.sigproc.spec import PreprocessSpec

__all__ = [
    "CausalFilter",
    "PreprocessSpec",
    "design_sos",
    "flat_channels",
    "peak_to_peak",
    "preprocess",
    "reject_mask",
    "warmup_samples",
]
