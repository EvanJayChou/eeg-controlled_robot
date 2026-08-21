"""Re-referencing.

Only ever needed for foreign datasets. The DSI-7 already references to ear
clips, so nothing in the online path calls this.

## The nose-to-mastoid transform

Lee2019 is nose-referenced: each stored channel is ``V_ch - V_nose``.
Subtracting the mean of the two mastoids gives

    (V_ch - V_nose) - mean(V_TP9 - V_nose, V_TP10 - V_nose)
      = V_ch - mean(V_TP9, V_TP10)

so the nose term cancels exactly and the result is true linked-mastoid data --
the closest available proxy for the DSI-7's ear clips. This is not cosmetic:
covariance-based decoders are reference-sensitive, and training on
nose-referenced covariances would mis-specify the geometry the model learns.

The DSI-7's Fz common-mode follower is a *driven ground*, not a reference, and
needs no counterpart transform.
"""

from __future__ import annotations

import mne

from eegbot.constants import LEE2019_MASTOIDS


def rereference_to_mastoids(
    raw: mne.io.BaseRaw,
    mastoids: tuple[str, ...] = LEE2019_MASTOIDS,
) -> mne.io.BaseRaw:
    """Re-reference to linked mastoids.

    Must run **before** channel selection -- the mastoid channels have to still
    be present. Callers in `eegbot.datasets.lee2019` depend on that ordering.
    """
    missing = [ch for ch in mastoids if ch not in raw.ch_names]
    if missing:
        raise ValueError(
            f"mastoid channels {missing} absent; re-reference before picking channels"
        )
    out = raw.copy().set_eeg_reference(ref_channels=list(mastoids), verbose=False)
    return out
