#!/usr/bin/env python
"""Cross-validate a decoder and write a per-subject report.

Equivalent to `eegbot evaluate`.

    python scripts/evaluate.py --decoder riemann_ts_lr --protocol cross_session

`cross_session` is the number worth quoting: it answers whether yesterday's
calibration still works today. Anything above 85% mean should be treated as a
leak until proven otherwise.
"""

import sys

from eegbot.cli import main

if __name__ == "__main__":
    sys.exit(main(["evaluate", *sys.argv[1:]]))
