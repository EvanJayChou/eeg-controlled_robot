#!/usr/bin/env python
"""Download Lee2019 and harmonize it to DSI-7 shape.

Equivalent to `eegbot prepare-lee2019`. Kept as a script for discoverability.

    python scripts/prepare_lee2019.py --subjects 1 2 3

Downloads are large and slow; results are cached to data/processed/ and the
script is idempotent, so re-running skips whatever already exists.
"""

import sys

from eegbot.cli import main

if __name__ == "__main__":
    sys.exit(main(["prepare-lee2019", *sys.argv[1:]]))
