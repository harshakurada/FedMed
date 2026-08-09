"""Module 13 demo-mode configuration: a single, explicit toggle.

`DEMO_MODE=true` (the default -- see below): the demo round runs against small,
clearly-labeled synthetic (non-medical) data, generated in-memory, so the full
DP + CKKS + mTLS + dashboard pipeline can be demonstrated in seconds without needing a
real local BraTS2020 copy. `DEMO_MODE=false`: the exact same pipeline runs against real
local BraTS2020 data (`FEDMED_BRATS_ROOT` must already be set -- this project never
auto-downloads it).

Default is `true` -- this is the "safe" default the Module 13 spec asks for: it never
requires an external dataset to be configured, and it never risks accidentally kicking
off a run against real patient data just because someone ran the demo script without
first checking their environment variables.

Deliberately named `DEMO_MODE`, not `FEDMED_DEMO_MODE` like every other config in this
project (`server/federated/dp/dp_config.py`, `server/federated/encrypted/ckks_config.py`,
...) -- Module 13's own spec names this exact variable, and matching it exactly matters
more here than internal naming consistency.
"""

from __future__ import annotations

import os


def demo_mode_enabled() -> bool:
    raw = os.environ.get("DEMO_MODE")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}
