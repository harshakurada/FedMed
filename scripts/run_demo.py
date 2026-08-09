"""The single recruiter-facing FedMed demo command.

    .venv\\Scripts\\python.exe scripts\\run_demo.py
    .venv\\Scripts\\python.exe scripts\\run_demo.py --live

A thin wrapper around `server.demo.run_demo` (same pattern as
`scripts/generate_dev_certs.py`) so the demo can be launched without `python -m` module
syntax. See `server/demo/run_demo.py` for what it actually does, and
`docs/dashboard.md`/the README's "Demo" section for the full two-command demo (this
script for the backend, `npm start` in `dashboard/` for the frontend).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.demo.run_demo import main  # noqa: E402

if __name__ == "__main__":
    main()
