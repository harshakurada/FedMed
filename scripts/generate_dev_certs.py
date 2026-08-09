"""Development-only certificate generator for the FedMed mTLS coordination service.

    .venv\\Scripts\\python.exe scripts\\generate_dev_certs.py
    .venv\\Scripts\\python.exe scripts\\generate_dev_certs.py --force
    .venv\\Scripts\\python.exe scripts\\generate_dev_certs.py --out-dir certs --hospital-ids hospital_a hospital_b hospital_c

Generates a dev CA + server certificate + one client certificate per hospital under
`--out-dir` (default `certs/`, gitignored). Requires `openssl` on PATH -- this script
does not install software on your behalf, and fails clearly if `openssl` is missing.
Refuses to overwrite an existing certificate directory unless `--force` is passed.

These are LOCAL DEVELOPMENT certificates only. See docs/secure_communication.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Runs as a standalone script (not `python -m`), so the repo root must be added to
# sys.path explicitly for `server.federated...` to be importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.federated.grpc_service.certs import OpenSSLNotFoundError, generate_dev_certificates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="certs", help="Directory to write certificates into (default: certs/).")
    parser.add_argument(
        "--hospital-ids",
        nargs="+",
        default=["hospital_a", "hospital_b", "hospital_c"],
        help="Hospital identities to issue client certificates for.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing certificate directory.")
    args = parser.parse_args()

    try:
        paths = generate_dev_certificates(Path(args.out_dir), args.hospital_ids, force=args.force)
    except OpenSSLNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"CA certificate:      {paths.ca_cert}")
    print(f"CA private key:      {paths.ca_key}  (NEVER commit this)")
    print(f"Server certificate:  {paths.server_cert}")
    print(f"Server private key:  {paths.server_key}  (NEVER commit this)")
    for hospital_id, cert in paths.client_certs.items():
        print(f"{hospital_id} certificate: {cert}")
        print(f"{hospital_id} private key: {paths.client_keys[hospital_id]}  (NEVER commit this)")
    print(f"\n{Path(args.out_dir)} is covered by .gitignore -- verify with `git check-ignore -v {args.out_dir}\\ca.key`.")


if __name__ == "__main__":
    main()
