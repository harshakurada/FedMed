"""Development-only certificate generation for the FedMed mTLS coordination service.

Shells out to `openssl` (a system tool, not a new Python dependency) rather than adding a
Python crypto library. Generates a dev CA, a server certificate (SAN `localhost`/
`127.0.0.1`, used by `health_server.py`), and one client certificate per hospital
(CN = hospital id -- exactly what `health_server.py` reads back from the verified
connection to associate an authenticated identity with it). Never overwrites existing
certificates unless `force=True`.

These are LOCAL DEVELOPMENT certificates only: a self-signed CA, no revocation, no
external trust. Never use them for anything beyond this local simulation. Private keys
are written under the caller's `output_dir`, which must be gitignored (see `.gitignore`'s
`certs/` entry) -- never committed.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class OpenSSLNotFoundError(RuntimeError):
    """Raised when the `openssl` CLI is not on PATH. Never silently skipped or faked."""


@dataclass(frozen=True)
class CertPaths:
    ca_cert: Path
    ca_key: Path
    server_cert: Path
    server_key: Path
    client_certs: dict[str, Path]
    client_keys: dict[str, Path]


def _require_openssl() -> str:
    openssl = shutil.which("openssl")
    if openssl is None:
        raise OpenSSLNotFoundError(
            "openssl was not found on PATH. Install OpenSSL (e.g. Git for Windows bundles "
            "one) and ensure it's on PATH, then re-run this script. FedMed does not "
            "install software on your behalf."
        )
    return openssl


def _run(openssl: str, *args: str) -> None:
    result = subprocess.run([openssl, *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"openssl {' '.join(args)} failed:\n{result.stderr}")


def _generate_ca(openssl: str, out_dir: Path) -> tuple[Path, Path]:
    ca_key = out_dir / "ca.key"
    ca_cert = out_dir / "ca.crt"
    _run(
        openssl,
        "req",
        "-x509",
        "-newkey",
        "rsa:4096",
        "-days",
        "3650",
        "-nodes",
        "-keyout",
        str(ca_key),
        "-out",
        str(ca_cert),
        "-subj",
        "/CN=FedMed Dev CA",
    )
    return ca_cert, ca_key


def _generate_signed_cert(
    openssl: str,
    out_dir: Path,
    name: str,
    common_name: str,
    ca_cert: Path,
    ca_key: Path,
    san: str | None = None,
) -> tuple[Path, Path]:
    key_path = out_dir / f"{name}.key"
    csr_path = out_dir / f"{name}.csr"
    cert_path = out_dir / f"{name}.crt"

    _run(openssl, "genrsa", "-out", str(key_path), "2048")
    _run(openssl, "req", "-new", "-key", str(key_path), "-out", str(csr_path), "-subj", f"/CN={common_name}")

    sign_args = [
        "x509",
        "-req",
        "-in",
        str(csr_path),
        "-CA",
        str(ca_cert),
        "-CAkey",
        str(ca_key),
        "-CAcreateserial",
        "-out",
        str(cert_path),
        "-days",
        "825",
        "-sha256",
    ]
    extfile_path: Path | None = None
    if san:
        extfile_path = out_dir / f"{name}.ext.cnf"
        extfile_path.write_text(f"subjectAltName={san}\n")
        sign_args += ["-extfile", str(extfile_path)]

    _run(openssl, *sign_args)
    csr_path.unlink(missing_ok=True)
    if extfile_path is not None:
        extfile_path.unlink(missing_ok=True)
    return cert_path, key_path


def generate_dev_certificates(output_dir: Path, hospital_ids: list[str], force: bool = False) -> CertPaths:
    """Generate a dev CA + server cert + one client cert per hospital under `output_dir`.

    Raises `OpenSSLNotFoundError` if `openssl` isn't available, and `FileExistsError` if
    `output_dir` already has certificate material and `force` is not set -- never
    silently overwrites certificates a running server/client might already trust.
    """
    openssl = _require_openssl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("*.crt")) + list(output_dir.glob("*.key"))
    if existing and not force:
        raise FileExistsError(
            f"{output_dir} already contains certificate material ({len(existing)} file(s)). "
            "Pass force=True (or --force on the CLI) to regenerate -- this invalidates any "
            "certificates a currently running server/client already trusts."
        )
    for path in existing:
        path.unlink()

    ca_cert, ca_key = _generate_ca(openssl, output_dir)
    server_cert, server_key = _generate_signed_cert(
        openssl, output_dir, "server", "localhost", ca_cert, ca_key, san="DNS:localhost,IP:127.0.0.1"
    )

    client_certs: dict[str, Path] = {}
    client_keys: dict[str, Path] = {}
    for hospital_id in hospital_ids:
        cert, key = _generate_signed_cert(openssl, output_dir, hospital_id, hospital_id, ca_cert, ca_key)
        client_certs[hospital_id] = cert
        client_keys[hospital_id] = key

    return CertPaths(
        ca_cert=ca_cert,
        ca_key=ca_key,
        server_cert=server_cert,
        server_key=server_key,
        client_certs=client_certs,
        client_keys=client_keys,
    )
