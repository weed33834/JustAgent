"""Download and verify plugin package integrity before installation."""

from __future__ import annotations

import atexit
import base64
import shutil
import subprocess
import tempfile
from pathlib import Path

import structlog
from cryptography.exceptions import InvalidSignature  # pyright: ignore[reportMissingImports]
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,  # pyright: ignore[reportMissingImports]
)

from justagent.exceptions import PluginError
from justagent.utils.hashing import compute_sha256, pip_cmd

logger = structlog.get_logger("justagent")


class PackageVerificationError(PluginError):
    """Raised when a plugin package fails integrity or signature verification."""


class PackageDownloadError(PluginError):
    """Raised when a plugin package cannot be downloaded."""


def verify_package(
    package_path: Path,
    *,
    sha256_hex: str | None,
    signature_b64: str | None,
    public_key_b64: str | None,
) -> None:
    """Verify a downloaded package against an expected sha256 and/or signature.

    When ``signature_b64`` is provided, ``sha256_hex`` and ``public_key_b64``
    must also be provided. The signature is validated over the sha256 hex string.

    Raises:
        PackageVerificationError: If any check fails.
    """
    actual_sha256 = compute_sha256(package_path)

    if sha256_hex is not None and actual_sha256 != sha256_hex:
        raise PackageVerificationError(
            "Package sha256 mismatch",
            details={"expected": sha256_hex, "actual": actual_sha256},
        )

    if signature_b64 is not None:
        if sha256_hex is None:
            raise PackageVerificationError(
                "Package signature requires a sha256 hash",
            )
        if public_key_b64 is None:
            raise PackageVerificationError(
                "Package signature present but no public key configured",
            )
        try:
            public_key_bytes = base64.b64decode(public_key_b64, validate=True)
            signature_bytes = base64.b64decode(signature_b64, validate=True)
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            public_key.verify(signature_bytes, sha256_hex.encode("ascii"))
        except (ValueError, InvalidSignature) as exc:
            raise PackageVerificationError(
                "Package signature verification failed",
                details={"reason": str(exc)},
            ) from exc


def download_package(source_for_pip: str, output_dir: Path) -> Path:
    """Download a package to a directory without installing it.

    Returns the path to the downloaded wheel or sdist.

    Raises:
        PackageDownloadError: If the download fails or no file is produced.
    """
    cmd = pip_cmd()
    args = [*cmd, "download", "--no-deps", "--quiet", "-d", str(output_dir), source_for_pip]
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise PackageDownloadError(
            f"Failed to download package {source_for_pip}",
            details={"reason": str(exc), "stderr": getattr(exc, "stderr", "")},
        ) from exc

    if result.returncode != 0:
        raise PackageDownloadError(
            f"Failed to download package {source_for_pip}",
            details={"stderr": result.stderr},
        )

    # pip with PEP 658 (default on pip>=23) may also drop a ``.metadata``
    # sidecar next to the wheel/sdist, so filter to actual package files.
    packages = [f for f in output_dir.iterdir() if f.name.endswith((".whl", ".tar.gz", ".zip"))]
    if len(packages) != 1:
        raise PackageDownloadError(
            f"Expected exactly one downloaded file for {source_for_pip}, found {len(packages)}",
        )
    return packages[0]


def download_and_verify(
    source_for_pip: str,
    *,
    sha256_hex: str | None,
    signature_b64: str | None,
    public_key_b64: str | None,
    dest_dir: Path | None = None,
) -> Path:
    """Download a package and verify it before returning the local file path.

    The verified file is moved out of the transient download directory into a
    stable location so the caller can use it after this function returns.

    Parameters
    ----------
    dest_dir:
        Optional stable directory to place the verified file in. When
        provided, the caller owns the lifecycle of *dest_dir* (e.g. a plugin
        cache directory). When omitted, a fresh temporary directory is
        created and registered for removal at interpreter exit so that
        short-lived CLI processes do not leak empty directories even if the
        caller only deletes the returned file.

    The caller is responsible for deleting the returned file after
    installation. The file's parent directory is the temp dir (or *dest_dir*)
    and may also be removed once the file is no longer needed.
    """
    with tempfile.TemporaryDirectory(prefix="justagent-pkg-") as tmp:
        tmp_path = Path(tmp)
        package_path = download_package(source_for_pip, tmp_path)
        verify_package(
            package_path,
            sha256_hex=sha256_hex,
            signature_b64=signature_b64,
            public_key_b64=public_key_b64,
        )
        # Move out of the temporary directory so the caller can keep it.
        if dest_dir is not None:
            final_dir = dest_dir
        else:
            final_dir = Path(tempfile.mkdtemp(prefix="justagent-pkg-"))
            # Guarantee cleanup of the temp dir even if the caller only
            # unlinks the file (the previous mkdtemp dir leaked otherwise).
            atexit.register(shutil.rmtree, final_dir, ignore_errors=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / package_path.name
        shutil.move(str(package_path), str(final_path))
    return final_path
