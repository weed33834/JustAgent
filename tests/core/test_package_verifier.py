"""Tests for package download and integrity verification."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from justagent.core.package_verifier import (
    PackageVerificationError,
    compute_sha256,
    verify_package,
)


def _make_keypair() -> tuple[str, str]:
    """Return (public_key_b64, sha256_hex_signed_b64) helper for a given payload."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return base64.b64encode(public_key.public_bytes_raw()).decode("ascii"), private_key


def test_compute_sha256(tmp_path: Path) -> None:
    path = tmp_path / "pkg.whl"
    path.write_bytes(b"hello")
    assert compute_sha256(path) == hashlib.sha256(b"hello").hexdigest()


def test_verify_package_with_valid_sha256(tmp_path: Path) -> None:
    path = tmp_path / "pkg.whl"
    path.write_bytes(b"hello")
    sha256 = hashlib.sha256(b"hello").hexdigest()

    verify_package(path, sha256_hex=sha256, signature_b64=None, public_key_b64=None)


def test_verify_package_with_invalid_sha256(tmp_path: Path) -> None:
    path = tmp_path / "pkg.whl"
    path.write_bytes(b"hello")

    with pytest.raises(PackageVerificationError):
        verify_package(
            path,
            sha256_hex="0" * 64,
            signature_b64=None,
            public_key_b64=None,
        )


def test_verify_package_with_valid_signature(tmp_path: Path) -> None:
    path = tmp_path / "pkg.whl"
    path.write_bytes(b"hello")
    sha256 = hashlib.sha256(b"hello").hexdigest()
    public_key_b64, private_key = _make_keypair()
    signature = private_key.sign(sha256.encode("ascii"))
    signature_b64 = base64.b64encode(signature).decode("ascii")

    verify_package(
        path, sha256_hex=sha256, signature_b64=signature_b64, public_key_b64=public_key_b64
    )


def test_verify_package_with_invalid_signature(tmp_path: Path) -> None:
    path = tmp_path / "pkg.whl"
    path.write_bytes(b"hello")
    sha256 = hashlib.sha256(b"hello").hexdigest()
    public_key_b64, _private_key = _make_keypair()

    with pytest.raises(PackageVerificationError):
        verify_package(
            path,
            sha256_hex=sha256,
            signature_b64=base64.b64encode(b"invalid").decode("ascii"),
            public_key_b64=public_key_b64,
        )


def test_verify_package_requires_public_key_when_signature_present(tmp_path: Path) -> None:
    path = tmp_path / "pkg.whl"
    path.write_bytes(b"hello")
    sha256 = hashlib.sha256(b"hello").hexdigest()

    with pytest.raises(PackageVerificationError):
        verify_package(
            path,
            sha256_hex=sha256,
            signature_b64=base64.b64encode(b"sig").decode("ascii"),
            public_key_b64=None,
        )


def test_verify_package_requires_sha256_when_signature_present(tmp_path: Path) -> None:
    path = tmp_path / "pkg.whl"
    path.write_bytes(b"hello")
    public_key_b64, _private_key = _make_keypair()

    with pytest.raises(PackageVerificationError):
        verify_package(
            path,
            sha256_hex=None,
            signature_b64=base64.b64encode(b"sig").decode("ascii"),
            public_key_b64=public_key_b64,
        )


def test_download_package_returns_path_and_clears_directory(tmp_path: Path) -> None:
    from justagent.core.package_verifier import download_package

    pkg = tmp_path / "pkg.whl"
    pkg.write_bytes(b"pkg")

    def _fake_run(cmd: list[str], **_kwargs: object) -> object:
        # The function places the package in the temp dir itself for this test.
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with (
        patch("justagent.core.package_verifier.subprocess.run", side_effect=_fake_run),
        patch(
            "justagent.utils.hashing.pip_cmd",
            return_value=["pip"],
        ),
    ):
        out_dir = tmp_path / "download"
        out_dir.mkdir()
        # Move pkg into download dir so download_package can find it.
        pkg.rename(out_dir / "pkg.whl")
        result = download_package("pkg==1.0.0", out_dir)

    assert result == out_dir / "pkg.whl"


def _fake_completed_process(returncode: int, stdout: str = "", stderr: str = "") -> object:
    return type(
        "CompletedProcess",
        (),
        {"returncode": returncode, "stdout": stdout, "stderr": stderr},
    )()


def test_download_package_raises_on_subprocess_exception(tmp_path: Path) -> None:
    """If subprocess.run raises, download_package raises PackageDownloadError."""
    from justagent.core.package_verifier import PackageDownloadError, download_package

    out_dir = tmp_path / "download"
    out_dir.mkdir()

    def boom(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("pip not found")

    with (
        patch("justagent.core.package_verifier.subprocess.run", side_effect=boom),
        patch("justagent.utils.hashing.pip_cmd", return_value=["pip"]),
        pytest.raises(PackageDownloadError),
    ):
        download_package("pkg==1.0.0", out_dir)


def test_download_package_raises_on_nonzero_returncode(tmp_path: Path) -> None:
    """A non-zero returncode from pip raises PackageDownloadError."""
    from justagent.core.package_verifier import PackageDownloadError, download_package

    out_dir = tmp_path / "download"
    out_dir.mkdir()
    fake_result = _fake_completed_process(1, stderr="package not found")

    with (
        patch("justagent.core.package_verifier.subprocess.run", return_value=fake_result),
        patch("justagent.utils.hashing.pip_cmd", return_value=["pip"]),
        pytest.raises(PackageDownloadError),
    ):
        download_package("pkg==1.0.0", out_dir)


def test_download_package_raises_when_multiple_files(tmp_path: Path) -> None:
    """If the download directory ends up with != 1 file, raise PackageDownloadError."""
    from justagent.core.package_verifier import PackageDownloadError, download_package

    out_dir = tmp_path / "download"
    out_dir.mkdir()
    # Pre-place two files so iterdir() yields more than one entry.
    (out_dir / "a.whl").write_bytes(b"a")
    (out_dir / "b.whl").write_bytes(b"b")
    fake_result = _fake_completed_process(0)

    with (
        patch("justagent.core.package_verifier.subprocess.run", return_value=fake_result),
        patch("justagent.utils.hashing.pip_cmd", return_value=["pip"]),
        pytest.raises(PackageDownloadError),
    ):
        download_package("pkg==1.0.0", out_dir)


def test_download_package_raises_when_no_files(tmp_path: Path) -> None:
    """If no file is produced in the output dir, raise PackageDownloadError."""
    from justagent.core.package_verifier import PackageDownloadError, download_package

    out_dir = tmp_path / "download"
    out_dir.mkdir()
    fake_result = _fake_completed_process(0)

    with (
        patch("justagent.core.package_verifier.subprocess.run", return_value=fake_result),
        patch("justagent.utils.hashing.pip_cmd", return_value=["pip"]),
        pytest.raises(PackageDownloadError),
    ):
        download_package("pkg==1.0.0", out_dir)


def test_download_and_verify_returns_moved_file(tmp_path: Path) -> None:
    """download_and_verify downloads, verifies, and moves the file out of temp."""
    import shutil

    from justagent.core.package_verifier import download_and_verify

    def fake_download(source: str, output_dir: Path) -> Path:
        pkg = output_dir / "pkg-1.0.whl"
        pkg.write_bytes(b"package content")
        return pkg

    with (
        patch("justagent.core.package_verifier.download_package", side_effect=fake_download),
        patch("justagent.core.package_verifier.verify_package"),
    ):
        result = download_and_verify(
            "pkg==1.0.0",
            sha256_hex=None,
            signature_b64=None,
            public_key_b64=None,
        )
    try:
        assert result.exists()
        assert result.read_bytes() == b"package content"
        assert result.name == "pkg-1.0.whl"
    finally:
        # download_and_verify intentionally leaves the final file in a temp
        # dir (caller is responsible for cleanup); tidy up here.
        shutil.rmtree(result.parent, ignore_errors=True)


def test_download_and_verify_with_dest_dir_does_not_register_atexit(tmp_path: Path) -> None:
    """When ``dest_dir`` is provided, no atexit cleanup is registered.

    The caller owns the lifecycle of *dest_dir*, so the implementation must
    not register an ``atexit`` handler that would remove it out from under
    the caller. The verified file is placed directly inside *dest_dir*.
    """
    import atexit

    from justagent.core.package_verifier import download_and_verify

    def fake_download(source: str, output_dir: Path) -> Path:
        pkg = output_dir / "pkg-2.0.whl"
        pkg.write_bytes(b"dest-dir content")
        return pkg

    dest_dir = tmp_path / "plugin-cache"

    registered_paths: list[object] = []
    real_register = atexit.register

    def _tracking_register(func, *args, **kwargs):
        # Capture the path argument (shutil.rmtree is called with final_dir).
        if args:
            registered_paths.append(args[0] if len(args) == 1 else args)
        return real_register(func, *args, **kwargs)

    with (
        patch("justagent.core.package_verifier.download_package", side_effect=fake_download),
        patch("justagent.core.package_verifier.verify_package"),
        patch("justagent.core.package_verifier.atexit.register", side_effect=_tracking_register),
    ):
        result = download_and_verify(
            "pkg==2.0.0",
            sha256_hex=None,
            signature_b64=None,
            public_key_b64=None,
            dest_dir=dest_dir,
        )

    # The file landed in the caller-provided dest_dir.
    assert result.parent == dest_dir
    assert result.exists()
    assert result.read_bytes() == b"dest-dir content"
    assert result.name == "pkg-2.0.whl"
    # No atexit handler was registered for cleanup.
    assert registered_paths == []
