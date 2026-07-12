"""Tests for filesystem permission helpers."""

from __future__ import annotations

import logging
import stat
from pathlib import Path

from pyfakefs.fake_filesystem_unittest import Patcher

from autoship.utils.permissions import (
    ensure_dir_permissions,
    ensure_file_permissions,
    warn_if_too_broad,
)


def _make_file(fs, path_str: str, content: str = "content", mode: int = 0o644) -> Path:
    """Create a file in the fake filesystem with the given mode."""
    fs.create_dir(str(Path(path_str).parent))
    path = Path(path_str)
    path.write_text(content)
    path.chmod(mode)
    return path


class TestEnsureDirPermissions:
    """Tests for ensure_dir_permissions."""

    def test_creates_directory(self) -> None:
        with Patcher():
            path = Path("/fake/testdir")
            ensure_dir_permissions(path, 0o700)
            assert path.exists()
            assert path.is_dir()

    def test_creates_nested_directories(self) -> None:
        with Patcher():
            path = Path("/fake/a/b/c")
            ensure_dir_permissions(path, 0o700)
            assert path.exists()

    def test_sets_mode_on_new_directory(self) -> None:
        with Patcher():
            path = Path("/fake/testdir")
            ensure_dir_permissions(path, 0o700)
            current = stat.S_IMODE(path.stat().st_mode)
            assert current == 0o700

    def test_tightens_too_broad_permissions(self, caplog) -> None:
        with Patcher():
            path = Path("/fake/testdir")
            path.mkdir(parents=True)
            path.chmod(0o777)
            with caplog.at_level(logging.WARNING):
                ensure_dir_permissions(path, 0o700)
            assert "too broad" in caplog.text
            current = stat.S_IMODE(path.stat().st_mode)
            assert current == 0o700


class TestEnsureFilePermissions:
    """Tests for ensure_file_permissions."""

    def test_sets_mode_on_existing_file(self) -> None:
        with Patcher() as patcher:
            path = _make_file(patcher.fs, "/fake/testfile", mode=0o644)
            ensure_file_permissions(path, 0o600)
            current = stat.S_IMODE(path.stat().st_mode)
            assert current == 0o600

    def test_no_warning_when_permissions_are_already_tight(self, caplog) -> None:
        with Patcher() as patcher:
            path = _make_file(patcher.fs, "/fake/testfile", mode=0o600)
            with caplog.at_level(logging.WARNING):
                ensure_file_permissions(path, 0o700)
            assert "too broad" not in caplog.text

    def test_warns_on_too_broad_permissions(self, caplog) -> None:
        with Patcher() as patcher:
            path = _make_file(patcher.fs, "/fake/testfile", mode=0o777)
            with caplog.at_level(logging.WARNING):
                ensure_file_permissions(path, 0o600)
            assert "too broad" in caplog.text
            current = stat.S_IMODE(path.stat().st_mode)
            assert current == 0o600


class TestWarnIfTooBroad:
    """Tests for warn_if_too_broad."""

    def test_warns_when_permissions_are_too_broad(self, caplog) -> None:
        with Patcher() as patcher:
            path = _make_file(patcher.fs, "/fake/testfile", mode=0o777)
            with caplog.at_level(logging.WARNING):
                warn_if_too_broad(path, 0o600)
            assert "too broad" in caplog.text

    def test_no_warning_when_permissions_are_tight(self, caplog) -> None:
        with Patcher() as patcher:
            path = _make_file(patcher.fs, "/fake/testfile", mode=0o600)
            with caplog.at_level(logging.WARNING):
                warn_if_too_broad(path, 0o700)
            assert "too broad" not in caplog.text

    def test_no_warning_when_permissions_match(self, caplog) -> None:
        with Patcher() as patcher:
            path = _make_file(patcher.fs, "/fake/testfile", mode=0o600)
            with caplog.at_level(logging.WARNING):
                warn_if_too_broad(path, 0o600)
            assert "too broad" not in caplog.text
