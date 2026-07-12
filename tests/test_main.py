"""Tests for the __main__ entry point."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_main_module_imports() -> None:
    """Verify __main__.py can be imported."""
    import autoship.__main__ as m

    assert m is not None


def test_main_entry_point_is_cli_entrypoint() -> None:
    """__main__.py re-exports cli_entrypoint from cli.main."""
    from autoship.__main__ import cli_entrypoint as main_entry
    from autoship.cli.main import cli_entrypoint

    assert main_entry is cli_entrypoint


def test_main_module_has_name_guard() -> None:
    """__main__.py contains the __name__ == '__main__' guard."""
    import autoship.__main__ as main_module

    source_path = Path(main_module.__file__)
    source = source_path.read_text()
    assert 'if __name__ == "__main__"' in source
    assert "cli_entrypoint()" in source


def test_main_module_enter_guard_block() -> None:
    """Execute __main__.py as __main__ to cover the guard block (line 6)."""
    main_path = Path(__file__).resolve().parent.parent / "src" / "autoship" / "__main__.py"
    with patch("autoship.cli.main.cli_entrypoint", return_value=0):
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(main_path), run_name="__main__")
        assert exc_info.value.code == 0


def test_main_module_raises_systemexit() -> None:
    """When executed via python -m, the module raises SystemExit."""
    result = subprocess.run(
        [sys.executable, "-m", "autoship", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # --help exits with code 0
    assert result.returncode == 0


def test_main_module_run_in_subprocess() -> None:
    """Verify the module can be executed and produces expected output."""
    result = subprocess.run(
        [sys.executable, "-c", "import autoship.__main__"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Importing alone should not trigger the guard and should succeed
    assert result.returncode == 0
