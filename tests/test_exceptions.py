"""Tests for the exception hierarchy."""

from __future__ import annotations

from autoship.exceptions import AutoShipError, ConfigError, ExitCode


def test_exit_codes() -> None:
    assert ExitCode.SUCCESS == 0
    assert ExitCode.VERIFY_ERROR == 50
    assert ExitCode.UPLOAD_ERROR == 60


def test_base_error_default_code() -> None:
    err = AutoShipError("boom")
    assert err.code == ExitCode.USAGE_ERROR
    assert err.details == {}


def test_error_with_details() -> None:
    err = ConfigError("bad", details={"field": "missing"})
    assert err.code == ExitCode.CONFIG_ERROR
    assert err.details["field"] == "missing"


def test_all_error_codes_are_distinct() -> None:
    codes = [
        ExitCode.SUCCESS,
        ExitCode.USAGE_ERROR,
        ExitCode.CONFIG_ERROR,
        ExitCode.PLUGIN_ERROR,
        ExitCode.MODEL_GATEWAY_ERROR,
        ExitCode.GIT_ERROR,
        ExitCode.CLEAN_ERROR,
        ExitCode.VERIFY_ERROR,
        ExitCode.UPLOAD_ERROR,
        ExitCode.USER_ABORT,
    ]
    assert len(codes) == len(set(codes))
