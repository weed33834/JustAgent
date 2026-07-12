"""Unified exception hierarchy and exit codes for AutoShip-CLI."""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class ExitCode(IntEnum):
    """CLI exit status codes."""

    SUCCESS = 0
    USAGE_ERROR = 1
    CONFIG_ERROR = 2
    PERMISSION_DENIED = 3
    PLUGIN_ERROR = 10
    MODEL_GATEWAY_ERROR = 20
    GIT_ERROR = 30
    CLEAN_ERROR = 40
    VERIFY_ERROR = 50
    UPLOAD_ERROR = 60
    SECURITY_ERROR = 70
    USER_ABORT = 130


class AutoShipError(Exception):
    """Base exception for all AutoShip errors."""

    code: ExitCode = ExitCode.USAGE_ERROR

    def __init__(
        self,
        message: str,
        code: ExitCode | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code if code is not None else self.code
        self.details = details or {}


class ConfigError(AutoShipError):
    """Raised when configuration is invalid or missing."""

    code = ExitCode.CONFIG_ERROR


class PermissionDeniedError(AutoShipError):
    """Raised when an RBAC permission check denies access."""

    code = ExitCode.PERMISSION_DENIED


class ToolChainError(AutoShipError):
    """Raised when an external tool (autoflake, black, etc.) fails."""

    code = ExitCode.CLEAN_ERROR


class GitError(AutoShipError):
    """Raised when a Git operation fails."""

    code = ExitCode.GIT_ERROR


class ModelGatewayError(AutoShipError):
    """Raised when a local model service is unavailable or returns an error."""

    code = ExitCode.MODEL_GATEWAY_ERROR


class PluginError(AutoShipError):
    """Raised when a plugin fails in a way that should abort the command."""

    code = ExitCode.PLUGIN_ERROR


class UploadError(AutoShipError):
    """Raised when an upload/publish operation fails."""

    code = ExitCode.UPLOAD_ERROR


class VerifyError(AutoShipError):
    """Raised when a verification command fails."""

    code = ExitCode.VERIFY_ERROR


class SecurityScanError(AutoShipError):
    """Raised when a security scan finds issues above the configured threshold."""

    code = ExitCode.SECURITY_ERROR


class SandboxError(AutoShipError):
    """Raised when a required sandbox cannot be enforced."""

    code = ExitCode.SECURITY_ERROR


class RegistryError(AutoShipError):
    """Raised when the plugin registry index cannot be verified or trusted."""

    code = ExitCode.SECURITY_ERROR
