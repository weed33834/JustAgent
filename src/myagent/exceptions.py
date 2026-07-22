"""Unified exception hierarchy and exit codes for MyAgent-CLI."""

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


class MyAgentError(Exception):
    """Base exception for all MyAgent errors."""

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


class ConfigError(MyAgentError):
    """Raised when configuration is invalid or missing."""

    code = ExitCode.CONFIG_ERROR


class PermissionDeniedError(MyAgentError):
    """Raised when an RBAC permission check denies access."""

    code = ExitCode.PERMISSION_DENIED


class ToolChainError(MyAgentError):
    """Raised when an external tool (autoflake, black, etc.) fails."""

    code = ExitCode.CLEAN_ERROR


class GitError(MyAgentError):
    """Raised when a Git operation fails."""

    code = ExitCode.GIT_ERROR


class ModelGatewayError(MyAgentError):
    """Raised when a local model service is unavailable or returns an error."""

    code = ExitCode.MODEL_GATEWAY_ERROR


class PluginError(MyAgentError):
    """Raised when a plugin fails in a way that should abort the command."""

    code = ExitCode.PLUGIN_ERROR


class UploadError(MyAgentError):
    """Raised when an upload/publish operation fails."""

    code = ExitCode.UPLOAD_ERROR


class VerifyError(MyAgentError):
    """Raised when a verification command fails."""

    code = ExitCode.VERIFY_ERROR


class SecurityScanError(MyAgentError):
    """Raised when a security scan finds issues above the configured threshold."""

    code = ExitCode.SECURITY_ERROR


class SandboxError(MyAgentError):
    """Raised when a required sandbox cannot be enforced."""

    code = ExitCode.SECURITY_ERROR


class RegistryError(MyAgentError):
    """Raised when the plugin registry index cannot be verified or trusted."""

    code = ExitCode.SECURITY_ERROR
