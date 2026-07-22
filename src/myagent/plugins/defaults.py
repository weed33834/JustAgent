"""Default built-in plugin implementations."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from myagent.adapters.model_gateway import ChatMessage
from myagent.core.context import CommandContext
from myagent.core.model_router import ModelRouter
from myagent.exceptions import ModelGatewayError, VerifyError
from myagent.hookspec import hookimpl
from myagent.utils.redaction import redact_paths


@dataclass
class FixSuggestion:
    """A suggested fix returned by an ``on_error`` hook.

    Attributes:
        description: Human-readable explanation of the fix.
        patch: Optional unified diff that can be applied to the project.
    """

    description: str
    patch: str | None = None


_DIFF_BLOCK_RE = re.compile(r"```diff\s*\n(.*?)\n```\s*(?:\n|$)", re.DOTALL)


class BuiltinPlugins:
    """Built-in plugins providing default AI-assisted behaviours."""

    @hookimpl
    def pre_commit(self, context: CommandContext) -> None:
        """Run security scans by delegating to the security-scan plugin."""
        from myagent.plugins import security_scan

        security_scan.plugin.pre_commit(context)

    @hookimpl
    def on_error(self, context: CommandContext, error: Exception) -> FixSuggestion | None:
        """Suggest a fix when a command fails and the user requested --fix."""
        if not context.extras.get("fix"):
            return None

        stdout, stderr = "", ""
        if isinstance(error, VerifyError):
            stdout = error.details.get("stdout", "")
            stderr = error.details.get("stderr", "")
        elif isinstance(error, subprocess.CalledProcessError):
            stdout = error.stdout or ""
            stderr = error.stderr or ""

        # Mask absolute local paths before sending stdout/stderr to the model
        # so that the user's project layout is not leaked to the remote backend.
        project_root = getattr(context, "project_root", None)
        if project_root is None:
            project_root = getattr(getattr(context, "config", None), "project_root", None)
        stdout = redact_paths(stdout, project_root)
        stderr = redact_paths(stderr, project_root)

        command = context.extras.get("verify_command", "")
        prompt = (
            "The following verification command failed. "
            "Suggest a concise fix in one or two sentences. "
            "If you can provide a unified diff patch, include it after the description "
            "under a line starting with '```diff'.\n\n"
            f"Command: {command}\n\n"
            f"stdout:\n{stdout[-4000:]}\n\n"
            f"stderr:\n{stderr[-4000:]}"
        )

        with ModelRouter(context.config) as router:
            try:
                suggestion = router.chat(
                    [
                        ChatMessage(
                            role="system", content="You are a helpful debugging assistant."
                        ),
                        ChatMessage(role="user", content=prompt),
                    ],
                    "verify-fix",
                )
            except ModelGatewayError:
                return None

            return _parse_suggestion(suggestion)


def _parse_suggestion(text: str) -> FixSuggestion:
    """Split a model response into description and optional diff patch."""
    matches = _DIFF_BLOCK_RE.findall(text)
    if matches:
        patch = "\n".join(matches).strip()
        description = _DIFF_BLOCK_RE.sub("", text).strip()
        return FixSuggestion(description=description, patch=patch)
    return FixSuggestion(description=text.strip())


plugin = BuiltinPlugins()
