"""Tests for the RBAC module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoship.core.rbac import (
    BUILTIN_ROLES,
    Permission,
    has_permission,
    merge_roles,
    require_permission,
    resolve_role,
)
from autoship.core.sso import Identity
from autoship.exceptions import ExitCode, PermissionDeniedError
from autoship.models.config import AppConfig, RbacConfig, RoleBinding, RoleConfig


def _identity(user: str = "alice", groups: list[str] | None = None) -> Identity:
    return Identity(
        user=user,
        subject=user,
        groups=groups or [],
        provider="stub",
        expires_at=None,
    )


def _rbac(
    *,
    enabled: bool = True,
    roles: dict[str, RoleConfig] | None = None,
    bindings: list[RoleBinding] | None = None,
) -> RbacConfig:
    return RbacConfig(
        enabled=enabled,
        roles=roles or {},
        bindings=bindings or [],
    )


def test_has_permission_disabled_returns_true() -> None:
    rbac = _rbac(enabled=False)
    identity = _identity()
    assert has_permission(identity, rbac, Permission.COMMIT_RUN) is True


def test_has_permission_admin_returns_true() -> None:
    rbac = _rbac(
        enabled=True,
        bindings=[RoleBinding(role="admin", users=["alice"])],
    )
    identity = _identity()
    assert has_permission(identity, rbac, Permission.COMMIT_RUN) is True
    assert has_permission(identity, rbac, Permission.AUDIT_CLEANUP) is True


def test_has_permission_developer_can_commit() -> None:
    rbac = _rbac(
        enabled=True,
        bindings=[RoleBinding(role="developer", users=["alice"])],
    )
    identity = _identity()
    assert has_permission(identity, rbac, Permission.COMMIT_RUN) is True


def test_has_permission_developer_cannot_cleanup_audit() -> None:
    rbac = _rbac(
        enabled=True,
        bindings=[RoleBinding(role="developer", users=["alice"])],
    )
    identity = _identity()
    assert has_permission(identity, rbac, Permission.AUDIT_CLEANUP) is False


def test_has_permission_maintainer_can_auto_push() -> None:
    rbac = _rbac(
        enabled=True,
        bindings=[RoleBinding(role="maintainer", users=["alice"])],
    )
    identity = _identity()
    assert has_permission(identity, rbac, Permission.COMMIT_AUTO_PUSH) is True


def test_resolve_role_by_user_match() -> None:
    rbac = _rbac(
        enabled=True,
        bindings=[RoleBinding(role="developer", users=["alice", "bob"])],
    )
    identity = _identity("alice")
    assert resolve_role(identity, rbac) == "developer"


def test_resolve_role_by_group_match() -> None:
    rbac = _rbac(
        enabled=True,
        bindings=[RoleBinding(role="developer", groups=["eng"])],
    )
    identity = _identity("alice", groups=["eng", "staff"])
    assert resolve_role(identity, rbac) == "developer"


def test_resolve_role_first_match_wins() -> None:
    rbac = _rbac(
        enabled=True,
        bindings=[
            RoleBinding(role="viewer", users=["alice"]),
            RoleBinding(role="admin", users=["alice"]),
        ],
    )
    identity = _identity("alice")
    assert resolve_role(identity, rbac) == "viewer"


def test_resolve_role_no_match_returns_none() -> None:
    rbac = _rbac(
        enabled=True,
        bindings=[RoleBinding(role="developer", users=["bob"])],
    )
    identity = _identity("alice")
    assert resolve_role(identity, rbac) is None


def test_role_inheritance_via_parent() -> None:
    rbac = _rbac(
        enabled=True,
        roles={
            "intern": RoleConfig(permissions=[], parent="developer"),
        },
        bindings=[RoleBinding(role="intern", users=["alice"])],
    )
    identity = _identity("alice")
    # intern has no own perms but inherits developer's commit:run.
    assert has_permission(identity, rbac, Permission.COMMIT_RUN) is True
    # And not the maintainer-only auto_push.
    assert has_permission(identity, rbac, Permission.COMMIT_AUTO_PUSH) is False


def test_custom_role_overrides_builtin() -> None:
    custom_viewer = RoleConfig(permissions=["clean:run"])
    rbac = _rbac(
        enabled=True,
        roles={"viewer": custom_viewer},
        bindings=[RoleBinding(role="viewer", users=["alice"])],
    )
    merged = merge_roles(rbac)
    assert merged["viewer"] is custom_viewer
    # Custom viewer lost the inherited verify:run permission.
    identity = _identity("alice")
    assert has_permission(identity, rbac, Permission.CLEAN_RUN) is True
    assert has_permission(identity, rbac, Permission.VERIFY_RUN) is False


def test_require_permission_denied_raises(tmp_path: Path) -> None:
    config = AppConfig(
        project_root=tmp_path,
        rbac=_rbac(
            enabled=True,
            bindings=[RoleBinding(role="viewer", users=["alice"])],
        ),
    )
    audit = MagicMock()
    ctx = MagicMock()
    ctx.obj = {
        "config": config,
        "audit_logger": audit,
        "identity": _identity("alice"),
        "role": "viewer",
    }
    ctx.info_name = "upload"

    with pytest.raises(PermissionDeniedError) as exc_info:
        require_permission(ctx, Permission.UPLOAD_RUN)
    assert exc_info.value.code == ExitCode.PERMISSION_DENIED
    audit.record.assert_called_once()
    call_args = audit.record.call_args
    assert call_args.args[0] == "rbac.denied"
    payload = call_args.args[1]
    assert payload["permission"] == "upload:run"
    assert payload["user"] == "alice"
    assert payload["role"] == "viewer"
    assert payload["command"] == "upload"


def test_require_permission_allowed_passes_silently(tmp_path: Path) -> None:
    config = AppConfig(
        project_root=tmp_path,
        rbac=_rbac(
            enabled=True,
            bindings=[RoleBinding(role="developer", users=["alice"])],
        ),
    )
    audit = MagicMock()
    ctx = MagicMock()
    ctx.obj = {
        "config": config,
        "audit_logger": audit,
        "identity": _identity("alice"),
        "role": "developer",
    }
    ctx.info_name = "upload"
    # Should not raise; audit should not record a denial.
    require_permission(ctx, Permission.UPLOAD_RUN)
    audit.record.assert_not_called()


def test_builtin_roles_contain_expected_permissions() -> None:
    # Sanity-check the builtin role table so downstream tests are not mysterious.
    assert "*" in BUILTIN_ROLES["admin"].permissions
    assert "commit:run" in BUILTIN_ROLES["developer"].permissions
    assert "commit:auto_push" in BUILTIN_ROLES["maintainer"].permissions
    assert "commit:auto_push" not in BUILTIN_ROLES["developer"].permissions
    assert "audit:export" in BUILTIN_ROLES["viewer"].permissions


# ---------------------------------------------------------------------------
# scope:* wildcard matching (H4)
# ---------------------------------------------------------------------------


def test_scope_wildcard_grants_all_actions_under_scope() -> None:
    """A 'commit:*' permission grants every action under the 'commit' scope."""
    rbac = _rbac(
        enabled=True,
        roles={"custom": RoleConfig(permissions=["commit:*"])},
        bindings=[RoleBinding(role="custom", users=["alice"])],
    )
    identity = _identity("alice")
    assert has_permission(identity, rbac, Permission.COMMIT_RUN) is True
    assert has_permission(identity, rbac, Permission.COMMIT_AUTO_PUSH) is True
    # A different scope is not granted.
    assert has_permission(identity, rbac, Permission.CLEAN_RUN) is False


def test_scope_wildcard_does_not_match_bare_permission() -> None:
    """A 'scope:*' entry does not match a permission without a scope prefix."""
    from autoship.core.rbac import _matches

    # "commit:*" should not match a bare permission without ':'.
    assert _matches("admin", "commit:*") is False
    # It should match permissions under the scope.
    assert _matches("commit:run", "commit:*") is True
    # And not match a different scope.
    assert _matches("clean:run", "commit:*") is False
