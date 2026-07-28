"""Tests for the RBAC module."""

from __future__ import annotations

import pytest

from myagent.security.rbac import (
    AccessDecision,
    Permission,
    RBACEngine,
    RBACError,
    Role,
    ResourceType,
    User,
    UserStatus,
)


class TestRBACEngine:
    def test_builtin_roles_exist(self) -> None:
        engine = RBACEngine()
        roles = engine.list_roles()
        role_names = [r.name for r in roles]
        assert "Administrator" in role_names
        assert "Viewer" in role_names
        assert "Editor" in role_names

    def test_admin_role_has_admin_permission(self) -> None:
        engine = RBACEngine()
        engine.register_user(User(username="admin_user", roles=[RBACEngine.ADMIN_ROLE]))
        assert engine.check_access("admin_user", ResourceType.DOCUMENT, Permission.DELETE)
        assert engine.check_access("admin_user", ResourceType.DATABASE, Permission.ADMIN)
        assert engine.check_access("admin_user", ResourceType.AGENT, Permission.EXECUTE)

    def test_viewer_role_read_only(self) -> None:
        engine = RBACEngine()
        engine.register_user(User(username="viewer", roles=[RBACEngine.VIEWER_ROLE]))
        assert engine.check_access("viewer", ResourceType.DOCUMENT, Permission.READ)
        assert not engine.check_access("viewer", ResourceType.DOCUMENT, Permission.WRITE)
        assert not engine.check_access("viewer", ResourceType.DOCUMENT, Permission.DELETE)

    def test_editor_role_read_write(self) -> None:
        engine = RBACEngine()
        engine.register_user(User(username="editor", roles=[RBACEngine.EDITOR_ROLE]))
        assert engine.check_access("editor", ResourceType.DOCUMENT, Permission.READ)
        assert engine.check_access("editor", ResourceType.DOCUMENT, Permission.WRITE)
        assert not engine.check_access("editor", ResourceType.DOCUMENT, Permission.DELETE)

    def test_editor_inherits_viewer(self) -> None:
        engine = RBACEngine()
        editor_role = engine.get_role(RBACEngine.EDITOR_ROLE)
        assert RBACEngine.VIEWER_ROLE in editor_role.inherits_from

    def test_create_custom_role(self) -> None:
        engine = RBACEngine()
        role = Role(
            name="Manager",
            description="Department manager",
            permissions={
                ResourceType.DOCUMENT: {Permission.READ, Permission.WRITE, Permission.DELETE},
            },
        )
        engine.create_role(role)
        engine.register_user(User(username="mgr", roles=[role.id]))
        assert engine.check_access("mgr", ResourceType.DOCUMENT, Permission.DELETE)
        assert not engine.check_access("mgr", ResourceType.DATABASE, Permission.ADMIN)

    def test_create_duplicate_role_raises(self) -> None:
        engine = RBACEngine()
        role = Role(id="custom1", name="Custom", permissions={})
        engine.create_role(role)
        with pytest.raises(RBACError, match="Role already exists"):
            engine.create_role(Role(id="custom1", name="Dup", permissions={}))

    def test_delete_custom_role(self) -> None:
        engine = RBACEngine()
        role = Role(name="Temp", permissions={})
        engine.create_role(role)
        deleted = engine.delete_role(role.id)
        assert deleted is not None
        assert engine.get_role(role.id) is None

    def test_delete_system_role_raises(self) -> None:
        engine = RBACEngine()
        with pytest.raises(RBACError, match="Cannot delete system role"):
            engine.delete_role(RBACEngine.ADMIN_ROLE)

    def test_assign_and_revoke_role(self) -> None:
        engine = RBACEngine()
        engine.register_user(User(username="user1"))
        engine.assign_role("user1", RBACEngine.VIEWER_ROLE)
        assert engine.check_access("user1", ResourceType.DOCUMENT, Permission.READ)
        engine.revoke_role("user1", RBACEngine.VIEWER_ROLE)
        assert not engine.check_access("user1", ResourceType.DOCUMENT, Permission.READ)

    def test_assign_unknown_role_raises(self) -> None:
        engine = RBACEngine()
        engine.register_user(User(username="user1"))
        with pytest.raises(RBACError, match="Role not found"):
            engine.assign_role("user1", "nonexistent")

    def test_assign_to_unknown_user_raises(self) -> None:
        engine = RBACEngine()
        with pytest.raises(RBACError, match="User not found"):
            engine.assign_role("ghost", RBACEngine.VIEWER_ROLE)

    def test_register_duplicate_user_raises(self) -> None:
        engine = RBACEngine()
        engine.register_user(User(username="dup"))
        with pytest.raises(RBACError, match="User already exists|Username already taken"):
            engine.register_user(User(username="dup"))

    def test_get_user_by_username(self) -> None:
        engine = RBACEngine()
        user = User(username="alice", display_name="Alice")
        engine.register_user(user)
        found = engine.get_user("alice")
        assert found is not None
        assert found.display_name == "Alice"

    def test_evaluate_access_returns_decision(self) -> None:
        engine = RBACEngine()
        engine.register_user(User(username="alice", roles=[RBACEngine.VIEWER_ROLE]))
        decision = engine.evaluate_access("alice", ResourceType.DOCUMENT, Permission.READ)
        assert isinstance(decision, AccessDecision)
        assert decision.allowed is True
        assert decision.matched_role == RBACEngine.VIEWER_ROLE
        assert decision.reason

    def test_evaluate_access_denied_returns_decision(self) -> None:
        engine = RBACEngine()
        engine.register_user(User(username="alice", roles=[RBACEngine.VIEWER_ROLE]))
        decision = engine.evaluate_access("alice", ResourceType.DOCUMENT, Permission.DELETE)
        assert decision.allowed is False
        assert decision.matched_role is None

    def test_inactive_user_denied(self) -> None:
        engine = RBACEngine()
        user = User(username="inactive", roles=[RBACEngine.ADMIN_ROLE], status=UserStatus.INACTIVE)
        engine.register_user(user)
        assert not engine.check_access("inactive", ResourceType.DOCUMENT, Permission.READ)

    def test_suspended_user_denied(self) -> None:
        engine = RBACEngine()
        engine.register_user(User(username="suspended", roles=[RBACEngine.ADMIN_ROLE]))
        engine.update_user_status("suspended", UserStatus.SUSPENDED)
        assert not engine.check_access("suspended", ResourceType.DOCUMENT, Permission.READ)

    def test_unknown_user_denied(self) -> None:
        engine = RBACEngine()
        assert not engine.check_access("ghost", ResourceType.DOCUMENT, Permission.READ)

    def test_role_inheritance_transitive(self) -> None:
        engine = RBACEngine()
        # Create A inherits from B, B inherits from C (viewer).
        role_c = Role(name="RoleC", permissions={ResourceType.DOCUMENT: {Permission.EXPORT}})
        engine.create_role(role_c)
        role_b = Role(name="RoleB", permissions={}, inherits_from=[role_c.id])
        engine.create_role(role_b)
        role_a = Role(name="RoleA", permissions={}, inherits_from=[role_b.id])
        engine.create_role(role_a)
        engine.register_user(User(username="chain_user", roles=[role_a.id]))
        # Should inherit EXPORT from role_c.
        assert engine.check_access("chain_user", ResourceType.DOCUMENT, Permission.EXPORT)

    def test_role_inheritance_cycle_safe(self) -> None:
        engine = RBACEngine()
        role1 = Role(id="cyc1", name="Cyc1", permissions={}, inherits_from=["cyc2"])
        engine.create_role(role1)
        role2 = Role(id="cyc2", name="Cyc2", permissions={}, inherits_from=["cyc1"])
        engine.create_role(role2)
        engine.register_user(User(username="cyc_user", roles=["cyc1"]))
        # Should not hang or crash.
        engine.check_access("cyc_user", ResourceType.DOCUMENT, Permission.READ)

    def test_list_users(self) -> None:
        engine = RBACEngine()
        engine.register_user(User(username="u1"))
        engine.register_user(User(username="u2"))
        users = engine.list_users()
        assert len(users) == 2

    def test_access_decision_bool(self) -> None:
        d_true = AccessDecision(allowed=True)
        d_false = AccessDecision(allowed=False)
        assert bool(d_true) is True
        assert bool(d_false) is False

    def test_user_is_active_property(self) -> None:
        active = User(username="a")
        inactive = User(username="b", status=UserStatus.INACTIVE)
        assert active.is_active
        assert not inactive.is_active

    def test_role_grants_admin_implies_all(self) -> None:
        role = Role(
            name="TestAdmin",
            permissions={ResourceType.DOCUMENT: {Permission.ADMIN}},
        )
        assert role.grants(ResourceType.DOCUMENT, Permission.READ)
        assert role.grants(ResourceType.DOCUMENT, Permission.WRITE)
        assert role.grants(ResourceType.DOCUMENT, Permission.DELETE)
        assert not role.grants(ResourceType.DATABASE, Permission.READ)
