"""Tests for the enterprise permission engine.

Covers data-level, resource-level, and composite permission checks, the
:class:`EnterpriseDecision` result type, the :func:`create_enterprise_engine`
factory, :class:`UserContext` helpers, and thread safety.
"""

from __future__ import annotations

import threading

import pytest

from myagent.permissions import PermissionAction, PermissionEngine, PermissionRule
from myagent.permissions.enterprise import (
    ClearanceLevel,
    DataOperation,
    DataPermission,
    DataResource,
    EnterpriseDecision,
    EnterprisePermissionEngine,
    HardwareResource,
    ResourceCriticality,
    ResourceOperation,
    ResourcePermission,
    UserContext,
    create_enterprise_engine,
)


# ---------------------------------------------------------------------------
# UserContext
# ---------------------------------------------------------------------------


class TestUserContext:
    def test_has_role_true_when_role_present(self) -> None:
        user = UserContext(user_id="alice", roles={"engineer", "viewer"})
        assert user.has_role("engineer") is True
        assert user.has_role("viewer") is True

    def test_has_role_false_when_role_absent(self) -> None:
        user = UserContext(user_id="alice", roles={"engineer"})
        assert user.has_role("admin") is False

    def test_has_role_false_with_empty_roles(self) -> None:
        user = UserContext(user_id="alice")
        assert user.has_role("anything") is False

    def test_is_admin_true_for_admin_role(self) -> None:
        user = UserContext(user_id="bob", roles={"admin"})
        assert user.is_admin() is True

    def test_is_admin_true_for_superadmin_role(self) -> None:
        user = UserContext(user_id="carol", roles={"superadmin"})
        assert user.is_admin() is True

    def test_is_admin_true_when_admin_among_other_roles(self) -> None:
        user = UserContext(user_id="carol", roles={"engineer", "admin", "viewer"})
        assert user.is_admin() is True

    def test_is_admin_false_for_non_admin_roles(self) -> None:
        user = UserContext(user_id="dave", roles={"engineer", "viewer"})
        assert user.is_admin() is False

    def test_is_admin_false_for_empty_roles(self) -> None:
        user = UserContext(user_id="eve")
        assert user.is_admin() is False

    def test_defaults(self) -> None:
        user = UserContext(user_id="alice")
        assert user.username == ""
        assert user.department == ""
        assert user.roles == set()
        assert user.clearance is ClearanceLevel.INTERNAL
        assert user.metadata == {}


# ---------------------------------------------------------------------------
# Data-level permissions
# ---------------------------------------------------------------------------


class TestDataPermissions:
    def test_owner_has_full_access_for_any_operation(self) -> None:
        engine = EnterprisePermissionEngine()
        owner = UserContext(user_id="alice", clearance=ClearanceLevel.PUBLIC)
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.RESTRICTED,
            owner="alice",
            department="other",
        )
        for op in DataOperation:
            decision = engine.check_data_access(owner, data, op)
            assert decision.action is PermissionAction.ALLOW, op
            assert decision.allowed is True, op
            assert decision.data_decision is PermissionAction.ALLOW, op

    def test_owner_access_bypasses_clearance_check(self) -> None:
        # Owner with PUBLIC clearance owns RESTRICTED data -> still ALLOW.
        engine = EnterprisePermissionEngine()
        owner = UserContext(user_id="alice", clearance=ClearanceLevel.PUBLIC)
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.RESTRICTED,
            owner="alice",
        )
        decision = engine.check_data_access(owner, data, DataOperation.READ)
        assert decision.action is PermissionAction.ALLOW
        assert "owner" in decision.reason.lower()

    def test_admin_override_allows_any_data_operation(self) -> None:
        engine = EnterprisePermissionEngine()
        admin = UserContext(
            user_id="admin-1", roles={"admin"}, clearance=ClearanceLevel.PUBLIC
        )
        data = DataResource(
            resource_id="secret",
            classification=ClearanceLevel.RESTRICTED,
            department="other",
        )
        for op in DataOperation:
            decision = engine.check_data_access(admin, data, op)
            assert decision.action is PermissionAction.ALLOW, op
            assert decision.reason == "Administrative override"

    def test_superadmin_override_allows_data(self) -> None:
        engine = EnterprisePermissionEngine()
        superadmin = UserContext(
            user_id="root", roles={"superadmin"}, clearance=ClearanceLevel.PUBLIC
        )
        data = DataResource(
            resource_id="secret",
            classification=ClearanceLevel.RESTRICTED,
        )
        decision = engine.check_data_access(superadmin, data, DataOperation.DELETE)
        assert decision.action is PermissionAction.ALLOW

    def test_low_clearance_denied_access_to_high_classification(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", clearance=ClearanceLevel.INTERNAL)
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.RESTRICTED,
        )
        decision = engine.check_data_access(user, data, DataOperation.READ)
        assert decision.action is PermissionAction.DENY
        assert decision.data_decision is PermissionAction.DENY
        assert "clearance" in decision.reason.lower()

    def test_clearance_equal_to_classification_is_not_denied(self) -> None:
        # Equal clearance passes the clearance gate and falls through to the
        # same-department default (read -> ALLOW).
        engine = EnterprisePermissionEngine()
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.CONFIDENTIAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.CONFIDENTIAL,
            department="engineering",
        )
        decision = engine.check_data_access(user, data, DataOperation.READ)
        assert decision.action is PermissionAction.ALLOW

    def test_same_department_read_allowed(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="engineering",
        )
        decision = engine.check_data_access(user, data, DataOperation.READ)
        assert decision.action is PermissionAction.ALLOW
        assert "Same-department" in decision.reason

    def test_same_department_write_asks(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="engineering",
        )
        decision = engine.check_data_access(user, data, DataOperation.WRITE)
        assert decision.action is PermissionAction.ASK
        assert "requires confirmation" in decision.reason

    def test_same_department_export_asks(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="engineering",
        )
        decision = engine.check_data_access(user, data, DataOperation.EXPORT)
        assert decision.action is PermissionAction.ASK

    def test_cross_department_read_asks(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="marketing",
        )
        decision = engine.check_data_access(user, data, DataOperation.READ)
        assert decision.action is PermissionAction.ASK
        assert "Cross-department" in decision.reason

    def test_cross_department_write_denied(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="marketing",
        )
        decision = engine.check_data_access(user, data, DataOperation.WRITE)
        assert decision.action is PermissionAction.DENY
        assert decision.data_decision is PermissionAction.DENY

    def test_cross_department_export_denied(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="marketing",
        )
        decision = engine.check_data_access(user, data, DataOperation.EXPORT)
        assert decision.action is PermissionAction.DENY

    def test_explicit_data_rule_overrides_same_department_default(self) -> None:
        engine = EnterprisePermissionEngine()
        # Same-dept read would normally ALLOW; rule overrides to DENY.
        engine.add_data_permission(
            DataPermission(
                name="deny-internal-read",
                operations=frozenset({DataOperation.READ}),
                classification=ClearanceLevel.INTERNAL,
                action=PermissionAction.DENY,
            )
        )
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="engineering",
        )
        decision = engine.check_data_access(user, data, DataOperation.READ)
        assert decision.action is PermissionAction.DENY
        assert decision.data_decision is PermissionAction.DENY
        assert any("deny-internal-read" in line for line in decision.audit_trail)

    def test_explicit_data_rule_allows_cross_department(self) -> None:
        engine = EnterprisePermissionEngine()
        # Cross-dept read would normally ASK; rule overrides to ALLOW.
        engine.add_data_permission(
            DataPermission(
                name="allow-cross-read",
                operations=frozenset({DataOperation.READ}),
                classification=ClearanceLevel.INTERNAL,
                action=PermissionAction.ALLOW,
            )
        )
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="marketing",
        )
        decision = engine.check_data_access(user, data, DataOperation.READ)
        assert decision.action is PermissionAction.ALLOW

    def test_data_rule_department_filter(self) -> None:
        engine = EnterprisePermissionEngine()
        # Rule only applies to the "engineering" department.
        engine.add_data_permission(
            DataPermission(
                name="eng-allow-read",
                operations=frozenset({DataOperation.READ}),
                classification=ClearanceLevel.INTERNAL,
                department="engineering",
                action=PermissionAction.ALLOW,
            )
        )
        eng_user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        mkt_user = UserContext(
            user_id="bob",
            department="marketing",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="finance",  # cross-dept for both users
        )
        # engineering user matches the rule -> ALLOW
        assert (
            engine.check_data_access(eng_user, data, DataOperation.READ).action
            is PermissionAction.ALLOW
        )
        # marketing user does not match -> cross-dept read -> ASK
        assert (
            engine.check_data_access(mkt_user, data, DataOperation.READ).action
            is PermissionAction.ASK
        )

    def test_data_rule_classification_filter(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_data_permission(
            DataPermission(
                name="confidential-read-ask",
                operations=frozenset({DataOperation.READ}),
                classification=ClearanceLevel.CONFIDENTIAL,
                action=PermissionAction.ASK,
            )
        )
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.CONFIDENTIAL,
        )
        confidential = DataResource(
            resource_id="c",
            classification=ClearanceLevel.CONFIDENTIAL,
            department="engineering",  # same dept -> would ALLOW without rule
        )
        internal = DataResource(
            resource_id="i",
            classification=ClearanceLevel.INTERNAL,
            department="engineering",
        )
        # CONFIDENTIAL matches rule -> ASK (overrides same-dept ALLOW)
        assert (
            engine.check_data_access(user, confidential, DataOperation.READ).action
            is PermissionAction.ASK
        )
        # INTERNAL does not match rule -> same-dept ALLOW
        assert (
            engine.check_data_access(user, internal, DataOperation.READ).action
            is PermissionAction.ALLOW
        )

    def test_non_matching_data_rule_is_skipped(self) -> None:
        engine = EnterprisePermissionEngine()
        # Rule for EXPORT only; should not affect READ.
        engine.add_data_permission(
            DataPermission(
                name="export-deny",
                operations=frozenset({DataOperation.EXPORT}),
                classification=ClearanceLevel.INTERNAL,
                action=PermissionAction.DENY,
            )
        )
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="engineering",
        )
        # READ not in rule operations -> same-dept ALLOW.
        assert (
            engine.check_data_access(user, data, DataOperation.READ).action
            is PermissionAction.ALLOW
        )

    def test_restricted_data_export_denied_by_default_rule(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_data_permission(
            DataPermission(
                name="restricted-export-deny",
                operations=frozenset({DataOperation.EXPORT}),
                classification=ClearanceLevel.RESTRICTED,
                action=PermissionAction.DENY,
                description="Non-admin export of RESTRICTED data is denied",
            )
        )
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.RESTRICTED,
        )
        data = DataResource(
            resource_id="top-secret",
            classification=ClearanceLevel.RESTRICTED,
            department="engineering",  # same dept: EXPORT would ASK without rule
        )
        decision = engine.check_data_access(user, data, DataOperation.EXPORT)
        assert decision.action is PermissionAction.DENY
        assert decision.data_decision is PermissionAction.DENY


# ---------------------------------------------------------------------------
# Resource-level permissions
# ---------------------------------------------------------------------------


class TestResourcePermissions:
    def test_admin_override_allows_any_resource_operation(self) -> None:
        engine = EnterprisePermissionEngine()
        admin = UserContext(
            user_id="admin-1", roles={"admin"}, department="engineering"
        )
        resource = HardwareResource(
            resource_id="srv-1",
            resource_type="server",
            criticality=ResourceCriticality.CRITICAL,
            department="marketing",
        )
        for op in ResourceOperation:
            decision = engine.check_resource_access(admin, resource, op)
            assert decision.action is PermissionAction.ALLOW, op
            assert decision.resource_decision is PermissionAction.ALLOW, op

    def test_critical_resource_configure_denied_for_non_admin(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.CRITICAL,
            department="engineering",  # same dept, but critical configure is blocked
        )
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.CONFIGURE
        )
        assert decision.action is PermissionAction.DENY
        assert "admin" in decision.reason.lower()
        assert decision.resource_decision is PermissionAction.DENY

    def test_critical_resource_admin_denied_for_non_admin(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.CRITICAL,
            department="engineering",
        )
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.ADMIN
        )
        assert decision.action is PermissionAction.DENY

    def test_critical_resource_view_not_blocked_by_critical_check(self) -> None:
        # Only CONFIGURE/ADMIN are blocked for critical resources; VIEW is fine.
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.CRITICAL,
            department="engineering",
        )
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.VIEW
        )
        assert decision.action is PermissionAction.ALLOW

    def test_non_critical_resource_configure_not_blocked_by_critical_check(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.HIGH,
            department="engineering",
        )
        # Same-dept CONFIGURE on a HIGH (non-critical) resource -> ASK.
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.CONFIGURE
        )
        assert decision.action is PermissionAction.ASK

    def test_same_department_view_allowed(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.MEDIUM,
            department="engineering",
        )
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.VIEW
        )
        assert decision.action is PermissionAction.ALLOW

    def test_same_department_operate_allowed(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.MEDIUM,
            department="engineering",
        )
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.OPERATE
        )
        assert decision.action is PermissionAction.ALLOW

    def test_same_department_allocate_asks(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.MEDIUM,
            department="engineering",
        )
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.ALLOCATE
        )
        assert decision.action is PermissionAction.ASK

    def test_same_department_configure_asks(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.MEDIUM,
            department="engineering",
        )
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.CONFIGURE
        )
        assert decision.action is PermissionAction.ASK

    def test_cross_department_view_asks(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.MEDIUM,
            department="marketing",
        )
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.VIEW
        )
        assert decision.action is PermissionAction.ASK
        assert "Cross-department" in decision.reason

    def test_cross_department_operate_denied(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.MEDIUM,
            department="marketing",
        )
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.OPERATE
        )
        assert decision.action is PermissionAction.DENY
        assert decision.resource_decision is PermissionAction.DENY

    def test_cross_department_configure_denied(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.MEDIUM,
            department="marketing",
        )
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.CONFIGURE
        )
        assert decision.action is PermissionAction.DENY

    def test_explicit_resource_rule_overrides_default(self) -> None:
        engine = EnterprisePermissionEngine()
        # Cross-dept VIEW would normally ASK; rule overrides to ALLOW.
        engine.add_resource_permission(
            ResourcePermission(
                name="allow-cross-view",
                operations=frozenset({ResourceOperation.VIEW}),
                action=PermissionAction.ALLOW,
            )
        )
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.MEDIUM,
            department="marketing",
        )
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.VIEW
        )
        assert decision.action is PermissionAction.ALLOW
        assert any("allow-cross-view" in line for line in decision.audit_trail)

    def test_resource_rule_filters_by_type(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_resource_permission(
            ResourcePermission(
                name="gpu-allocate-allow",
                operations=frozenset({ResourceOperation.ALLOCATE}),
                resource_type="gpu",
                action=PermissionAction.ALLOW,
            )
        )
        user = UserContext(user_id="alice", department="engineering")
        gpu = HardwareResource(
            resource_id="gpu-1",
            resource_type="gpu",
            criticality=ResourceCriticality.HIGH,
            department="marketing",  # cross-dept
        )
        server = HardwareResource(
            resource_id="srv-1",
            resource_type="server",
            criticality=ResourceCriticality.HIGH,
            department="marketing",
        )
        # gpu matches rule -> ALLOW
        assert (
            engine.check_resource_access(user, gpu, ResourceOperation.ALLOCATE).action
            is PermissionAction.ALLOW
        )
        # server does not match -> cross-dept ALLOCATE -> DENY
        assert (
            engine.check_resource_access(user, server, ResourceOperation.ALLOCATE).action
            is PermissionAction.DENY
        )

    def test_resource_rule_filters_by_criticality(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_resource_permission(
            ResourcePermission(
                name="critical-operate-deny",
                operations=frozenset({ResourceOperation.OPERATE}),
                criticality=ResourceCriticality.CRITICAL,
                action=PermissionAction.DENY,
            )
        )
        user = UserContext(user_id="alice", department="engineering")
        critical_res = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.CRITICAL,
            department="engineering",  # same dept -> would ALLOW without rule
        )
        medium_res = HardwareResource(
            resource_id="srv-2",
            criticality=ResourceCriticality.MEDIUM,
            department="engineering",
        )
        # CRITICAL matches rule -> DENY (overrides same-dept ALLOW)
        assert (
            engine.check_resource_access(user, critical_res, ResourceOperation.OPERATE).action
            is PermissionAction.DENY
        )
        # MEDIUM does not match -> same-dept ALLOW
        assert (
            engine.check_resource_access(user, medium_res, ResourceOperation.OPERATE).action
            is PermissionAction.ALLOW
        )


# ---------------------------------------------------------------------------
# Composite check
# ---------------------------------------------------------------------------


class TestCompositeCheck:
    def test_tool_check_only_when_no_user_data_resource(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_tool_rule(
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        )
        decision = engine.check("read_file", {"path": "/x"})
        assert decision.action is PermissionAction.ALLOW
        assert decision.tool_decision is not None
        assert decision.data_decision is None
        assert decision.resource_decision is None

    def test_tool_and_data_check_combined_allow(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_tool_rule(
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        )
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="engineering",
        )
        decision = engine.check(
            "read_file",
            {"path": "/x"},
            user=user,
            data=data,
            data_operation=DataOperation.READ,
        )
        assert decision.action is PermissionAction.ALLOW
        assert decision.tool_decision is not None
        assert decision.tool_decision.action is PermissionAction.ALLOW
        assert decision.data_decision is PermissionAction.ALLOW

    def test_tool_and_data_check_most_restrictive_wins(self) -> None:
        # Tool ALLOW + data DENY -> DENY.
        engine = EnterprisePermissionEngine()
        engine.add_tool_rule(
            PermissionRule(tool="export_tool", action=PermissionAction.ALLOW)
        )
        user = UserContext(user_id="alice", clearance=ClearanceLevel.INTERNAL)
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.RESTRICTED,  # above clearance -> DENY
        )
        decision = engine.check(
            "export_tool",
            {"path": "/x"},
            user=user,
            data=data,
            data_operation=DataOperation.READ,
        )
        assert decision.action is PermissionAction.DENY
        assert decision.tool_decision is not None
        assert decision.tool_decision.action is PermissionAction.ALLOW
        assert decision.data_decision is PermissionAction.DENY

    def test_tool_and_resource_check_combined_allow(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_tool_rule(
            PermissionRule(tool="operate_server", action=PermissionAction.ALLOW)
        )
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.MEDIUM,
            department="engineering",
        )
        decision = engine.check(
            "operate_server",
            {"resource_id": "srv-1"},
            user=user,
            resource=resource,
            resource_operation=ResourceOperation.OPERATE,
        )
        assert decision.action is PermissionAction.ALLOW
        assert decision.resource_decision is PermissionAction.ALLOW

    def test_tool_and_resource_check_most_restrictive_wins(self) -> None:
        # Tool ALLOW + resource DENY -> DENY.
        engine = EnterprisePermissionEngine()
        engine.add_tool_rule(
            PermissionRule(tool="configure_server", action=PermissionAction.ALLOW)
        )
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.CRITICAL,
            department="engineering",
        )
        decision = engine.check(
            "configure_server",
            {"resource_id": "srv-1"},
            user=user,
            resource=resource,
            resource_operation=ResourceOperation.CONFIGURE,
        )
        assert decision.action is PermissionAction.DENY
        assert decision.tool_decision is not None
        assert decision.tool_decision.action is PermissionAction.ALLOW
        assert decision.resource_decision is PermissionAction.DENY

    def test_deny_beats_ask_and_allow(self) -> None:
        # Tool ASK (default) + data DENY -> DENY.
        engine = EnterprisePermissionEngine(default_action=PermissionAction.ASK)
        user = UserContext(user_id="alice", clearance=ClearanceLevel.INTERNAL)
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.RESTRICTED,
        )
        decision = engine.check(
            "some_tool",
            {"x": 1},
            user=user,
            data=data,
            data_operation=DataOperation.READ,
        )
        assert decision.action is PermissionAction.DENY

    def test_ask_beats_allow(self) -> None:
        # Tool ALLOW + data ASK -> ASK.
        engine = EnterprisePermissionEngine()
        engine.add_tool_rule(
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        )
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="marketing",  # cross-dept read -> ASK
        )
        decision = engine.check(
            "read_file",
            {"path": "/x"},
            user=user,
            data=data,
            data_operation=DataOperation.READ,
        )
        assert decision.action is PermissionAction.ASK
        assert decision.tool_decision is not None
        assert decision.tool_decision.action is PermissionAction.ALLOW
        assert decision.data_decision is PermissionAction.ASK

    def test_all_layers_allow_returns_allow(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_tool_rule(
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        )
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="engineering",
        )
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.MEDIUM,
            department="engineering",
        )
        decision = engine.check(
            "read_file",
            {"path": "/x"},
            user=user,
            data=data,
            data_operation=DataOperation.READ,
            resource=resource,
            resource_operation=ResourceOperation.VIEW,
        )
        assert decision.action is PermissionAction.ALLOW
        assert decision.tool_decision is not None
        assert decision.data_decision is PermissionAction.ALLOW
        assert decision.resource_decision is PermissionAction.ALLOW

    def test_audit_trail_contains_all_evaluation_steps(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_tool_rule(
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        )
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="engineering",
        )
        resource = HardwareResource(
            resource_id="srv-1",
            criticality=ResourceCriticality.MEDIUM,
            department="engineering",
        )
        decision = engine.check(
            "read_file",
            {"path": "/x"},
            user=user,
            data=data,
            data_operation=DataOperation.READ,
            resource=resource,
            resource_operation=ResourceOperation.VIEW,
        )
        # Tool-level step.
        assert any("Tool check" in line for line in decision.audit_trail)
        # Data-level step.
        assert any("Data access" in line for line in decision.audit_trail)
        # Resource-level step.
        assert any("Resource access" in line for line in decision.audit_trail)

    def test_audit_trail_contains_data_rule_match(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_tool_rule(
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        )
        engine.add_data_permission(
            DataPermission(
                name="deny-read",
                operations=frozenset({DataOperation.READ}),
                classification=ClearanceLevel.INTERNAL,
                action=PermissionAction.DENY,
            )
        )
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="engineering",
        )
        decision = engine.check(
            "read_file",
            {"path": "/x"},
            user=user,
            data=data,
            data_operation=DataOperation.READ,
        )
        assert any("Data rule matched" in line for line in decision.audit_trail)
        assert any("deny-read" in line for line in decision.audit_trail)

    def test_reason_includes_layer_actions(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_tool_rule(
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        )
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="marketing",  # cross-dept -> ASK
        )
        decision = engine.check(
            "read_file",
            {"path": "/x"},
            user=user,
            data=data,
            data_operation=DataOperation.READ,
        )
        assert "Composite decision" in decision.reason
        assert "tool=" in decision.reason
        assert "data=" in decision.reason


# ---------------------------------------------------------------------------
# EnterpriseDecision
# ---------------------------------------------------------------------------


class TestEnterpriseDecision:
    def test_allowed_true_for_allow(self) -> None:
        decision = EnterpriseDecision(action=PermissionAction.ALLOW)
        assert decision.allowed is True

    def test_allowed_false_for_deny(self) -> None:
        decision = EnterpriseDecision(action=PermissionAction.DENY)
        assert decision.allowed is False

    def test_allowed_false_for_ask(self) -> None:
        decision = EnterpriseDecision(action=PermissionAction.ASK)
        assert decision.allowed is False

    def test_audit_trail_is_populated(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", clearance=ClearanceLevel.INTERNAL)
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.RESTRICTED,
        )
        decision = engine.check_data_access(user, data, DataOperation.READ)
        assert len(decision.audit_trail) > 0
        assert any("Data access" in line for line in decision.audit_trail)

    def test_reason_is_human_readable(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", clearance=ClearanceLevel.INTERNAL)
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.RESTRICTED,
        )
        decision = engine.check_data_access(user, data, DataOperation.READ)
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0
        assert "clearance" in decision.reason.lower()

    def test_default_fields(self) -> None:
        decision = EnterpriseDecision(action=PermissionAction.ASK)
        assert decision.tool_decision is None
        assert decision.data_decision is None
        assert decision.resource_decision is None
        assert decision.reason == ""
        assert decision.audit_trail == []


# ---------------------------------------------------------------------------
# create_enterprise_engine factory
# ---------------------------------------------------------------------------


class TestCreateEnterpriseEngine:
    def test_creates_engine_with_default_rules(self) -> None:
        engine = create_enterprise_engine()
        assert isinstance(engine, EnterprisePermissionEngine)
        # Data rule: restricted-export-deny.
        data_rule_names = {r.name for r in engine.data_rules}
        assert "restricted-export-deny" in data_rule_names
        # Resource rule: critical-allocate-ask.
        resource_rule_names = {r.name for r in engine.resource_rules}
        assert "critical-allocate-ask" in resource_rule_names

    def test_restricted_export_denied(self) -> None:
        engine = create_enterprise_engine()
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.RESTRICTED,
        )
        data = DataResource(
            resource_id="top-secret",
            classification=ClearanceLevel.RESTRICTED,
            department="engineering",  # same dept: EXPORT would ASK without rule
        )
        decision = engine.check_data_access(user, data, DataOperation.EXPORT)
        assert decision.action is PermissionAction.DENY

    def test_critical_allocate_asked(self) -> None:
        engine = create_enterprise_engine()
        user = UserContext(user_id="alice", department="engineering")
        resource = HardwareResource(
            resource_id="gpu-cluster-1",
            resource_type="gpu",
            criticality=ResourceCriticality.CRITICAL,
            department="marketing",  # cross-dept: would DENY without rule
        )
        decision = engine.check_resource_access(
            user, resource, ResourceOperation.ALLOCATE
        )
        assert decision.action is PermissionAction.ASK

    def test_read_only_tools_allowed(self) -> None:
        engine = create_enterprise_engine()
        for tool in ("read_file", "search", "web_fetch", "ask_question"):
            decision = engine.check_tool(tool, {"path": "/x"})
            assert decision.action is PermissionAction.ALLOW, tool

    def test_read_only_tools_allowed_via_composite_check(self) -> None:
        engine = create_enterprise_engine()
        decision = engine.check("read_file", {"path": "/x"})
        assert decision.action is PermissionAction.ALLOW
        assert decision.allowed is True

    def test_destructive_tools_asked_by_default(self) -> None:
        engine = create_enterprise_engine()
        # Act mode asks for destructive tools (no rule -> default ASK).
        decision = engine.check_tool("write_to_file", {"path": "/x"})
        assert decision.action is PermissionAction.ASK

    def test_default_rule_actions(self) -> None:
        engine = create_enterprise_engine()
        restricted_rule = next(
            r for r in engine.data_rules if r.name == "restricted-export-deny"
        )
        assert restricted_rule.action is PermissionAction.DENY
        assert restricted_rule.classification is ClearanceLevel.RESTRICTED

        critical_rule = next(
            r for r in engine.resource_rules if r.name == "critical-allocate-ask"
        )
        assert critical_rule.action is PermissionAction.ASK
        assert critical_rule.criticality is ResourceCriticality.CRITICAL


# ---------------------------------------------------------------------------
# Rule management
# ---------------------------------------------------------------------------


class TestRuleManagement:
    def test_add_and_list_data_rules(self) -> None:
        engine = EnterprisePermissionEngine()
        rule = DataPermission(
            name="r1",
            operations=frozenset({DataOperation.READ}),
            action=PermissionAction.ALLOW,
        )
        engine.add_data_permission(rule)
        assert engine.data_rules == [rule]

    def test_remove_data_rule(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_data_permission(
            DataPermission(
                name="r1",
                operations=frozenset({DataOperation.READ}),
                action=PermissionAction.ALLOW,
            )
        )
        assert engine.remove_data_permission("r1") is True
        assert engine.data_rules == []

    def test_remove_nonexistent_data_rule_returns_false(self) -> None:
        engine = EnterprisePermissionEngine()
        assert engine.remove_data_permission("nope") is False

    def test_add_and_list_resource_rules(self) -> None:
        engine = EnterprisePermissionEngine()
        rule = ResourcePermission(
            name="r1",
            operations=frozenset({ResourceOperation.VIEW}),
            action=PermissionAction.ALLOW,
        )
        engine.add_resource_permission(rule)
        assert engine.resource_rules == [rule]

    def test_remove_resource_rule(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_resource_permission(
            ResourcePermission(
                name="r1",
                operations=frozenset({ResourceOperation.VIEW}),
                action=PermissionAction.ALLOW,
            )
        )
        assert engine.remove_resource_permission("r1") is True
        assert engine.resource_rules == []

    def test_remove_nonexistent_resource_rule_returns_false(self) -> None:
        engine = EnterprisePermissionEngine()
        assert engine.remove_resource_permission("nope") is False

    def test_data_rules_returns_a_copy(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_data_permission(
            DataPermission(
                name="r1",
                operations=frozenset({DataOperation.READ}),
                action=PermissionAction.ALLOW,
            )
        )
        rules = engine.data_rules
        rules.clear()
        # Mutating the returned list must not affect the engine's state.
        assert len(engine.data_rules) == 1

    def test_clear_all_removes_every_layer(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_tool_rule(
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        )
        engine.add_data_permission(
            DataPermission(
                name="r1",
                operations=frozenset({DataOperation.READ}),
                action=PermissionAction.ALLOW,
            )
        )
        engine.add_resource_permission(
            ResourcePermission(
                name="r2",
                operations=frozenset({ResourceOperation.VIEW}),
                action=PermissionAction.ALLOW,
            )
        )
        engine.clear_all()
        assert engine.data_rules == []
        assert engine.resource_rules == []
        assert engine.base_engine.rules == []

    def test_base_engine_property(self) -> None:
        engine = EnterprisePermissionEngine()
        assert isinstance(engine.base_engine, PermissionEngine)

    def test_check_tool_delegates_to_base(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_tool_rule(
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        )
        decision = engine.check_tool("read_file", {"path": "/x"})
        assert decision.action is PermissionAction.ALLOW


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_data_checks_do_not_crash(self) -> None:
        engine = EnterprisePermissionEngine()
        engine.add_data_permission(
            DataPermission(
                name="internal-read-ask",
                operations=frozenset({DataOperation.READ}),
                classification=ClearanceLevel.INTERNAL,
                action=PermissionAction.ASK,
            )
        )
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="engineering",
        )
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    decision = engine.check_data_access(user, data, DataOperation.READ)
                    assert decision.action is PermissionAction.ASK
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_concurrent_rule_addition_and_check(self) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.INTERNAL,
        )
        data = DataResource(
            resource_id="doc-1",
            classification=ClearanceLevel.INTERNAL,
            department="engineering",
        )
        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(100):
                    engine.check_data_access(user, data, DataOperation.READ)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def writer() -> None:
            try:
                for i in range(100):
                    engine.add_data_permission(
                        DataPermission(
                            name=f"rule-{i}",
                            operations=frozenset({DataOperation.READ}),
                            classification=ClearanceLevel.INTERNAL,
                            action=PermissionAction.ASK,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads += [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_concurrent_composite_checks(self) -> None:
        engine = create_enterprise_engine()
        user = UserContext(
            user_id="alice",
            department="engineering",
            clearance=ClearanceLevel.RESTRICTED,
        )
        data = DataResource(
            resource_id="top-secret",
            classification=ClearanceLevel.RESTRICTED,
            department="engineering",
        )
        resource = HardwareResource(
            resource_id="gpu-1",
            resource_type="gpu",
            criticality=ResourceCriticality.CRITICAL,
            department="engineering",
        )
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    decision = engine.check(
                        "read_file",
                        {"path": "/x"},
                        user=user,
                        data=data,
                        data_operation=DataOperation.EXPORT,
                        resource=resource,
                        resource_operation=ResourceOperation.ALLOCATE,
                    )
                    # data EXPORT -> DENY (rule), so composite must be DENY.
                    assert decision.action is PermissionAction.DENY
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# Enum sanity (clearance ordering underpins the clearance check)
# ---------------------------------------------------------------------------


class TestClearanceOrdering:
    def test_weights_are_ordered(self) -> None:
        assert ClearanceLevel.PUBLIC.weight < ClearanceLevel.INTERNAL.weight
        assert ClearanceLevel.INTERNAL.weight < ClearanceLevel.CONFIDENTIAL.weight
        assert ClearanceLevel.CONFIDENTIAL.weight < ClearanceLevel.RESTRICTED.weight

    @pytest.mark.parametrize(
        ("user_clearance", "data_classification", "denied"),
        [
            (ClearanceLevel.PUBLIC, ClearanceLevel.PUBLIC, False),
            (ClearanceLevel.PUBLIC, ClearanceLevel.INTERNAL, True),
            (ClearanceLevel.INTERNAL, ClearanceLevel.CONFIDENTIAL, True),
            (ClearanceLevel.RESTRICTED, ClearanceLevel.RESTRICTED, False),
        ],
    )
    def test_clearance_matrix(
        self,
        user_clearance: ClearanceLevel,
        data_classification: ClearanceLevel,
        denied: bool,
    ) -> None:
        engine = EnterprisePermissionEngine()
        user = UserContext(user_id="alice", clearance=user_clearance)
        data = DataResource(
            resource_id="doc-1",
            classification=data_classification,
            # No department match -> falls to cross-dept defaults, but the
            # clearance gate runs first and decides DENY when insufficient.
            department="other",
        )
        decision = engine.check_data_access(user, data, DataOperation.READ)
        if denied:
            assert decision.action is PermissionAction.DENY
        else:
            # Equal/higher clearance -> not denied by clearance; cross-dept
            # read -> ASK.
            assert decision.action is PermissionAction.ASK
