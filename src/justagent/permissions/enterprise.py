"""Enterprise permission extensions — data-level and resource-level access control.

Extends the base :class:`~justagent.permissions.PermissionEngine` with
enterprise-grade access control that goes beyond tool-level rules:

* **Data-level permissions** — control who can read, write, or export
  specific documents, knowledge bases, and data sets based on data
  classification (PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED) and the
  user's clearance level.
* **Resource-level permissions** — control who can allocate, operate,
  or administer hardware resources (servers, storage, GPU clusters,
  databases) based on department ownership and role assignments.
* **Context-aware evaluation** — permission decisions incorporate the
  requesting user's identity, department, role hierarchy, and the
  sensitivity of the target data or resource.
* **Policy composition** — multiple policy sources (RBAC, ABAC, data
  classification, resource ownership) are composed into a single
  decision with a full audit trail.

Design:

* :class:`DataPermission` — a rule governing access to data by
  classification and owner.
* :class:`ResourcePermission` — a rule governing access to hardware
  resources by type and owner.
* :class:`EnterprisePermissionEngine` — composes tool-level rules
  (from the base engine) with data-level and resource-level rules,
  returning a rich :class:`EnterpriseDecision` with full context.
* :class:`UserContext` — the requesting user's identity, department,
  roles, and clearance level, used as input to the evaluation.
* :class:`DataResource` — a description of the data being accessed
  (classification, owner, department, tags).
* :class:`HardwareResource` — a description of the hardware being
  accessed (type, owner, department, criticality).

The enterprise engine is designed to be used alongside the base
:class:`PermissionEngine` — it delegates tool-level checks to the base
engine and adds data/resource checks on top.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from justagent.permissions import (
    PermissionAction,
    PermissionDecision,
    PermissionEngine,
    PermissionRule,
    PermissionScope,
)

logger = logging.getLogger("justagent.permissions.enterprise")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ClearanceLevel(str, Enum):  # noqa: UP042
    """User clearance level for data access.

    Higher levels grant access to more sensitive data classifications.
    The ordering is: PUBLIC < INTERNAL < CONFIDENTIAL < RESTRICTED.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    @property
    def weight(self) -> int:
        """Numeric weight for comparison (higher = more access)."""

        return _CLEARANCE_WEIGHTS[self]


_CLEARANCE_WEIGHTS: dict[ClearanceLevel, int] = {
    ClearanceLevel.PUBLIC: 0,
    ClearanceLevel.INTERNAL: 1,
    ClearanceLevel.CONFIDENTIAL: 2,
    ClearanceLevel.RESTRICTED: 3,
}


class DataOperation(str, Enum):  # noqa: UP042
    """Operations that can be performed on data resources."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXPORT = "export"
    SHARE = "share"
    ADMIN = "admin"


class ResourceOperation(str, Enum):  # noqa: UP042
    """Operations that can be performed on hardware resources."""

    VIEW = "view"
    ALLOCATE = "allocate"
    OPERATE = "operate"
    CONFIGURE = "configure"
    ADMIN = "admin"


class ResourceCriticality(str, Enum):  # noqa: UP042
    """Criticality level of a hardware resource."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class UserContext:
    """The requesting user's identity and access context.

    Attributes:
        user_id: Unique user identifier.
        username: Display name.
        department: Department code (e.g. ``"engineering"``).
        roles: Set of role names assigned to the user.
        clearance: Maximum data classification the user can access.
        metadata: Additional context (e.g. ``{"location": "HQ"}``).
    """

    user_id: str
    username: str = ""
    department: str = ""
    roles: set[str] = field(default_factory=set)
    clearance: ClearanceLevel = ClearanceLevel.INTERNAL
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        """Check if the user has a specific role."""

        return role in self.roles

    def is_admin(self) -> bool:
        """Check if the user has an administrative role."""

        return "admin" in self.roles or "superadmin" in self.roles


@dataclass
class DataResource:
    """A description of the data being accessed.

    Attributes:
        resource_id: Unique identifier for the data resource.
        classification: Sensitivity level of the data.
        owner: User ID of the data owner.
        department: Department that owns the data.
        tags: Free-form tags for additional access control.
    """

    resource_id: str
    classification: ClearanceLevel = ClearanceLevel.INTERNAL
    owner: str = ""
    department: str = ""
    tags: set[str] = field(default_factory=set)


@dataclass
class HardwareResource:
    """A description of the hardware resource being accessed.

    Attributes:
        resource_id: Unique identifier for the hardware resource.
        resource_type: Type of hardware (server, storage, gpu, etc.).
        criticality: How critical the resource is to operations.
        owner: Department that owns the resource.
        department: Department assigned to the resource.
        tags: Free-form tags for additional access control.
    """

    resource_id: str
    resource_type: str = "server"
    criticality: ResourceCriticality = ResourceCriticality.MEDIUM
    owner: str = ""
    department: str = ""
    tags: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Rule types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataPermission:
    """A rule governing access to data resources.

    Attributes:
        name: Human-readable rule name.
        operations: Set of operations this rule applies to.
        classification: Data classification this rule targets
            (``None`` means all classifications).
        department: Restrict to a specific department
            (``""`` means all departments).
        action: ALLOW, DENY, or ASK.
        description: Optional human-readable note.
    """

    name: str
    operations: frozenset[DataOperation]
    classification: ClearanceLevel | None = None
    department: str = ""
    action: PermissionAction = PermissionAction.DENY
    description: str = ""


@dataclass(frozen=True)
class ResourcePermission:
    """A rule governing access to hardware resources.

    Attributes:
        name: Human-readable rule name.
        operations: Set of operations this rule applies to.
        resource_type: Type of hardware this rule targets
            (``""`` means all types).
        criticality: Minimum criticality level this rule applies to
            (``None`` means all levels).
        department: Restrict to a specific department
            (``""`` means all departments).
        action: ALLOW, DENY, or ASK.
        description: Optional human-readable note.
    """

    name: str
    operations: frozenset[ResourceOperation]
    resource_type: str = ""
    criticality: ResourceCriticality | None = None
    department: str = ""
    action: PermissionAction = PermissionAction.DENY
    description: str = ""


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


@dataclass
class EnterpriseDecision:
    """The enterprise engine's decision for a single access check.

    Attributes:
        action: The final ALLOW / DENY / ASK decision.
        tool_decision: The decision from the base tool-level engine.
        data_decision: The decision from data-level rules (if applicable).
        resource_decision: The decision from resource-level rules (if applicable).
        reason: Human-readable explanation.
        audit_trail: List of evaluation steps for auditability.
    """

    action: PermissionAction
    tool_decision: PermissionDecision | None = None
    data_decision: PermissionAction | None = None
    resource_decision: PermissionAction | None = None
    reason: str = ""
    audit_trail: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        """Convenience: True if the action is ALLOW."""

        return self.action is PermissionAction.ALLOW


# ---------------------------------------------------------------------------
# Enterprise permission engine
# ---------------------------------------------------------------------------


class EnterprisePermissionEngine:
    """Composes tool-level, data-level, and resource-level permission checks.

    The engine wraps a base :class:`PermissionEngine` for tool-level
    checks and adds two additional evaluation layers:

    1. **Data-level** — when a tool operates on a known data resource,
       the engine checks :class:`DataPermission` rules against the
       user's clearance level and the data's classification.
    2. **Resource-level** — when a tool operates on a hardware resource,
       the engine checks :class:`ResourcePermission` rules against the
       user's department and the resource's criticality.

    The most restrictive decision wins: if any layer returns DENY, the
    final decision is DENY. If all layers return ALLOW, the final
    decision is ALLOW. If any layer returns ASK and none return DENY,
    the final decision is ASK.

    Example::

        from justagent.permissions.enterprise import (
            EnterprisePermissionEngine,
            UserContext,
            ClearanceLevel,
            DataResource,
            DataOperation,
            DataPermission,
        )
        from justagent.permissions import PermissionAction

        engine = EnterprisePermissionEngine()
        engine.add_data_permission(DataPermission(
            name="confidential-read",
            operations=frozenset({DataOperation.READ}),
            classification=ClearanceLevel.CONFIDENTIAL,
            action=PermissionAction.ASK,
        ))

        user = UserContext(user_id="alice", clearance=ClearanceLevel.CONFIDENTIAL)
        data = DataResource(resource_id="doc-1", classification=ClearanceLevel.CONFIDENTIAL)
        decision = engine.check_data_access(user, data, DataOperation.READ)
        assert decision.action == PermissionAction.ASK
    """

    def __init__(
        self,
        *,
        base_engine: PermissionEngine | None = None,
        default_action: PermissionAction = PermissionAction.ASK,
    ) -> None:
        self._base = base_engine or PermissionEngine(default_action=default_action)
        self._data_rules: list[DataPermission] = []
        self._resource_rules: list[ResourcePermission] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Base engine delegation
    # ------------------------------------------------------------------

    @property
    def base_engine(self) -> PermissionEngine:
        """The underlying tool-level permission engine."""

        return self._base

    def add_tool_rule(self, rule: PermissionRule) -> None:
        """Add a tool-level rule to the base engine."""

        self._base.add_rule(rule)

    def check_tool(
        self,
        tool: str,
        tool_input: dict[str, Any],
    ) -> PermissionDecision:
        """Delegate a tool-level check to the base engine."""

        return self._base.check(tool, tool_input)

    # ------------------------------------------------------------------
    # Data-level permissions
    # ------------------------------------------------------------------

    def add_data_permission(self, rule: DataPermission) -> None:
        """Add a data-level permission rule."""

        with self._lock:
            self._data_rules.append(rule)

    def remove_data_permission(self, name: str) -> bool:
        """Remove a data-level permission rule by name. Returns True if removed."""

        with self._lock:
            before = len(self._data_rules)
            self._data_rules = [r for r in self._data_rules if r.name != name]
            return len(self._data_rules) < before

    def check_data_access(
        self,
        user: UserContext,
        data: DataResource,
        operation: DataOperation,
    ) -> EnterpriseDecision:
        """Evaluate data-level access for a user.

        The evaluation considers:

        * **Clearance** — the user's clearance level must be >= the
          data's classification.
        * **Ownership** — the data owner always has full access.
        * **Department** — same-department users may have broader access.
        * **Rules** — explicit :class:`DataPermission` rules override
          defaults.

        Args:
            user: The requesting user's context.
            data: The data resource being accessed.
            operation: The operation being performed.

        Returns:
            An :class:`EnterpriseDecision` with the final action and
            audit trail.
        """

        audit: list[str] = []
        audit.append(
            f"Data access: user={user.user_id} data={data.resource_id} "
            f"op={operation.value} classification={data.classification.value}"
        )

        # Owner always has full access.
        if data.owner and data.owner == user.user_id:
            audit.append(f"Owner match — ALLOW (user={user.user_id})")
            return EnterpriseDecision(
                action=PermissionAction.ALLOW,
                data_decision=PermissionAction.ALLOW,
                reason="Data owner has full access",
                audit_trail=audit,
            )

        # Admin override.
        if user.is_admin():
            audit.append("Admin role — ALLOW")
            return EnterpriseDecision(
                action=PermissionAction.ALLOW,
                data_decision=PermissionAction.ALLOW,
                reason="Administrative override",
                audit_trail=audit,
            )

        # Clearance check.
        if user.clearance.weight < data.classification.weight:
            audit.append(
                f"Clearance denied — user={user.clearance.value} "
                f"data={data.classification.value}"
            )
            return EnterpriseDecision(
                action=PermissionAction.DENY,
                data_decision=PermissionAction.DENY,
                reason=(
                    f"User clearance ({user.clearance.value}) below "
                    f"data classification ({data.classification.value})"
                ),
                audit_trail=audit,
            )

        # Evaluate explicit data rules.
        rule_decision = self._evaluate_data_rules(user, data, operation, audit)
        if rule_decision is not None:
            return EnterpriseDecision(
                action=rule_decision,
                data_decision=rule_decision,
                reason=f"Matched data permission rule",
                audit_trail=audit,
            )

        # Default: same department → allow read, ask for write/export.
        if data.department and data.department == user.department:
            if operation in (DataOperation.READ,):
                audit.append(
                    f"Same department ({user.department}) read — ALLOW"
                )
                return EnterpriseDecision(
                    action=PermissionAction.ALLOW,
                    data_decision=PermissionAction.ALLOW,
                    reason=f"Same-department read access ({user.department})",
                    audit_trail=audit,
                )
            audit.append(
                f"Same department ({user.department}) {operation.value} — ASK"
            )
            return EnterpriseDecision(
                action=PermissionAction.ASK,
                data_decision=PermissionAction.ASK,
                reason=f"Same-department {operation.value} requires confirmation",
                audit_trail=audit,
            )

        # Cross-department access: ask for read, deny for write/export.
        if operation in (DataOperation.READ,):
            audit.append("Cross-department read — ASK")
            return EnterpriseDecision(
                action=PermissionAction.ASK,
                data_decision=PermissionAction.ASK,
                reason="Cross-department read requires confirmation",
                audit_trail=audit,
            )

        audit.append(f"Cross-department {operation.value} — DENY")
        return EnterpriseDecision(
            action=PermissionAction.DENY,
            data_decision=PermissionAction.DENY,
            reason=f"Cross-department {operation.value} denied",
            audit_trail=audit,
        )

    def _evaluate_data_rules(
        self,
        user: UserContext,
        data: DataResource,
        operation: DataOperation,
        audit: list[str],
    ) -> PermissionAction | None:
        """Evaluate explicit data permission rules. Returns None if no rule matches."""

        with self._lock:
            rules = list(self._data_rules)

        for rule in rules:
            if operation not in rule.operations:
                continue
            if rule.classification is not None and rule.classification != data.classification:
                continue
            if rule.department and rule.department != user.department:
                continue
            audit.append(
                f"Data rule matched: {rule.name} → {rule.action.value}"
            )
            return rule.action
        return None

    # ------------------------------------------------------------------
    # Resource-level permissions
    # ------------------------------------------------------------------

    def add_resource_permission(self, rule: ResourcePermission) -> None:
        """Add a resource-level permission rule."""

        with self._lock:
            self._resource_rules.append(rule)

    def remove_resource_permission(self, name: str) -> bool:
        """Remove a resource-level permission rule by name."""

        with self._lock:
            before = len(self._resource_rules)
            self._resource_rules = [r for r in self._resource_rules if r.name != name]
            return len(self._resource_rules) < before

    def check_resource_access(
        self,
        user: UserContext,
        resource: HardwareResource,
        operation: ResourceOperation,
    ) -> EnterpriseDecision:
        """Evaluate resource-level access for a user.

        The evaluation considers:

        * **Criticality** — critical resources require admin role for
          configure/admin operations.
        * **Department** — same-department users have broader access.
        * **Rules** — explicit :class:`ResourcePermission` rules override
          defaults.

        Args:
            user: The requesting user's context.
            resource: The hardware resource being accessed.
            operation: The operation being performed.

        Returns:
            An :class:`EnterpriseDecision` with the final action and
            audit trail.
        """

        audit: list[str] = []
        audit.append(
            f"Resource access: user={user.user_id} resource={resource.resource_id} "
            f"op={operation.value} type={resource.resource_type} "
            f"criticality={resource.criticality.value}"
        )

        # Admin override.
        if user.is_admin():
            audit.append("Admin role — ALLOW")
            return EnterpriseDecision(
                action=PermissionAction.ALLOW,
                resource_decision=PermissionAction.ALLOW,
                reason="Administrative override",
                audit_trail=audit,
            )

        # Critical resources: configure/admin require admin role.
        if (
            resource.criticality is ResourceCriticality.CRITICAL
            and operation in (ResourceOperation.CONFIGURE, ResourceOperation.ADMIN)
            and not user.is_admin()
        ):
            audit.append(
                f"Critical resource {operation.value} without admin — DENY"
            )
            return EnterpriseDecision(
                action=PermissionAction.DENY,
                resource_decision=PermissionAction.DENY,
                reason=(
                    f"Critical resource {operation.value} requires admin role"
                ),
                audit_trail=audit,
            )

        # Evaluate explicit resource rules.
        rule_decision = self._evaluate_resource_rules(user, resource, operation, audit)
        if rule_decision is not None:
            return EnterpriseDecision(
                action=rule_decision,
                resource_decision=rule_decision,
                reason="Matched resource permission rule",
                audit_trail=audit,
            )

        # Same department: allow view/operate, ask for allocate/configure.
        if resource.department and resource.department == user.department:
            if operation in (ResourceOperation.VIEW, ResourceOperation.OPERATE):
                audit.append(
                    f"Same department ({user.department}) {operation.value} — ALLOW"
                )
                return EnterpriseDecision(
                    action=PermissionAction.ALLOW,
                    resource_decision=PermissionAction.ALLOW,
                    reason=f"Same-department {operation.value} access",
                    audit_trail=audit,
                )
            audit.append(
                f"Same department ({user.department}) {operation.value} — ASK"
            )
            return EnterpriseDecision(
                action=PermissionAction.ASK,
                resource_decision=PermissionAction.ASK,
                reason=f"Same-department {operation.value} requires confirmation",
                audit_trail=audit,
            )

        # Cross-department: ask for view, deny for operate/configure/admin.
        if operation is ResourceOperation.VIEW:
            audit.append("Cross-department view — ASK")
            return EnterpriseDecision(
                action=PermissionAction.ASK,
                resource_decision=PermissionAction.ASK,
                reason="Cross-department view requires confirmation",
                audit_trail=audit,
            )

        audit.append(f"Cross-department {operation.value} — DENY")
        return EnterpriseDecision(
            action=PermissionAction.DENY,
            resource_decision=PermissionAction.DENY,
            reason=f"Cross-department {operation.value} denied",
            audit_trail=audit,
        )

    def _evaluate_resource_rules(
        self,
        user: UserContext,
        resource: HardwareResource,
        operation: ResourceOperation,
        audit: list[str],
    ) -> PermissionAction | None:
        """Evaluate explicit resource permission rules."""

        with self._lock:
            rules = list(self._resource_rules)

        for rule in rules:
            if operation not in rule.operations:
                continue
            if rule.resource_type and rule.resource_type != resource.resource_type:
                continue
            if rule.criticality is not None and rule.criticality != resource.criticality:
                continue
            if rule.department and rule.department != user.department:
                continue
            audit.append(
                f"Resource rule matched: {rule.name} → {rule.action.value}"
            )
            return rule.action
        return None

    # ------------------------------------------------------------------
    # Composite check
    # ------------------------------------------------------------------

    def check(
        self,
        tool: str,
        tool_input: dict[str, Any],
        *,
        user: UserContext | None = None,
        data: DataResource | None = None,
        resource: HardwareResource | None = None,
        data_operation: DataOperation | None = None,
        resource_operation: ResourceOperation | None = None,
    ) -> EnterpriseDecision:
        """Composite check across all permission layers.

        Evaluates tool-level, data-level, and resource-level permissions
        and returns the most restrictive decision.

        Args:
            tool: Tool name for the base engine.
            tool_input: Tool input dict for the base engine.
            user: User context for data/resource checks.
            data: Data resource being accessed (if any).
            resource: Hardware resource being accessed (if any).
            data_operation: Data operation being performed (if any).
            resource_operation: Resource operation being performed (if any).

        Returns:
            An :class:`EnterpriseDecision` with the composite result.
        """

        audit: list[str] = []
        actions: list[PermissionAction] = []

        # Tool-level check.
        tool_decision = self._base.check(tool, tool_input)
        audit.append(f"Tool check: {tool} → {tool_decision.action.value}")
        actions.append(tool_decision.action)

        # Data-level check.
        data_action: PermissionAction | None = None
        if user is not None and data is not None and data_operation is not None:
            data_decision = self.check_data_access(user, data, data_operation)
            data_action = data_decision.action
            actions.append(data_action)
            audit.extend(data_decision.audit_trail)

        # Resource-level check.
        resource_action: PermissionAction | None = None
        if user is not None and resource is not None and resource_operation is not None:
            resource_decision = self.check_resource_access(
                user, resource, resource_operation
            )
            resource_action = resource_decision.action
            actions.append(resource_action)
            audit.extend(resource_decision.audit_trail)

        # Most restrictive decision wins.
        if PermissionAction.DENY in actions:
            final = PermissionAction.DENY
        elif PermissionAction.ASK in actions:
            final = PermissionAction.ASK
        else:
            final = PermissionAction.ALLOW

        reason = f"Composite decision: {final.value} (tool={tool_decision.action.value}"
        if data_action is not None:
            reason += f", data={data_action.value}"
        if resource_action is not None:
            reason += f", resource={resource_action.value}"
        reason += ")"

        logger.debug("Enterprise permission check: %s", reason)

        return EnterpriseDecision(
            action=final,
            tool_decision=tool_decision,
            data_decision=data_action,
            resource_decision=resource_action,
            reason=reason,
            audit_trail=audit,
        )

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    @property
    def data_rules(self) -> list[DataPermission]:
        """Return a copy of the data-level rules."""

        with self._lock:
            return list(self._data_rules)

    @property
    def resource_rules(self) -> list[ResourcePermission]:
        """Return a copy of the resource-level rules."""

        with self._lock:
            return list(self._resource_rules)

    def clear_all(self) -> None:
        """Clear all rules (tool, data, and resource)."""

        with self._lock:
            self._base.clear_rules()
            self._data_rules.clear()
            self._resource_rules.clear()


# ---------------------------------------------------------------------------
# Convenience: create enterprise engines for common scenarios
# ---------------------------------------------------------------------------


def create_enterprise_engine(
    *,
    default_action: PermissionAction = PermissionAction.ASK,
) -> EnterprisePermissionEngine:
    """Create an enterprise permission engine with sensible defaults.

    The default configuration:

    * Allows read-only tool operations (read_file, search, web_fetch).
    * Asks for destructive tool operations (write, edit, run_command).
    * Denies export of RESTRICTED data by non-admins.
    * Asks for allocation of CRITICAL resources by non-admins.
    """

    engine = EnterprisePermissionEngine(default_action=default_action)

    # Tool-level defaults (same as act mode).
    from justagent.permissions import create_act_mode_engine

    engine._base = create_act_mode_engine()

    # Data-level defaults.
    engine.add_data_permission(DataPermission(
        name="restricted-export-deny",
        operations=frozenset({DataOperation.EXPORT}),
        classification=ClearanceLevel.RESTRICTED,
        action=PermissionAction.DENY,
        description="Non-admin export of RESTRICTED data is denied",
    ))

    # Resource-level defaults.
    engine.add_resource_permission(ResourcePermission(
        name="critical-allocate-ask",
        operations=frozenset({ResourceOperation.ALLOCATE}),
        criticality=ResourceCriticality.CRITICAL,
        action=PermissionAction.ASK,
        description="Allocation of CRITICAL resources requires confirmation",
    ))

    return engine


__all__ = [
    # Enums
    "ClearanceLevel",
    "DataOperation",
    "ResourceCriticality",
    "ResourceOperation",
    # Models
    "DataPermission",
    "DataResource",
    "EnterpriseDecision",
    "HardwareResource",
    "ResourcePermission",
    "UserContext",
    # Engine
    "EnterprisePermissionEngine",
    "create_enterprise_engine",
]
