"""Role-based access control — permissions, roles, users and enforcement.

Provides a thread-safe, in-memory RBAC engine for the JustAgent platform.
Access decisions are computed by resolving a user's assigned roles (and
their inheritance chain) against a permission matrix keyed by resource
type.

Design:

* :class:`Permission` — the set of actions that can be granted or denied.
* :class:`ResourceType` — the set of protectable resource categories.
* :class:`Role` — a named bundle of permissions per resource type, with
  optional inheritance from other roles.
* :class:`User` — a platform identity with assigned role ids.
* :class:`AccessDecision` — a structured result explaining *why* access
  was allowed or denied.
* :class:`RBACEngine` — the thread-safe manager that owns roles and
  users, evaluates access and supports assignment / revocation.

Role inheritance is transitive: if role *A* inherits from *B* and *B*
inherits from *C*, a user with *A* effectively holds the union of all
three roles' permissions. Inheritance cycles are detected and broken.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from justagent.utils import now

logger = logging.getLogger("justagent.security.rbac")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RBACError(Exception):
    """Raised for invalid RBAC operations (unknown role, duplicate user...)."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Permission(str, Enum):  # noqa: UP042 - match existing codebase style
    """Actions that can be granted or denied on a resource.

    Attributes:
        READ: View or retrieve a resource.
        WRITE: Create or modify a resource.
        DELETE: Remove a resource.
        ADMIN: Full administrative control over a resource.
        EXECUTE: Run / invoke a resource (workflows, agents).
        DELEGATE: Grant a subset of one's own permissions to others.
        SHARE: Share a resource with other users.
        EXPORT: Export / download a resource outside the platform.
        REVIEW: Review / examine a resource.
        SEAL: Affix an official seal.
        ARCHIVE: File a resource into archival storage.
        SERVE: Serve / deliver a legal document.
    """

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"
    DELEGATE = "delegate"
    SHARE = "share"
    EXPORT = "export"
    REVIEW = "review"
    SEAL = "seal"
    ARCHIVE = "archive"
    SERVE = "serve"


class ResourceType(str, Enum):  # noqa: UP042
    """Protectable resource categories in the platform.

    Attributes:
        DOCUMENT: Knowledge-base documents and files.
        CHANNEL: Communication channels and conversations.
        MEETING: Scheduled meetings and their artifacts.
        DATABASE: Connected databases and their records.
        STORAGE: Object/file storage buckets.
        AGENT: AI agent instances and their configurations.
        WORKFLOW: Automated workflow definitions and runs.
        CONFIG: System and project configuration.
        USER: User accounts and profiles.
        REPORT: Generated reports and analytics.
        CASE_FILE: Judicial case files (案卷).
        EVIDENCE: Evidence materials (证据).
        LEGAL_DOCUMENT: Legal documents / instruments (法律文书).
        COURT_RECORD: Court trial transcripts (庭审笔录).
        JUDGGMENT: Judgment / adjudication documents (裁判文书).
    """

    DOCUMENT = "document"
    CHANNEL = "channel"
    MEETING = "meeting"
    DATABASE = "database"
    STORAGE = "storage"
    AGENT = "agent"
    WORKFLOW = "workflow"
    CONFIG = "config"
    USER = "user"
    REPORT = "report"
    CASE_FILE = "case_file"
    EVIDENCE = "evidence"
    LEGAL_DOCUMENT = "legal_document"
    COURT_RECORD = "court_record"
    JUDGGMENT = "judggment"


class UserStatus(str, Enum):  # noqa: UP042
    """Lifecycle status of a user account.

    Attributes:
        ACTIVE: The user can authenticate and operate.
        INACTIVE: The user is disabled (cannot authenticate).
        SUSPENDED: The user is temporarily blocked (e.g. security hold).
    """

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Role(BaseModel):
    """A named bundle of permissions keyed by resource type.

    Attributes:
        id: Unique role identifier (auto-generated UUID4 hex).
        name: Human-readable role name (e.g. ``"editor"``).
        description: What the role is for.
        permissions: Mapping of :class:`ResourceType` to the set of
            :class:`Permission` values granted by this role.
        inherits_from: IDs of roles whose permissions are implicitly
            included (transitive, cycle-safe).
        created_at: Unix timestamp of creation.
        system: Whether this is a built-in system role (cannot be deleted).
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: str = ""
    permissions: dict[ResourceType, set[Permission]] = Field(default_factory=dict)
    inherits_from: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=now)
    system: bool = False

    def grants(
        self,
        resource_type: ResourceType,
        permission: Permission,
    ) -> bool:
        """True if *this role* (ignoring inheritance) grants the permission."""

        perms = self.permissions.get(resource_type, set())
        return permission in perms or Permission.ADMIN in perms


class User(BaseModel):
    """A platform identity with assigned role ids.

    Attributes:
        id: Unique user identifier (auto-generated UUID4 hex).
        username: Login / handle (unique within the engine).
        display_name: Full name for display.
        email: Primary email address.
        department: Organisational unit.
        roles: List of role IDs assigned to this user.
        status: Current :class:`UserStatus`.
        metadata: Arbitrary structured metadata.
        created_at: Unix timestamp of creation.
        last_active: Unix timestamp of last activity (0 = never).
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    username: str
    display_name: str = ""
    email: str = ""
    department: str = ""
    roles: list[str] = Field(default_factory=list)
    status: UserStatus = UserStatus.ACTIVE
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=now)
    last_active: float = 0.0

    @property
    def is_active(self) -> bool:
        """True if the user can currently operate (status is ACTIVE)."""

        return self.status is UserStatus.ACTIVE


# ---------------------------------------------------------------------------
# Access decision
# ---------------------------------------------------------------------------


@dataclass
class AccessDecision:
    """Structured result of an access check.

    Attributes:
        allowed: Whether access was granted.
        reason: Human-readable explanation of the decision.
        matched_role: The role ID that satisfied the request, or
            ``None`` when denied.
        permission: The :class:`Permission` that was checked.
        resource_type: The :class:`ResourceType` that was checked.
    """

    allowed: bool
    reason: str = ""
    matched_role: str | None = None
    permission: Permission = Permission.READ
    resource_type: ResourceType = ResourceType.DOCUMENT

    def __bool__(self) -> bool:
        return self.allowed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RBAC engine
# ---------------------------------------------------------------------------


class RBACEngine:
    """Thread-safe role-based access control engine.

    Owns the role and user registries and evaluates access requests by
    resolving a user's roles (with transitive inheritance) against the
    permission matrix.

    Built-in system roles are registered on construction:

    * ``admin`` — :attr:`Permission.ADMIN` on every resource type.
    * ``viewer`` — :attr:`Permission.READ` on every resource type.
    * ``editor`` — :attr:`Permission.READ` and :attr:`Permission.WRITE`
      on every resource type, inherits from ``viewer``.

    Example::

        engine = RBACEngine()
        engine.register_user(User(username="alice", roles=["admin"]))
        assert engine.check_access("alice", ResourceType.DOCUMENT, Permission.DELETE)
    """

    #: IDs of the built-in system roles.
    ADMIN_ROLE = "role_admin"
    VIEWER_ROLE = "role_viewer"
    EDITOR_ROLE = "role_editor"

    def __init__(self) -> None:
        self._roles: dict[str, Role] = {}
        self._users: dict[str, User] = {}
        self._username_index: dict[str, str] = {}
        self._lock = threading.RLock()
        self._init_system_roles()

    def _init_system_roles(self) -> None:
        """Register the built-in system roles."""

        admin_perms = {rt: {Permission.ADMIN} for rt in ResourceType}
        viewer_perms = {rt: {Permission.READ} for rt in ResourceType}
        editor_perms = {rt: {Permission.READ, Permission.WRITE} for rt in ResourceType}

        self._roles[self.ADMIN_ROLE] = Role(
            id=self.ADMIN_ROLE,
            name="Administrator",
            description="Full administrative access to all resources.",
            permissions=admin_perms,
            system=True,
        )
        self._roles[self.VIEWER_ROLE] = Role(
            id=self.VIEWER_ROLE,
            name="Viewer",
            description="Read-only access to all resources.",
            permissions=viewer_perms,
            system=True,
        )
        self._roles[self.EDITOR_ROLE] = Role(
            id=self.EDITOR_ROLE,
            name="Editor",
            description="Read and write access to all resources.",
            permissions=editor_perms,
            inherits_from=[self.VIEWER_ROLE],
            system=True,
        )

    # ------------------------------------------------------------------
    # Role management
    # ------------------------------------------------------------------

    def create_role(self, role: Role) -> Role:
        """Register a new role.

        Raises:
            RBACError: If a role with the same ID already exists.
        """

        with self._lock:
            if role.id in self._roles:
                raise RBACError(f"Role already exists: {role.id}")
            self._roles[role.id] = role
        logger.info("Created role %s (%s)", role.id, role.name)
        return role

    def get_role(self, role_id: str) -> Role | None:
        """Return a role by id, or ``None``."""

        with self._lock:
            return self._roles.get(role_id)

    def list_roles(self) -> list[Role]:
        """Return all registered roles."""

        with self._lock:
            return list(self._roles.values())

    def delete_role(self, role_id: str) -> Role | None:
        """Delete a custom role. System roles cannot be deleted.

        Returns:
            The removed role, or ``None`` if not found.

        Raises:
            RBACError: If the role is a system role.
        """

        with self._lock:
            role = self._roles.get(role_id)
            if role is None:
                return None
            if role.system:
                raise RBACError(f"Cannot delete system role: {role_id}")
            self._roles.pop(role_id, None)
            # Remove the role from any user's role list.
            for user in self._users.values():
                if role_id in user.roles:
                    user.roles = [r for r in user.roles if r != role_id]
        logger.info("Deleted role %s", role_id)
        return role

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------

    def register_user(self, user: User) -> User:
        """Register a new user.

        Raises:
            RBACError: If the user ID or username is already taken.
        """

        with self._lock:
            if user.id in self._users:
                raise RBACError(f"User already exists: {user.id}")
            if user.username in self._username_index:
                raise RBACError(f"Username already taken: {user.username}")
            self._users[user.id] = user
            self._username_index[user.username] = user.id
        logger.info("Registered user %s (%s)", user.username, user.id)
        return user

    def get_user(self, user_id_or_username: str) -> User | None:
        """Return a user by id or username, or ``None``.

        The lookup tries the user-id index first, then the username index.
        """

        with self._lock:
            return self._resolve_user(user_id_or_username)

    def list_users(self) -> list[User]:
        """Return all registered users."""

        with self._lock:
            return list(self._users.values())

    def update_user_status(self, user_id: str, status: UserStatus) -> User:
        """Change a user's status.

        *user_id* may be the user ID or the username.

        Raises:
            RBACError: If the user is unknown.
        """

        with self._lock:
            user = self._resolve_user(user_id)
            if user is None:
                raise RBACError(f"User not found: {user_id}")
            user.status = status
        logger.info("User %s status -> %s", user_id, status.value)
        return user

    # ------------------------------------------------------------------
    # Role assignment
    # ------------------------------------------------------------------

    def assign_role(self, user_id: str, role_id: str) -> User:
        """Assign a role to a user (idempotent).

        *user_id* may be the user ID or the username.

        Raises:
            RBACError: If the user or role is unknown.
        """

        with self._lock:
            user = self._resolve_user(user_id)
            if user is None:
                raise RBACError(f"User not found: {user_id}")
            if role_id not in self._roles:
                raise RBACError(f"Role not found: {role_id}")
            if role_id not in user.roles:
                user.roles.append(role_id)
        logger.info("Assigned role %s to user %s", role_id, user_id)
        return user

    def revoke_role(self, user_id: str, role_id: str) -> User:
        """Revoke a role from a user (idempotent).

        *user_id* may be the user ID or the username.

        Raises:
            RBACError: If the user is unknown.
        """

        with self._lock:
            user = self._resolve_user(user_id)
            if user is None:
                raise RBACError(f"User not found: {user_id}")
            user.roles = [r for r in user.roles if r != role_id]
        logger.info("Revoked role %s from user %s", role_id, user_id)
        return user

    # ------------------------------------------------------------------
    # Access evaluation
    # ------------------------------------------------------------------

    def check_access(
        self,
        user: str | User,
        resource_type: ResourceType,
        permission: Permission,
    ) -> bool:
        """Return ``True`` if *user* may perform *permission* on *resource_type*.

        *user* may be a user ID, username or a :class:`User` instance.
        Inactive and suspended users are always denied.
        """

        decision = self.evaluate_access(user, resource_type, permission)
        return decision.allowed

    def evaluate_access(
        self,
        user: str | User,
        resource_type: ResourceType,
        permission: Permission,
    ) -> AccessDecision:
        """Return a detailed :class:`AccessDecision`.

        The decision includes the matched role and a human-readable
        reason, making it suitable for audit logging.
        """

        with self._lock:
            user_obj = user if isinstance(user, User) else self._resolve_user(user)

            if user_obj is None:
                return AccessDecision(
                    allowed=False,
                    reason=f"User not found: {user}",
                    permission=permission,
                    resource_type=resource_type,
                )

            if not user_obj.is_active:
                return AccessDecision(
                    allowed=False,
                    reason=f"User is {user_obj.status.value} (not active)",
                    permission=permission,
                    resource_type=resource_type,
                )

            # Resolve all roles (with inheritance) for this user.
            all_role_ids = self._resolve_role_chain(user_obj.roles)
            for role_id in all_role_ids:
                role = self._roles.get(role_id)
                if role is None:
                    continue
                if role.grants(resource_type, permission):
                    return AccessDecision(
                        allowed=True,
                        reason=f"Granted by role '{role.name}' ({role_id})",
                        matched_role=role_id,
                        permission=permission,
                        resource_type=resource_type,
                    )

            return AccessDecision(
                allowed=False,
                reason=(f"No role grants {permission.value} on {resource_type.value}"),
                permission=permission,
                resource_type=resource_type,
            )

    def _resolve_role_chain(self, role_ids: list[str]) -> list[str]:
        """Return the transitive closure of role IDs (cycle-safe).

        Performs a depth-first traversal of the inheritance graph,
        skipping already-visited nodes to break cycles.
        """

        visited: set[str] = set()
        result: list[str] = []

        def _visit(rid: str) -> None:
            if rid in visited:
                return
            visited.add(rid)
            result.append(rid)
            role = self._roles.get(rid)
            if role is not None:
                for parent_id in role.inherits_from:
                    _visit(parent_id)

        for rid in role_ids:
            _visit(rid)
        return result

    def _resolve_user(self, user_id_or_username: str) -> User | None:
        """Look up a user by id or username (caller must hold the lock).

        Tries the user-id index first, then the username index.
        """

        if user_id_or_username in self._users:
            return self._users[user_id_or_username]
        uid = self._username_index.get(user_id_or_username)
        if uid is not None:
            return self._users.get(uid)
        return None

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_users_by_role(self, role_id: str) -> list[User]:
        """Return all users who have *role_id* assigned (directly)."""

        with self._lock:
            return [user for user in self._users.values() if role_id in user.roles]

    def list_permissions(
        self,
        user: str | User,
        resource_type: ResourceType,
    ) -> set[Permission]:
        """Return the set of permissions *user* has on *resource_type*."""

        with self._lock:
            user_obj = user if isinstance(user, User) else self._resolve_user(user)
            if user_obj is None or not user_obj.is_active:
                return set()

            all_role_ids = self._resolve_role_chain(user_obj.roles)
            result: set[Permission] = set()
            for role_id in all_role_ids:
                role = self._roles.get(role_id)
                if role is None:
                    continue
                perms = role.permissions.get(resource_type, set())
                result |= perms
                # ADMIN implies all other permissions.
                if Permission.ADMIN in result:
                    return set(Permission)
            return result

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    @property
    def role_count(self) -> int:
        """Total number of registered roles."""

        with self._lock:
            return len(self._roles)

    @property
    def user_count(self) -> int:
        """Total number of registered users."""

        with self._lock:
            return len(self._users)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary suitable for dashboards."""

        with self._lock:
            active_users = sum(1 for u in self._users.values() if u.is_active)
            return {
                "roles": len(self._roles),
                "users": len(self._users),
                "active_users": active_users,
                "system_roles": sum(1 for r in self._roles.values() if r.system),
            }


__all__ = [
    "AccessDecision",
    "Permission",
    "RBACEngine",
    "RBACError",
    "ResourceType",
    "Role",
    "User",
    "UserStatus",
]
