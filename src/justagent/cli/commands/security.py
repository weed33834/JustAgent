"""justagent security command — RBAC, encryption, DLP and compliance.

Exposes the :mod:justagent.security package through the CLI:

* justagent security rbac         — role / user / permission management.
* justagent security encrypt      — authenticated file encryption (AEAD).
* justagent security decrypt      — file decryption.
* justagent security dlp          — PII scanning and data-loss prevention.
* justagent security compliance   — framework checks and audit trail.

The security engines (:class:RBACEngine, :class:KeyManager,
:class:ComplianceChecker, :class:AuditTrailManager) are in-memory by
design.  To make the CLI usable across invocations, this module
transparently persists state to a JSON file
(<project_root>/.justagent/security_state.json by default, overridable
via the JUSTAGENT_SECURITY_STATE environment variable).  Each command
loads the state, performs its operation, and writes the state back
(read-only commands skip the write).

Optional dependencies (e.g. cryptography for the encryption
subsystem) are imported independently per subsystem so that a missing
library degrades only the affected command group rather than the whole
module.

The module follows the same conventions as the other justagent
commands: a register(parent: typer.Typer) entry point, ctx.obj
for shared config / audit / verbosity, Rich tables for listings, and
Google-style docstrings with full type hints.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from justagent.cli.commands import _common as common
from justagent.cli.display import get_console

# ---------------------------------------------------------------------------
# Subsystem imports — each imported independently so a missing optional
# dependency (e.g. cryptography) degrades only the affected group.
# ---------------------------------------------------------------------------

try:
    from justagent.security.rbac import (
        AccessDecision,
        Permission,
        RBACEngine,
        RBACError,
        ResourceType,
        Role,
        User,
    )

    _RBAC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _RBAC_AVAILABLE = False

try:
    from justagent.security.encryption import (
        EncryptedPayload,
        EncryptionAlgorithm,
        EncryptionEngine,
        EncryptionError,
        EncryptionKey,
        KeyManager,
    )

    _ENCRYPTION_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _ENCRYPTION_AVAILABLE = False

try:
    from justagent.security.data_protection import (
        DataSanitizer,
        DataSensitivityLevel,
        DLPScanner,
        PIIFinding,
    )

    _DLP_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _DLP_AVAILABLE = False

try:
    from justagent.security.compliance import (
        AuditResult,
        AuditTrail,
        AuditTrailManager,
        ComplianceChecker,
        ComplianceFramework,
        ComplianceRule,
        PolicyDecision,
        Severity,
    )

    _COMPLIANCE_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _COMPLIANCE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Typer sub-apps
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="security",
    help="企业安全子系统：RBAC 权限、加密、DLP 数据防泄漏与合规审计。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

rbac_app = typer.Typer(
    name="rbac",
    help="基于角色的访问控制（角色 / 用户 / 权限管理）。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

encrypt_app = typer.Typer(
    name="encrypt",
    help="文件加密（AEAD 认证加密）。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

decrypt_app = typer.Typer(
    name="decrypt",
    help="文件解密。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

dlp_app = typer.Typer(
    name="dlp",
    help="数据防泄漏（PII 扫描与脱敏）。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

compliance_app = typer.Typer(
    name="compliance",
    help="合规检查与审计追踪。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def register(parent: typer.Typer) -> None:
    """Register the security command group and its sub-groups."""

    app.add_typer(rbac_app, name="rbac")
    app.add_typer(encrypt_app, name="encrypt")
    app.add_typer(decrypt_app, name="decrypt")
    app.add_typer(dlp_app, name="dlp")
    app.add_typer(compliance_app, name="compliance")
    parent.add_typer(app, name="security")


# ---------------------------------------------------------------------------
# Context accessors — defensive (build fallbacks when ctx.obj is missing
# keys, e.g. when a command is invoked directly in tests).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Persistence layer
# ---------------------------------------------------------------------------


def _state_path(ctx: typer.Context) -> Path:
    """Resolve the security state file path.

    Priority: JUSTAGENT_SECURITY_STATE (or legacy
    MYAGENT_SECURITY_STATE) env var >
    <project_root>/.justagent/security_state.json >
    ./.justagent/security_state.json.
    """

    env = os.environ.get("JUSTAGENT_SECURITY_STATE") or os.environ.get("MYAGENT_SECURITY_STATE")
    if env:
        return Path(env).expanduser()
    config = common.get_config(ctx)
    root = Path(getattr(config, "project_root", ".") or ".")
    return root / ".justagent" / "security_state.json"


class _SecurityState:
    """Container holding the security engines with JSON save/restore.

    The underlying engines (:class:RBACEngine, :class:KeyManager,
    :class:ComplianceChecker, :class:AuditTrailManager) only offer
    incremental mutation APIs.  To round-trip persisted state we restore
    the internal registries directly — this is the integration layer's
    concern and is clearly isolated here rather than spread across
    commands.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rbac_engine: RBACEngine | None = RBACEngine() if _RBAC_AVAILABLE else None
        self.key_manager: KeyManager | None = KeyManager() if _ENCRYPTION_AVAILABLE else None
        self.compliance_checker: ComplianceChecker | None = (
            ComplianceChecker() if _COMPLIANCE_AVAILABLE else None
        )
        self.audit_manager: AuditTrailManager | None = (
            AuditTrailManager() if _COMPLIANCE_AVAILABLE else None
        )

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> _SecurityState:
        """Load state from *path*, or return an empty state on any error."""

        state = cls(path)
        if not path.exists():
            return state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            get_console().print(
                f"[yellow]⚠ 无法读取安全状态文件 {path}：{exc}（将以空状态启动）[/yellow]"
            )
            return state
        state._restore(data)
        return state

    def save(self) -> None:
        """Persist the current registries to the state file."""

        data = self._snapshot()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            get_console().print(
                f"[red]✗ 无法保存安全状态文件 {self.path}：{exc}[/red]", style="red"
            )

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        """Serialise all registries to a JSON-friendly dict."""

        result: dict[str, Any] = {
            "schema_version": 1,
            "saved_at": time.time(),
        }
        if self.rbac_engine is not None:
            engine = self.rbac_engine
            result["roles"] = [r.model_dump(mode="json") for r in engine._roles.values()]
            result["users"] = [u.model_dump(mode="json") for u in engine._users.values()]
        if self.key_manager is not None:
            result["keys"] = [k.model_dump(mode="json") for k in self.key_manager._keys.values()]
        if self.compliance_checker is not None:
            result["compliance_rules"] = [
                r.model_dump(mode="json") for r in self.compliance_checker._rules.values()
            ]
        if self.audit_manager is not None:
            result["audit_entries"] = [
                e.model_dump(mode="json") for e in self.audit_manager._entries
            ]
        return result

    def _restore(self, data: dict[str, Any]) -> None:
        """Rebuild the engines' internal registries from *data*."""

        # --- RBAC ---
        if self.rbac_engine is not None:
            engine = self.rbac_engine
            engine._roles.clear()
            engine._users.clear()
            engine._username_index.clear()
            for raw in data.get("roles", []):
                role = Role.model_validate(raw)
                engine._roles[role.id] = role
            for raw in data.get("users", []):
                user = User.model_validate(raw)
                engine._users[user.id] = user
                engine._username_index[user.username] = user.id

        # --- Key manager ---
        if self.key_manager is not None:
            self.key_manager._keys.clear()
            for raw in data.get("keys", []):
                key_obj = EncryptionKey.model_validate(raw)
                self.key_manager._keys[key_obj.key_id] = key_obj

        # --- Compliance checker ---
        if self.compliance_checker is not None:
            self.compliance_checker._rules.clear()
            for raw in data.get("compliance_rules", []):
                rule = ComplianceRule.model_validate(raw)
                self.compliance_checker._rules[rule.id] = rule

        # --- Audit trail ---
        if self.audit_manager is not None:
            self.audit_manager._entries.clear()
            for raw in data.get("audit_entries", []):
                entry = AuditTrail.model_validate(raw)
                self.audit_manager._entries.append(entry)


@contextmanager
def _state_session(ctx: typer.Context, *, save: bool = True) -> Iterator[_SecurityState]:
    """Load state, yield it, and persist on exit (unless dry-run or read-only).

    Args:
        ctx: The Typer context (for path resolution and dry-run flag).
        save: When True (default) the state is written back on a clean
            exit.  Read-only commands pass False to avoid needless writes.
    """

    state = _SecurityState.load(_state_path(ctx))
    try:
        yield state
    finally:
        if save and not common.get_dry_run(ctx):
            state.save()


# ---------------------------------------------------------------------------
# Output / formatting helpers
# ---------------------------------------------------------------------------


def _id_short(value: str, width: int = 8) -> str:
    """Show the leading characters of an ID for compact table display."""

    return value[:width] if value else "-"


def _parse_enum(value: str, enum_cls: type[Any], label: str) -> Any:
    """Parse *value* into an enum member, raising BadParameter on failure."""

    try:
        return enum_cls(value)
    except ValueError:
        valid = ", ".join(str(m.value) for m in enum_cls)
        raise typer.BadParameter(f"无效的 {label}：{value!r}（可选值：{valid}）") from None


def _require_rbac() -> None:
    """Raise a Typer error if the RBAC subsystem is unavailable."""

    if not _RBAC_AVAILABLE:
        raise typer.BadParameter(
            "RBAC 子系统不可用（未能导入 justagent.security.rbac）。请确认 pydantic 已安装。"
        )


def _require_encryption() -> None:
    """Raise a Typer error if the encryption subsystem is unavailable."""

    if not _ENCRYPTION_AVAILABLE:
        raise typer.BadParameter(
            "加密子系统不可用（未能导入 justagent.security.encryption）。"
            "请安装 cryptography：pip install cryptography"
        )


def _require_dlp() -> None:
    """Raise a Typer error if the DLP subsystem is unavailable."""

    if not _DLP_AVAILABLE:
        raise typer.BadParameter(
            "DLP 子系统不可用（未能导入 justagent.security.data_protection）。"
            "请确认 pydantic 已安装。"
        )


def _require_compliance() -> None:
    """Raise a Typer error if the compliance subsystem is unavailable."""

    if not _COMPLIANCE_AVAILABLE:
        raise typer.BadParameter(
            "合规子系统不可用（未能导入 justagent.security.compliance）。请确认 pydantic 已安装。"
        )


def _record_security_audit(
    state: _SecurityState,
    actor: str,
    action: str,
    resource: str = "",
    result: AuditResult = AuditResult.SUCCESS,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record a security audit event to the AuditTrailManager (best-effort)."""

    if state.audit_manager is None:
        return
    with suppress(Exception):
        event = AuditTrail(
            actor=actor,
            action=action,
            resource=resource,
            result=result,
            metadata=metadata or {},
        )
        state.audit_manager.record(event)


def _resolve_role(engine: RBACEngine, role_id_or_name: str) -> Role:
    """Return a role by ID or name, raising BadParameter if not found."""

    role = engine.get_role(role_id_or_name)
    if role is not None:
        return role
    # Fall back to a name match.
    matches = [r for r in engine.list_roles() if r.name == role_id_or_name]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise typer.BadParameter(
            f"角色名称 '{role_id_or_name}' 不唯一（{len(matches)} 个匹配），请使用角色 ID。"
        )
    raise typer.BadParameter(f"未找到角色：{role_id_or_name}")


def _parse_permissions(specs: list[str]) -> dict[Any, set[Any]]:
    """Parse permission specs like document:read,write into a dict.

    Each spec must have the form <resource_type>:<perm1>,<perm2>,....
    Multiple specs for the same resource type are merged.
    """

    result: dict[Any, set[Any]] = {}
    for spec in specs:
        if ":" not in spec:
            raise typer.BadParameter(
                f"权限格式应为 resource:perm1,perm2，例如 document:read,write；得到：{spec!r}"
            )
        rt_str, perms_str = spec.split(":", 1)
        rt = _parse_enum(rt_str.strip(), ResourceType, "资源类型")
        perms: set[Any] = set()
        for p_str in perms_str.split(","):
            p_str = p_str.strip()
            if p_str:
                perms.add(_parse_enum(p_str, Permission, "权限"))
        if not perms:
            raise typer.BadParameter(f"权限规格 {spec!r} 未指定任何权限")
        if rt in result:
            result[rt] |= perms
        else:
            result[rt] = perms
    return result


def _format_permissions(permissions: dict[Any, set[Any]]) -> str:
    """Render a permissions dict as a compact rt:p1,p2; … string."""

    parts: list[str] = []
    for rt, perms in sorted(permissions.items(), key=lambda kv: kv[0].value):
        perm_str = ",".join(sorted(p.value for p in perms))
        parts.append(f"{rt.value}:{perm_str}")
    return "; ".join(parts) if parts else "(none)"


# ---------------------------------------------------------------------------
# RBAC commands
# ---------------------------------------------------------------------------


@rbac_app.command("add-role", help="添加一个自定义角色。")
def rbac_add_role(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", "-n", help="角色名称，例如 auditor"),
    description: str = typer.Option("", "--description", "-d", help="角色描述"),
    permission: list[str] = typer.Option(
        ...,
        "--permission",
        "-p",
        help=(
            "权限规格，格式 resource:perm1,perm2。可重复指定。"
            "例如 --permission document:read,write --permission evidence:review"
        ),
    ),
    inherits_from: list[str] = typer.Option(
        [],
        "--inherits-from",
        help="继承的角色 ID（可重复指定）",
    ),
) -> None:
    """添加一个自定义 RBAC 角色并为其分配权限。

    权限以 resource:perm1,perm2 格式指定，可多次使用 --permission
    为不同资源类型设置权限。角色可通过 --inherits-from 继承已有角色
    的权限（传递式，自动检测循环）。
    """

    if not _RBAC_AVAILABLE:
        _require_rbac()
        return

    verbose = common.get_verbose(ctx)
    dry_run = common.get_dry_run(ctx)

    permissions = _parse_permissions(permission)

    if dry_run:
        get_console().print(
            Panel(
                f"[dry-run] 将创建角色\n名称: {name}\n描述: {description or '(无)'}\n"
                f"权限: {_format_permissions(permissions)}\n"
                f"继承: {', '.join(inherits_from) or '(无)'}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    with _state_session(ctx) as state:
        engine = state.rbac_engine
        assert engine is not None  # _RBAC_AVAILABLE guaranteed above
        role = Role(
            name=name,
            description=description,
            permissions=permissions,
            inherits_from=list(inherits_from),
            system=False,
        )
        try:
            engine.create_role(role)
        except RBACError as exc:
            get_console().print(f"[red]✗ {exc}[/red]")
            raise typer.Exit(code=1) from exc

        _record_security_audit(
            state,
            actor="cli",
            action="security.rbac.add_role",
            resource=role.id,
            metadata={"role_name": name, "permissions": _format_permissions(permissions)},
        )

    common.audit(
        ctx,
        "security.rbac.add_role",
        {"role_id": role.id, "role_name": name},
    )

    console = get_console()
    if verbose:
        console.print(
            Panel(
                f"角色 ID:    {role.id}\n"
                f"名称:       {role.name}\n"
                f"描述:       {role.description or '-'}\n"
                f"权限:       {_format_permissions(role.permissions)}\n"
                f"继承自:     {', '.join(role.inherits_from) or '(无)'}\n"
                f"系统角色:   {'是' if role.system else '否'}\n"
                f"创建时间:   {common.format_ts(role.created_at)}",
                title=f"已创建角色 {role.name}",
                border_style="green",
            )
        )
    else:
        console.print(f"[green]✓[/green] 已创建角色 [bold]{role.name}[/bold]（ID: {role.id}）")


@rbac_app.command("list-roles", help="列出所有角色。")
def rbac_list_roles(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """列出所有已注册的 RBAC 角色（含系统内置角色）。"""

    if not _RBAC_AVAILABLE:
        _require_rbac()
        return

    with _state_session(ctx, save=False) as state:
        engine = state.rbac_engine
        assert engine is not None
        roles = engine.list_roles()

    if json_output:
        rows = [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "system": r.system,
                "permissions": {
                    rt.value: sorted(p.value for p in perms) for rt, perms in r.permissions.items()
                },
                "inherits_from": r.inherits_from,
                "created_at": r.created_at,
            }
            for r in roles
        ]
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    console = get_console()
    if not roles:
        console.print("[dim]暂无角色。使用 security rbac add-role 创建一个。[/dim]")
        return

    table = Table(title=f"角色列表（共 {len(roles)} 个）", border_style="cyan")
    table.add_column("ID", style="dim", width=12)
    table.add_column("名称", style="bold")
    table.add_column("描述")
    table.add_column("权限概览")
    table.add_column("继承", style="dim")
    table.add_column("系统", justify="center")

    for r in roles:
        table.add_row(
            _id_short(r.id, 12),
            r.name,
            common.short(r.description, 30),
            common.short(_format_permissions(r.permissions), 40),
            ", ".join(_id_short(rid, 8) for rid in r.inherits_from) or "-",
            Text("✓" if r.system else "—", style="dim" if r.system else "white"),
        )
    console.print(table)


@rbac_app.command("assign", help="为用户分配角色。")
def rbac_assign(
    ctx: typer.Context,
    user: str = typer.Option(..., "--user", "-u", help="用户名或用户 ID"),
    role: str = typer.Option(..., "--role", "-r", help="角色 ID 或角色名称"),
    display_name: str = typer.Option(
        "", "--display-name", help="用户显示名（仅在用户不存在时自动注册时使用）"
    ),
    email: str = typer.Option("", "--email", help="用户邮箱（仅在用户不存在时自动注册时使用）"),
    department: str = typer.Option(
        "", "--department", help="用户部门（仅在用户不存在时自动注册时使用）"
    ),
) -> None:
    """为用户分配一个角色（幂等）。

    如果指定的用户尚未注册，将自动以其用户名创建一个新用户。
    角色可通过 ID 或名称指定。分配操作是幂等的 — 重复分配同一角色
    不会产生错误。
    """

    if not _RBAC_AVAILABLE:
        _require_rbac()
        return

    verbose = common.get_verbose(ctx)
    dry_run = common.get_dry_run(ctx)

    if dry_run:
        get_console().print(
            Panel(
                f"[dry-run] 将为用户 '{user}' 分配角色 '{role}'",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    with _state_session(ctx) as state:
        engine = state.rbac_engine
        assert engine is not None
        resolved_role = _resolve_role(engine, role)

        # Resolve or auto-register the user.
        user_obj = engine.get_user(user)
        user_created = False
        if user_obj is None:
            user_obj = User(
                username=user,
                display_name=display_name or user,
                email=email,
                department=department,
            )
            try:
                engine.register_user(user_obj)
                user_created = True
            except RBACError as exc:
                get_console().print(f"[red]✗ 注册用户失败：{exc}[/red]")
                raise typer.Exit(code=1) from exc

        try:
            engine.assign_role(user_obj.id, resolved_role.id)
        except RBACError as exc:
            get_console().print(f"[red]✗ 分配角色失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc

        _record_security_audit(
            state,
            actor="cli",
            action="security.rbac.assign",
            resource=resolved_role.id,
            metadata={
                "user": user_obj.username,
                "user_id": user_obj.id,
                "role_name": resolved_role.name,
                "user_created": user_created,
            },
        )

    common.audit(
        ctx,
        "security.rbac.assign",
        {"user": user, "role": resolved_role.name, "role_id": resolved_role.id},
    )

    console = get_console()
    created_note = "（新用户已自动注册）" if user_created else ""
    if verbose:
        console.print(
            Panel(
                f"用户:       {user_obj.username}（ID: {user_obj.id}）{created_note}\n"
                f"显示名:     {user_obj.display_name or '-'}\n"
                f"邮箱:       {user_obj.email or '-'}\n"
                f"部门:       {user_obj.department or '-'}\n"
                f"已分配角色: {resolved_role.name}（ID: {resolved_role.id}）\n"
                f"当前角色数: {len(user_obj.roles)}",
                title=f"角色分配结果 — {user_obj.username}",
                border_style="green",
            )
        )
    else:
        console.print(
            f"[green]✓[/green] 已为用户 [bold]{user_obj.username}[/bold]"
            f"分配角色 [bold]{resolved_role.name}[/bold]{created_note}"
        )


@rbac_app.command("check", help="检查用户是否拥有指定权限。")
def rbac_check(
    ctx: typer.Context,
    user: str = typer.Option(..., "--user", "-u", help="用户名或用户 ID"),
    resource: str = typer.Option(
        ...,
        "--resource",
        "-r",
        help=(
            "资源类型（document/channel/meeting/database/storage/agent/"
            "workflow/config/user/report/case_file/evidence/legal_document/"
            "court_record/judggment）"
        ),
    ),
    permission: str = typer.Option(
        ...,
        "--permission",
        "-p",
        help=(
            "权限（read/write/delete/admin/execute/delegate/share/export/"
            "review/seal/archive/serve）"
        ),
    ),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """检查用户是否拥有对指定资源类型的指定权限。

    返回一个结构化的访问决策，包括是否允许、匹配的角色和原因说明。
    非活跃或已暂停的用户始终被拒绝。
    """

    if not _RBAC_AVAILABLE:
        _require_rbac()
        return

    rt = _parse_enum(resource, ResourceType, "资源类型")
    perm = _parse_enum(permission, Permission, "权限")

    with _state_session(ctx, save=False) as state:
        engine = state.rbac_engine
        assert engine is not None
        decision: AccessDecision = engine.evaluate_access(user, rt, perm)

    common.audit(
        ctx,
        "security.rbac.check",
        {
            "user": user,
            "resource": rt.value,
            "permission": perm.value,
            "allowed": decision.allowed,
        },
    )

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "user": user,
                    "resource_type": rt.value,
                    "permission": perm.value,
                    "allowed": decision.allowed,
                    "reason": decision.reason,
                    "matched_role": decision.matched_role,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    console = get_console()
    if decision.allowed:
        console.print(
            Panel(
                f"[green]✓ 允许[/green]\n\n"
                f"[bold]用户:[/bold]       {user}\n"
                f"[bold]资源类型:[/bold]   {rt.value}\n"
                f"[bold]权限:[/bold]       {perm.value}\n"
                f"[bold]原因:[/bold]       {decision.reason}\n"
                f"[bold]匹配角色:[/bold]   {decision.matched_role or '-'}",
                title="访问决策",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[red]✗ 拒绝[/red]\n\n"
                f"[bold]用户:[/bold]       {user}\n"
                f"[bold]资源类型:[/bold]   {rt.value}\n"
                f"[bold]权限:[/bold]       {perm.value}\n"
                f"[bold]原因:[/bold]       {decision.reason}",
                title="访问决策",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Encryption commands
# ---------------------------------------------------------------------------


@encrypt_app.command("file", help="加密一个文件。")
def encrypt_file(
    ctx: typer.Context,
    file_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="要加密的文件路径",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="加密载荷输出路径（JSON，默认在原文件名后加 .enc）",
    ),
    algorithm: str = typer.Option(
        "aes_256_gcm",
        "--algorithm",
        "-a",
        help="加密算法（aes_256_gcm / chacha20_poly1305 / fernet）",
    ),
    password: str | None = typer.Option(
        None,
        "--password",
        help="从口令派生密钥（PBKDF2）；不指定则使用或生成随机密钥",
    ),
    associated_data: str = typer.Option(
        "",
        "--associated-data",
        help="AEAD 关联数据（被认证但不被加密，仅 AES-GCM / ChaCha20 支持）",
    ),
) -> None:
    """使用 AEAD 认证加密对文件进行加密。

    加密结果为一个自描述的 JSON 载荷（包含密钥 ID、算法、nonce、
    密文和认证标签），可使用 security decrypt file 还原。

    密钥管理：
    - 指定 --password 时，使用 PBKDF2 从口令派生密钥并持久化。
    - 不指定时，自动使用已有活动密钥或生成新的随机密钥。

    .. warning::

        密钥存储在安全状态文件中。如果该文件丢失，加密数据将无法解密。
    """

    if not _ENCRYPTION_AVAILABLE:
        _require_encryption()
        return

    verbose = common.get_verbose(ctx)
    dry_run = common.get_dry_run(ctx)
    algo = _parse_enum(algorithm, EncryptionAlgorithm, "加密算法")

    if dry_run:
        get_console().print(
            Panel(
                f"[dry-run] 将加密文件 {file_path}\n"
                f"算法: {algo.value}\n"
                f"口令派生: {'是' if password else '否（随机密钥）'}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    try:
        plaintext = file_path.read_bytes()
    except OSError as exc:
        get_console().print(f"[red]✗ 读取文件失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc

    with _state_session(ctx) as state:
        km = state.key_manager
        assert km is not None
        engine = EncryptionEngine(km)

        aad = associated_data.encode("utf-8") if associated_data else b""

        # Resolve or create the key.
        if password is not None:
            key = km.create_key(algo, password=password)
        else:
            active: EncryptionKey | None = km.get_active_key(algo)
            key = km.create_key(algo) if active is None else active

        try:
            payload = engine.encrypt(
                plaintext,
                algorithm=algo,
                associated_data=aad,
                key_id=key.key_id,
            )
        except EncryptionError as exc:
            get_console().print(f"[red]✗ 加密失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc

        _record_security_audit(
            state,
            actor="cli",
            action="security.encrypt.file",
            resource=str(file_path),
            metadata={
                "key_id": payload.key_id,
                "algorithm": payload.algorithm.value,
                "plaintext_bytes": len(plaintext),
                "ciphertext_bytes": len(payload.ciphertext),
            },
        )

    common.audit(
        ctx,
        "security.encrypt.file",
        {"file": str(file_path), "algorithm": algo.value, "key_id": payload.key_id},
    )

    # Write the encrypted payload as JSON.
    out_path = output or file_path.with_suffix(file_path.suffix + ".enc")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        get_console().print(f"[red]✗ 写入输出文件失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console = get_console()
    if verbose:
        console.print(
            Panel(
                f"源文件:       {file_path}\n"
                f"输出文件:     {out_path}\n"
                f"算法:         {payload.algorithm.value}\n"
                f"密钥 ID:      {payload.key_id}\n"
                f"明文大小:     {len(plaintext)} 字节\n"
                f"密文大小:     {len(payload.ciphertext)} 字节\n"
                f"关联数据:     {associated_data or '(无)'}\n"
                f"加密时间:     {common.format_ts(payload.created_at)}",
                title=f"已加密 — {file_path.name}",
                border_style="green",
            )
        )
    else:
        console.print(
            f"[green]✓[/green] 已加密 [bold]{file_path.name}[/bold]"
            f"（{payload.algorithm.value}，密钥: {_id_short(payload.key_id)}）"
            f" -> {out_path}"
        )


@decrypt_app.command("file", help="解密一个文件。")
def decrypt_file(
    ctx: typer.Context,
    file_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="加密载荷文件路径（JSON）",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="解密明文输出路径（默认在原文件名后加 .dec）",
    ),
) -> None:
    """解密一个由 security encrypt file 生成的加密载荷。

    从 JSON 载荷中读取密钥 ID、算法、nonce 和密文，通过安全状态中
    持久化的密钥进行解密。

    .. note::

        解密所用的密钥必须存在于安全状态文件中（即加密时使用的
        同一状态文件）。如果密钥已撤销但仍存在于状态中，解密仍可进行。
    """

    if not _ENCRYPTION_AVAILABLE:
        _require_encryption()
        return

    verbose = common.get_verbose(ctx)
    dry_run = common.get_dry_run(ctx)

    if dry_run:
        get_console().print(
            Panel(
                f"[dry-run] 将解密文件 {file_path}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    try:
        raw = file_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        get_console().print(f"[red]✗ 读取或解析加密载荷失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc

    try:
        payload = EncryptedPayload.from_dict(data)
    except (KeyError, ValueError) as exc:
        get_console().print(f"[red]✗ 加密载荷格式无效：{exc}[/red]")
        raise typer.Exit(code=1) from exc

    with _state_session(ctx) as state:
        km = state.key_manager
        assert km is not None
        engine = EncryptionEngine(km)

        try:
            plaintext = engine.decrypt(payload)
        except EncryptionError as exc:
            get_console().print(f"[red]✗ 解密失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc

        _record_security_audit(
            state,
            actor="cli",
            action="security.decrypt.file",
            resource=str(file_path),
            metadata={
                "key_id": payload.key_id,
                "algorithm": payload.algorithm.value,
                "plaintext_bytes": len(plaintext),
            },
        )

    common.audit(
        ctx,
        "security.decrypt.file",
        {"file": str(file_path), "key_id": payload.key_id},
    )

    out_path = output or file_path.with_suffix(file_path.suffix + ".dec")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(plaintext)
    except OSError as exc:
        get_console().print(f"[red]✗ 写入输出文件失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console = get_console()
    if verbose:
        console.print(
            Panel(
                f"源文件:       {file_path}\n"
                f"输出文件:     {out_path}\n"
                f"算法:         {payload.algorithm.value}\n"
                f"密钥 ID:      {payload.key_id}\n"
                f"密文大小:     {len(payload.ciphertext)} 字节\n"
                f"明文大小:     {len(plaintext)} 字节",
                title=f"已解密 — {file_path.name}",
                border_style="green",
            )
        )
    else:
        console.print(
            f"[green]✓[/green] 已解密 [bold]{file_path.name}[/bold]"
            f"（{payload.algorithm.value}，密钥: {_id_short(payload.key_id)}）"
            f" -> {out_path}"
        )


# ---------------------------------------------------------------------------
# DLP commands
# ---------------------------------------------------------------------------


@dlp_app.command("scan", help="扫描文本或文件中的 PII（个人敏感信息）。")
def dlp_scan(
    ctx: typer.Context,
    text: str | None = typer.Argument(None, help="要扫描的文本（与 --file 二选一）"),
    file: Path | None = typer.Option(
        None,
        "--file",
        "-f",
        exists=True,
        dir_okay=False,
        readable=True,
        help="从文件读取要扫描的内容",
    ),
    redact: bool = typer.Option(False, "--redact", help="输出完全脱敏后的内容（替换为占位符）"),
    mask: bool = typer.Option(False, "--mask", help="输出部分遮罩后的内容（保留类型上下文）"),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """扫描文本或文件中的个人敏感信息（PII）。

    内置检测以下 PII 类型：邮箱、电话、美国 SSN、信用卡号、IP 地址、
    护照号、银行账号、身份证号、地址，以及中国特有的居民身份证号、
    统一社会信用代码、官方案件编号、营业执照号等。

    使用 --redact 查看脱敏后的内容，或 --mask 查看部分遮罩的内容。
    """

    if not _DLP_AVAILABLE:
        _require_dlp()
        return

    if text is None and file is None:
        raise typer.BadParameter("必须提供文本参数或 --file 选项之一")

    if file is not None:
        try:
            content = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            get_console().print(f"[red]✗ 读取文件失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc
    else:
        content = text or ""

    scanner = DLPScanner()
    findings: list[PIIFinding] = scanner.scan_text(content)

    common.audit(
        ctx,
        "security.dlp.scan",
        {
            "source": str(file) if file else "(text)",
            "total_findings": len(findings),
        },
    )

    if json_output:
        rows = [
            {
                "pii_type": f.pii_type.value,
                "value": f.value,
                "start": f.start,
                "end": f.end,
                "sensitivity": f.sensitivity.value,
                "rule_id": f.rule_id,
            }
            for f in findings
        ]
        typer.echo(
            json.dumps(
                {
                    "total_findings": len(findings),
                    "findings": rows,
                    "summary": scanner.summary(content),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    console = get_console()

    # Sanitised output modes.
    if redact or mask:
        sanitizer = DataSanitizer(scanner)
        result = sanitizer.redact_pii(content) if redact else sanitizer.mask_partial(content)
        console.print(Panel(result, title="脱敏结果", border_style="yellow"))
        if findings:
            console.print(f"[dim]检测到 {len(findings)} 处 PII，已脱敏。[/dim]")
        else:
            console.print("[dim]未检测到 PII。[/dim]")
        return

    # Default: findings table.
    if not findings:
        console.print("[dim]未检测到 PII。内容是安全的。[/dim]")
        return

    table = Table(title=f"PII 扫描结果（共 {len(findings)} 处）", border_style="cyan")
    table.add_column("#", justify="right", style="dim")
    table.add_column("类型", style="bold")
    table.add_column("匹配值")
    table.add_column("敏感度", style="bold")
    table.add_column("位置", justify="right")

    for idx, f in enumerate(findings, 1):
        sens_style = {
            DataSensitivityLevel.CRITICAL: "red",
            DataSensitivityLevel.HIGH: "red",
            DataSensitivityLevel.MEDIUM: "yellow",
            DataSensitivityLevel.LOW: "green",
        }.get(f.sensitivity, "white")
        table.add_row(
            str(idx),
            f.pii_type.value,
            common.short(f.value, 40),
            Text(f.sensitivity.value, style=sens_style),
            f"{f.start}-{f.end}",
        )
    console.print(table)

    # Summary.
    summary = scanner.summary(content)
    console.print(
        f"[dim]按类型: {summary['by_type']}  |  按敏感度: {summary['by_sensitivity']}[/dim]"
    )
    if summary.get("has_critical"):
        console.print("[red]⚠ 检测到 CRITICAL 级别的敏感信息！[/red]")


# ---------------------------------------------------------------------------
# Compliance commands
# ---------------------------------------------------------------------------


@compliance_app.command("check", help="检查数据访问操作的合规性。")
def compliance_check(
    ctx: typer.Context,
    user: str = typer.Option(..., "--user", "-u", help="执行操作的用户"),
    data_type: str = typer.Option(
        ...,
        "--data-type",
        "-t",
        help="数据类型（pii/phi/credit_card/pci/confidential...）",
    ),
    action: str = typer.Option(
        ...,
        "--action",
        "-a",
        help="操作类型（read/write/export/share/delete...）",
    ),
    framework: str | None = typer.Option(
        None, "--framework", help="按合规框架过滤（gdpr/hipaa/pci_dss/soc2/...）"
    ),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """检查指定的数据访问操作是否符合合规规则。

    评估所有已启用的合规规则，返回一个策略决策（允许 / 拒绝）及
    违反的规则列表和建议。HIGH 和 CRITICAL 级别的违规将阻止操作。
    """

    if not _COMPLIANCE_AVAILABLE:
        _require_compliance()
        return

    fw_filter = (
        _parse_enum(framework, ComplianceFramework, "合规框架") if framework is not None else None
    )

    with _state_session(ctx, save=False) as state:
        checker = state.compliance_checker
        assert checker is not None
        decision: PolicyDecision = checker.check_data_access(user, data_type, action)

        # If a framework filter is given, narrow violated rules to that framework.
        if fw_filter is not None and decision.violated_rules:
            decision.violated_rules = [
                r for r in decision.violated_rules if r.framework is fw_filter
            ]
            has_blocking = any(
                r.severity in (Severity.HIGH, Severity.CRITICAL) for r in decision.violated_rules
            )
            decision.allowed = not has_blocking

        # Record the compliance check in the audit trail.
        if state.audit_manager is not None:
            _record_security_audit(
                state,
                actor=user,
                action="security.compliance.check",
                resource=data_type,
                result=AuditResult.SUCCESS if decision.allowed else AuditResult.DENIED,
                metadata={
                    "data_type": data_type,
                    "operation": action,
                    "framework": fw_filter.value if fw_filter else None,
                    "allowed": decision.allowed,
                    "violated_count": len(decision.violated_rules),
                },
            )

    common.audit(
        ctx,
        "security.compliance.check",
        {
            "user": user,
            "data_type": data_type,
            "action": action,
            "allowed": decision.allowed,
        },
    )

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "user": user,
                    "data_type": data_type,
                    "action": action,
                    "allowed": decision.allowed,
                    "violated_rules": [
                        {
                            "id": r.id,
                            "framework": r.framework.value,
                            "requirement": r.requirement,
                            "description": r.description,
                            "severity": r.severity.value,
                        }
                        for r in decision.violated_rules
                    ],
                    "recommendations": decision.recommendations,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    console = get_console()
    if decision.allowed:
        header = (
            f"[green]✓ 允许[/green]\n\n"
            f"[bold]用户:[/bold]       {user}\n"
            f"[bold]数据类型:[/bold]   {data_type}\n"
            f"[bold]操作:[/bold]       {action}\n"
        )
        if decision.violated_rules:
            header += f"\n[bold]非阻断性违规（{len(decision.violated_rules)} 条）：[/bold]\n"
            for r in decision.violated_rules:
                header += f"  - [{r.severity.value}] {r.framework.value}: {r.requirement}\n"
        console.print(Panel(header, title="合规决策", border_style="green"))
    else:
        header = (
            f"[red]✗ 拒绝[/red]\n\n"
            f"[bold]用户:[/bold]       {user}\n"
            f"[bold]数据类型:[/bold]   {data_type}\n"
            f"[bold]操作:[/bold]       {action}\n"
        )
        if decision.violated_rules:
            header += f"\n[bold red]阻断性违规（{len(decision.violated_rules)} 条）：[/bold red]\n"
            for r in decision.violated_rules:
                header += (
                    f"  - [{r.severity.value}] {r.framework.value}: "
                    f"{r.requirement}\n    {common.short(r.description, 70)}\n"
                )
        if decision.recommendations:
            header += "\n[bold]建议：[/bold]\n"
            for rec in decision.recommendations:
                header += f"  - {common.short(rec, 70)}\n"
        console.print(Panel(header, title="合规决策", border_style="red"))


@compliance_app.command("audit", help="查看安全审计追踪。")
def compliance_audit(
    ctx: typer.Context,
    actor: str | None = typer.Option(None, "--actor", help="按操作者过滤"),
    action: str | None = typer.Option(None, "--action", help="按操作类型过滤"),
    resource: str | None = typer.Option(None, "--resource", help="按资源过滤"),
    result: str | None = typer.Option(
        None, "--result", help="按结果过滤（success/failure/denied/error）"
    ),
    limit: int = typer.Option(50, "--limit", help="返回条目数量上限"),
    verify: bool = typer.Option(False, "--verify", help="验证审计链完整性（SHA-256 哈希链）"),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """查看安全审计追踪记录。

    审计追踪是一个仅追加的、防篡改的哈希链（每条记录的 SHA-256 哈希
    依赖前一条记录的哈希）。使用 --verify 可验证整条链的完整性。

    支持按操作者、操作类型、资源和结果过滤。
    """

    if not _COMPLIANCE_AVAILABLE:
        _require_compliance()
        return

    result_filter = _parse_enum(result, AuditResult, "结果") if result is not None else None

    with _state_session(ctx, save=False) as state:
        manager = state.audit_manager
        assert manager is not None

        # Build filters.
        filters: dict[str, Any] = {}
        if actor is not None:
            filters["actor"] = actor
        if action is not None:
            filters["action"] = action
        if resource is not None:
            filters["resource"] = resource
        if result_filter is not None:
            filters["result"] = result_filter

        entries = manager.query(filters if filters else None)

        # Apply limit (most recent first).
        if limit > 0:
            entries = entries[-limit:]
        entries = list(reversed(entries))

        chain_valid = manager.verify_chain() if verify else None
        summary = manager.summary()

    common.audit(
        ctx,
        "security.compliance.audit",
        {"filters": filters, "returned": len(entries)},
    )

    if json_output:
        rows = [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp,
                "actor": e.actor,
                "action": e.action,
                "resource": e.resource,
                "result": e.result.value,
                "ip_address": e.ip_address,
                "entry_hash": e.entry_hash[:16] + "..." if e.entry_hash else "",
                "metadata": e.metadata,
            }
            for e in entries
        ]
        typer.echo(
            json.dumps(
                {
                    "total_entries": summary["total_entries"],
                    "returned": len(rows),
                    "by_result": summary["by_result"],
                    "by_action": summary["by_action"],
                    "chain_valid": chain_valid,
                    "entries": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    console = get_console()

    # Chain verification banner.
    if chain_valid is not None:
        if chain_valid:
            console.print("[green]✓ 审计链完整性验证通过[/green]")
        else:
            console.print("[red]✗ 审计链完整性验证失败！链可能已被篡改。[/red]")

    if not entries:
        console.print("[dim]暂无审计记录。[/dim]")
        return

    table = Table(
        title=f"安全审计追踪（显示 {len(entries)} 条，共 {summary['total_entries']} 条）",
        border_style="cyan",
    )
    table.add_column("时间", style="dim", width=16)
    table.add_column("操作者", style="bold")
    table.add_column("操作")
    table.add_column("资源")
    table.add_column("结果", style="bold")
    table.add_column("哈希", style="dim")

    for e in entries:
        result_style = {
            AuditResult.SUCCESS: "green",
            AuditResult.FAILURE: "yellow",
            AuditResult.DENIED: "red",
            AuditResult.ERROR: "red",
        }.get(e.result, "white")
        table.add_row(
            common.format_ts(e.timestamp),
            common.short(e.actor, 16),
            common.short(e.action, 28),
            common.short(e.resource, 24),
            Text(e.result.value, style=result_style),
            _id_short(e.entry_hash, 12),
        )
    console.print(table)

    # Summary line.
    by_result_str = ", ".join(f"{k}: {v}" for k, v in sorted(summary["by_result"].items()))
    console.print(f"[dim]结果分布: {by_result_str or '(无)'}[/dim]")


__all__ = [
    "app",
    "compliance_app",
    "decrypt_app",
    "dlp_app",
    "encrypt_app",
    "rbac_app",
    "register",
]
