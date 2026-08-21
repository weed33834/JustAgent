"""Lightweight multi-user accounts & roles for the JustAgent Web console.

Stores users in ``~/.justagent/web_users.json`` (created on demand), issues
session tokens persisted to ``~/.justagent/web_sessions.json``, and supports
roles: ``admin`` / ``editor`` / ``viewer``.

- ``viewer``: read-only
- ``editor``: read + write (create/edit cases, laws, docs, tasks)
- ``admin``: everything (incl. user management)

A default ``admin`` account is created on first use with the password from
``JUSTAGENT_WEB_ADMIN_PASSWORD`` (or a generated one printed once).

Passwords are hashed with PBKDF2-HMAC-SHA256 (600k iterations). Legacy
single-HMAC hashes are verified transparently and re-hashed on the next
successful login.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from justagent.utils.atomic_write import atomic_write_text

ROLES = ("admin", "editor", "viewer")

WRITE_ROLES = {"admin", "editor"}
ADMIN_ROLES = {"admin"}

# OWASP 2023 recommendation for PBKDF2-HMAC-SHA256.
PBKDF2_ITERATIONS = 600_000
_PBKDF2_PREFIX = "pbkdf2_sha256"
_PBKDF2_FORMAT = "{prefix}${iter}${salt}${digest}"


def _users_path() -> Path:
    env = os.environ.get("JUSTAGENT_WEB_USERS_FILE", "")
    return Path(env) if env else Path.home() / ".justagent" / "web_users.json"


def _sessions_path() -> Path:
    env = os.environ.get("JUSTAGENT_WEB_SESSIONS_FILE", "")
    return Path(env) if env else Path.home() / ".justagent" / "web_sessions.json"


@dataclass
class User:
    username: str
    role: str = "viewer"
    password_hash: str = ""
    created_at: float = 0.0


@dataclass
class UserStore:
    path: Path = field(default_factory=_users_path)

    def _load(self) -> dict:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                return dict(data) if isinstance(data, dict) else {}
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _hash(password: str) -> str:
        """Hash with PBKDF2-HMAC-SHA256; format ``pbkdf2_sha256$<iter>$<salt>$<digest>``."""
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
        return _PBKDF2_FORMAT.format(
            prefix=_PBKDF2_PREFIX,
            iter=PBKDF2_ITERATIONS,
            salt=base64.b64encode(salt).decode("ascii"),
            digest=base64.b64encode(digest).decode("ascii"),
        )

    @staticmethod
    def _legacy_hash(password: str, salt: str) -> str:
        """Pre-3.1 scheme (single HMAC-SHA256); kept only to verify and migrate old rows."""
        digest = hmac.new(salt.encode(), password.encode(), "sha256").hexdigest()
        return f"{salt}${digest}"

    @staticmethod
    def _is_legacy(stored: str) -> bool:
        return not stored.startswith(_PBKDF2_PREFIX + "$")

    def _check(self, password: str, stored: str) -> bool:
        try:
            if self._is_legacy(stored):
                return hmac.compare_digest(self._legacy_hash(password, stored.split("$", 1)[0]), stored)
            _, iter_str, salt_b64, digest_b64 = stored.split("$", 3)
            digest = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), base64.b64decode(salt_b64), int(iter_str)
            )
            return hmac.compare_digest(digest, base64.b64decode(digest_b64))
        except (ValueError, TypeError):
            return False

    def authenticate(self, username: str, password: str) -> User | None:
        data = self._load()
        row = data.get(username)
        stored = row.get("password_hash", "") if row else ""
        if not row or not self._check(password, stored):
            return None
        if self._is_legacy(stored):
            # Transparent upgrade: legacy hash verified → re-hash with PBKDF2.
            row["password_hash"] = self._hash(password)
            self._save(data)
        return User(username=username, role=row.get("role", "viewer"))

    def ensure_admin(self) -> None:
        data = self._load()
        if "admin" in data:
            return
        password = os.environ.get("JUSTAGENT_WEB_ADMIN_PASSWORD", "")
        if not password:
            password = secrets.token_urlsafe(12)
        data["admin"] = {
            "username": "admin",
            "role": "admin",
            "password_hash": self._hash(password),
            "created_at": time.time(),
        }
        self._save(data)

    def list_users(self) -> list[dict]:
        return [
            {"username": u, "role": row.get("role", "viewer")}
            for u, row in self._load().items()
        ]

    def set_role(self, username: str, role: str, actor_role: str) -> bool:
        if actor_role not in ADMIN_ROLES:
            return False
        if role not in ROLES:
            return False
        data = self._load()
        if username not in data:
            return False
        data[username]["role"] = role
        self._save(data)
        return True


class TokenManager:
    """Session tokens (token -> {username, role, expires}), persisted to disk.

    Persistence lets sessions survive process restarts. If the session file
    cannot be written, the manager degrades to memory-only mode.
    """

    def __init__(self, ttl: float = 12 * 3600, path: Path | None = None) -> None:
        self._ttl = ttl
        self._path = path or _sessions_path()
        self._tokens = self._load()

    def _load(self) -> dict[str, dict]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        now = time.time()
        return {t: info for t, info in data.items() if info.get("expires", 0) > now}

    def _save(self) -> None:
        # Degrade to memory-only mode on write failure; sessions won't survive restarts.
        with contextlib.suppress(OSError):
            atomic_write_text(self._path, json.dumps(self._tokens, ensure_ascii=False))

    def issue(self, user: User) -> str:
        token = secrets.token_urlsafe(32)
        self._tokens[token] = {
            "username": user.username,
            "role": user.role,
            "expires": time.time() + self._ttl,
        }
        self._save()
        return token

    def resolve(self, token: str) -> dict | None:
        info = self._tokens.get(token)
        if not info:
            return None
        if info["expires"] < time.time():
            self._tokens.pop(token, None)
            self._save()
            return None
        return info

    def revoke(self, token: str) -> None:
        self._tokens.pop(token, None)
        self._save()
