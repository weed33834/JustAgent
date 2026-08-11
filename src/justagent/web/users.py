"""Lightweight multi-user accounts & roles for the JustAgent Web console.

Stores users in ``data/users.json`` (created on demand), issues in-memory
tokens on login, and supports roles: ``admin`` / ``editor`` / ``viewer``.

- ``viewer``: read-only
- ``editor``: read + write (create/edit cases, laws, docs, tasks)
- ``admin``: everything (incl. user management)

A default ``admin`` account is created on first use with the password from
``JUSTAGENT_WEB_ADMIN_PASSWORD`` (or a generated one printed once).
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

ROLES = ("admin", "editor", "viewer")

WRITE_ROLES = {"admin", "editor"}
ADMIN_ROLES = {"admin"}


@dataclass
class User:
    username: str
    role: str = "viewer"
    password_hash: str = ""
    created_at: float = 0.0


@dataclass
class UserStore:
    path: Path = field(default_factory=lambda: Path.home() / ".justagent" / "web_users.json")

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _hash(password: str, salt: str | None = None) -> str:
        salt = salt or secrets.token_hex(8)
        digest = hmac.new(salt.encode(), password.encode(), "sha256").hexdigest()
        return f"{salt}${digest}"

    def _check(self, password: str, stored: str) -> bool:
        try:
            salt, digest = stored.split("$", 1)
            return hmac.compare_digest(self._hash(password, salt), stored)
        except ValueError:
            return False

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

    def authenticate(self, username: str, password: str) -> User | None:
        data = self._load()
        row = data.get(username)
        if not row or not self._check(password, row.get("password_hash", "")):
            return None
        return User(username=username, role=row.get("role", "viewer"))

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
    """In-memory session tokens (token -> {username, role, expires})."""

    def __init__(self, ttl: float = 12 * 3600) -> None:
        self._ttl = ttl
        self._tokens: dict[str, dict] = {}

    def issue(self, user: User) -> str:
        token = secrets.token_urlsafe(32)
        self._tokens[token] = {
            "username": user.username,
            "role": user.role,
            "expires": time.time() + self._ttl,
        }
        return token

    def resolve(self, token: str) -> dict | None:
        info = self._tokens.get(token)
        if not info:
            return None
        if info["expires"] < time.time():
            self._tokens.pop(token, None)
            return None
        return info

    def revoke(self, token: str) -> None:
        self._tokens.pop(token, None)
