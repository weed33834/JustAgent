"""基于 JSON 文件的简易缓存（不依赖 diskcache）。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class DiskCache:
    """文件系统缓存，使用 JSON 存储键值对。"""

    def __init__(self, cache_dir: Path | None = None, default_ttl: int = 3600) -> None:
        self.cache_dir = cache_dir or (Path.home() / ".justagent" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl

    def _cache_path(self, key: str) -> Path:
        # 将 key 中的特殊字符替换为下划线，保证文件名安全
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.cache_dir / f"{safe}.json"

    def get(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if data.get("expires_at") is not None and time.time() > data["expires_at"]:
            path.unlink(missing_ok=True)
            return None
        return data.get("value")

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        path = self._cache_path(key)
        effective_ttl = ttl if ttl is not None else self.default_ttl
        # ``effective_ttl is not None`` (rather than truthiness) so that an
        # explicit ``ttl=0`` records ``expires_at = now`` and is treated as
        # immediately expired on the next read, matching the test contract.
        expires_at = (
            time.time() + effective_ttl if effective_ttl is not None else None
        )
        data = {"value": value, "expires_at": expires_at}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def invalidate(self, key: str) -> None:
        self._cache_path(key).unlink(missing_ok=True)

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.json"):
            f.unlink(missing_ok=True)
