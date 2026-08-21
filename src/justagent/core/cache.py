"""基于 diskcache 的简易缓存，替换自建 JSON 文件缓存（原实现现已删除）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from diskcache import Cache as _DiskCache

logger = logging.getLogger("justagent.cache")


class Cache:
    """文件系统缓存，使用 diskcache 存储键值对。

    提供与旧版 JSON 文件缓存兼容的 ``get``/``set``/``invalidate``/``clear`` 接口。
    底层使用 ``diskcache.Cache``，支持 TTL 过期、LRU 淘汰和线程安全并发访问。
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        size_limit: int = 2**30,  # 1 GiB
        eviction_policy: str = "LRU",
    ) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "justagent"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = _DiskCache(
            directory=str(self._cache_dir),
            size_limit=size_limit,
            eviction_policy=eviction_policy,
        )
        logger.debug("Cache initialized at %s (size_limit=%d)", self._cache_dir, size_limit)

    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值。键不存在或已过期时返回 *default*。"""
        return self._cache.get(key, default=default)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """存储缓存值，可选的生存时间（秒）。

        Args:
            key: 缓存键。
            value: 要缓存的值（必须可被 pickle 序列化）。
            ttl: 生存时间（秒）。为 ``None`` 时条目永不过期（但可能被 LRU 淘汰）。
        """
        self._cache.set(key, value, expire=ttl)

    def invalidate(self, key: str) -> None:
        """使指定的缓存键失效。键不存在时静默跳过。"""
        self._cache.delete(key, ignore_missing=True)

    def clear(self) -> None:
        """清空所有缓存条目。"""
        self._cache.clear()

    @property
    def directory(self) -> str:
        return str(self._cache_dir)

    @property
    def size(self) -> int:
        """当前缓存大小（字节）。"""
        volume = self._cache.volume()
        return volume if volume is not None else 0