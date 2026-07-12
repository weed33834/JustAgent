"""Upload/publish adapters for AutoShip-CLI."""

from __future__ import annotations

from autoship.adapters.upload.base import UploadAdapter
from autoship.adapters.upload.registry import get_uploader, register_uploader

__all__ = ["UploadAdapter", "get_uploader", "register_uploader"]
