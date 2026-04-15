from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, Optional
import uuid


class TempVisualizationStore:
    def __init__(self, ttl_sec: int, max_items: int):
        self._ttl_sec = ttl_sec
        self._max_items = max_items
        self._items: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def _prune_locked(self, now: datetime) -> None:
        expired_keys = [k for k, v in self._items.items() if v["expires_at"] <= now]
        for key in expired_keys:
            self._items.pop(key, None)

        overflow = len(self._items) - self._max_items
        if overflow > 0:
            sorted_keys = sorted(self._items.items(), key=lambda kv: kv[1]["expires_at"])
            for key, _ in sorted_keys[:overflow]:
                self._items.pop(key, None)

    def save(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self._ttl_sec)
        token = uuid.uuid4().hex

        with self._lock:
            self._prune_locked(now)
            self._items[token] = {
                "image": image_bytes,
                "mime_type": mime_type,
                "expires_at": expires_at,
            }

        return {
            "token": token,
            "mime_type": mime_type,
            "byte_size": len(image_bytes),
            "ttl_seconds": self._ttl_sec,
            "expires_at_utc": expires_at.isoformat().replace("+00:00", "Z"),
        }

    def get(self, token: str) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._prune_locked(now)
            item = self._items.get(token)
            if item is None:
                return None
            if item["expires_at"] <= now:
                self._items.pop(token, None)
                return None
            return item
