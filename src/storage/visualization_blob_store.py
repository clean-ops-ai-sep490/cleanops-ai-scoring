from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import uuid
from typing import Any, Dict

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings, PublicAccess


@dataclass(frozen=True)
class VisualizationBlobConfig:
    enabled: bool
    connection_string: str
    container: str
    prefix: str


class VisualizationBlobStore:
    def __init__(self, config: VisualizationBlobConfig, logger):
        self._config = config
        self._logger = logger
        self._container_client = None

        if not self._config.enabled:
            return

        if not self._config.connection_string.strip():
            self._logger.warning("Visualization blob storage enabled but connection string is empty.")
            return

        blob_service = BlobServiceClient.from_connection_string(self._config.connection_string)
        self._container_client = blob_service.get_container_client(self._config.container)
        self._ensure_container()

    def _ensure_container(self) -> None:
        if self._container_client is None:
            return

        try:
            self._container_client.create_container(public_access=PublicAccess.Blob)
        except ResourceExistsError:
            return
        except Exception as exc:
            # Container may already exist with different policy; do not block uploads.
            self._logger.warning("Unable to enforce public access on visualization container: %s", exc)

    def _build_object_key(self, source_type: str, source: str, env_key: str) -> str:
        now = datetime.now(timezone.utc)
        clean_env = (env_key or "unknown").strip().lower()
        source_hash = hashlib.sha256((source or "unknown").encode("utf-8")).hexdigest()[:16]
        uid = uuid.uuid4().hex[:12]

        prefix = (self._config.prefix or "").strip("/")
        suffix = f"{clean_env}/{now:%Y/%m/%d}/{source_type}_{source_hash}_{uid}.jpg"
        return f"{prefix}/{suffix}" if prefix else suffix

    def upload_visualization(
        self,
        image_bytes: bytes,
        source_type: str,
        source: str,
        env_key: str,
    ) -> Dict[str, Any]:
        if self._container_client is None:
            raise RuntimeError("Visualization blob storage is not configured.")

        object_key = self._build_object_key(source_type, source, env_key)
        blob_client = self._container_client.get_blob_client(object_key)

        blob_client.upload_blob(
            image_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type="image/jpeg"),
        )

        return {
            "url": blob_client.url,
            "mime_type": "image/jpeg",
            "byte_size": len(image_bytes),
            "object_key": object_key,
            "container": self._config.container,
        }
