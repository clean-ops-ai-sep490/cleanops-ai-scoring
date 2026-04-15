from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple


@dataclass(frozen=True)
class ObjectStorageConfig:
    enabled: bool
    connection_string: str
    container: str
    force_refresh: bool


class ObjectStorageModelLoader:
    def __init__(self, config: ObjectStorageConfig, logger):
        self._config = config
        self._logger = logger
        self._container_client: Any = None
        self._not_found_error_cls: Any = Exception
        self._request_error_cls: Any = Exception

        if self._config.enabled:
            if not self._config.connection_string:
                self._logger.warning("MODEL_STORAGE_CONNECTION_STRING is empty. Blob storage is disabled.")
                return

            blob_module = importlib.import_module("azure.storage.blob")
            exceptions_module = importlib.import_module("azure.core.exceptions")

            blob_service_client_cls = getattr(blob_module, "BlobServiceClient")
            self._not_found_error_cls = getattr(exceptions_module, "ResourceNotFoundError", Exception)
            self._request_error_cls = getattr(exceptions_module, "AzureError", Exception)

            blob_service_client = blob_service_client_cls.from_connection_string(
                self._config.connection_string
            )
            self._container_client = blob_service_client.get_container_client(self._config.container)

    @staticmethod
    def _etag_path(local_path: Path) -> Path:
        return local_path.with_suffix(local_path.suffix + ".etag")

    @staticmethod
    def _read_etag(etag_file: Path) -> Optional[str]:
        if not etag_file.exists():
            return None
        value = etag_file.read_text(encoding="utf-8").strip()
        return value if value else None

    @staticmethod
    def _write_etag(etag_file: Path, etag: str) -> None:
        etag_file.parent.mkdir(parents=True, exist_ok=True)
        etag_file.write_text(etag, encoding="utf-8")

    def _download_if_needed(self, object_key: str, local_path: Path) -> Tuple[Optional[Path], str]:
        if not self._config.enabled or self._container_client is None:
            return None, "storage-disabled"

        local_path.parent.mkdir(parents=True, exist_ok=True)
        etag_file = self._etag_path(local_path)

        try:
            blob_client = self._container_client.get_blob_client(object_key)
            metadata = blob_client.get_blob_properties()
            remote_etag = str(getattr(metadata, "etag", "")).strip('"')
            local_etag = self._read_etag(etag_file)

            needs_download = (
                self._config.force_refresh
                or not local_path.exists()
                or (remote_etag and remote_etag != local_etag)
            )

            if needs_download:
                self._logger.info("Downloading model '%s' from object storage into %s", object_key, local_path)
                with local_path.open("wb") as handle:
                    download_stream = blob_client.download_blob()
                    download_stream.readinto(handle)
                if remote_etag:
                    self._write_etag(etag_file, remote_etag)
                return local_path, "downloaded"

            return local_path, "cache-hit"
        except self._not_found_error_cls:
            self._logger.warning("Blob '%s' not found in container '%s'", object_key, self._config.container)
            return None, "storage-error:not-found"
        except self._request_error_cls as ex:
            self._logger.warning("Failed to read blob '%s' from storage: %s", object_key, ex)
            return None, "storage-error:request"
        except OSError as ex:
            self._logger.warning("Failed to sync object '%s' from storage: %s", object_key, ex)
            return None, "storage-error"
        except Exception as ex:
            self._logger.warning("Unexpected error while syncing '%s': %s", object_key, ex)
            return None, "storage-error"

    def resolve_model_path(
        self,
        object_key: str,
        cache_path: Path,
        fallback_paths: Iterable[Path],
    ) -> Tuple[Optional[str], str]:
        synced_path, source = self._download_if_needed(object_key, cache_path)
        if synced_path is not None and synced_path.exists():
            return str(synced_path), f"object-storage:{source}"

        if cache_path.exists():
            return str(cache_path), "cache-fallback"

        for path in fallback_paths:
            if path.exists() and path.is_file():
                return str(path), "local-fallback"

        return None, "not-found"
