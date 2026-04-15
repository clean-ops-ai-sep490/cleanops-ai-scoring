from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def _patch_binary_content_media_type(node: Any) -> None:
    if isinstance(node, dict):
        if (
            node.get("type") == "string"
            and node.get("contentMediaType") == "application/octet-stream"
        ):
            node.pop("contentMediaType", None)
            node["format"] = "binary"

        for value in node.values():
            _patch_binary_content_media_type(value)
    elif isinstance(node, list):
        for item in node:
            _patch_binary_content_media_type(item)


def apply_custom_openapi(app: FastAPI) -> None:
    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            openapi_version="3.0.3",
            routes=app.routes,
            tags=app.openapi_tags,
        )

        _patch_binary_content_media_type(schema)
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi
