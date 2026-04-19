from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ImageURL(BaseModel):
    url: str


class EvaluateVisualizeRequest(BaseModel):
    url: str
    env: Optional[str] = "LOBBY_CORRIDOR"


class PpeEvaluateRequest(BaseModel):
    image_urls: list[str] = Field(..., min_length=1)
    validation_type: str = "all_required"
    required_objects: list[str] = Field(..., min_length=1)
    min_confidence: float = 0.25
