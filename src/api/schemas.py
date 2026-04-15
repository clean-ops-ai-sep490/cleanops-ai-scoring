from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ImageURL(BaseModel):
    url: str


class EvaluateVisualizeRequest(BaseModel):
    url: str
    env: Optional[str] = "LOBBY_CORRIDOR"
