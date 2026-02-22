from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimilarityResult:
    """Structured response from similarity inference."""

    text_embedding: Any
    scores: list[float]
