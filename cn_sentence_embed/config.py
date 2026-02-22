from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "nlp_gte_sentence-embedding_chinese-base"


@dataclass(frozen=True)
class SentenceEmbeddingConfig:
    """Runtime configuration for sentence embedding inference."""

    model_path: Path = DEFAULT_MODEL_PATH
    device: str = "cpu"
    sequence_length: int = 512
    quiet: bool = True

    @classmethod
    def from_values(
        cls,
        model_path: str | Path | None = None,
        device: str = "cpu",
        sequence_length: int = 512,
        quiet: bool = True,
    ) -> "SentenceEmbeddingConfig":
        resolved_path = Path(model_path).expanduser() if model_path else DEFAULT_MODEL_PATH
        return cls(
            model_path=resolved_path,
            device=device,
            sequence_length=sequence_length,
            quiet=quiet,
        )

    def resolved_model_path(self) -> Path:
        return self.model_path.expanduser().resolve()
