from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from .config import SentenceEmbeddingConfig
from .runtime import configure_runtime, suppress_output
from .types import SimilarityResult


class SentenceEmbeddingClient:
    """Reusable client for Chinese sentence embedding and similarity tasks."""

    def __init__(
        self,
        config: SentenceEmbeddingConfig | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "cpu",
        sequence_length: int = 512,
        quiet: bool = True,
    ) -> None:
        self.config = config or SentenceEmbeddingConfig.from_values(
            model_path=model_path,
            device=device,
            sequence_length=sequence_length,
            quiet=quiet,
        )

        configure_runtime(self.config.quiet)

        resolved_model_path = self.config.resolved_model_path()
        if not resolved_model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {resolved_model_path}. "
                "Please check the model directory or pass --model-path."
            )

        with suppress_output(self.config.quiet):
            self._pipeline = pipeline(
                task=Tasks.sentence_embedding,
                model=str(resolved_model_path),
                sequence_length=self.config.sequence_length,
                device=self.config.device,
            )

    def _run_pipeline(self, payload: dict[str, Any]) -> dict[str, Any]:
        with suppress_output(self.config.quiet):
            return self._pipeline(input=payload)

    @staticmethod
    def _normalize_sentences(sentences: Sequence[str] | str, field_name: str) -> list[str]:
        if isinstance(sentences, str):
            normalized = [sentences]
        else:
            normalized = list(sentences)

        if not normalized:
            raise ValueError(f"{field_name} cannot be empty.")

        if any(not isinstance(text, str) for text in normalized):
            raise TypeError(f"{field_name} must contain only strings.")

        return normalized

    def similarity(
        self,
        source_sentences: Sequence[str] | str,
        sentences_to_compare: Sequence[str] | str,
    ) -> SimilarityResult:
        source_list = self._normalize_sentences(source_sentences, "source_sentences")
        compare_list = self._normalize_sentences(sentences_to_compare, "sentences_to_compare")

        result = self._run_pipeline(
            {
                "source_sentence": source_list,
                "sentences_to_compare": compare_list,
            }
        )
        scores = [float(score) for score in result["scores"]]
        return SimilarityResult(text_embedding=result["text_embedding"], scores=scores)

    def compute_similarity(
        self,
        source_sentences: Sequence[str] | str,
        sentences_to_compare: Sequence[str] | str,
    ) -> tuple[Any, list[float]]:
        """Backward-compatible API used by legacy scripts."""
        result = self.similarity(source_sentences, sentences_to_compare)
        return result.text_embedding, result.scores

    def batch_similarity(
        self,
        source_list: Sequence[str] | str,
        compare_list: Sequence[str] | str,
    ) -> list[dict[str, Any]]:
        sources = self._normalize_sentences(source_list, "source_list")
        compare_sentences = self._normalize_sentences(compare_list, "compare_list")

        all_results: list[dict[str, Any]] = []
        for source_sentence in sources:
            single_result = self.similarity([source_sentence], compare_sentences)
            all_results.append({"source": source_sentence, "scores": single_result.scores})
        return all_results

    def encode(self, sentences: Sequence[str] | str) -> Any:
        source_list = self._normalize_sentences(sentences, "sentences")
        result = self._run_pipeline({"source_sentence": source_list})
        return result["text_embedding"]
