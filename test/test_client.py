from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from cn_sentence_embed import SentenceEmbeddingClient, SentenceEmbeddingConfig


class TestSentenceEmbeddingClient(TestCase):
    @patch("cn_sentence_embed.client.pipeline")
    def test_client_initialization_and_similarity(self, mocked_pipeline: MagicMock) -> None:
        mocked_executor = MagicMock()
        mocked_executor.return_value = {
            "text_embedding": [[0.1, 0.2, 0.3]],
            "scores": [0.95, 0.67],
        }
        mocked_pipeline.return_value = mocked_executor

        config = SentenceEmbeddingConfig.from_values(model_path=Path(__file__), quiet=False)
        client = SentenceEmbeddingClient(config=config)

        result = client.similarity("问题A", ["候选B", "候选C"])

        self.assertEqual(result.scores, [0.95, 0.67])
        mocked_pipeline.assert_called_once()
        mocked_executor.assert_called_once()

    @patch("cn_sentence_embed.client.pipeline")
    def test_compute_similarity_keeps_backward_compatibility(self, mocked_pipeline: MagicMock) -> None:
        mocked_executor = MagicMock()
        mocked_executor.return_value = {
            "text_embedding": [[0.11, 0.22]],
            "scores": [0.88],
        }
        mocked_pipeline.return_value = mocked_executor

        config = SentenceEmbeddingConfig.from_values(model_path=Path(__file__), quiet=False)
        client = SentenceEmbeddingClient(config=config)

        embeddings, scores = client.compute_similarity(["A"], ["B"])

        self.assertEqual(embeddings, [[0.11, 0.22]])
        self.assertEqual(scores, [0.88])

    @patch("cn_sentence_embed.client.pipeline")
    def test_encode_and_input_validation(self, mocked_pipeline: MagicMock) -> None:
        mocked_executor = MagicMock()
        mocked_executor.return_value = {
            "text_embedding": [[0.2, 0.4]],
            "scores": [],
        }
        mocked_pipeline.return_value = mocked_executor

        config = SentenceEmbeddingConfig.from_values(model_path=Path(__file__), quiet=False)
        client = SentenceEmbeddingClient(config=config)

        embeddings = client.encode(["文本1"])
        self.assertEqual(embeddings, [[0.2, 0.4]])

        with self.assertRaises(ValueError):
            client.similarity([], ["候选"])
