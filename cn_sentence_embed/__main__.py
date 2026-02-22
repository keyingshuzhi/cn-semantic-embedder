from __future__ import annotations

import argparse
import json

from .client import SentenceEmbeddingClient
from .config import SentenceEmbeddingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chinese sentence embedding toolkit")
    parser.add_argument("--model-path", type=str, default=None, help="Local model directory")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device, e.g. cpu or cuda")
    parser.add_argument("--sequence-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--no-quiet", action="store_true", help="Show model initialization logs")

    subparsers = parser.add_subparsers(dest="command", required=True)

    similarity_parser = subparsers.add_parser("similarity", help="Compute sentence similarity")
    similarity_parser.add_argument("--source", nargs="+", required=True, help="Source sentence(s)")
    similarity_parser.add_argument("--compare", nargs="+", required=True, help="Compared sentence(s)")

    encode_parser = subparsers.add_parser("encode", help="Encode sentence(s) into embeddings")
    encode_parser.add_argument("--text", nargs="+", required=True, help="Sentence(s) to encode")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = SentenceEmbeddingConfig.from_values(
        model_path=args.model_path,
        device=args.device,
        sequence_length=args.sequence_length,
        quiet=not args.no_quiet,
    )
    client = SentenceEmbeddingClient(config=config)

    if args.command == "similarity":
        result = client.similarity(args.source, args.compare)
        payload = {
            "scores": result.scores,
            "embedding_shape": list(getattr(result.text_embedding, "shape", [])),
        }
    else:
        embeddings = client.encode(args.text)
        payload = {
            "embedding_shape": list(getattr(embeddings, "shape", [])),
        }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
