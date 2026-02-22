from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cn_sentence_embed import SentenceEmbeddingClient


def main() -> None:
    client = SentenceEmbeddingClient(device="cpu")

    source_sentence = "吃完海鲜可以喝牛奶吗?"
    candidates = [
        "不可以，早晨喝牛奶不科学",
        "吃了海鲜后不能再喝牛奶，建议间隔数小时。",
        "吃海鲜后可以搭配清淡主食。",
        "吃海鲜不建议同时吃富含维生素C的水果。",
    ]

    embeddings, scores = client.compute_similarity([source_sentence], candidates)
    print("句子向量形状:", embeddings.shape)
    print("相似度分数:")
    for sentence, score in zip(candidates, scores):
        print(f"- {source_sentence} vs {sentence}: {score:.4f}")


if __name__ == "__main__":
    main()
