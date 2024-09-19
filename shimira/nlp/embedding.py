import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional


def get_sentence_embedding(
    sentence: str,
    model: Optional[SentenceTransformer] = None,
    model_path: str = "pkshatech/GLuCoSE-base-ja",
) -> np.ndarray:
    if model is None:
        model = SentenceTransformer(model_name_or_path=model_path)

    return model.encode(sentences=sentence)
