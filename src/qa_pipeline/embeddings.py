"""向量后端：优先 sentence-transformers，否则 TF-IDF，供语义切分与 SemDeDup 使用。"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self) -> None:
        self.backend = "tfidf"
        self._model = None
        self._vectorizer = None
        model_name = os.environ.get("QA_PIPELINE_EMBED_MODEL")
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(model_name)
                self.backend = "bge"
            except Exception as exc:
                logger.info("embedding fallback to tfidf: %s", exc)
                self.backend = "tfidf"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 8), dtype=np.float32)
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
            mat = self._vectorizer.fit_transform(texts)
        else:
            try:
                mat = self._vectorizer.transform(texts)
            except Exception:
                mat = self._vectorizer.fit_transform(texts)
        return np.asarray(normalize(mat).toarray(), dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return a @ b.T
