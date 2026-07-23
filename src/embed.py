import re
import unicodedata
from typing import List, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingManager:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.vectorizer = TfidfVectorizer(
            analyzer=self._analyze_text,
            min_df=1,
            norm="l2",
        )
        self._fitted = False

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip()

    @classmethod
    def _analyze_text(cls, text: str) -> List[str]:
        normalized_text = cls._normalize_text(text).lower()
        tokens = re.findall(r"\w+", normalized_text)
        if len(tokens) < 2:
            return tokens

        bigrams = [f"{first}__{second}" for first, second in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        normalized_texts = [self._normalize_text(text) for text in texts]
        if not normalized_texts:
            return np.empty((0, 0), dtype=float)

        if not self._fitted:
            raise RuntimeError("EmbeddingManager must be fitted with document chunks before embedding queries.")

        matrix = self.vectorizer.transform(normalized_texts)
        return matrix.astype(float).toarray()

    def fit_and_embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        texts = [self._normalize_text(chunk["text"]) for chunk in chunks]
        if not texts:
            self._fitted = False
            return np.empty((0, 0), dtype=float)

        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.astype(float).toarray()

    def embed_query(self, query: str) -> np.ndarray:
        normalized_query = self._normalize_text(query)
        if not normalized_query:
            raise ValueError("Query string cannot be empty.")

        if not self._fitted:
            raise RuntimeError("No document chunks have been indexed yet.")

        return self._embed_texts([normalized_query])[0]
