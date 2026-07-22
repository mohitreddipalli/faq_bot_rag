import numpy as np
from typing import List, Dict, Any, Union

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.use_fallback = False
        self.tfidf_vectorizer = None

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.use_fallback = True
        else:
            self.use_fallback = True

    def fit_and_embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        texts = [chunk["text"] for chunk in chunks]
        if not texts:
            return np.empty((0, 384))

        if not self.use_fallback and self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return embeddings / norms
        else:
            if HAS_SKLEARN:
                self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                matrix = self.tfidf_vectorizer.fit_transform(texts)
                return matrix.toarray()
            else:
                raise RuntimeError("Neither sentence-transformers nor scikit-learn is available.")

    def embed_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        if not self.use_fallback and self.model is not None:
            embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        else:
            if self.tfidf_vectorizer is not None:
                matrix = self.tfidf_vectorizer.transform([query])
                return matrix.toarray()[0]
            else:
                raise RuntimeError("Embedding model/vectorizer has not been initialized.")
