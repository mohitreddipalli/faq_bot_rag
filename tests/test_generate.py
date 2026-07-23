import os
import unittest
from unittest.mock import patch

from src.generate import generate_answer
from src.embed import EmbeddingManager


class TestLiveApiRequirements(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_generate_answer_falls_back_without_groq_configuration(self):
        answer = generate_answer(
            query="What is Git?",
            retrieved_results=[{"chunk": {"id": 1, "text": "Git is a distributed version control system."}, "score": 0.99}],
        )

        self.assertIn("Git is a distributed version control system.", answer)

    @patch.dict(os.environ, {}, clear=True)
    def test_embedding_manager_uses_local_embeddings(self):
        manager = EmbeddingManager()
        chunks = [
            {"id": 1, "text": "Git is a distributed version control system.", "word_count": 7, "char_count": 44},
            {"id": 2, "text": "A branch is a line of development.", "word_count": 7, "char_count": 34},
        ]

        chunk_vecs = manager.fit_and_embed_chunks(chunks)
        query_vec = manager.embed_query("What is Git?")

        self.assertGreater(chunk_vecs.shape[0], 0)
        self.assertEqual(chunk_vecs.shape[0], len(chunks))
        self.assertEqual(query_vec.shape[0], chunk_vecs.shape[1])


if __name__ == "__main__":
    unittest.main()
