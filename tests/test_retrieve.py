import unittest
import numpy as np
from src.retrieve import compute_cosine_similarity, retrieve_top_k


class TestRetrieve(unittest.TestCase):
    def test_compute_cosine_similarity(self):
        query_vec = np.array([1.0, 0.0, 0.0])
        chunk_vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0]
        ])

        scores = compute_cosine_similarity(query_vec, chunk_vecs)
        self.assertTrue(np.isclose(scores[0], 1.0))
        self.assertTrue(np.isclose(scores[1], 0.0))
        self.assertTrue(np.isclose(scores[2], -1.0))

    def test_retrieve_top_k(self):
        chunks = [
            {"id": 1, "text": "Remote work stipend is $1000", "word_count": 5, "char_count": 30},
            {"id": 2, "text": "Health insurance covers 90%", "word_count": 4, "char_count": 27},
            {"id": 3, "text": "PTO is 20 days per year", "word_count": 6, "char_count": 23}
        ]

        chunk_vecs = np.array([
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 0.9]
        ])

        query_vec = np.array([0.95, 0.05, 0.0])

        res = retrieve_top_k(query_vec, chunk_vecs, chunks, top_k=2, threshold=0.5)

        self.assertTrue(res["is_relevant"])
        self.assertEqual(len(res["results"]), 2)
        self.assertEqual(res["results"][0]["chunk"]["id"], 1)
        self.assertGreater(res["results"][0]["score"], 0.8)


if __name__ == "__main__":
    unittest.main()
