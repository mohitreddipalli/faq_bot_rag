import unittest
from src.chunk import chunk_text


class TestChunk(unittest.TestCase):
    def test_chunk_short_text(self):
        text = "This is a short single paragraph document."
        chunks = chunk_text(text, chunk_size=250, overlap=50)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["id"], 1)
        self.assertEqual(chunks[0]["text"], text)
        self.assertEqual(chunks[0]["word_count"], 7)

    def test_chunk_overlapping_text(self):
        words = [f"word{i}" for i in range(300)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=200, overlap=50)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["word_count"], 200)

        chunk1_words = set(chunks[0]["text"].split())
        chunk2_words = set(chunks[1]["text"].split())
        intersection = chunk1_words.intersection(chunk2_words)

        self.assertGreater(len(intersection), 0)

    def test_chunk_empty_input(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])


if __name__ == "__main__":
    unittest.main()
