from typing import List, Dict, Any
import re


def chunk_text(
    text: str,
    chunk_size: int = 250,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Splits continuous document text into overlapping chunks based on word count.
    """
    if not text or not text.strip():
        return []

    clean_text = re.sub(r'\s+', ' ', text).strip()
    words = clean_text.split(' ')

    if len(words) <= chunk_size:
        return [{
            "id": 1,
            "text": clean_text,
            "word_count": len(words),
            "char_count": len(clean_text)
        }]

    # Ensure the sliding window advances by chunk_size - overlap
    # so that chunks overlap by the requested amount.
    # Use max(1, ...) to avoid zero or negative step sizes.
    step = max(1, chunk_size - overlap)

    chunks = []
    chunk_id = 1

    for start_idx in range(0, len(words), step):
        end_idx = min(start_idx + chunk_size, len(words))
        chunk_words = words[start_idx:end_idx]

        if len(chunk_words) < max(20, overlap // 2) and chunks:
            break

        chunk_str = " ".join(chunk_words)
        chunks.append({
            "id": chunk_id,
            "text": chunk_str,
            "word_count": len(chunk_words),
            "char_count": len(chunk_str)
        })
        chunk_id += 1

        if end_idx >= len(words):
            break

    return chunks
