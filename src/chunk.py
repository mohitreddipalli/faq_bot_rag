import re
from typing import List, Dict, Any


def chunk_text(
    text: str,
    chunk_size: int = 180,
    overlap: int = 30
) -> List[Dict[str, Any]]:
    """
    Splits continuous document text into overlapping chunks based on word count.
    """
    if not text or not text.strip():
        return []

    clean_text = re.sub(r'\s+', ' ', text).strip()
    if not clean_text:
        return []

    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    step = max(1, chunk_size - overlap)

    # Prefer paragraph and sentence boundaries before falling back to raw
    # word windows so definitions stay intact when possible.
    paragraphs = [part.strip() for part in re.split(r'\n{2,}', text) if part.strip()]
    units: List[str] = []
    for paragraph in paragraphs:
        sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', paragraph) if sentence.strip()]
        if sentences:
            units.extend(sentences)
        else:
            units.append(paragraph.strip())

    words = clean_text.split(' ')

    if len(words) <= chunk_size:
        return [{
            "id": 1,
            "text": clean_text,
            "word_count": len(words),
            "char_count": len(clean_text)
        }]

    chunks = []
    chunk_id = 1
    current_words: List[str] = []

    def _append_chunk(chunk_words: List[str]) -> None:
        nonlocal chunk_id
        chunk_str = " ".join(chunk_words).strip()
        if not chunk_str:
            return
        chunks.append({
            "id": chunk_id,
            "text": chunk_str,
            "word_count": len(chunk_words),
            "char_count": len(chunk_str)
        })
        chunk_id += 1

    for unit in units:
        unit_words = unit.split()
        if not unit_words:
            continue

        if len(unit_words) > chunk_size:
            if current_words:
                _append_chunk(current_words)
                current_words = current_words[-overlap:] if overlap else []

            for start_idx in range(0, len(unit_words), step):
                part_words = unit_words[start_idx:start_idx + chunk_size]
                if len(part_words) < max(20, overlap // 2) and chunks:
                    break
                _append_chunk(part_words)
                if start_idx + chunk_size >= len(unit_words):
                    break
            current_words = []
            continue

        if current_words and len(current_words) + len(unit_words) > chunk_size:
            _append_chunk(current_words)
            current_words = current_words[-overlap:] if overlap else []

        current_words.extend(unit_words)

    if current_words:
        _append_chunk(current_words)

    return chunks
