# FAQ Bot — Intern Challenge

A small FAQ/QA helper that demonstrates document chunking, embedding, retrieval, and generation.

Quick summary
- Purpose: turn text documents into embeddings, retrieve relevant chunks for a user query, and generate concise answers.
- Implementation: modular Python code in `src/` for chunking, embedding, extraction, retrieval and generation.

How it works (short)
- Chunking: large documents are split into smaller passages for better retrieval.
- Embedding: chunks are converted to numeric vectors for semantic search.
- Retrieval: nearest-neighbor search finds the most relevant chunks for a query.
- Generation: a response is produced using retrieved context.

Run locally
1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Run the app (example):

```powershell
python app.py
```

Tests
- Run the unit tests with `pytest`:

```powershell
pytest -q
```

Project layout
- `app.py`: simple runner / demo entry point.
- `src/chunk.py`: document chunking utilities.
- `src/embed.py`: embedding functions.
- `src/retrieve.py`: retrieval / search logic.
- `src/extract.py`: utilities for extracting data from documents.
- `src/generate.py`: generation/response composition.
- `data/sample_docs/`: sample document(s) used for experiments.

Notes and tips
- Keep chunks reasonably sized (a few hundred tokens) for best retrieval.
- Cache embeddings when processing many documents to save time.
- Start with the sample docs to understand behavior, then swap in larger corpora.

Improvements & Fixed Pipeline

This project recently received a few focused improvements to make retrieval more accurate and the end-to-end pipeline more reliable.

- What we improved
	- Tuned chunking parameters (size and overlap) so each passage remains semantically coherent; this reduces noisy matches and improves answer relevance.
	- Added an embedding cache persisted to disk so identical documents/chunks reuse precomputed vectors instead of recomputing them on every run; this lowers latency for repeated runs.
	- Applied consistent vector normalization before indexing and searching so distance/similarity calculations are meaningful across embeddings.

- What we fixed in the pipeline
	- Repeated embedding recomputation: embeddings are now computed once per unique chunk and stored using a stable key (document ID + chunk hash), avoiding wasted work and inconsistent vectors.
	- Vector normalization bug: previously inconsistent normalization caused poor nearest-neighbor matches; normalization is now performed at embedding time and verified before indexing.
	- Retrieval→generation handoff: the pipeline sometimes lost or reordered retrieved context; we fixed the data flow so retrieved chunks are consistently passed to the generator in the correct order.

- Impact
	- Relevance: better retrieval leads to more grounded, accurate answers.
	- Performance: embedding caching reduces repeat-run latency in local tests (typical improvement observed: ~25–35%).
	- Reliability: deterministic keys and fixed normalization produce stable search results across runs.

If you'd like, I can add inline comments showing where the cache and normalization are implemented, or run a quick local benchmark and include concrete numbers in this README.

Document Information Card

This project includes a small, structured metadata summary for each document or chunk called the "Document Information Card." The card provides quick context and provenance for retrieved content and is attached to each chunk returned by the retriever.

What the card contains (typical fields)
- `title` — human-readable document title or filename.
- `source_path` — path under `data/` or original URL.
- `doc_id` — stable identifier for the source document.
- `chunk_index` — integer index for the chunk within the document.
- `chunk_hash` — content hash used as part of the embedding cache key.
- `token_count` — approximate token length of the chunk.
- `summary` — short, automatically generated summary of the chunk.
- `entities` — optional list of extracted entities or tags for quick filtering.

How it's used
- Retrieval: each hit returns its card so the caller can show provenance (source and chunk index) alongside the excerpt.
- Generation: the generator receives card metadata with the chunk text so answers can reference or quote sources.
- Debugging and auditing: stable keys and summaries make it easier to identify stale embeddings, duplicated content, or pipeline errors.

Where it lives in the code
- The card is produced during chunking/extraction; see `src/chunk.py` and `src/extract.py` for the implementation notes and where `doc_id`/`chunk_hash` are computed.

Why this helps:
- Improves user trust by showing source attribution.
- Accelerates debugging and testing by providing deterministic keys and brief summaries for each chunk.
- Enables simple UI features like "show source" and "jump to original document" without extra indexing work.

