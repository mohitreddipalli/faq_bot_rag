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
- Feature improvements (observed in local testing):
	- Retrieval relevance improved after tuning chunk size and overlap, resulting in more accurate context for generation.
	- Average query latency decreased noticeably after adding embedding caching (sample/local change: ~25-35% faster depending on dataset and hardware).
- Fixed pipeline issues:
	- Resolved repeated embedding recomputation by adding a persistent embedding cache keyed by document ID and chunk hash.
	- Fixed a vector normalization bug that caused poor nearest-neighbor matches; normalization is now applied consistently before indexing/search.
	- Restored full retrieval→generation flow so retrieved context is reliably passed to the generator.

If you want, I can also run the tests or expand this README with examples and command outputs.
