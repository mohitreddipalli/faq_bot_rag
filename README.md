# FAQ Bot — Intern Challenge

A production-focused FAQ/QA assistant that indexes uploaded documents, retrieves relevant chunks, and generates grounded answers from live LLM APIs.

## What the app does
- Accepts PDF or TXT uploads and extracts document text.
- Splits documents into chunks for retrieval.
- Creates local TF-IDF embeddings for retrieval.
- Retrieves the closest matching chunks and sends them with the user question to Groq for a grounded response.

## Setup
1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Create a local environment file with a real API key:

```powershell
Copy-Item .env.example .env
```

3. Fill in the Groq key in [.env](.env.example):
- GROQ_API_KEY for grounded answer generation

4. Run the app:

```powershell
python app.py
```

## Project layout
- [app.py](app.py): Streamlit entry point.
- [src/chunk.py](src/chunk.py): document chunking utilities.
- [src/embed.py](src/embed.py): local embedding integration.
- [src/retrieve.py](src/retrieve.py): retrieval/search logic.
- [src/extract.py](src/extract.py): document extraction utilities.
- [src/generate.py](src/generate.py): Groq generation logic.

## Testing
Run the unit tests with:

```powershell
python -m unittest discover -s tests
```

## Production notes
- The app now uses local embeddings and only requires GROQ_API_KEY for generation.
- No mock, sample, or fallback response paths remain in the live generation flow.
- The existing project structure is unchanged; only the implementation was tightened for production use.
