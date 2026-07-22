# 🛠️ AIML Internship Challenge: Debug the RAG Pipeline

Welcome! This repository contains a prototype single-document Retrieval-Augmented Generation (RAG) system for a FAQ Bot. However, **several subtle bugs have been introduced into the core pipeline**, causing unexpected behaviors and test failures.

Your mission as an AIML intern is to diagnose, debug, and fix the codebase until all unit tests pass and the Streamlit application returns accurate, grounded answers.

---

## 🎯 Objectives & Symptoms

When running the application or unit test suite, you will notice three main issues:

1. **Text Gaps & Skipping:** The text chunking module is skipping large sections of the document instead of creating overlapping windows.
2. **Irrelevant Context Retrieval:** The similarity search is returning the *least relevant* chunks instead of the top matching chunks!
3. **Empty Context Prompts:** The LLM generator reports that it cannot find the answer because context chunks are missing from the prompt body.

---

## 🧪 How to Test & Debug

1. Run the test suite:
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```
2. Inspect failing tests to pinpoint which module (`src/chunk.py`, `src/retrieve.py`, or `src/generate.py`) contains logic flaws.
3. Fix the bugs and ensure all 8 unit tests pass cleanly.
4. Run `streamlit run app.py` to verify the UI functionality.

Good luck!
