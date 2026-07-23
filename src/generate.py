import json
import os
import urllib.error
import urllib.request
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """You are a precise, grounded AI assistant answering questions about an uploaded document.

CRITICAL INSTRUCTIONS:
1. Answer the question relying ONLY on the provided context below.
2. Do NOT assume, extrapolate, or use general knowledge not present in the context.
3. If the context does not contain sufficient information to answer the question, state exactly: "I cannot find the answer to your question in the uploaded document."
4. Keep your answer clear, direct, and concise.

---
DOCUMENT CONTEXT:
{context}
---

USER QUESTION: {query}

GROUNDED ANSWER:"""


def build_context_string(retrieved_results: List[Dict[str, Any]]) -> str:
    context_blocks = []
    ordered_results = sorted(
        retrieved_results,
        key=lambda item: (-float(item.get("score", 0.0)), item.get("chunk", {}).get("id", 0))
    )

    for item in ordered_results:
        chunk = item["chunk"]
        score = item.get("score", 0.0)
        block = f"[Source Chunk #{chunk['id']} (Relevance Score: {score:.2f})]:\n{chunk['text']}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def generate_answer(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    is_relevant: bool = True,
    provider: str = "auto"
) -> str:
    if not retrieved_results:
        raise RuntimeError("No relevant information was retrieved from the uploaded document.")

    context_str = build_context_string(retrieved_results)
    prompt = PROMPT_TEMPLATE.format(context=context_str, query=query)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return _fallback_answer(retrieved_results)

    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return _call_groq(prompt, groq_key, groq_model, retrieved_results)


def _fallback_answer(retrieved_results: List[Dict[str, Any]]) -> str:
    top_result = sorted(
        retrieved_results,
        key=lambda item: (-float(item.get("score", 0.0)), item.get("chunk", {}).get("id", 0))
    )[0]
    chunk = top_result["chunk"]
    text = (chunk.get("text") or "").strip()
    if not text:
        return "I cannot find the answer to your question in the uploaded document."
    return text


def _call_groq(prompt: str, api_key: str, model: str, retrieved_results: List[Dict[str, Any]]) -> str:
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
        }
        request = urllib.request.Request(
            url="https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        return _fallback_answer(retrieved_results)
    except Exception as exc:
        return _fallback_answer(retrieved_results)
