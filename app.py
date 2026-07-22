import os
import streamlit as st
import numpy as np
from datetime import datetime
import math
from dotenv import load_dotenv

load_dotenv()

from src.extract import extract_text_from_file
from src.chunk import chunk_text
from src.embed import EmbeddingManager
from src.retrieve import retrieve_top_k
from src.generate import generate_answer


def _format_size(bytes_size: int) -> str:
    """Return human-readable file size."""
    if bytes_size is None:
        return "N/A"
    if bytes_size < 1024:
        return f"{bytes_size} B"
    kb = bytes_size / 1024.0
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024.0
    return f"{mb:.2f} MB"


def _estimate_pdf_pages_from_text(text: str) -> int:
    """Estimate PDF page count from extracted text by counting page breaks.

    This mirrors the simple extraction that joins pages with double newlines.
    Falls back to 1 if text exists but no clear separators are found.
    """
    if not text:
        return 0
    pages = text.count("\n\n") + 1
    return max(1, pages)


def _render_document_info_card(doc: dict):
    """Render a styled document information card using Streamlit markdown.

    The card shows file metadata and simple statistics in two columns.
    """
    filename = doc.get("filename", "<unknown>")
    file_type = (doc.get("file_type") or "").upper() or "N/A"
    size = _format_size(doc.get("file_size_bytes"))
    text = doc.get("text", "")
    chunks = doc.get("chunks", [])
    total_lines = text.count("\n") + 1 if text else 0
    total_words = len(text.split()) if text else 0
    total_chars = len(text)
    total_chunks = len(chunks)
    uploaded_at_iso = doc.get("uploaded_at")
    try:
        uploaded_at = datetime.fromisoformat(uploaded_at_iso) if uploaded_at_iso else None
        uploaded_str = uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if uploaded_at else "N/A"
    except Exception:
        uploaded_str = uploaded_at_iso or "N/A"

    # Page count: for TXT files show N/A, for PDF estimate from text
    if file_type.lower() == "pdf":
        page_count = _estimate_pdf_pages_from_text(text)
    elif file_type.lower() == "txt":
        page_count = "N/A"
    else:
        page_count = "N/A"

    # Build HTML for a clean two-column card
    card_html = f"""
    <div style="border:1px solid rgba(255,255,255,0.06); padding:16px; border-radius:8px; background-color:rgba(255,255,255,0.02);">
      <h3 style="margin:0 0 8px 0">📄 Document Information</h3>
      <div style="display:flex; gap:24px;">
        <div style="flex:1;">
          <p>📄 <strong>Document Name</strong>: {filename}</p>
          <p>📁 <strong>File Type</strong>: {file_type}</p>
          <p>📦 <strong>File Size</strong>: {size}</p>
          <p>📑 <strong>Total Pages</strong>: {page_count}</p>
        </div>
        <div style="flex:1;">
          <p>📄 <strong>Total Lines</strong>: {total_lines}</p>
          <p>📝 <strong>Total Words</strong>: {total_words}</p>
          <p>🔤 <strong>Total Characters</strong>: {total_chars}</p>
          <p>✂️ <strong>Total Chunks Created</strong>: {total_chunks}</p>
        </div>
      </div>
      <div style="margin-top:12px; display:flex; justify-content:space-between; align-items:center;">
        <div>📅 <strong>Upload Time</strong>: {uploaded_str}</div>
        <div style="color:#4BB543;">✅ <strong>Processing Status</strong>: Processed Successfully</div>
      </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


st.set_page_config(
    page_title="FAQ Bot - Single Document RAG",
    page_icon="🤖",
    layout="wide"
)

if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = None

if "document_data" not in st.session_state:
    st.session_state.document_data = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------- SIDEBAR ----------------
st.sidebar.title("🤖 FAQ Bot Configuration")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Document (PDF or TXT)",
    type=["pdf", "txt"],
    help="Upload a document to index and query."
)

# Allow pasting raw document text directly and indexing it like an upload.
pasted_text = st.sidebar.text_area(
    "Or paste your document text here",
    value="",
    height=200,
    help="Paste the full text of a document and click 'Index Pasted Text' to index it for querying."
)

if st.sidebar.button("Index Pasted Text"):
    content = pasted_text.strip()
    if not content:
        st.sidebar.error("No text to index. Paste your document text into the textarea first.")
    else:
        try:
            chunk_size = int(os.getenv("CHUNK_SIZE", 250))
            overlap = int(os.getenv("CHUNK_OVERLAP", 50))
            chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)

            if st.session_state.embedding_manager is None:
                st.session_state.embedding_manager = EmbeddingManager()

            embeddings = st.session_state.embedding_manager.fit_and_embed_chunks(chunks)
            file_size_bytes = len(content.encode("utf-8"))
            uploaded_at = datetime.now()

            st.session_state.document_data = {
                "filename": "pasted_text",
                "text": content,
                "chunks": chunks,
                "embeddings": embeddings,
                "file_size_bytes": file_size_bytes,
                "file_type": "txt",
                "uploaded_at": uploaded_at.isoformat(),
            }
            st.session_state.processing_failed = False
            st.session_state.chat_history = []
            st.toast(f"Successfully indexed {len(chunks)} chunks from pasted text!", icon="✅")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error indexing pasted text: {str(e)}")

st.sidebar.markdown("### ⚙️ RAG Configuration")
# Hyperparameters and LLM provider are intentionally not editable via the UI.
# Use environment variables to configure these values when launching the app.
threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.25))
top_k = int(os.getenv("TOP_K", 3))
provider = os.getenv("LLM_PROVIDER", "auto")

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Session & History", use_container_width=True):
    st.session_state.document_data = None
    st.session_state.chat_history = []
    st.rerun()

# ---------------- CORE LOGIC ----------------
if uploaded_file is not None:
    current_filename = uploaded_file.name
    if (
        st.session_state.document_data is None or
        st.session_state.document_data["filename"] != current_filename
    ):
        with st.spinner(f"Extracting & Indexing '{current_filename}'..."):
            try:
                raw_text = extract_text_from_file(uploaded_file, current_filename)
                chunk_size = int(os.getenv("CHUNK_SIZE", 250))
                overlap = int(os.getenv("CHUNK_OVERLAP", 50))
                chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
                
                if st.session_state.embedding_manager is None:
                    st.session_state.embedding_manager = EmbeddingManager()
                
                embeddings = st.session_state.embedding_manager.fit_and_embed_chunks(chunks)
                # Gather document metadata for the information card
                try:
                    file_size_bytes = uploaded_file.size
                except Exception:
                    file_size_bytes = len(raw_text.encode("utf-8"))

                file_ext = os.path.splitext(current_filename)[1].lower().lstrip('.')
                uploaded_at = datetime.now()

                st.session_state.document_data = {
                    "filename": current_filename,
                    "text": raw_text,
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "file_size_bytes": file_size_bytes,
                    "file_type": file_ext,
                    "uploaded_at": uploaded_at.isoformat(),
                }
                st.session_state.processing_failed = False
                st.session_state.chat_history = []
                st.toast(f"Successfully indexed {len(chunks)} chunks!", icon="✅")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                # Mark processing as failed so the UI can display a failure card
                st.session_state.processing_failed = True

# ---------------- MAIN DASHBOARD ----------------
st.title("📚 Single-Document FAQ Assistant")
st.markdown("Ask natural language questions about your uploaded document.")

doc = st.session_state.document_data

if doc is None:
    st.info("👈 Please upload a PDF or TXT file using the sidebar to begin.")
    
    if st.button("📄 Or click here to load the Sample Policy Document"):
        sample_path = "data/sample_docs/sample_policy.txt"
        if os.path.exists(sample_path):
            with open(sample_path, "r") as f:
                content = f.read()
            chunks = chunk_text(content, chunk_size=250, overlap=50)
            if st.session_state.embedding_manager is None:
                st.session_state.embedding_manager = EmbeddingManager()
            embeddings = st.session_state.embedding_manager.fit_and_embed_chunks(chunks)
            file_size_bytes = os.path.getsize(sample_path)
            uploaded_at = datetime.now()

            st.session_state.document_data = {
                "filename": "sample_policy.txt",
                "text": content,
                "chunks": chunks,
                "embeddings": embeddings,
                "file_size_bytes": file_size_bytes,
                "file_type": "txt",
                "uploaded_at": uploaded_at.isoformat(),
            }
            st.session_state.processing_failed = False
            st.rerun()
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Indexed Document", doc["filename"])
    col2.metric("Total Character Count", f"{len(doc['text']):,}")
    col3.metric("Indexed Chunks", f"{len(doc['chunks'])}")
    
    st.markdown("---")

    # Show Document Information Card (or failure card if processing failed)
    if st.session_state.get("processing_failed"):
        st.error("❌ Document Processing Failed")
    else:
        _render_document_info_card(doc)


    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "retrieved_results" in message and message["retrieved_results"]:
                with st.expander("🔍 View Retrieved Source Chunks"):
                    for item in message["retrieved_results"]:
                        c = item["chunk"]
                        score = item["score"]
                        st.markdown(f"**Chunk #{c['id']}** | *Similarity Score: `{score:.4f}`*")
                        st.caption(c["text"])
                        st.divider()

    if user_question := st.chat_input("Ask a question about the document..."):
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context & generating grounded answer..."):
                query_vec = st.session_state.embedding_manager.embed_query(user_question)

                retrieval = retrieve_top_k(
                    query_vec=query_vec,
                    chunk_vecs=doc["embeddings"],
                    chunks=doc["chunks"],
                    top_k=top_k,
                    threshold=threshold
                )
                # Provide the original query text so retrieve_top_k can
                # use a token-overlap fallback if vector similarities are
                # all zero (e.g., TF-IDF produced empty vectors).
                retrieval = retrieve_top_k(
                    query_vec=query_vec,
                    chunk_vecs=doc["embeddings"],
                    chunks=doc["chunks"],
                    top_k=top_k,
                    threshold=threshold,
                    query_text=user_question
                )

                answer = generate_answer(
                    query=user_question,
                    retrieved_results=retrieval["results"],
                    is_relevant=retrieval["is_relevant"],
                    provider=provider
                )

                st.markdown(answer)

                if retrieval["results"]:
                    with st.expander("🔍 View Retrieved Source Chunks"):
                        for item in retrieval["results"]:
                            c = item["chunk"]
                            score = item["score"]
                            st.markdown(f"**Chunk #{c['id']}** | *Similarity Score: `{score:.4f}`*")
                            st.caption(c["text"])
                            st.divider()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "retrieved_results": retrieval["results"]
        })
