import os
import io
from typing import Union

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


def extract_text_from_file(file_source: Union[str, bytes, io.BytesIO], filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        return _extract_txt(file_source)
    elif ext == ".pdf":
        return _extract_pdf(file_source)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Only .pdf and .txt files are supported.")


def _extract_txt(file_source: Union[str, bytes, io.BytesIO]) -> str:
    if isinstance(file_source, str):
        with open(file_source, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    elif isinstance(file_source, bytes):
        content = file_source.decode("utf-8", errors="replace")
    elif hasattr(file_source, "read"):
        data = file_source.read()
        if isinstance(data, bytes):
            content = data.decode("utf-8", errors="replace")
        else:
            content = str(data)
    else:
        raise ValueError("Invalid file source type for TXT extraction.")

    content = content.strip()
    if not content:
        raise ValueError("The uploaded TXT document is empty.")
    return content


def _extract_pdf(file_source: Union[str, bytes, io.BytesIO]) -> str:
    text_pages = []

    if isinstance(file_source, str):
        stream = file_source
    elif isinstance(file_source, bytes):
        stream = io.BytesIO(file_source)
    else:
        stream = file_source

    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(stream) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_pages.append(extracted.strip())
        except Exception:
            text_pages = []

    if not text_pages and HAS_PYPDF:
        try:
            if isinstance(stream, io.BytesIO):
                stream.seek(0)
            reader = pypdf.PdfReader(stream)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_pages.append(extracted.strip())
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    full_text = "\n\n".join(text_pages).strip()
    if not full_text:
        raise ValueError(
            "Could not extract readable text from the PDF document. "
            "It may be empty, image-only (scanned), or password-protected."
        )
    return full_text
