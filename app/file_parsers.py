from io import BytesIO
from typing import Optional

import docx
import pdfplumber


def read_txt_file(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def read_docx_file(file_bytes: bytes) -> str:
    try:
        bio = BytesIO(file_bytes)
        document = docx.Document(bio)
        parts = []
        for para in document.paragraphs:
            txt = para.text.strip()
            if txt:
                parts.append(txt)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def read_pdf_file(file_bytes: bytes) -> str:
    try:
        bio = BytesIO(file_bytes)
        parts = []
        with pdfplumber.open(bio) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                txt = txt.strip()
                if txt:
                    parts.append(txt)
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


def load_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    name = (uploaded_file.name or "").lower()
    file_bytes = uploaded_file.read()

    if name.endswith(".pdf"):
        return read_pdf_file(file_bytes)
    if name.endswith(".docx"):
        return read_docx_file(file_bytes)
    if name.endswith(".txt"):
        return read_txt_file(file_bytes)

    return ""
