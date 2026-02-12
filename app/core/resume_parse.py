from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pathlib
from docx import Document  # python-docx


@dataclass
class ResumeContent:
    raw_text: str
    source: str  # "docx" | "text"


def extract_text_from_docx(path: str | pathlib.Path) -> str:
    doc = Document(str(path))
    parts = []

    # Extract paragraphs
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)

    # Extract tables if present
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                (cell.text or "").strip().replace("\n", " ")
                for cell in row.cells
            ).strip()
            if row_text:
                parts.append(row_text)

    return "\n".join(parts).strip()


def load_resume(docx_path: Optional[str], pasted_text: Optional[str]) -> ResumeContent:
    if docx_path:
        txt = extract_text_from_docx(docx_path)
        return ResumeContent(raw_text=txt, source="docx")

    if pasted_text and pasted_text.strip():
        return ResumeContent(raw_text=pasted_text.strip(), source="text")

    raise ValueError("Provide either a DOCX file or pasted résumé text.")
