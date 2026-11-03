"""Safe, in-memory extraction for supported document formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class DocumentParseError(ValueError):
    """Raised when a document cannot be parsed or is not supported."""


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    text: str
    page_count: int | None = None
    file_type: str = ""


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    """Parse a PDF, TXT, or DOCX byte payload without writing it to disk."""
    suffix = Path(filename).suffix.lower()
    if not data:
        raise DocumentParseError("The uploaded file is empty.")
    if suffix == ".pdf":
        return _parse_pdf(filename, data)
    if suffix == ".txt":
        return _parse_txt(filename, data)
    if suffix == ".docx":
        return _parse_docx(filename, data)
    raise DocumentParseError(
        f"Unsupported file type '{suffix or 'unknown'}'. Upload PDF, TXT, or DOCX."
    )


def _parse_pdf(filename: str, data: bytes) -> ParsedDocument:
    try:
        import pymupdf
    except ImportError as exc:
        raise DocumentParseError(
            "PDF support requires the PyMuPDF dependency. Run: "
            "python -m pip install PyMuPDF"
        ) from exc

    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentParseError(
            "This file is not a valid PDF or is damaged. Try opening it in Preview "
            "and exporting it again as a new PDF."
        ) from exc

    try:
        if pdf.needs_pass:
            raise DocumentParseError(
                "This PDF is password-protected. Unlock it in Preview or Adobe Acrobat "
                "and upload the unlocked copy."
            )
        text = "\n".join(page.get_text("text") for page in pdf)
        page_count = len(pdf)
    except Exception as exc:
        if isinstance(exc, DocumentParseError):
            raise
        raise DocumentParseError(
            "The PDF could not be read. It may contain damaged pages or unsupported "
            "content; try exporting a fresh copy from Preview."
        ) from exc
    finally:
        pdf.close()
    if not text.strip():
        raise DocumentParseError(
            "The PDF contains no extractable text. It may be scanned; OCR is required."
        )
    return ParsedDocument(filename, text, page_count, "pdf")


def _parse_txt(filename: str, data: bytes) -> ParsedDocument:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = data.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise DocumentParseError(
                "The text file is not valid UTF-8 or UTF-16 text."
            ) from exc
    if not text.strip():
        raise DocumentParseError("The text file is empty.")
    return ParsedDocument(filename, text, None, "txt")


def _parse_docx(filename: str, data: bytes) -> ParsedDocument:
    try:
        from io import BytesIO

        from docx import Document

        document = Document(BytesIO(data))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    except ImportError as exc:
        raise DocumentParseError(
            "DOCX support requires the optional 'python-docx' dependency."
        ) from exc
    except Exception as exc:
        raise DocumentParseError("The DOCX file is corrupted or cannot be read.") from exc
    if not text.strip():
        raise DocumentParseError("The DOCX file contains no readable paragraphs.")
    return ParsedDocument(filename, text, None, "docx")
