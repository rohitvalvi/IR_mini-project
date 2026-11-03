import pytest

from modules.document_parser import DocumentParseError, parse_document


def test_parse_utf8_text():
    parsed = parse_document("notes.txt", "Hello café".encode("utf-8"))
    assert parsed.text == "Hello café"
    assert parsed.file_type == "txt"


def test_parse_empty_and_unsupported():
    with pytest.raises(DocumentParseError, match="empty"):
        parse_document("empty.txt", b"")
    with pytest.raises(DocumentParseError, match="Unsupported"):
        parse_document("image.png", b"data")


def test_parse_corrupt_pdf():
    with pytest.raises(DocumentParseError, match="damaged|PyMuPDF"):
        parse_document("broken.pdf", b"not a pdf")
