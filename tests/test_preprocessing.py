from modules.text_preprocessing import chunk_text, clean_text, split_sentences


def test_clean_text_preserves_unicode_and_removes_noise():
    assert clean_text("  Café\u0000  is\n\n useful. ") == "Café is\nuseful."


def test_sentence_conscious_chunking_covers_all_content():
    text = "First sentence. Second sentence! Third sentence? " * 10
    chunks = chunk_text(text, max_tokens=20)
    assert len(chunks) > 1
    assert "First sentence." in " ".join(chunks)
    assert "Third sentence?" in " ".join(chunks)
