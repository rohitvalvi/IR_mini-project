from modules.evaluation import compression_percentage, tfidf_baseline, word_count


def test_metrics_and_baseline():
    original = "Cats are animals. Cats need food. Cats sleep often. Dogs bark."
    summary = "Cats are animals."
    assert word_count(original) == 11
    assert compression_percentage(original, summary) > 0
    assert "Cats" in tfidf_baseline(original, sentence_count=2)
