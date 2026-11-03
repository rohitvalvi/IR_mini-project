from modules.summarizer import DocumentSummarizer, SummaryLength


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def __call__(self, text, **kwargs):
        return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

    def decode(self, output, skip_special_tokens=True):
        return "Generated summary."


class FakeTensor:
    def to(self, device):
        return self


class FakeModel:
    def generate(self, **kwargs):
        return [["summary"]]


def test_summarizer_processes_long_input_without_loading_real_model():
    summarizer = DocumentSummarizer(FakeTokenizer(), FakeModel())
    result = summarizer.summarize("Important sentence. " * 300, SummaryLength.SHORT)
    assert result == "Generated summary."
