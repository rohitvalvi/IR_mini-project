# AI-Based Document Summarization System

A beginner-friendly college mini-project that summarizes PDF, TXT, and DOCX documents
locally with Streamlit, PyMuPDF, Hugging Face Transformers, and PyTorch.

## Features

- In-memory PDF/TXT/DOCX extraction with validation and clear errors.
- Unicode-safe cleanup and tokenizer-aware, sentence-conscious chunking.
- Abstractive summarization using `sshleifer/distilbart-cnn-12-6`.
- Hierarchical summarization: every chunk is summarized, then partial summaries are
  summarized until one result remains. Long input is never silently truncated.
- Short, medium, and detailed modes; word counts, compression, and elapsed time.
- TXT download, model caching, MPS acceleration on Apple Silicon, and a simple TF-IDF
  extractive baseline for academic comparison.

## Installation on macOS (M2)

```bash
cd /Users/rohitvalvi/Desktop/IR_Mini_Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

The first summary request downloads model files from Hugging Face. Model files are
cached by the Transformers library and are not committed to this repository. On
Apple Silicon, PyTorch uses the MPS backend when available; CPU remains supported.

## Architecture and data flow

```mermaid
flowchart LR
  U[Streamlit upload] --> P[document_parser]
  P --> C[text_preprocessing]
  C --> S[Tokenizer-aware chunks]
  S --> M[DistilBART model]
  M --> H[Hierarchical reduction]
  H --> E[evaluation metrics]
  E --> V[Summary, metrics, TXT download]
```

The interface keeps uploaded bytes in memory. The app does not write source documents
to a database or application folder. Hugging Face's model cache is separate and
contains model weights, not user documents.

## Project structure

```text
app.py
modules/
  document_parser.py
  evaluation.py
  summarizer.py
  text_preprocessing.py
tests/
sample_documents/
requirements.txt
```

## Troubleshooting

- **Model download fails:** check internet access and run
  `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')"`
  after activating `.venv`.
- **MPS memory error:** close other applications or run with CPU by changing the
  `summarizer._device()` return value to `"cpu"`.
- **Scanned PDF:** PyMuPDF can read embedded text but does not perform OCR. OCR the PDF
  first (for example with macOS Preview or an approved OCR tool), then upload it.
- **DOCX error:** confirm `python-docx` is installed and the file is not password-protected.

## Testing and evaluation

Run `python -m pytest -q`. Parser and preprocessing tests do not download a model;
the summarizer test uses fake tokenizer/model objects. For a report, provide reference
summaries and call `modules.evaluation.rouge_scores(reference, generated)` to calculate
ROUGE-1, ROUGE-2, and ROUGE-L F1. Compression and processing time are application
metrics, not measures of factual quality.

## Limitations and ethics

The selected model is a general news summarizer and may not be optimal for every domain.
Generated text can omit important information or contain inaccuracies. Do not use it as
the sole basis for medical, legal, financial, or academic claims. The project does not
train a model and has no paid API requirement. Very large documents can take substantial
time and memory because every chunk is processed.

## GitHub publishing

```bash
git init
git add app.py modules tests sample_documents requirements.txt README.md .gitignore
git commit -m "Build document summarization mini-project"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository>.git
git push -u origin main
```

Do not commit `.venv`, model caches, private documents, secrets, or generated results.

## Verifiable references

1. Lewis et al., “BART: Denoising Sequence-to-Sequence Pre-training for Natural Language
   Generation, Translation, and Comprehension,” ACL 2020,
   https://aclanthology.org/2020.acl-main.703/
2. Wolf et al., “Transformers: State-of-the-Art Natural Language Processing,” EMNLP 2020,
   https://aclanthology.org/2020.emnlp-demos.6/
3. Lin, “ROUGE: A Package for Automatic Evaluation of Summaries,” 2004,
   https://aclanthology.org/W04-1013/
4. PyMuPDF documentation, https://pymupdf.readthedocs.io/
5. Streamlit documentation, https://docs.streamlit.io/
