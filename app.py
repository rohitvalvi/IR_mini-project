"""Streamlit entry point for the AI-Based Document Summarization System."""

from __future__ import annotations

import time

import streamlit as st

from modules.document_parser import DocumentParseError, parse_document
from modules.evaluation import compression_percentage, word_count
from modules.summarizer import SummaryLength, SummarizationError, get_summarizer
from modules.text_preprocessing import clean_text

st.set_page_config(
    page_title="AI Document Summarizer",
    page_icon="📝",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1200px; padding-top: 2rem; }
    .metric-card { padding: 1rem; border-radius: .6rem; background: #f5f7fb; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_summarizer():
    """Load one model instance per Streamlit process."""
    return get_summarizer()


def main() -> None:
    st.title("📝 AI-Based Document Summarization System")
    st.caption(
        "Private, local document summarization with a pretrained abstractive model. "
        "Uploaded bytes are processed in memory and are not stored by this app."
    )

    with st.sidebar:
        st.header("Summary settings")
        length = st.selectbox(
            "Summary length",
            options=[SummaryLength.SHORT, SummaryLength.MEDIUM, SummaryLength.DETAILED],
            format_func=lambda item: item.value.title(),
            index=1,
        )
        st.info(
            "The first run downloads the model from Hugging Face and can take a few "
            
        )
        st.markdown("**Supported:** PDF, TXT, and DOCX")

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx"],
        help="Maximum size: 25 MB. Scanned PDFs need OCR and are reported clearly.",
    )
    if uploaded is None:
        st.markdown(
            "### How it works\n"
            "1. Upload a document  ·  2. Extract and clean text  ·  "
            "3. Summarize every chunk  ·  4. Download the result"
        )
        return

    if uploaded.size > 25 * 1024 * 1024:
        st.error("This file is larger than the 25 MB safety limit.")
        return

    try:
        document = parse_document(uploaded.name, uploaded.getvalue())
    except DocumentParseError as exc:
        st.error(str(exc))
        return

    cleaned = clean_text(document.text)
    if not cleaned:
        st.error(
            "No usable text was found. This may be an empty or scanned PDF. "
            "Run OCR on the PDF and upload the searchable version."
        )
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Extracted text preview")
        st.text_area("Preview", cleaned[:5000], height=260, label_visibility="collapsed")
        st.caption(f"{document.page_count or 1} page(s) · {word_count(cleaned):,} words")
    with right:
        st.subheader("Generated summary")
        if st.button("Generate summary", type="primary", use_container_width=True):
            started = time.perf_counter()
            try:
                summarizer = load_summarizer()
                with st.status("Summarizing the complete document...", expanded=True) as status:
                    summary = summarizer.summarize(cleaned, length)
                    status.update(label="Summary ready", state="complete")
            except SummarizationError as exc:
                st.error(str(exc))
                return
            except Exception:
                st.error(
                    "The model could not complete this request. Check your internet "
                    "connection/model cache and try again."
                )
                return

            elapsed = time.perf_counter() - started
            st.session_state["last_summary"] = summary
            st.session_state["last_metrics"] = (
                word_count(cleaned),
                word_count(summary),
                compression_percentage(cleaned, summary),
                elapsed,
            )

        summary = st.session_state.get("last_summary")
        if summary:
            st.text_area("Summary", summary, height=260, label_visibility="collapsed")
            original, summarized, compression, elapsed = st.session_state["last_metrics"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Original words", f"{original:,}")
            m2.metric("Summary words", f"{summarized:,}")
            m3.metric("Compression", f"{compression:.1f}%")
            m4.metric("Processing time", f"{elapsed:.1f}s")
            st.download_button(
                "Download summary as TXT",
                data=summary.encode("utf-8"),
                file_name=f"{uploaded.name.rsplit('.', 1)[0]}_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

    with st.expander("Important limitation"):
        st.write(
            "Abstractive summaries may omit details or contain inaccuracies. "
            "Review the generated text before using it for academic, legal, or "
            "medical decisions."
        )


if __name__ == "__main__":
    main()