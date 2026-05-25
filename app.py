import html
from io import BytesIO

import streamlit as st
from pypdf import PdfReader

from analyzer import load_models, analyze_document


def extract_pdf_text(uploaded_file):
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pages = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return "\n\n".join(pages)


def highlight_important_sentences(text, important_sentences):
    highlighted_text = ""
    last_end = 0

    for sentence in sorted(important_sentences, key=lambda item: item["start"]):
        highlighted_text += html.escape(text[last_end:sentence["start"]])
        highlighted_text += f"<mark>{html.escape(text[sentence['start']:sentence['end']])}</mark>"
        last_end = sentence["end"]

    highlighted_text += html.escape(text[last_end:])
    return highlighted_text.replace("\n", "<br>")


def build_report(results):
    topics = ", ".join([word for word, count in results["top_words"]])
    sentiment = results["sentiment"]

    return f"""# Document Analysis Report

## Stats
- Word count: {results["stats"]["word_count"]}
- Sentence count: {results["stats"]["sentence_count"]}
- Reading time: {results["stats"]["reading_time"]} minute(s)

## Sentiment
{sentiment["label"].title()} ({sentiment["confidence"]:.0%} confidence)

## Key Topics
{topics}

## Important Sentences
{results["spacy_summary"]}

## Transformer Summary
{results["transformer_summary"] or results["length_warning"] or "No transformer summary generated."}
"""


st.set_page_config(layout="wide")
st.title("document analyzer")

st.markdown(
    """
    <style>
        .sentiment-positive {
            color: green;
        }
        .sentiment-negative {
            color: red;
        }
        .sentiment-neutral {
            color: grey;
        }
    </style>
    """,
    unsafe_allow_html=True
)

#initialize models
nlp_model, tokenizer, model, sent_tok, sent_model = load_models()

#input ui widget
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
pdf_text = ""

if uploaded_file:
    try:
        pdf_text = extract_pdf_text(uploaded_file)
        if not pdf_text.strip():
            st.warning("Could not extract readable text from that PDF.")
    except Exception as exc:
        st.error(f"Could not read PDF: {exc}")

user_text = st.text_area("Paste your text here", value=pdf_text, height=300)

if st.button("analyze doc") and user_text:

    with st.spinner("analyzing docs"):
        results = analyze_document(user_text, nlp_model, tokenizer, model, sent_model, sent_tok)

        st.success("complete !")

    stat_cols = st.columns(3)
    stats = results["stats"]
    stat_cards = [
        ("Words", stats["word_count"]),
        ("Reading time", f"{stats['reading_time']} min"),
        ("Sentences", stats["sentence_count"])
    ]

    for col, (label, value) in zip(stat_cols, stat_cards):
        with col:
            st.metric(label, value)

    sentiment = results["sentiment"]
    st.subheader("sentiment")
    st.markdown(
        f"<span class='sentiment-{sentiment['color']}'><b>{sentiment['label'].title()}</b> ({sentiment['confidence']:.0%} confidence)</span>",
        unsafe_allow_html=True
    )

    st.subheader("key topics")
    for word, count in results["top_words"]:
        st.write(f"- **{word}**: {count} occurrences")

    st.subheader("important sentences in context")
    st.markdown(
        highlight_important_sentences(user_text, results["important_sentences"]),
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("spacy summary")
        st.write(results["spacy_summary"] or "Not enough text to create a summary.")

    with col2:
        st.subheader("transformer summary")
        if results["length_warning"]:
            st.warning(results["length_warning"])
        else:
            st.write(results["transformer_summary"])

    st.download_button(
        "download report",
        data=build_report(results),
        file_name="document_analysis_report.md",
        mime="text/markdown"
    )
