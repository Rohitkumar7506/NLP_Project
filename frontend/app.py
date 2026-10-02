"""
Frontend: Streamlit web app for the Intelligent Study Assistant (RAG).

Run from the project root:
    streamlit run frontend/app.py

The UI mirrors the Jupyter notebook
(IntelligentStudyAssistant_Fixed.ipynb):
  - Home            : project overview
  - Ask a Question  : extractive Q&A over the PDF (notebook section 10)
  - Retrieve        : semantic FAISS vs TF-IDF retrieval (sections 7-8)
  - Evaluate        : Precision@1 comparison + charts (sections 9, 12-13)
  - Data & Chunks   : extracted pages, chunks and text stats (sections 2-4)
"""

import os
import sys
import tempfile

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Make the backend package importable when running from anywhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.rag_engine import RAGEngine  # noqa: E402

DEFAULT_PDF = os.path.join(
    PROJECT_ROOT, "data", "study_material", "NLP_Study_Notes_for_RAG_Test.pdf"
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

st.set_page_config(
    page_title="Intelligent Study Assistant (RAG)",
    page_icon="📚",
    layout="wide",
)


# ----------------------------------------------------------------------
# Cached engine builder (backend call from the frontend)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def loadEngine(pdfPath, modelSignature, chunkSize, overlap):
    """Build the RAG index once per PDF / chunking configuration."""
    engine = RAGEngine()
    engine.buildFromPdf(pdfPath, chunkSize=chunkSize, overlap=overlap)
    return engine


def resolvePdf():
    """Return a filesystem path for the selected PDF (default or upload)."""
    uploaded = st.sidebar.file_uploader("Upload a study PDF", type=["pdf"])
    if uploaded is not None:
        fd, tmpPath = tempfile.mkstemp(suffix=".pdf", prefix=uploaded.name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(uploaded.getbuffer())
        return tmpPath, uploaded.name, True

    if os.path.exists(DEFAULT_PDF):
        return DEFAULT_PDF, os.path.basename(DEFAULT_PDF), False

    st.error("No PDF found. Upload one in the sidebar or restore the default file.")
    st.stop()


# ----------------------------------------------------------------------
# Sidebar: configuration + build index
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

pdfPath, pdfName, isUpload = resolvePdf()

chunkSize = st.sidebar.number_input("Chunk size (characters)", 200, 2000, 800, step=100)
overlap = st.sidebar.number_input("Chunk overlap (characters)", 0, 1000, 200, step=50)

st.sidebar.caption(
    "Model: `all-MiniLM-L6-v2` · Index: FAISS `IndexFlatIP` (cosine similarity)"
)

with st.spinner("Building embeddings and FAISS index… (first run downloads the model)"):
    try:
        engine = loadEngine(pdfPath, RAGEngine().model_name, int(chunkSize), int(overlap))
    except Exception as error:
        st.error(f"Failed to build the index: {error}")
        st.stop()

st.sidebar.success(
    f"✅ Indexed '{pdfName}'\n\n"
    f"{len(engine.pages)} pages · {len(engine.chunks)} chunks · "
        f"{engine.embeddings.shape[1]}-D vectors"
)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "💬 Ask a Question", "🔎 Retrieve Passages",
     "📊 Evaluate & Visualize", "📄 Data & Chunks"],
)

# ----------------------------------------------------------------------
# Page: Home
# ----------------------------------------------------------------------
if page == "🏠 Home":
    st.title("📚 Intelligent Study Assistant")
    st.caption(
        "BCA 5th Semester NLP Project — Retrieval-Augmented Generation (RAG) "
        "over PDF study material, built with Streamlit (frontend) and a "
        "FAISS / sentence-transformers backend."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pages indexed", len(engine.pages))
    col2.metric("Text chunks", len(engine.chunks))
    col3.metric("Tokens", f"{engine.tokenCount:,}")
    col4.metric("Index build time", f"{engine.buildTime:.2f} s")

    st.markdown("### How it works")
    st.markdown(
        """
1. **Extract** — text is read from the PDF page by page so every answer keeps its source page.
2. **Chunk** — each page is split into overlapping chunks (size / stride configurable in the sidebar).
3. **Embed** — `all-MiniLM-L6-v2` maps chunks to dense vectors; similar meanings sit close in vector space.
4. **Index** — L2-normalized vectors are stored in a FAISS inner-product index (inner product = cosine similarity).
5. **Retrieve** — a question is embedded and the top passages are ranked by similarity.
6. **Answer** — no generative model is used: the most relevant *complete sentence* from the retrieved
   passages is returned, grounded in the PDF with its source page.
        """
    )

    st.info(
        "👉 Go to **Ask a Question** in the sidebar to start studying, or open "
        "**Evaluate & Visualize** to compare TF-IDF against semantic retrieval."
    )

# ----------------------------------------------------------------------
# Page: Ask a Question
# ----------------------------------------------------------------------
elif page == "💬 Ask a Question":
    st.title("💬 Ask a Question")
    st.caption(
        "Extractive answering (notebook section 10): the assistant retrieves the top "
        "passages and selects the single most relevant sentence from the PDF."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    preset = st.selectbox(
        "Or pick a sample question",
        ["— Type your own question below —"] + list(RAGEngine.defaultTestQueries().keys()),
    )
    userQuery = st.text_input("Your question", key="queryInput")

    if preset != "— Type your own question below —":
        st.session_state.queryInput = preset

    def ask(query):
        result = engine.generateAnswer(query)
        st.session_state.chat_history.append(
            {"role": "user", "text": query}
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "text": result["answer"],
                "page": result["source_page"],
                "score": result["retrieval_score"],
            }
        )

    if st.button("Ask 🚀", type="primary") or (preset != "— Type your own question below —"):
        query = userQuery.strip() if userQuery.strip() else (
            preset if preset != "— Type your own question below —" else ""
        )
        if query:
            ask(query)
        else:
            st.warning("Please enter a question first.")

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["text"])
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**{message['text']}**")
                meta = []
                if message["page"]:
                    meta.append(f"📄 Source: **page {message['page']}** of `{pdfName}`")
                if message["score"] is not None:
                    meta.append(f"🎯 Retrieval similarity: **{message['score']:.4f}**")
                if meta:
                    st.caption(" · ".join(meta))

    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

# ----------------------------------------------------------------------
# Page: Retrieve Passages
# ----------------------------------------------------------------------
elif page == "🔎 Retrieve Passages":
    st.title("🔎 Compare Retrieval Methods")
    st.caption(
        "Notebook sections 7–8: semantic search (embeddings + FAISS) versus the "
        "TF-IDF word-overlap baseline, side by side."
    )

    query = st.text_input("Search query", value="What is Retrieval-Augmented Generation?")
    topK = st.slider("Number of results (top-K)", 1, 10, 3)

    if query.strip():
        faissResults = engine.semanticSearch(query, topK=topK)
        tfidfResults = engine.tfidfSearch(query, topK=topK)

        left, right = st.columns(2)

        with left:
            st.subheader("🧠 Embeddings + FAISS (semantic)")
            for hit in faissResults:
                with st.container(border=True):
                    st.markdown(
                        f"**Result {hit['rank']}** · Similarity: `{hit['score']:.4f}` · "
                        f"Page **{hit['page']}**"
                    )
                    st.write(hit["text"][:400] + ("…" if len(hit["text"]) > 400 else ""))

        with right:
            st.subheader("🔤 TF-IDF (lexical baseline)")
            for hit in tfidfResults:
                with st.container(border=True):
                    st.markdown(
                        f"**Result {hit['rank']}** · Score: `{hit['score']:.4f}` · "
                        f"Page **{hit['page']}**"
                    )
                    st.write(hit["text"][:400] + ("…" if len(hit["text"]) > 400 else ""))

        st.caption(
            "Similarity scores are a ranking signal, not proof that a passage fully "
            "answers the question."
        )

# ----------------------------------------------------------------------
# Page: Evaluate & Visualize
# ----------------------------------------------------------------------
elif page == "📊 Evaluate & Visualize":
    st.title("📊 Evaluation & Visualization")
    st.caption(
        "Notebook sections 9, 12–13: Precision@1 on five labelled test questions, "
        "plus the comparison bar chart and the query-to-chunk similarity heatmap. "
        "Expected page labels are manual relevance judgements for the six-page test PDF."
    )

    if st.button("▶️ Run evaluation", type="primary"):
        with st.spinner("Evaluating both retrieval methods…"):
            evaluation = engine.evaluateRetrieval()
            st.session_state.evaluation = evaluation

    evaluation = st.session_state.get("evaluation")

    if evaluation:
        evalFrame = pd.DataFrame(evaluation["rows"])

        col1, col2 = st.columns(2)
        col1.metric("TF-IDF Precision@1", f"{evaluation['tfidfPrecisionAt1']:.0%}")
        col2.metric("Embeddings + FAISS Precision@1", f"{evaluation['faissPrecisionAt1']:.0%}")

        st.dataframe(evalFrame, use_container_width=True, hide_index=True)

        # Section 12 - retrieval methods comparison bar chart
        st.subheader("Retrieval Methods Comparison")
        figBar, axBar = plt.subplots(figsize=(7, 4))
        methods = ["TF-IDF", "Embeddings + FAISS"]
        precisionScores = [evaluation["tfidfPrecisionAt1"], evaluation["faissPrecisionAt1"]]
        bars = axBar.bar(methods, precisionScores, color=["#f4a261", "#2a9d8f"])
        axBar.set_ylabel("Precision@1")
        axBar.set_ylim(0, 1)
        for bar, score in zip(bars, precisionScores):
            axBar.text(
                bar.get_x() + bar.get_width() / 2,
                min(score + 0.03, 0.97),
                f"{score:.0%}",
                ha="center",
            )
        figBar.tight_layout()
        st.pyplot(figBar)

        # Section 13 - query-to-chunk similarity heatmap
        st.subheader("Query-to-Document Chunk Similarity")
        queryTexts = list(RAGEngine.defaultTestQueries().keys())
        similarityMatrix = engine.queryChunkSimilarity(queryTexts)

        figHeat, axHeat = plt.subplots(figsize=(14, 6))
        image = axHeat.imshow(similarityMatrix, aspect="auto", cmap="YlGnBu")
        axHeat.set_xlabel("Document chunks")
        axHeat.set_xticks(range(len(engine.chunks)))
        axHeat.set_xticklabels(
            [f"Chunk {i + 1}" for i in range(len(engine.chunks))], rotation=90
        )
        axHeat.set_yticks(range(len(queryTexts)))
        axHeat.set_yticklabels(queryTexts, fontsize=8)
        figHeat.colorbar(image, ax=axHeat, label="Cosine similarity")
        figHeat.tight_layout()
        st.pyplot(figHeat)

        # Section 14 - save results
        st.subheader("Download results")
        csvEval = evalFrame.to_csv(index=False).encode()
        st.download_button(
            "⬇️ retrieval_evaluation.csv", csvEval,
            file_name="retrieval_evaluation.csv", mime="text/csv",
        )

        answerRows = []
        for testQuery in queryTexts:
            result = engine.generateAnswer(testQuery)
            answerRows.append({
                "Question": testQuery,
                "Extractive answer": result["answer"],
                "Source page": result["source_page"],
                "Retrieved chunk score": (
                    round(result["retrieval_score"], 4)
                    if result["retrieval_score"] is not None else None
                ),
            })
        answerFrame = pd.DataFrame(answerRows)
        st.dataframe(answerFrame, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ answer_generation_results.csv",
            answerFrame.to_csv(index=False).encode(),
            file_name="answer_generation_results.csv", mime="text/csv",
        )

        # Also persist to results/ like the notebook does
        if st.button("💾 Save CSVs to results/ folder"):
            os.makedirs(RESULTS_DIR, exist_ok=True)
            evalFrame.to_csv(os.path.join(RESULTS_DIR, "retrieval_evaluation.csv"), index=False)
            answerFrame.to_csv(os.path.join(RESULTS_DIR, "answer_generation_results.csv"), index=False)
            st.success(f"Saved to `{os.path.relpath(RESULTS_DIR, PROJECT_ROOT)}/`")
    else:
        st.info("Click **Run evaluation** to compute the metrics and charts.")

# ----------------------------------------------------------------------
# Page: Data & Chunks
# ----------------------------------------------------------------------
elif page == "📄 Data & Chunks":
    st.title("📄 Extracted Data & Chunks")
    st.caption("Notebook sections 2–4: what the pipeline actually reads from the PDF.")

    tabStats, tabPages, tabChunks = st.tabs(["📈 Text statistics", "📑 Pages", "🧩 Chunks"])

    with tabStats:
        col1, col2, col3 = st.columns(3)
        col1.metric("Original text length", f"{engine.rawTextLength:,} chars")
        col2.metric("Clean text length", f"{engine.cleanTextLength:,} chars")
        col3.metric("Token count", f"{engine.tokenCount:,}")
        st.write("")
        st.metric("Embedding matrix shape", str(engine.embeddings.shape))
        st.metric("FAISS vectors stored", engine.vectorIndex.ntotal)
        st.metric("Vector dimensions", engine.vectorIndex.d)

    with tabPages:
        for pageEntry in engine.pages:
            with st.expander(f"Page {pageEntry['page']}"):
                st.text(pageEntry["text"][:1500])

    with tabChunks:
        chunkFrame = pd.DataFrame(
            {
                "Chunk #": range(1, len(engine.chunks) + 1),
                "Source page": [c["page"] for c in engine.chunks],
                "Length (chars)": [len(c["text"]) for c in engine.chunks],
                "Preview": [c["text"][:120] + "…" for c in engine.chunks],
            }
        )
        st.dataframe(chunkFrame, use_container_width=True, hide_index=True)

        selected = st.selectbox(
            "Inspect a chunk", chunkFrame["Chunk #"].tolist(),
            format_func=lambda n: f"Chunk {n} (page {engine.chunks[n - 1]['page']})",
        )
        st.text(engine.chunks[selected - 1]["text"])
