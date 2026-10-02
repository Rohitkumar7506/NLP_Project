# 📚 Intelligent Study Assistant — Streamlit Web App (Frontend + Backend)

BCA 5th Semester NLP Project: a Retrieval-Augmented Generation (RAG) study assistant
built from `IntelligentStudyAssistant_Fixed.ipynb`, converted into a proper
**frontend (Streamlit)** + **backend (Python RAG engine)** web application.

## Project structure

```
├── backend/                  # BACKEND — all NLP / RAG logic
│   ├── __init__.py
│   └── rag_engine.py         # RAGEngine class: PDF extraction → chunking →
│                             # embeddings → FAISS index → retrieval → evaluation
│                             # → extractive answer generation
├── frontend/                 # FRONTEND — Streamlit web app
│   └── app.py                # UI pages calling the backend engine
├── data/study_material/      # Input PDF(s)
├── results/                  # Evaluation CSVs (saved from the app or notebook)
├── src/pdf_reader.py         # Original helper module
├── IntelligentStudyAssistant_Fixed.ipynb  # Source notebook
└── requirements.txt
```

## Notebook → App mapping

| Notebook section            | Backend method (`rag_engine.py`)     | Frontend page (`app.py`)        |
|-----------------------------|--------------------------------------|---------------------------------|
| 2. Extract text from PDF    | `buildFromPdf()`                     | Data & Chunks → Pages           |
| 3. Cleaning & tokenization  | `buildFromPdf()` (stats attributes)  | Home / Data & Chunks metrics    |
| 4. Overlapping chunking     | `buildFromPdf(chunkSize, overlap)`   | Sidebar chunk-size controls     |
| 5. Embeddings (MiniLM)      | `buildFromPdf()`                     | — (automatic on build)          |
| 6. FAISS vector index       | `buildFromPdf()`                     | Data & Chunks stats             |
| 7. Semantic retrieval       | `semanticSearch()`                   | Retrieve Passages               |
| 8. TF-IDF baseline          | `tfidfSearch()`                      | Retrieve Passages               |
| 9. Retrieval comparison     | `evaluateRetrieval()`                | Evaluate & Visualize            |
| 10. Extractive answering    | `generateAnswer()`                   | Ask a Question (chat UI)        |
| 12. Bar chart               | evaluation metrics                   | Evaluate & Visualize            |
| 13. Similarity heatmap      | `queryChunkSimilarity()`             | Evaluate & Visualize            |
| 14. Save results            | —                                    | Download buttons + save to `results/` |

## Run the website

```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

Then open http://localhost:8501

### Pages
- **🏠 Home** — project overview and live indexing stats
- **💬 Ask a Question** — chat-style extractive Q&A; every answer shows its source page and similarity score
- **🔎 Retrieve Passages** — side-by-side Embeddings+FAISS vs TF-IDF results for any query
- **📊 Evaluate & Visualize** — Precision@1 table, comparison bar chart, query-to-chunk heatmap, CSV download/save
- **📄 Data & Chunks** — text statistics, extracted pages, and the chunk table

You can also upload a different study PDF from the sidebar; the index rebuilds automatically.

> Note: the first run downloads the `all-MiniLM-L6-v2` sentence-transformer model.
