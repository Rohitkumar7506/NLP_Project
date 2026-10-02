"""
Backend: RAG engine for the Intelligent Study Assistant.

This module packages every step of the Jupyter notebook
(IntelligentStudyAssistant_Fixed.ipynb) into a reusable class:

  1. PDF text extraction, page by page          (notebook section 2)
  2. Text cleaning and tokenization             (notebook section 3)
  3. Overlapping chunking with page metadata    (notebook section 4)
  4. Sentence-transformer embeddings            (notebook section 5)
  5. FAISS inner-product vector index           (notebook section 6)
  6. Semantic retrieval                         (notebook section 7)
  7. TF-IDF retrieval baseline                  (notebook section 8)
  8. Retrieval evaluation (Precision@1)         (notebook section 9)
  9. Extractive answer generation               (notebook section 10)
"""

import os
import re
import time

import numpy as np
import pymupdf
import nltk
import faiss

from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEngine:
    """Encapsulates indexing, retrieval and extractive answering."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.pages = []
        self.chunks = []
        self.chunkTexts = []
        self.embeddings = None
        self.normalizedEmbeddings = None
        self.vectorIndex = None
        self.tfidfVectorizer = None
        self.tfidfMatrix = None
        self.tokenCount = 0
        self.rawTextLength = 0
        self.cleanTextLength = 0
        self.buildTime = None

    # ------------------------------------------------------------------
    # Indexing pipeline (notebook sections 2-6)
    # ------------------------------------------------------------------
    def buildFromPdf(self, pdfPath, chunkSize=800, overlap=200):
        if not os.path.exists(pdfPath):
            raise FileNotFoundError(
                f"PDF not found at {pdfPath}. Check the data/study_material folder."
            )
        startTime = time.time()

        # Section 2 - extract text page by page so sources keep page numbers
        pages = []
        with pymupdf.open(pdfPath) as document:
            for pageNumber, page in enumerate(document, start=1):
                pageText = page.get_text()
                if pageText.strip():
                    pages.append({"page": pageNumber, "text": pageText})

        if not pages:
            raise ValueError(
                "No selectable text was found in the PDF. "
                "If it is a scanned PDF, OCR is needed."
            )
        self.pages = pages

        # Section 3 - cleaning and tokenization
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        rawText = "\n".join(page["text"] for page in pages)
        cleanText = re.sub(r"\s+", " ", rawText).strip()
        tokens = word_tokenize(cleanText)

        self.rawTextLength = len(rawText)
        self.cleanTextLength = len(cleanText)
        self.tokenCount = len(tokens)

        # Section 4 - overlapping chunks with page metadata
        if chunkSize <= 0 or overlap < 0 or overlap >= chunkSize:
            raise ValueError("Use chunkSize > overlap >= 0.")

        chunks = []
        for page in pages:
            pageText = re.sub(r"\s+", " ", page["text"]).strip()
            start = 0
            while start < len(pageText):
                end = start + chunkSize
                chunkText = pageText[start:end].strip()
                if chunkText:
                    chunks.append({"page": page["page"], "text": chunkText})
                start += chunkSize - overlap

        if not chunks:
            raise ValueError("No text chunks were created.")

        self.chunks = chunks
        self.chunkTexts = [chunk["text"] for chunk in chunks]

        # Section 5 - embeddings
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

        self.embeddings = self.model.encode(
            self.chunkTexts, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")

        # Section 6 - FAISS index over L2-normalized vectors
        embeddingDimension = self.embeddings.shape[1]
        self.normalizedEmbeddings = self.embeddings.copy()
        faiss.normalize_L2(self.normalizedEmbeddings)

        self.vectorIndex = faiss.IndexFlatIP(embeddingDimension)
        self.vectorIndex.add(self.normalizedEmbeddings)

        # Section 8 - TF-IDF baseline
        self.tfidfVectorizer = TfidfVectorizer(stop_words="english")
        self.tfidfMatrix = self.tfidfVectorizer.fit_transform(self.chunkTexts)

        self.buildTime = time.time() - startTime
        return self

    # ------------------------------------------------------------------
    # Retrieval (notebook sections 7 and 8)
    # ------------------------------------------------------------------
    def semanticSearch(self, query, topK=3):
        """Return ranked dicts: {rank, score, page, text} (section 7)."""
        queryEmbedding = self.model.encode(
            [query], convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(queryEmbedding)

        k = min(topK, len(self.chunks))
        scores, indices = self.vectorIndex.search(queryEmbedding, k=k)

        results = []
        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
            chunk = self.chunks[int(index)]
            results.append({
                "rank": rank,
                "score": float(score),
                "page": chunk["page"],
                "text": chunk["text"],
            })
        return results

    def tfidfSearch(self, query, topK=3):
        """Return ranked dicts using the TF-IDF baseline (section 8)."""
        queryVector = self.tfidfVectorizer.transform([query])
        scores = cosine_similarity(queryVector, self.tfidfMatrix)[0]
        topIndices = scores.argsort()[::-1][:min(topK, len(self.chunks))]

        results = []
        for rank, index in enumerate(topIndices, start=1):
            chunk = self.chunks[int(index)]
            results.append({
                "rank": rank,
                "score": float(scores[index]),
                "page": chunk["page"],
                "text": chunk["text"],
            })
        return results

    # ------------------------------------------------------------------
    # Evaluation (notebook section 9)
    # ------------------------------------------------------------------
    @staticmethod
    def defaultTestQueries():
        return {
            "What is Retrieval-Augmented Generation?": [5],
            "What is the role of embeddings?": [4],
            "Why is text preprocessing needed?": [3],
            "How can retrieval quality be evaluated?": [6],
            "What are the limitations of a RAG system?": [6],
        }

    def evaluateRetrieval(self, relevantPages=None):
        """Compare TF-IDF vs FAISS Precision@1 on labelled questions."""
        if relevantPages is None:
            relevantPages = self.defaultTestQueries()

        rows = []
        for testQuery, expectedPages in relevantPages.items():
            tfidfScores = cosine_similarity(
                self.tfidfVectorizer.transform([testQuery]), self.tfidfMatrix
            )[0]
            tfidfIndex = int(tfidfScores.argmax())
            tfidfPage = self.chunks[tfidfIndex]["page"]

            queryEmbedding = self.model.encode(
                [testQuery], convert_to_numpy=True
            ).astype("float32")
            faiss.normalize_L2(queryEmbedding)
            faissScores, faissIndices = self.vectorIndex.search(queryEmbedding, k=1)
            faissIndex = int(faissIndices[0][0])
            faissPage = self.chunks[faissIndex]["page"]

            rows.append({
                "Question": testQuery,
                "Expected relevant page(s)": ", ".join(map(str, expectedPages)),
                "TF-IDF top page": tfidfPage,
                "TF-IDF hit": tfidfPage in expectedPages,
                "FAISS top page": faissPage,
                "FAISS hit": faissPage in expectedPages,
                "FAISS similarity": round(float(faissScores[0][0]), 4),
            })

        tfidfHits = [row["TF-IDF hit"] for row in rows]
        faissHits = [row["FAISS hit"] for row in rows]
        return {
            "rows": rows,
            "tfidfPrecisionAt1": float(np.mean(tfidfHits)) if rows else 0.0,
            "faissPrecisionAt1": float(np.mean(faissHits)) if rows else 0.0,
        }

    # ------------------------------------------------------------------
    # Extractive answer generation (notebook section 10)
    # ------------------------------------------------------------------
    @staticmethod
    def _cleanCandidateSentence(sentence):
        sentence = re.sub(r"^[\s•\-–]+", "", sentence).strip()
        sentence = re.sub(
            r"^NLP Study Notes\s*[—-]\s*Intelligent Study Assistant\s*Page\s*\d+\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        )
        headingPatterns = [
            r"^\d+\.\s*Introduction to NLP\s*",
            r"^\d+\.\s*Text Preprocessing\s*",
            r"^\d+\.\s*Text Representations and Embeddings\s*",
            r"^\d+\.\s*Retrieval-Augmented Generation\s*\(RAG\)\s*",
            r"^\d+\.\s*Evaluating a Study Assistant\s*",
            r"^Limitations and responsible use\s*",
            r"^Possible evaluation measures\s*",
            r"^Why preprocessing matters\s*",
            r"^Main stages of a RAG system\s*",
            r"^Cosine similarity\s*",
            r"^Common applications\s*",
            r"^Typical NLP pipeline\s*",
            r"^Learning objectives\s*",
        ]
        for pattern in headingPatterns:
            sentence = re.sub(pattern, "", sentence, flags=re.IGNORECASE).strip()

        # Remove a duplicated heading phrase repeated by PDF extraction.
        sentence = re.sub(
            r"^(.{8,75}?)\s+\1\b",
            r"\1",
            sentence,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", sentence).strip()

    def generateAnswer(self, query, topK=5):
        """Retrieve top passages, then pick the most relevant sentence."""
        if not query or not query.strip():
            return {"answer": "Please enter a question.",
                    "source_page": None, "retrieval_score": None}

        query = query.strip()
        queryEmbedding = self.model.encode(
            [query], convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(queryEmbedding)

        k = min(topK, len(self.chunks))
        retrievalScores, retrievalIndices = self.vectorIndex.search(queryEmbedding, k=k)

        candidates, candidatePages, candidateChunkScores = [], [], []
        for retrievalScore, chunkIndex in zip(retrievalScores[0], retrievalIndices[0]):
            chunk = self.chunks[int(chunkIndex)]
            sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])

            for sentence in sentences:
                sentence = self._cleanCandidateSentence(sentence)
                if len(sentence.split()) < 8:
                    continue
                if sentence.endswith(":") or "?" in sentence:
                    continue
                if re.fullmatch(r"[\W\d_]+", sentence):
                    continue
                candidates.append(sentence)
                candidatePages.append(chunk["page"])
                candidateChunkScores.append(float(retrievalScore))

        if not candidates:
            bestChunkIndex = int(retrievalIndices[0][0])
            return {
                "answer": self.chunks[bestChunkIndex]["text"],
                "source_page": self.chunks[bestChunkIndex]["page"],
                "retrieval_score": float(retrievalScores[0][0]),
            }

        sentenceEmbeddings = self.model.encode(
            candidates, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        faiss.normalize_L2(sentenceEmbeddings)

        semanticScores = sentenceEmbeddings @ queryEmbedding[0]

        lexicalVectorizer = TfidfVectorizer(stop_words="english")
        try:
            lexicalMatrix = lexicalVectorizer.fit_transform(candidates + [query])
            lexicalScores = cosine_similarity(
                lexicalMatrix[-1], lexicalMatrix[:-1]
            )[0]
        except ValueError:
            lexicalScores = np.zeros(len(candidates), dtype="float32")

        # Semantic relevance is the main signal; lexical overlap breaks close ties.
        combinedScores = 0.8 * semanticScores + 0.2 * lexicalScores
        bestIndex = int(np.argmax(combinedScores))

        return {
            "answer": candidates[bestIndex],
            "source_page": candidatePages[bestIndex],
            "retrieval_score": candidateChunkScores[bestIndex],
        }

    # ------------------------------------------------------------------
    # Similarity matrix for the heatmap (notebook section 13)
    # ------------------------------------------------------------------
    def queryChunkSimilarity(self, queryTexts):
        queryEmbeddings = self.model.encode(
            queryTexts, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(queryEmbeddings)
        return queryEmbeddings @ self.normalizedEmbeddings.T
