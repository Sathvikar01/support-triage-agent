import hashlib
import json
import logging
import pickle
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from corpus_loader import Document
from config import VECTOR_DB_DIR

logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CACHE_VERSION = "2026-05-02-v3-query-expansion"

QUERY_EXPANSION_MAP = {
    "stolen": "lost stolen card replacement",
    "card stolen": "lost stolen card replacement block",
    "identity stolen": "identity theft fraud unauthorized",
    "not working": "not working error broken issue",
    "can't access": "cannot access login blocked locked out",
    "can't login": "cannot login password reset access",
    "score": "score assessment test results grading",
    "refund": "refund billing payment charge money back",
    "delete account": "delete account remove close deactivate",
    "how to": "guide steps instructions tutorial",
}


class HybridRetriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.texts = [d.content for d in documents]
        self.tfidf = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = None
        self.embedder = None
        self.embeddings = None
        self.faiss_index = None
        self.reranker = None
        self.score_mode = "tfidf"
        self._company_indices: dict = {}

    def build(self):
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        faiss_path = VECTOR_DB_DIR / "index.faiss"
        tfidf_path = VECTOR_DB_DIR / "tfidf.pkl"
        embeddings_path = VECTOR_DB_DIR / "embeddings.npy"
        manifest_path = VECTOR_DB_DIR / "manifest.json"

        manifest = self._manifest()
        if self._cache_is_valid(manifest_path, manifest) and tfidf_path.exists():
            logger.info("Loading existing retrieval cache...")
            with open(tfidf_path, "rb") as f:
                tfidf_data = pickle.load(f)
                self.tfidf = tfidf_data["vectorizer"]
                self.tfidf_matrix = tfidf_data["matrix"]

            self._load_models()

            if HAS_FAISS and faiss_path.exists() and self.embedder:
                self.faiss_index = faiss.read_index(str(faiss_path))
                self.score_mode = "rrf"
                logger.info("Loaded FAISS index with %d vectors.", self.faiss_index.ntotal)
            elif embeddings_path.exists() and self.embedder:
                self.embeddings = np.load(embeddings_path)
                self.score_mode = "rrf"

            if self.reranker:
                self.score_mode = "rerank"

            self._build_company_index()
            logger.info("Retriever ready (cache, mode=%s).", self.score_mode)
            return

        for stale_path in (faiss_path, embeddings_path):
            if stale_path.exists():
                stale_path.unlink()

        logger.info("Building TF-IDF index from scratch...")
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)
        with open(tfidf_path, "wb") as f:
            pickle.dump({"vectorizer": self.tfidf, "matrix": self.tfidf_matrix}, f)
        logger.info("TF-IDF index built: %s", self.tfidf_matrix.shape)

        self._load_models()
        if self.embedder:
            logger.info("Encoding %d chunks with %s...", len(self.texts), EMBEDDING_MODEL)
            self.embeddings = self.embedder.encode(
                self.texts, show_progress_bar=True, batch_size=256, convert_to_numpy=True
            )
            self.embeddings = np.array(self.embeddings, dtype="float32")
            self._normalize(self.embeddings)

            if HAS_FAISS:
                dim = self.embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(self.embeddings)
                faiss.write_index(self.faiss_index, str(faiss_path))
                logger.info("FAISS index built with %d vectors, dim=%d", self.faiss_index.ntotal, dim)
            else:
                np.save(embeddings_path, self.embeddings)

            self.score_mode = "rerank" if self.reranker else "rrf"

        self._build_company_index()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Retriever ready (built, mode=%s).", self.score_mode)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        company: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        allowed_indices = self._indices_for_company(company)
        expanded_query = self._expand_query(query)
        tfidf_results = self._tfidf_search(expanded_query, top_k=50, allowed_indices=allowed_indices)
        embedding_results = self._embedding_search(expanded_query, top_k=50, allowed_indices=allowed_indices)

        if embedding_results:
            fused = self._rrf_fusion(tfidf_results, embedding_results, k=60)
        else:
            fused = tfidf_results

        reranked = self._rerank(query, fused[:30])
        boosted = self._metadata_boost(reranked, company)
        return boosted[:top_k]

    def estimate_confidence(self, results: List[Tuple[Document, float]]) -> float:
        if not results:
            return 0.0
        top_score = results[0][1]
        if self.score_mode == "rerank":
            return max(0.0, min(0.95, (top_score + 10.0) / 20.0))
        if self.score_mode == "rrf":
            return max(0.0, min(0.95, top_score / 0.03))
        return max(0.0, min(0.95, top_score / 0.25))

    def _expand_query(self, query: str) -> str:
        lowered = query.lower()
        expansions = []
        for key, expansion in QUERY_EXPANSION_MAP.items():
            if key in lowered:
                expansions.append(expansion)
        if expansions:
            return query + " " + " ".join(expansions)
        return query

    def _metadata_boost(
        self, results: List[Tuple[Document, float]], company: Optional[str]
    ) -> List[Tuple[Document, float]]:
        if not results or company in (None, "", "None", "Unknown"):
            return results
        boosted = []
        for doc, score in results:
            doc_company = doc.metadata.get("company", "")
            if doc_company == company:
                score *= 1.5
            boosted.append((doc, score))
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def _build_company_index(self):
        for i, doc in enumerate(self.documents):
            c = doc.metadata.get("company", "Unknown")
            self._company_indices.setdefault(c, set()).add(i)

    def _load_models(self):
        if not HAS_ST:
            logger.info("sentence-transformers unavailable; using TF-IDF-only retrieval.")
            return
        try:
            logger.info("Loading embeddings model: %s", EMBEDDING_MODEL)
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as exc:
            logger.warning("Embedding model unavailable (%s); using TF-IDF-only retrieval.", type(exc).__name__)
            self.embedder = None
            return

        try:
            logger.info("Loading reranker: %s", RERANKER_MODEL)
            self.reranker = CrossEncoder(RERANKER_MODEL)
        except Exception as exc:
            logger.warning("Reranker unavailable (%s); using fused retrieval scores.", type(exc).__name__)
            self.reranker = None

    def _indices_for_company(self, company: Optional[str]) -> Optional[set]:
        if company in (None, "", "None", "Unknown"):
            return None
        return self._company_indices.get(company) or None

    def _tfidf_search(
        self,
        query: str,
        top_k: int = 50,
        allowed_indices: Optional[set] = None,
    ) -> List[Tuple[int, float]]:
        q_vec = self.tfidf.transform([query])
        scores = (self.tfidf_matrix @ q_vec.T).toarray().flatten()
        if allowed_indices is not None:
            mask = np.ones(scores.shape, dtype=bool)
            mask[list(allowed_indices)] = False
            scores[mask] = 0.0
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def _embedding_search(
        self,
        query: str,
        top_k: int = 50,
        allowed_indices: Optional[set] = None,
    ) -> List[Tuple[int, float]]:
        if not self.embedder:
            return []

        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        q_emb = np.array(q_emb, dtype="float32")
        self._normalize(q_emb)

        if self.embeddings is not None:
            scores = (self.embeddings @ q_emb.T).flatten()
            if allowed_indices is not None:
                mask = np.ones(scores.shape, dtype=bool)
                mask[list(allowed_indices)] = False
                scores[mask] = -np.inf
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [(int(i), float(scores[i])) for i in top_idx if scores[i] > -np.inf]

        if self.faiss_index:
            search_k = top_k if allowed_indices is None else min(len(self.documents), max(top_k * 20, 200))
            D, I = self.faiss_index.search(q_emb, search_k)
            results = []
            for idx, score in zip(I[0], D[0]):
                idx = int(idx)
                if idx < 0:
                    continue
                if allowed_indices is not None and idx not in allowed_indices:
                    continue
                results.append((idx, float(score)))
                if len(results) >= top_k:
                    break
            return results

        return []

    def _rrf_fusion(
        self,
        tfidf_results: List[Tuple[int, float]],
        embedding_results: List[Tuple[int, float]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        rrf_scores = {}
        for rank, (idx, _) in enumerate(tfidf_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(embedding_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _rerank(self, query: str, candidates: List[Tuple[int, float]]) -> List[Tuple[Document, float]]:
        if not candidates:
            return []
        if not self.reranker:
            return [(self.documents[idx], score) for idx, score in candidates]

        pairs = [(query, self.documents[idx].content) for idx, _ in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in scored]

    def _manifest(self) -> dict:
        digest = hashlib.sha256()
        for doc in self.documents:
            meta = doc.metadata
            digest.update(str(meta.get("source", "")).encode("utf-8"))
            digest.update(str(meta.get("chunk_id", "")).encode("utf-8"))
            digest.update(doc.content.encode("utf-8", errors="ignore"))
        return {
            "cache_version": CACHE_VERSION,
            "document_count": len(self.documents),
            "corpus_hash": digest.hexdigest(),
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
        }

    def _cache_is_valid(self, manifest_path, expected: dict) -> bool:
        if not manifest_path.exists():
            return False
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return actual == expected

    def _normalize(self, vectors: np.ndarray):
        if HAS_FAISS:
            faiss.normalize_L2(vectors)
            return
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms
