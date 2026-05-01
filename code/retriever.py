import numpy as np
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from corpus_loader import Document

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_ST = True
except ImportError:
    HAS_ST = False


class HybridRetriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.texts = [d.content for d in documents]
        self.tfidf = TfidfVectorizer(max_features=50000, stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = None
        self.embedder = None
        self.embeddings = None
        self.faiss_index = None
        self.reranker = None
        self._use_embeddings = True
        self._use_reranker = True

    def build(self):
        print("Building TF-IDF index...")
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)
        print(f"TF-IDF index built: {self.tfidf_matrix.shape}")

        print("Loading all-MiniLM-L6-v2 embeddings model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"Encoding {len(self.texts)} chunks...")
        self.embeddings = self.embedder.encode(
            self.texts, show_progress_bar=True, batch_size=256, convert_to_numpy=True
        )
        self.embeddings = np.array(self.embeddings, dtype="float32")
        faiss.normalize_L2(self.embeddings)

        if HAS_FAISS:
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)
            print(f"FAISS index built with {self.faiss_index.ntotal} vectors, dim={dim}")

        print("Loading cross-encoder reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Retriever ready.")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        tfidf_results = self._tfidf_search(query, top_k=50)
        embedding_results = self._embedding_search(query, top_k=50)
        rrf_results = self._rrf_fusion(tfidf_results, embedding_results, k=60)
        top_rrf = rrf_results[:30]
        reranked = self._rerank(query, top_rrf)
        return reranked[:top_k]

    def _tfidf_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        q_vec = self.tfidf.transform([query])
        scores = (self.tfidf_matrix @ q_vec.T).toarray().flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def _embedding_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        q_emb = np.array(q_emb, dtype="float32")
        faiss.normalize_L2(q_emb)
        if self.faiss_index:
            D, I = self.faiss_index.search(q_emb, top_k)
            return [(int(I[0][i]), float(D[0][i])) for i in range(len(I[0]))]
        scores = (self.embeddings @ q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

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
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def _rerank(self, query: str, candidates: List[Tuple[int, float]]) -> List[Tuple[Document, float]]:
        if not candidates:
            return []
        pairs = [(query, self.documents[idx].content) for idx, _ in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in scored]
