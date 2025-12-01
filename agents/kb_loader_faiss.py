# agents/kb_loader_faiss.py
"""
Embedding + FAISS KB loader.

Usage:
  kb = EmbeddingKB("data/kb.csv", index_path="data/faiss_index.idx", embed_model="all-MiniLM-L6-v2")
  kb.build_index()            # builds & saves index (slow first time)
  rows = kb.retrieve("tomato late blight", topk=5)

Requirements:
  pip install sentence-transformers faiss-cpu numpy pandas

Notes:
 - embed_model can be given as either "all-MiniLM-L6-v2" or
   "sentence-transformers/all-MiniLM-L6-v2".
 - The first build_index() run will download the embedding model and may take time.
"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import os

_HAS_ST = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
    _HAS_ST = False

_HAS_FAISS = True
try:
    import faiss
except Exception:
    faiss = None
    _HAS_FAISS = False


class EmbeddingKB:
    def __init__(
        self,
        csv_path: str,
        index_path: str = "data/faiss_index.idx",
        embed_model: str = "all-MiniLM-L6-v2",
        vector_dim: Optional[int] = None,
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"KB CSV not found: {csv_path}")
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.fillna("")

        # Combined text used for embedding
        cols = []
        for c in [
            "disease_name",
            "symptoms",
            "solution_text",
            "chemical_recommendations",
            "organic_recommendations",
            "notes",
        ]:
            if c in self.df.columns:
                cols.append(c)
        if not cols:
            cols = list(self.df.columns)
        # create combined field (use " . " as separator to keep sentences distinct)
        self.df["_combined"] = self.df[cols].agg(" . ".join, axis=1)

        self.index_path = Path(index_path)
        self.embed_model_id = embed_model
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self.vector_dim = vector_dim

    def _ensure_model(self):
        if self.model is None:
            if not _HAS_ST or SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is not installed. Install with: pip install sentence-transformers"
                )
            mid = self.embed_model_id
            # allow model id with or without 'sentence-transformers/' prefix
            if not mid.startswith("sentence-transformers/") and "/" not in mid:
                mid = f"sentence-transformers/{mid}"
            self.model = SentenceTransformer(mid)

    def build_index(self, force_rebuild: bool = False):
        """
        Compute embeddings for KB and build FAISS index. Save to disk for reuse.
        """
        if not _HAS_FAISS or faiss is None:
            raise RuntimeError("faiss (faiss-cpu) is not installed. Install with: pip install faiss-cpu")

        self._ensure_model()

        # try load existing index (unless force_rebuild)
        if self.index_path.exists() and not force_rebuild:
            try:
                self.faiss_index = faiss.read_index(str(self.index_path))
                emb_path = self.index_path.with_suffix(".npy")
                if emb_path.exists():
                    self.embeddings = np.load(str(emb_path))
                    self.vector_dim = self.embeddings.shape[1]
                return
            except Exception:
                # fall through to rebuild
                pass

        texts = self.df["_combined"].tolist()
        # encode -> numpy
        emb = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        if emb.ndim == 1:
            emb = emb.reshape(-1, emb.shape[0])

        emb = np.asarray(emb).astype("float32", copy=False)
        # normalize vectors for inner-product-based similarity (cosine)
        faiss.normalize_L2(emb)

        self.embeddings = emb
        self.vector_dim = emb.shape[1]

        # build FAISS index (inner-product on normalized vectors ~ cosine)
        index = faiss.IndexFlatIP(self.vector_dim)
        index.add(self.embeddings)
        self.faiss_index = index

        # persist index and embeddings
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(self.index_path))
        np.save(str(self.index_path.with_suffix(".npy")), self.embeddings)

    def load_index(self):
        if not _HAS_FAISS or faiss is None:
            raise RuntimeError("faiss (faiss-cpu) is not installed.")
        if not self.index_path.exists():
            raise FileNotFoundError("Index file not found; run build_index() first.")
        self.faiss_index = faiss.read_index(str(self.index_path))
        emb_path = self.index_path.with_suffix(".npy")
        if emb_path.exists():
            self.embeddings = np.load(str(emb_path))
            self.vector_dim = self.embeddings.shape[1]

    def _embed_query(self, q: str):
        self._ensure_model()
        vec = self.model.encode([q], convert_to_numpy=True)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        vec = np.asarray(vec).astype("float32", copy=False)
        faiss.normalize_L2(vec)
        return vec

    def retrieve(self, query: str, topk: int = 5) -> List[Dict]:
        """
        Return topk KB rows (as dicts) ranked by semantic similarity.
        """
        if self.faiss_index is None:
            # try to load index; if missing, build it
            try:
                self.load_index()
            except Exception:
                self.build_index()

        qv = self._embed_query(query)
        # Guard topk against dataset size
        k = min(int(topk), len(self.df))
        if k <= 0:
            return []

        D, I = self.faiss_index.search(qv, k)
        idxs = I[0].tolist()
        results: List[Dict] = []
        for i in idxs:
            if i < 0 or i >= len(self.df):
                continue
            row = self.df.iloc[int(i)].to_dict()
            results.append(row)
        return results

    def list_all(self):
        return list(self.df["_combined"].values)
