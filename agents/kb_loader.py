# agents/kb_loader.py
"""
Simple KB loader + retriever (TF-IDF fallback).

Usage:
    kb = KBLoader("data/kb.csv")
    rows = kb.retrieve_solutions("apple_apple_scab", topk=3)

Expected CSV columns (recommended):
  - disease_id (or disease_label)
  - disease_name
  - crop
  - symptoms
  - solution_text
  - chemical_recommendations
  - organic_recommendations
  - notes
"""

import os
import json
import re
from typing import List, Dict, Optional
from pathlib import Path

import pandas as pd

# Try to use scikit-learn's TF-IDF + cosine if available
_HAS_SKLEARN = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None
    _HAS_SKLEARN = False


def _to_json_serializable(v):
    """Convert numpy / pandas scalar types to Python native types for JSON serialization."""
    try:
        # pandas / numpy scalars have .item()
        if hasattr(v, "item"):
            return v.item()
        # else leave as-is
        return v
    except Exception:
        try:
            return str(v)
        except Exception:
            return None


def _simple_tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer returning lowercase tokens."""
    if not isinstance(text, str):
        return []
    # remove punctuation (keep internal hyphens), split by whitespace
    text = re.sub(r"[^\w\s\-]", " ", text)
    toks = [t.strip().lower() for t in text.split() if t.strip()]
    return toks


class KBLoader:
    """
    Loads a CSV KB containing disease -> solution entries and provides simple retrieval.

    Methods:
      - retrieve_solutions(disease_label, topk=3) -> List[Dict] (JSON-serializable dicts)
      - list_all_diseases() -> List[str]
    """

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"KB CSV not found: {self.csv_path}")

        # load dataframe
        self.df = pd.read_csv(self.csv_path, dtype=str).fillna("")

        # normalize column names (strip and lowercase to ease lookups)
        new_cols = {c: c.strip() for c in self.df.columns}
        self.df.rename(columns=new_cols, inplace=True)
        self.df.columns = [c.lower() for c in self.df.columns]

        # Build a combined text field for retrieval from the most useful columns
        text_cols = []
        for col in ["disease_name", "symptoms", "solution_text", "notes", "chemical_recommendations", "organic_recommendations"]:
            if col in self.df.columns:
                text_cols.append(col)
        if not text_cols:
            # fallback: use all columns
            text_cols = list(self.df.columns)

        # Ensure column exists; fillna done above
        self.df["_rb_combined"] = self.df[text_cols].agg(" . ".join, axis=1).astype(str)

        # TF-IDF retriever (optional)
        self.vectorizer = None
        self.tfidf_matrix = None
        if _HAS_SKLEARN and TfidfVectorizer is not None:
            try:
                self.vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
                self.tfidf_matrix = self.vectorizer.fit_transform(self.df["_rb_combined"].values)
            except Exception:
                self.vectorizer = None
                self.tfidf_matrix = None

    def _simple_overlap_scores(self, query: str) -> List[float]:
        """Fallback scoring by token overlap ratio."""
        q_tokens = set(_simple_tokenize(query))
        scores = []
        for txt in self.df["_rb_combined"].values:
            tset = set(_simple_tokenize(str(txt)))
            if not tset:
                scores.append(0.0)
            else:
                inter = q_tokens.intersection(tset)
                # normalized by average token count to avoid bias to long docs
                denom = (len(tset) + len(q_tokens) + 1e-8) / 2.0
                scores.append(len(inter) / denom)
        return scores

    def retrieve_solutions(self, disease_label: str, topk: int = 3) -> List[Dict]:
        """
        Retrieve top-k KB rows that match the disease_label.

        Behavior:
          1) Try exact matching on 'disease_id' (preferred) or 'disease_name'.
          2) If no exact hit, perform TF-IDF similarity over combined text (if sklearn available).
          3) Otherwise fallback to simple token-overlap retrieval.

        Returns a list of JSON-serializable dicts (keys are original CSV columns).
        """
        if not disease_label:
            return []

        # clamp topk
        try:
            k = max(1, int(topk))
        except Exception:
            k = 3

        # Exact match on disease_id or disease_label (case-insensitive)
        candidates = None
        label_norm = str(disease_label).strip().lower()

        if "disease_id" in self.df.columns:
            exact_mask = self.df["disease_id"].astype(str).str.strip().str.lower() == label_norm
            exact = self.df[exact_mask]
            if not exact.empty:
                candidates = exact

        if (candidates is None or candidates.empty) and "disease_label" in self.df.columns:
            exact_mask = self.df["disease_label"].astype(str).str.strip().str.lower() == label_norm
            exact = self.df[exact_mask]
            if not exact.empty:
                candidates = exact

        if (candidates is None or candidates.empty) and "disease_name" in self.df.columns:
            exact_mask = self.df["disease_name"].astype(str).str.strip().str.lower() == label_norm
            exact = self.df[exact_mask]
            if not exact.empty:
                candidates = exact

        if candidates is not None and not candidates.empty:
            # return up to topk exact matches (converted to serializable dicts)
            out = []
            for _, row in candidates.head(k).iterrows():
                d = {str(col): _to_json_serializable(val) for col, val in row.to_dict().items()}
                out.append(d)
            return out

        # Fallback: use TF-IDF if available
        query = str(disease_label).strip()
        if _HAS_SKLEARN and (self.vectorizer is not None) and (self.tfidf_matrix is not None):
            try:
                q_vec = self.vectorizer.transform([query])
                scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
                idxs = scores.argsort()[::-1][:k]
                out = []
                for i in idxs:
                    row = self.df.iloc[int(i)]
                    out.append({str(col): _to_json_serializable(val) for col, val in row.to_dict().items()})
                return out
            except Exception:
                # fallback to overlap
                pass

        # final fallback: token overlap
        try:
            scores = self._simple_overlap_scores(query)
            idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            out = []
            for i in idxs:
                row = self.df.iloc[int(i)]
                out.append({str(col): _to_json_serializable(val) for col, val in row.to_dict().items()})
            return out
        except Exception:
            # In worst case, return empty list
            return []

    def list_all_diseases(self) -> List[str]:
        """Return a list of disease ids/names present in the KB (for debugging)."""
        if "disease_id" in self.df.columns:
            return [str(_to_json_serializable(v)) for v in self.df["disease_id"].dropna().unique()]
        if "disease_name" in self.df.columns:
            return [str(_to_json_serializable(v)) for v in self.df["disease_name"].dropna().unique()]
        # fallback: indices
        return [str(i) for i in self.df.index.astype(str).values]
