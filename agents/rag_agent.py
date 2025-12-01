# agents/rag_agent.py
"""
RAGAgent: Compose KB content into farmer-friendly answers using an LLM.

Usage:
    from agents.rag_agent import RAGAgent
    rag = RAGAgent()
    out = rag.answer_for_label("tomato_early_blight", user_question="What should I do now?")
    print(out)

Behavior:
 - Prefers FAISS/EmbeddingKB if available (semantic retrieval).
 - Falls back to KBLoader (TF-IDF / exact match).
 - Uses OpenAI >=1.0.0 client when LLM_PROVIDER=openai and OPENAI_API_KEY is set.
 - Default provider is 'mock' for offline development.
"""

import os
import json
import time
from typing import List, Dict, Optional, Any
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Try to import EmbeddingKB (FAISS) first
_EMBEDDING_KB_AVAILABLE = False
try:
    from agents.kb_loader_faiss import EmbeddingKB  # type: ignore
    _EMBEDDING_KB_AVAILABLE = True
except Exception:
    EmbeddingKB = None
    _EMBEDDING_KB_AVAILABLE = False

# Fallback KB loader
from agents.kb_loader import KBLoader
from agents.tools import map_label_to_disease_id

# Try to detect OpenAI library (>=1.0.0)
_HAS_OPENAI = False
try:
    from openai import OpenAI as OpenAIClient  # type: ignore
    _HAS_OPENAI = True
except Exception:
    OpenAIClient = None
    _HAS_OPENAI = False


def _safe_usage_obj(obj: Any) -> Dict:
    """
    Normalize provider-specific usage objects into a plain dict that's JSON-serializable.
    """
    if obj is None:
        return {}
    try:
        # some SDK objects provide to_dict()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        # if it's already a dict-like, return shallow copy
        if isinstance(obj, dict):
            return dict(obj)
        # if it has attributes, try to __dict__ then fallback to str
        if hasattr(obj, "__dict__"):
            return dict(getattr(obj, "__dict__", {}) or {})
    except Exception:
        pass
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        try:
            return {"usage": str(obj)}
        except Exception:
            return {}


class LLMClient:
    """
    Minimal LLM client abstraction.
    Supports:
      - provider="openai": uses openai.OpenAI client (>=1.0.0)
      - provider="mock": deterministic mocked response (for offline dev)
    """

    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        self.provider = (provider or "mock").lower()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if self.provider == "openai" and not _HAS_OPENAI:
            raise RuntimeError("OpenAI client not available. Install openai>=1.0.0 or use provider='mock'.")
        if self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not found in environment for OpenAI provider.")
            # Initialize client (OpenAI(api_key=...))
            self._client = OpenAIClient(api_key=key)
        else:
            self._client = None

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 512) -> Dict[str, Any]:
        """
        Returns dict:
            { "text": "<assistant reply>", "usage": {...}, "raw": <provider raw> }
        """
        if self.provider == "mock":
            txt = (
                f"[MOCK LLM ANSWER]\nSystem: {system[:120]}...\nUser: {user[:240]}...\n\n"
                "NOTE: This is a mock reply. Install OpenAI and set OPENAI_API_KEY to get real responses."
            )
            return {"text": txt, "usage": {}, "raw": None}

        if self.provider == "openai":
            client = self._client
            if client is None:
                raise RuntimeError("OpenAI client not initialized")

            # Use the new-style OpenAI client chat completions
            # The exact shape may vary by provider/model; we defensively extract text and usage.
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                raise RuntimeError(f"OpenAI chat call failed: {e}")

            # Extract assistant text robustly
            text = ""
            try:
                # prefer structured attributes
                if hasattr(resp, "choices") and len(resp.choices) > 0:
                    ch0 = resp.choices[0]
                    # some SDK representations: ch0.message.content
                    if hasattr(ch0, "message") and hasattr(ch0.message, "content"):
                        text = ch0.message.content
                    # fallback to dict-like
                    else:
                        rd = resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)
                        if "choices" in rd and len(rd["choices"]) > 0:
                            c0 = rd["choices"][0]
                            if isinstance(c0, dict) and "message" in c0 and "content" in c0["message"]:
                                text = c0["message"]["content"]
                            elif isinstance(c0, dict) and "text" in c0:
                                text = c0["text"]
                else:
                    # last-effort: string conversion
                    text = str(resp)
            except Exception:
                try:
                    text = str(resp)
                except Exception:
                    text = ""

            # Extract usage into plain dict
            try:
                usage_obj = getattr(resp, "usage", None)
                if usage_obj is None and hasattr(resp, "to_dict"):
                    usage_obj = resp.to_dict().get("usage", None)
                usage = _safe_usage_obj(usage_obj)
            except Exception:
                usage = {}

            return {"text": text, "usage": usage, "raw": resp}

        raise RuntimeError(f"Unknown LLM provider: {self.provider}")


class RAGAgent:
    """
    RAGAgent composes KB content + LLM to produce farmer-friendly responses.
    Primary method:
       answer_for_label(model_label: str, user_question: Optional[str]) -> dict
    """

    def __init__(
        self,
        kb_csv: str = os.getenv("KB_CSV", "data/kb.csv"),
        llm_provider: str = os.getenv("LLM_PROVIDER", "mock"),
        llm_model: Optional[str] = None,
        topk: int = 3,
    ):
        self.kb_csv = kb_csv
        self.topk = topk
        self.llm = LLMClient(provider=llm_provider, model=llm_model)

        # prefer EmbeddingKB (FAISS) if available
        self.embedding_kb = None
        if _EMBEDDING_KB_AVAILABLE:
            try:
                self.embedding_kb = EmbeddingKB(self.kb_csv, index_path=os.getenv("FAISS_INDEX_PATH", "data/faiss_index.idx"))
                # attempt to load index (will raise if missing)
                try:
                    self.embedding_kb.load_index()
                except Exception:
                    # if loading fails, build it (may take time)
                    self.embedding_kb.build_index()
            except Exception as e:
                # fallback to old KBLoader
                print("Warning: failed to init EmbeddingKB, falling back to KBLoader:", e)
                self.embedding_kb = None

        # fallback KB loader
        self.kb = KBLoader(self.kb_csv)

    def _build_system_prompt(self) -> str:
        return (
            "You are a helpful agricultural assistant specialized in diagnosing plant leaf diseases and "
            "recommending safe, practical remedies and next steps for smallholder farmers. You MUST follow these rules:\n"
            "1) For any prescriptive, actionable, or numeric recommendations (e.g., which chemical to apply, dosage, timing, how to apply), you MUST ONLY use verifiable information drawn from the provided KB snippets. Do NOT hallucinate any prescription.\n"
            "2) You may answer general agricultural questions conversationally, but clearly mark when your answer is general advice versus KB-sourced prescriptive steps.\n"
            "3) Always provide concise practical steps, cite the KB entries used (by their index numbers), and include a short summary sentence at the top.\n"
            "4) If the KB is insufficient to give safe prescriptive guidance, explicitly say so and recommend seeking local extension services.\n"
            "5) Keep language simple, actionable, and respectful for smallholder farmers. Avoid technical jargon or, if used, explain terms briefly.\n"
        )

    def _retrieve_kb_rows(self, model_label: str, topk: int) -> List[Dict]:
        # Try semantic retrieval first if embedding KB is present
        if self.embedding_kb is not None:
            query = model_label.replace("_", " ")
            try:
                rows = self.embedding_kb.retrieve(query, topk=topk)
                if rows:
                    return rows
            except Exception:
                # fallthrough to fallback loader
                pass

        # Fallback to KBLoader (exact match by disease_id)
        disease_id = map_label_to_disease_id(model_label)
        rows = self.kb.retrieve_solutions(disease_id, topk=topk)
        return rows

    def _build_user_prompt(self, model_label: str, kb_rows: List[Dict], user_question: Optional[str]) -> str:
        kb_blocks = []
        for i, row in enumerate(kb_rows, start=1):
            block = (
                f"[[KB_{i}]] disease_id: {row.get('disease_id','')}\n"
                f"name: {row.get('disease_name','')}\n"
                f"crop: {row.get('crop','')}\n"
                f"symptoms: {row.get('symptoms','')}\n"
                f"solution_text: {row.get('solution_text','')}\n"
                f"chemical_recommendations: {row.get('chemical_recommendations','')}\n"
                f"organic_recommendations: {row.get('organic_recommendations','')}\n"
                f"notes: {row.get('notes','')}\n"
            )
            kb_blocks.append(block)
        kb_text = "\n\n".join(kb_blocks) if kb_blocks else "(no KB snippets found)"
        user_q_text = user_question.strip() if user_question else "Please provide farmer-friendly guidance for this disease."

        prompt = (
            f"You are given the following KB snippets (numbered). Use them as your only source for prescriptive steps.\n\n"
            f"KB SNIPPETS:\n{kb_text}\n\n"
            f"Task:\n"
            f"- The model detected: **{model_label}**.\n"
            f"- User question: {user_q_text}\n\n"
            "Produce a JSON object with these fields:\n"
            "  summary: short 1-2 sentence explanation of what the disease is and urgency.\n"
            "  prescription: an ordered list of actionable steps. EACH PRESCRIPTIVE STEP MUST BE DERIVED FROM THE KB and reference the KB index in parentheses, e.g., (KB_1).\n"
            "  sources: list of KB indices used (e.g., [\"KB_1\"]).\n"
            "  general_advice: optional general non-prescriptive advice (you may provide this conversationally).\n\n"
            "If you cannot produce safe prescriptive steps because the KB lacks necessary details, put prescription=[] and explain what additional info is needed.\n\n"
            "Finally, after the JSON, include a brief human-friendly paragraph the farmer can read (2-4 sentences) that summarizes the steps.\n"
        )
        return prompt

    def answer_for_label(
        self,
        model_label: str,
        user_question: Optional[str] = None,
        topk: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        topk = topk or self.topk
        kb_rows = self._retrieve_kb_rows(model_label, topk=topk)

        system = self._build_system_prompt()
        user_prompt = self._build_user_prompt(model_label, kb_rows, user_question)

        llm_resp = self.llm.chat(system=system, user=user_prompt, temperature=temperature, max_tokens=512)
        raw_text = llm_resp.get("text", "")

        json_obj = None
        try:
            txt = raw_text.strip()
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                jtxt = txt[start:end+1]
                json_obj = json.loads(jtxt)
        except Exception:
            json_obj = None

        if not json_obj:
            json_obj = {
                "summary": None,
                "prescription": [],
                "sources": [],
                "general_advice": None,
                "note": "LLM reply could not be parsed as JSON; see raw_llm for full response.",
            }

        out = {
            "model_label": model_label,
            "disease_id": map_label_to_disease_id(model_label),
            "kb_rows": kb_rows,
            # make sure usage is JSON-serializable
            "llm_usage": _safe_usage_obj(llm_resp.get("usage", {})),
            "raw_llm": raw_text,
            "parsed": json_obj,
            "timestamp": time.time(),
        }
        return out
