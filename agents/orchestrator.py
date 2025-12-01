# agents/orchestrator.py
from dotenv import load_dotenv
load_dotenv()

import os
import time
import json
from pathlib import Path
from typing import Any, Dict, Optional

# Metrics (safe no-op stubs if prometheus is not available)
from agents.metrics import CLASSIFICATION_COUNT, RAG_CALL_COUNT, set_followups_pending

from agents.tools import classify_leaf_tool

# Optional message bus (best-effort)
try:
    from agents.message_bus import bus as message_bus  # type: ignore
except Exception:
    message_bus = None

# --- Lazy singletons: do NOT construct heavy objects at import time ---
_RAG_INSTANCE = None
_MEM_INSTANCE = None
_FOLLOW_INSTANCE = None


def get_rag():
    """
    Lazily create and return RAGAgent instance.
    """
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        try:
            from agents.rag_agent import RAGAgent  # local import to avoid heavy init at import time
            KB_PATH = os.getenv("KB_CSV", "data/kb.csv")
            RAG_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
            RAG_MODEL = os.getenv("OPENAI_MODEL", None)
            _RAG_INSTANCE = RAGAgent(kb_csv=KB_PATH, llm_provider=RAG_PROVIDER, llm_model=RAG_MODEL, topk=3)
        except Exception as e:
            print("Warning: failed to initialize RAGAgent:", e)
            _RAG_INSTANCE = None
    return _RAG_INSTANCE


def get_mem():
    """
    Lazily create and return MemoryBank instance.
    """
    global _MEM_INSTANCE
    if _MEM_INSTANCE is None:
        try:
            from agents.memory import MemoryBank
            _MEM_INSTANCE = MemoryBank("data/memory_bank.json")
        except Exception as e:
            print("Warning: failed to initialize MemoryBank:", e)
            _MEM_INSTANCE = None
    return _MEM_INSTANCE


def get_follow():
    """
    Lazily create and return FollowUpAgent instance (wired to MemoryBank).
    """
    global _FOLLOW_INSTANCE
    if _FOLLOW_INSTANCE is None:
        try:
            from agents.followup_agent import FollowUpAgent
            _FOLLOW_INSTANCE = FollowUpAgent(persist_path="data/followups.json", memory_bank=get_mem())
        except Exception as e:
            print("Warning: failed to initialize FollowUpAgent:", e)
            _FOLLOW_INSTANCE = None
    return _FOLLOW_INSTANCE


def _safe_serialize(obj: Any) -> Any:
    """
    Convert an arbitrary object into a JSON-serializable form.
    We use json.dumps with default=str then json.loads to produce
    a JSON-compatible structure where non-serializable items become strings.
    """
    try:
        return json.loads(json.dumps(obj, default=str, ensure_ascii=False))
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


def run_pipeline(
    image_path: str,
    user_id: str,
    create_followup_days: float = 14.0,
    schedule_followup: bool = True,
) -> Dict[str, Any]:
    """
    1) classify image
    2) if leaf and confident, call RAG to compose reply
    3) log memory and optionally schedule followup (now includes model_label)
    """
    image_path = str(image_path)
    # ------------- classification -------------
    res = classify_leaf_tool(image_path)
    if res.get("error"):
        return {"error": res["error"], "classification": _safe_serialize(res)}

    # increment classification metric (best-effort)
    try:
        CLASSIFICATION_COUNT.labels(model_label=str(res.get("top1_label", "unknown")), is_leaf=str(res.get("is_leaf", False))).inc()
    except Exception:
        try:
            CLASSIFICATION_COUNT.inc()
        except Exception:
            pass

    if not res.get("is_leaf", False):
        # publish event that a non-leaf was received (best-effort)
        try:
            if message_bus is not None:
                message_bus.publish("classification", {"user_id": user_id, "image": image_path, "is_leaf": False, "raw": _safe_serialize(res)})
        except Exception:
            pass
        return {"message": "Not a leaf image. Request better photo.", "classification": _safe_serialize(res)}

    label = res.get("top1_label")
    prob = float(res.get("top1_prob", 0.0))

    thr = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.45"))
    if prob < thr:
        # publish low-confidence classification event
        try:
            if message_bus is not None:
                message_bus.publish("classification", {"user_id": user_id, "image": image_path, "is_leaf": True, "label": label, "prob": prob, "low_confidence": True})
        except Exception:
            pass
        return {"message": f"Low confidence ({prob:.2f}) — ask for another photo.", "classification": _safe_serialize(res)}

    # ------------- RAG call (best-effort) -------------
    rag_out = None
    rag_provider = os.getenv("LLM_PROVIDER", "mock")
    try:
        rag = get_rag()
        if rag is None:
            raise RuntimeError("RAGAgent is not initialized")
        rag_out = rag.answer_for_label(label, user_question="Provide a short farmer-friendly reply and instruct next steps.")
        # increment RAG metric (success)
        try:
            RAG_CALL_COUNT.labels(provider=rag_provider, status="success").inc()
        except Exception:
            try:
                RAG_CALL_COUNT.inc()
            except Exception:
                pass
    except Exception as e:
        # increment RAG metric (failure)
        try:
            RAG_CALL_COUNT.labels(provider=rag_provider, status="failed").inc()
        except Exception:
            try:
                RAG_CALL_COUNT.inc()
            except Exception:
                pass
        rag_out = {"error": str(e)}
        print("Warning: RAG call failed:", e)

    # Persist memory (classification + rag parsed if present)
    try:
        mem = get_mem()
        if mem is not None:
            memory_payload = {
                "image": str(image_path),
                "label": label,
                "prob": prob,
                "rag_parsed": _safe_serialize(rag_out.get("parsed") if isinstance(rag_out, dict) else None),
            }
            mem.add_memory(user_id, {"ts": time.time(), "type": "classification", "payload": memory_payload})
    except Exception as e:
        print("Warning: failed to persist memory:", e)

    # Publish classification event so other agents can react
    try:
        if message_bus is not None:
            event = {"user_id": user_id, "image": image_path, "label": label, "prob": prob, "rag": _safe_serialize(rag_out)}
            message_bus.publish("classification", event)
    except Exception:
        pass

    # ------------- schedule followup -------------
    fu = None
    if schedule_followup:
        try:
            follow = get_follow()
            if follow is None:
                raise RuntimeError("FollowUpAgent not initialized")
            fu = follow.create_followup(
                user_id=user_id,
                image_path=str(image_path),
                delay_days=create_followup_days,
                note="auto-scheduled after initial diagnosis",
                model_label=label,
            )
            # update followups pending gauge (best-effort)
            try:
                set_followups_pending(len(follow.list_pending()))
            except Exception:
                pass
        except Exception as e:
            print("Warning: failed to schedule followup:", e)
            fu = {"error": str(e)}

    # ------------- response -------------
    out = {
        "classification": _safe_serialize(res),
        "rag": _safe_serialize(rag_out),
        "followup": _safe_serialize(fu),
    }
    return out


# CLI helper unchanged but using sanitized output
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--days", type=float, default=14.0)
    args = ap.parse_args()
    out = run_pipeline(args.img, args.user, create_followup_days=args.days)
    print(json.dumps(out, indent=2, ensure_ascii=False))
