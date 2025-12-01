# agents/agent_main.py
"""
Agent bootstrap + CLI for the Plant Disease ADK prototype.

Features:
- Calls the local classify tool (models/local_inference.py via agents/tools.py)
- Uses RAGAgent (agents/rag_agent.py) to compose KB-sourced prescriptions
- Safety gate: only present prescriptive steps if KB sources are present
- Confidence thresholding for classifier outputs
- Simple logging of interactions to data/logs/interactions.log
- CLI helpers: test-image, create-followup, start-followup-worker
"""

from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import json
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ADK Agent import (optional; not required for CLI flows)
try:
    from google.adk.agents import Agent
except Exception:
    Agent = None  # ADK not installed — that's fine for local testing

# Tools and helpers
from agents.tools import classify_leaf_tool, map_label_to_disease_id
from agents.kb_loader import KBLoader
from agents.memory import InMemorySessionService, MemoryBank
from agents.followup_agent import FollowUpAgent
from agents.rag_agent import RAGAgent

# Configuration from env (with sensible defaults)
KB_CSV_PATH = os.getenv("KB_CSV", "data/kb.csv")
MODEL_CKPT = os.getenv("MODEL_CKPT", "models/effb3_320_curated_hardened_v6.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.45"))
RAG_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
RAG_MODEL = os.getenv("OPENAI_MODEL", None)

# Initialize singletons
# KB loader
kb = None
try:
    kb = KBLoader(KB_CSV_PATH)
except Exception:
    kb = None

# Session & memory
session_svc = InMemorySessionService()
memory_bank = MemoryBank("data/memory_bank.json")

# Follow-up agent
followup_agent = FollowUpAgent(persist_path="data/followups.json", memory_bank=memory_bank)

# RAG agent (create once)
try:
    _rag_agent = RAGAgent(kb_csv=KB_CSV_PATH, llm_provider=RAG_PROVIDER, llm_model=RAG_MODEL, topk=3)
except Exception as e:
    _rag_agent = None
    print("Warning: could not initialize RAGAgent:", e)


INSTRUCTION = (
    "You are a PlantDiseaseClassifierAgent. Use the provided classify_leaf_tool to "
    "classify incoming images. When a classification is returned and is_leaf==True "
    "look up solutions using the KB (via retrieve_solutions). If is_leaf==False, "
    "ask the user to upload a clearer leaf image."
)


def build_agent():
    """
    Build a minimal ADK Agent object (if ADK is available).
    This agent registers the classify_leaf_tool so the LLM can call it in conversation.
    """
    if Agent is None:
        raise RuntimeError("google-adk is not installed in this environment.")
    agent = Agent(
        name="classifier_agent",
        model=os.getenv("ADK_MODEL", "local-mock"),
        description="Agent that classifies leaf images and fetches KB solutions",
        instruction=INSTRUCTION,
        tools=[classify_leaf_tool],  # Agent may call this tool
    )
    return agent


def retrieve_and_format_solutions(model_label: str, topk: int = 3) -> Dict[str, Any]:
    """
    Helper function for CLI/testing: map model label to KB id (if applicable),
    run retrieval and return structured solutions.
    """
    disease_id = map_label_to_disease_id(model_label)
    if kb is None:
        return {"error": "KB not loaded"}
    results = kb.retrieve_solutions(disease_id, topk=topk)
    return {"disease_id": disease_id, "solutions": results}


def cli_classify_then_kb(img_path: str):
    """
    CLI helper: call the classify tool and then run RAG if allowed by safety gate.
    """
    img_path = str(img_path)
    print(f"[CLI] Classifying: {img_path}")
    resp = classify_leaf_tool(img_path)
    print("Model response:", json.dumps(resp, ensure_ascii=False, indent=2))

    if resp.get("error"):
        print("Model error:", resp["error"])
        return

    if not resp.get("is_leaf", False):
        print("The image does not appear to be a leaf image. Ask the user to upload a clearer leaf photo.")
        return

    top1_label = resp.get("top1_label")
    top1_prob = float(resp.get("top1_prob", 0.0))

    # confidence check
    if top1_prob < CONFIDENCE_THRESHOLD:
        print(f"Low confidence ({top1_prob:.2f}) for label '{top1_label}'.")
        print("I cannot safely provide prescriptive steps. Ask user to re-upload or provide more images.")
        return

    # Call the RAG agent to compose an answer
    if _rag_agent is None:
        print("RAG agent not available (not initialized). Falling back to KB retrieval only.")
        kb_res = retrieve_and_format_solutions(top1_label, topk=3)
        print("KB result:", json.dumps(kb_res, ensure_ascii=False, indent=2))
        return

    print(f"Calling RAGAgent for label '{top1_label}' (confidence={top1_prob:.2f})...")
    rag_out = _rag_agent.answer_for_label(top1_label, user_question="What should I do now?", topk=3, temperature=0.0)

    # Safeguard: ensure the parsed JSON contains sources that reference KB_*
    parsed = rag_out.get("parsed", {}) if isinstance(rag_out.get("parsed", {}), dict) else {}
    sources = parsed.get("sources") if isinstance(parsed, dict) else None
    allowed_prescription = False
    if isinstance(sources, list) and any(isinstance(s, str) and s.startswith("KB_") for s in sources):
        allowed_prescription = True

    # Print a neat user-facing message
    print("\n===== FARMER-FACING ANSWER =====\n")
    # Summary
    summary = parsed.get("summary") if isinstance(parsed, dict) else None
    if summary:
        print("Summary:", summary)
    else:
        # fallback to raw human-friendly text if available
        raw = rag_out.get("raw_llm", "")
        if isinstance(raw, str):
            idx = raw.rfind("}")
            tail = raw[idx+1:].strip() if idx != -1 else raw
            if tail:
                # print first paragraph of tail
                paras = [p for p in tail.split("\n\n") if p.strip()]
                if paras:
                    print("Summary (from LLM):", paras[0].strip())

    # Prescription (only if allowed)
    if allowed_prescription:
        pres = parsed.get("prescription", []) if isinstance(parsed, dict) else []
        if pres:
            print("\nRecommended steps (sourced from KB):")
            for i, step in enumerate(pres, start=1):
                print(f"  {i}. {step}")
        else:
            print("\nNo prescriptive steps found in parsed RAG output.")
    else:
        print("\nUnable to provide KB-sourced prescriptive steps (no KB source found).")
        print("Advice: Ask user to provide more images or consult local agricultural extension services.")

    # General advice block (non-prescriptive)
    gen = parsed.get("general_advice") if isinstance(parsed, dict) else None
    if gen:
        print("\nGeneral advice:", gen)

    # Logging summary for audits (very simple file append)
    try:
        log_dir = Path(os.getenv("LOG_DIR", "data/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "interactions.log"
        log_row = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "image": str(img_path),
            "label": top1_label,
            "prob": top1_prob,
            "allowed_prescription": allowed_prescription,
            "rag_parsed": parsed,
        }
        with open(log_path, "a", encoding="utf8") as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
    except Exception as e:
        print("Warning: failed to write log:", e)

    print("\n===== END =====\n")


def cli_create_followup(user_id: str, img_path: str, days_delay: float, note: str = "", model_label: Optional[str] = None):
    """
    Create a follow-up for a given user & image after days_delay days.
    This persists the followup to disk (via FollowUpAgent).
    If model_label is provided it will be saved in the followup so that followups can use RAG.
    """
    try:
        fu = followup_agent.create_followup(
            user_id=user_id,
            image_path=img_path,
            delay_days=float(days_delay),
            note=note,
            model_label=model_label,
        )
        print("Scheduled followup:", json.dumps(fu, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Failed to schedule followup:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-image", type=str, help="Path to a test image (calls tool and KB/RAG).")
    ap.add_argument("--start-followup-worker", action="store_true", help="Start the follow-up worker loop (blocking).")
    ap.add_argument("--create-followup", nargs=3, metavar=("USER_ID", "IMG_PATH", "DAYS"), help="Create a followup: USER_ID IMG_PATH DAYS")
    ap.add_argument("--label", type=str, default=None, help="Optional model label to attach to created followups (e.g. apple_apple_scab)")
    args = ap.parse_args()

    if args.test_image:
        cli_classify_then_kb(args.test_image)
        return

    if args.create_followup:
        user_id, img_path, days_str = args.create_followup
        try:
            days = float(days_str)
        except Exception:
            print("Invalid days value, must be numeric.")
            return
        cli_create_followup(user_id, img_path, days, note="cli-scheduled", model_label=args.label)
        return

    if args.start_followup_worker:
        print("Starting follow-up worker (Ctrl+C to exit).")
        # The worker prints triggers; you can override followup_agent.set_trigger_callback() to integrate with messaging.
        followup_agent.start_worker(blocking=True)
        return

    print("No action specified. Example usage:")
    print("  python -m agents.agent_main --test-image /path/to/leaf.jpg")
    print("  python -m agents.agent_main --create-followup farmer123 /path/img.jpg 15 --label apple_apple_scab")
    print("  python -m agents.agent_main --start-followup-worker")


if __name__ == "__main__":
    main()
