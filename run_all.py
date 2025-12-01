#!/usr/bin/env python3
"""
run_all.py

Single-command runner to exercise the whole PlantGuard demo locally.

Examples:
  # Run demo classification + followup (using demo image)
  python run_all.py --demo-image "B:/PlantGaurd/download (1).jpg" --demo-user demo_farmer --wait-seconds 90

  # Build FAISS index (first-run) and run demo:
  python run_all.py --build-faiss --demo-image "B:/PlantGaurd/download (1).jpg" --wait-seconds 90

Notes:
 - Requires your project files in place (agents/, models/, data/).
 - Uses MOCK LLM by default unless LLM env vars are set.
 - Followups created by orchestrator use 'delay_days' (default small value for demo).
"""

import os
import time
import signal
import argparse
import threading
import traceback

from dotenv import load_dotenv
load_dotenv()

# optional metrics
try:
    from agents.metrics import start_metrics_server, set_followups_pending
except Exception:
    start_metrics_server = None
    set_followups_pending = None

from agents.orchestrator import run_pipeline
from agents.rag_agent import RAGAgent
from agents.memory import MemoryBank
from agents.followup_agent import FollowUpAgent

# optional embedding KB builder
try:
    from agents.kb_loader_faiss import EmbeddingKB
except Exception:
    EmbeddingKB = None

# message bus (for debug)
try:
    from agents.message_bus import bus as message_bus
except Exception:
    message_bus = None

shutdown_event = threading.Event()

def default_sender(payload: dict):
    """
    Simple sender used by followup_agent.set_rag_sender().
    Prints the final message and metadata.
    """
    print("\n=== SENDER CALLED ===")
    print("To user:", payload.get("user_id"))
    fu = payload.get("followup")
    if fu:
        print("Followup ID:", fu.get("id"))
    print("Message:", payload.get("message"))
    print("rag_parsed keys:", list(payload.get("rag_parsed", {}).keys()) if payload.get("rag_parsed") else [])
    print("=====================\n")

def graceful_shutdown(signum=None, frame=None):
    print("Received shutdown signal. Stopping...")
    shutdown_event.set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-image", type=str, default=None, help="Path to demo image for classification.")
    ap.add_argument("--demo-user", type=str, default="demo_farmer", help="User id to attach demo run.")
    ap.add_argument("--wait-seconds", type=float, default=90.0, help="Seconds to wait for followup to trigger (for demo).")
    ap.add_argument("--followup-delay-min", type=float, default=0.05, help="Follow-up delay in minutes for demo scheduling (default 0.05 ~ 3s).")
    ap.add_argument("--build-faiss", action="store_true", help="Build FAISS index before running (first time only).")
    ap.add_argument("--metrics-port", type=int, default=8000, help="Prometheus metrics port.")
    args = ap.parse_args()

    # start metrics server (best-effort)
    if start_metrics_server is not None:
        try:
            start_metrics_server(args.metrics_port)
        except Exception as e:
            print("Failed to start metrics server:", e)

    # optionally build FAISS index
    if args.build_faiss:
        if EmbeddingKB is None:
            print("EmbeddingKB/faiss not available. Skipping build-faiss.")
        else:
            try:
                print("Building FAISS index (this may take time on first run)...")
                kb_csv = os.getenv("KB_CSV", "data/kb.csv")
                emb = EmbeddingKB(kb_csv, index_path=os.getenv("FAISS_INDEX_PATH", "data/faiss_index.idx"))
                emb.build_index()
                print("FAISS index built.")
            except Exception as e:
                print("FAISS build failed:", e)
                traceback.print_exc()

    # instantiate components
    print("Initializing RAGAgent (mock/openai depends on env)...")
    rag = None
    try:
        rag = RAGAgent(llm_provider=os.getenv("LLM_PROVIDER", "mock"), llm_model=os.getenv("OPENAI_MODEL", None))
    except Exception as e:
        print("Warning: could not initialize RAGAgent:", e)
        rag = None

    mem = MemoryBank(os.getenv("MEMORY_BANK_PATH", "data/memory_bank.json"))
    follow = FollowUpAgent(persist_path=os.getenv("FOLLOWUPS_PATH", "data/followups.json"), memory_bank=mem, poll_interval=2)

    # register rag sender (so followups use RAG to compose message)
    if rag is not None:
        try:
            follow.set_rag_sender(rag, default_sender)
        except Exception as e:
            print("Warning: could not set rag sender:", e)

    # register a simple message bus subscriber (optional)
    if message_bus is not None:
        def on_class(msg):
            print("[BUS] classification event:", msg.get("type", "N/A"))
        try:
            message_bus.subscribe("classification", on_class)
        except Exception:
            pass

    # start followup worker
    print("Starting follow-up worker (background)...")
    follow.start_worker(blocking=False)
    # update pending gauge (if metrics enabled)
    try:
        if set_followups_pending is not None:
            set_followups_pending(len(follow.list_pending()))
    except Exception:
        pass

    # install graceful shutdown handler
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # Run demo pipeline if demo-image provided
    if args.demo_image:
        demo_user = args.demo_user
        print("Running orchestrator pipeline on demo image:", args.demo_image)
        # Use small followup delay for demo (in minutes)
        try:
            out = run_pipeline(args.demo_image, demo_user, create_followup_days=(args.followup_delay_min / 1440.0), schedule_followup=True)
            print("Pipeline output (sanitized):")
            import json
            print(json.dumps(out, indent=2, ensure_ascii=False))
        except Exception as e:
            print("Orchestrator run failed:", e)
            traceback.print_exc()

    # Wait for followups to be triggered (demo)
    print(f"Waiting up to {args.wait_seconds} seconds for followups to trigger (Ctrl+C to exit early)...")
    start = time.time()
    try:
        while not shutdown_event.is_set():
            elapsed = time.time() - start
            if elapsed >= args.wait_seconds:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    # cleanup
    print("Shutting down follow-up worker...")
    try:
        follow.stop_worker()
    except Exception:
        pass

    print("Done. Exiting.")

if __name__ == "__main__":
    main()
