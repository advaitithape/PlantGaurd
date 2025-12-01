# agents/followup_agent.py
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Callable, List, Optional
from uuid import uuid4

DEFAULT_POLL_INTERVAL = 30  # seconds between checks (adjust as needed)


# Optional integrations: metrics and message bus. Wrapped in try/except so they remain optional.
try:
    from agents.metrics import FOLLOWUP_TRIGGERED_COUNT, set_followups_pending
except Exception:
    FOLLOWUP_TRIGGERED_COUNT = None
    set_followups_pending = None

try:
    from agents.message_bus import bus as message_bus  # type: ignore
except Exception:
    message_bus = None


class FollowUpAgent:
    """
    Simple follow-up scheduling agent.

    - create_followup(...) to schedule a follow-up (delay_days can be fractional)
    - start_worker(...) to start the background worker that triggers followups
    - follow-ups persist to a JSON file so the worker can resume after restart
    - when a follow-up is due, the agent calls `on_trigger` callback (provided by user) with the followup payload

    Additional RAG-driven behavior:
    - set_rag_sender(rag_agent, sender_fn) registers a RAGAgent instance and a sender callback.
      When follow-ups become due, the agent will use rag_agent to compose a short follow-up message and
      call sender_fn(payload_dict) where payload_dict contains user_id, followup, message, rag_parsed, kb_rows.
    """

    def __init__(
        self,
        persist_path: str = "data/followups.json",
        memory_bank=None,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ):
        self.persist_path = Path(persist_path)
        self.memory_bank = memory_bank
        self._lock = threading.RLock()
        self._load()

        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self.poll_interval = poll_interval
        self._paused = False

        # Default trigger callback simply prints; user should override via set_trigger_callback()
        self._trigger_callback: Callable[[Dict], None] = lambda payload: print("[FollowUp Triggered]", payload)

        # RAG + sender integration (set via set_rag_sender)
        self._rag_agent = None
        self._sender_fn: Optional[Callable[[Dict], None]] = None

    def _load(self):
        if self.persist_path.exists():
            try:
                with open(self.persist_path, "r", encoding="utf8") as f:
                    self.pending: List[Dict] = json.load(f)
            except Exception:
                self.pending = []
        else:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.pending = []

        # Update metrics gauge if available
        try:
            if set_followups_pending is not None:
                set_followups_pending(len(self.list_pending()))
        except Exception:
            pass

    def _persist(self):
        # Persist pending followups to disk and update followups pending gauge
        try:
            with open(self.persist_path, "w", encoding="utf8") as f:
                json.dump(self.pending, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print("Warning: failed to persist followups:", e)
        try:
            if set_followups_pending is not None:
                set_followups_pending(len(self.list_pending()))
        except Exception:
            pass

    def set_trigger_callback(self, cb: Callable[[Dict], None]):
        """
        Set a custom trigger callback for followups.
        The callback will be called with the followup payload when due.
        """
        self._trigger_callback = cb

    def set_rag_sender(self, rag_agent, sender_fn: Callable[[Dict], None]):
        """
        Register a RAG agent instance and a sender function.

        - rag_agent: an instance exposing answer_for_label(model_label, user_question, topk, temperature)
        - sender_fn: callable receiving a dict: { user_id, followup, message, rag_parsed, kb_rows, reason }

        When a followup triggers:
         - If followup has "model_label" in payload, rag_agent will be called to generate a follow-up message.
         - If no model_label present, sender_fn will be called with a simple "please upload new photo" message.
        """
        self._rag_agent = rag_agent
        self._sender_fn = sender_fn

    def create_followup(
        self,
        user_id: str,
        image_path: str,
        delay_days: float = 14.0,
        note: str = "",
        model_label: Optional[str] = None,
    ) -> Dict:
        """
        Schedule a follow-up after delay_days days. Returns the follow-up record.

        Optional `model_label` can be stored in the followup payload to let RAG compose context-aware followups.
        """
        with self._lock:
            fu = {
                "id": str(uuid4()),
                "user_id": user_id,
                "image_path": image_path,
                "note": note,
                "model_label": model_label,
                "created_ts": time.time(),
                "due_ts": time.time() + delay_days * 24 * 3600,
                "status": "pending",
            }
            self.pending.append(fu)
            self._persist()
            # Optionally record to memory bank
            if self.memory_bank is not None:
                try:
                    self.memory_bank.add_memory(user_id, {"ts": time.time(), "type": "followup_created", "payload": fu})
                except Exception:
                    pass
            return fu

    def list_pending(self) -> List[Dict]:
        with self._lock:
            return [p for p in self.pending if p.get("status") == "pending"]

    def cancel_followup(self, followup_id: str) -> bool:
        with self._lock:
            for p in self.pending:
                if p.get("id") == followup_id:
                    p["status"] = "cancelled"
                    self._persist()
                    return True
            return False

    def trigger_now(self, followup_id: str) -> bool:
        """
        Force trigger a followup immediately (useful for testing).
        Returns True if triggered, False otherwise.
        """
        with self._lock:
            for p in self.pending:
                if p.get("id") == followup_id and p.get("status") == "pending":
                    p["due_ts"] = time.time() - 1.0
                    # allow worker loop to pick it up; also call _check_and_trigger immediately
                    self._persist()
                    # call check_and_trigger synchronously for immediate effect
                    try:
                        self._check_and_trigger()
                        return True
                    except Exception:
                        return False
        return False

    def _default_trigger(self, payload: Dict):
        """
        Default trigger behavior used when no RAG/sender is set.
        Can be overridden by set_trigger_callback().
        """
        print("[FollowUp Default Trigger] payload:", payload)

    def _check_and_trigger(self):
        """
        Internal: check pending followups and trigger those due.
        This version supports RAG + sender integration if set via set_rag_sender.
        """
        now = time.time()
        to_trigger = []
        with self._lock:
            for p in self.pending:
                if p.get("status") != "pending":
                    continue
                if p.get("due_ts", 0) <= now:
                    p["status"] = "triggered"
                    p["triggered_ts"] = now
                    to_trigger.append(p)
            if to_trigger:
                self._persist()

        # Call callbacks outside lock
        for fu in to_trigger:
            try:
                # If both rag_agent and sender_fn are registered, use them to compose and send message
                if (self._rag_agent is not None) and (self._sender_fn is not None):
                    user_id = fu.get("user_id")
                    image_path = fu.get("image_path")
                    model_label = fu.get("model_label", None)

                    # If we have a model_label, use RAG to craft a short follow-up message
                    if model_label:
                        try:
                            rag_out = self._rag_agent.answer_for_label(
                                model_label,
                                user_question=(
                                    "Please generate a short (1-2 sentence) follow-up question asking the farmer "
                                    "about changes since the last advice and requesting a photo if helpful. "
                                    "Keep it friendly and brief."
                                ),
                                topk=2,
                                temperature=0.0,
                            )
                        except Exception as e:
                            # RAG failed; fallback to simple message
                            rag_out = {"parsed": {}, "raw_llm": "", "kb_rows": []}
                            print("Error calling RAG for followup:", e)

                        parsed = rag_out.get("parsed", {}) or {}
                        raw = rag_out.get("raw_llm", "") or ""
                        # Prefer a short human paragraph from raw_llm tail if present
                        msg = None
                        if isinstance(raw, str):
                            idx = raw.rfind("}")
                            tail = raw[idx + 1 :].strip() if idx != -1 else raw
                            if tail:
                                # take first paragraph
                                paras = [p for p in tail.split("\n\n") if p.strip()]
                                if paras:
                                    msg = paras[0].strip()

                        # fallback to parsed.general_advice or a default
                        if not msg:
                            ga = parsed.get("general_advice") if isinstance(parsed, dict) else None
                            if ga:
                                msg = ga if isinstance(ga, str) else str(ga)

                        if not msg:
                            msg = "Hi — this is a follow-up. How is your plant doing? Please reply and upload a photo if possible."

                        # Build payload for sender function
                        send_payload = {
                            "user_id": user_id,
                            "followup": fu,
                            "message": msg,
                            "rag_parsed": parsed,
                            "kb_rows": rag_out.get("kb_rows", []),
                            "reason": "followup_due",
                        }
                        try:
                            self._sender_fn(send_payload)
                            # record triggered event to memory bank
                            if self.memory_bank is not None:
                                try:
                                    self.memory_bank.add_memory(user_id, {"ts": time.time(), "type": "followup_triggered", "payload": fu})
                                except Exception:
                                    pass
                            # metrics
                            try:
                                if FOLLOWUP_TRIGGERED_COUNT is not None:
                                    FOLLOWUP_TRIGGERED_COUNT.inc()
                            except Exception:
                                pass
                            # publish to message bus for other agents
                            try:
                                if message_bus is not None:
                                    message_bus.publish("followup", send_payload)
                            except Exception:
                                pass
                        except Exception as e:
                            print("Error in sender_fn for followup:", e)
                            # fallback to default callback
                            try:
                                self._trigger_callback(fu)
                            except Exception as e2:
                                print("Error in fallback trigger callback:", e2)
                    else:
                        # No model_label: ask user to upload a fresh photo
                        msg = "Hi — we requested a follow-up. Please upload a fresh photo of the plant so we can reassess."
                        send_payload = {
                            "user_id": fu.get("user_id"),
                            "followup": fu,
                            "message": msg,
                            "reason": "need_new_image",
                        }
                        try:
                            self._sender_fn(send_payload)
                            if self.memory_bank is not None:
                                try:
                                    self.memory_bank.add_memory(fu["user_id"], {"ts": time.time(), "type": "followup_triggered", "payload": fu})
                                except Exception:
                                    pass
                            try:
                                if FOLLOWUP_TRIGGERED_COUNT is not None:
                                    FOLLOWUP_TRIGGERED_COUNT.inc()
                            except Exception:
                                pass
                            try:
                                if message_bus is not None:
                                    message_bus.publish("followup", send_payload)
                            except Exception:
                                pass
                        except Exception as e:
                            print("Error in sender_fn for followup:", e)
                            try:
                                self._trigger_callback(fu)
                            except Exception as e2:
                                print("Error in fallback trigger callback:", e2)
                else:
                    # fallback: call user-provided trigger callback
                    try:
                        self._trigger_callback(fu)
                        if self.memory_bank is not None:
                            try:
                                self.memory_bank.add_memory(fu["user_id"], {"ts": time.time(), "type": "followup_triggered", "payload": fu})
                            except Exception:
                                pass
                        # metrics & bus in fallback path as well (best-effort)
                        try:
                            if FOLLOWUP_TRIGGERED_COUNT is not None:
                                FOLLOWUP_TRIGGERED_COUNT.inc()
                        except Exception:
                            pass
                        try:
                            if message_bus is not None:
                                message_bus.publish("followup", {"user_id": fu.get("user_id"), "followup": fu, "message": None, "reason": "fallback_trigger"})
                        except Exception:
                            pass
                    except Exception as e:
                        print("Error in followup callback:", e)
            except Exception as e:
                print("Error handling followup:", e)

    def start_worker(self, blocking: bool = False):
        """
        Start background worker thread (non-blocking by default).
        If blocking=True, this call will run until terminated (useful for docker/container).
        """
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="FollowUpWorker")
        self._worker_thread.start()
        if blocking:
            try:
                while not self._stop_event.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop_worker()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            if not self._paused:
                try:
                    self._check_and_trigger()
                except Exception as e:
                    print("FollowUpAgent worker error:", e)
            # Sleep in small increments so stop can be responsive
            sleep_for = self.poll_interval
            for _ in range(int(max(1, sleep_for))):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def stop_worker(self):
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False
