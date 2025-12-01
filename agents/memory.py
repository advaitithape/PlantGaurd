# agents/memory.py
"""
Session service + simple persistent MemoryBank.

- InMemorySessionService: short-lived in-memory session store (not persisted)
- MemoryBank: persistent JSON-backed memory store for long-term memories
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional, List
import time


class InMemorySessionService:
    """
    Simple session store useful for short term conversation state.
    Sessions are stored in memory and not persisted.
    Each session: dict with arbitrary keys (e.g., conversation history, last_seen).
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, session_id: str, initial: Optional[Dict] = None) -> Dict:
        with self._lock:
            self._sessions[session_id] = initial or {}
            self._sessions[session_id]["created_at"] = time.time()
            return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[Dict]:
        with self._lock:
            # return a reference (caller should not mutate directly ideally)
            return self._sessions.get(session_id)

    def update_session(self, session_id: str, updates: Dict) -> Dict:
        with self._lock:
            s = self._sessions.setdefault(session_id, {})
            s.update(updates)
            s["updated_at"] = time.time()
            return s

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())


class MemoryBank:
    """
    Simple persistent MemoryBank for long-term memory.
    Backs up entries to a JSON file on every write.

    Schema:
      {
        "<user_id>": [ { "ts": epoch, "type": "followup", "payload": {...} }, ... ]
      }
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf8") as f:
                    self._data = json.load(f)
            except Exception:
                # if file is corrupt or unreadable, start fresh but don't crash
                self._data = {}
        else:
            # ensure parent dir exists
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._data = {}

    def _persist(self) -> None:
        try:
            with open(self.path, "w", encoding="utf8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # log to stdout instead of raising to avoid crashing background workers
            print("Warning: MemoryBank failed to persist to disk:", e)

    def add_memory(self, user_id: str, mem: Dict) -> None:
        """
        Add a memory entry for a user. If mem does not include 'ts', one will be added.
        """
        with self._lock:
            arr = self._data.setdefault(user_id, [])
            entry = dict(mem)  # shallow copy to avoid mutating caller
            if "ts" not in entry:
                entry["ts"] = time.time()
            arr.append(entry)
            self._persist()

    def get_memories(self, user_id: str) -> List[Dict]:
        with self._lock:
            return list(self._data.get(user_id, []))

    def clear_memories(self, user_id: str) -> None:
        with self._lock:
            self._data.pop(user_id, None)
            self._persist()
