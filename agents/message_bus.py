# agents/message_bus.py
"""
Simple in-memory Agent-to-Agent message bus.
Usage:
  from agents.message_bus import bus
  bus.subscribe("classification", handler_fn)
  bus.publish("rag", {"type":"classification", "payload": {...}})

Notes:
 - This is an in-memory bus for single-process coordination. For multi-process/distributed
   setups swap this out with Redis Pub/Sub, NATS, or another broker.
"""

import threading
import time
from typing import Callable, Dict, Any, List, Optional

_lock = threading.RLock()


class MessageBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._history: List[Dict[str, Any]] = []

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe a handler callable to a topic. Multiple handlers allowed.
        """
        with _lock:
            self._subs.setdefault(topic, []).append(handler)

    def subscribe_once(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe a handler that will be removed after it is called once.
        """
        def wrapper(msg: Dict[str, Any]) -> None:
            try:
                handler(msg)
            finally:
                # remove wrapper after first invocation
                try:
                    self.unsubscribe(topic, wrapper)
                except Exception:
                    pass

        self.subscribe(topic, wrapper)

    def unsubscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        with _lock:
            if topic in self._subs and handler in self._subs[topic]:
                self._subs[topic].remove(handler)

    def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Publish a message to a topic. Handlers are executed synchronously in the
        publisher thread. Handler errors are caught and logged to stdout.
        """
        # store history for debug
        with _lock:
            entry = {"ts": time.time(), "topic": topic, "message": message}
            self._history.append(entry)
            handlers = list(self._subs.get(topic, []))
        # call handlers outside lock
        for h in handlers:
            try:
                h(message)
            except Exception as e:
                print("MessageBus handler error:", e)

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with _lock:
            return list(self._history[-limit:])


# singleton bus
bus = MessageBus()
