# agents/metrics.py
"""
Prometheus metrics for PlantGuard agentic system.

This module is OPTIONAL.
If prometheus_client is missing, the system remains fully functional
and all metric calls degrade gracefully into no-ops.
"""

import os
import threading

# Try importing Prometheus client
try:
    from prometheus_client import (
        start_http_server,
        Counter,
        Gauge,
        Histogram,
    )
    _PROM = True
except Exception:
    _PROM = False

# ---------------------------------------------------------------------
# Metric Definitions (Safe Fallbacks)
# ---------------------------------------------------------------------

if _PROM:
    CLASSIFICATION_COUNT = Counter(
        "pg_classifications_total",
        "Number of classifications performed",
        labelnames=["model_label", "is_leaf"],
    )

    RAG_CALL_COUNT = Counter(
        "pg_rag_calls_total",
        "Number of RAG (LLM) calls",
        labelnames=["provider", "status"],
    )

    FOLLOWUP_TRIGGERED_COUNT = Counter(
        "pg_followups_triggered_total",
        "Number of followups triggered",
    )

    FOLLOWUPS_PENDING = Gauge(
        "pg_followups_pending",
        "Number of followups currently pending",
    )

    # Useful: track inference latency
    INFERENCE_LATENCY = Histogram(
        "pg_inference_latency_seconds",
        "Latency (seconds) of local inference",
        buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5),
    )

else:
    # No-op stubs
    class _Stub:
        def labels(self, *a, **k): return self
        def inc(self, *a, **k): pass
        def set(self, *a, **k): pass
        def observe(self, *a, **k): pass

    CLASSIFICATION_COUNT = _Stub()
    RAG_CALL_COUNT = _Stub()
    FOLLOWUP_TRIGGERED_COUNT = _Stub()
    FOLLOWUPS_PENDING = _Stub()
    INFERENCE_LATENCY = _Stub()

# ---------------------------------------------------------------------
# Metrics Server
# ---------------------------------------------------------------------

_metrics_started = False
_metrics_lock = threading.Lock()

def start_metrics_server(port: int = 8000):
    """
    Start the Prometheus metrics endpoint.
    Safe: can be called multiple times without errors.
    """
    global _metrics_started

    if not _PROM:
        print("Prometheus not installed. Metrics server not started.")
        return

    with _metrics_lock:
        if _metrics_started:
            print("Prometheus metrics server already running.")
            return

        try:
            start_http_server(port)
            _metrics_started = True
            print(f"[Metrics] Prometheus server started on port {port}")
        except Exception as e:
            print(f"[Metrics] Failed to start Prometheus server: {e}")

# Convenience wrapper: safe gauge update
def set_followups_pending(n: int):
    try:
        FOLLOWUPS_PENDING.set(n)
    except Exception:
        pass
