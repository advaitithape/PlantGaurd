# agents/tools.py
"""
Tooling helpers (ADK tools) that wrap the local model inference layer.

Exposes:
  - classify_leaf_tool(image_path_or_bytes) -> dict
  - map_label_to_disease_id(model_label) -> str
"""

import os
import time
import traceback
from pathlib import Path
from typing import Union, Optional

# Optional integrations (metrics & message bus) — best-effort imports
try:
    # metrics.py currently exposes CLASSIFICATION_COUNT, RAG_CALL_COUNT, FOLLOWUP_TRIGGERED_COUNT, FOLLOWUPS_PENDING
    from agents.metrics import CLASSIFICATION_COUNT  # type: ignore
    INFERENCE_LATENCY = None
except Exception:
    CLASSIFICATION_COUNT = None
    INFERENCE_LATENCY = None

try:
    from agents.message_bus import bus as message_bus  # type: ignore
except Exception:
    message_bus = None

# Import local wrapper (ensure models is a package or models/ is on sys.path)
try:
    from models.local_inference import LocalModelWrapper  # type: ignore
    _IMPORT_ERR = None
except Exception as e:
    LocalModelWrapper = None
    _IMPORT_ERR = e

_MODEL_WRAPPER: Optional[LocalModelWrapper] = None


def _get_wrapper() -> LocalModelWrapper:
    """
    Lazily construct and cache LocalModelWrapper.
    Raises ImportError with helpful message if LocalModelWrapper cannot be imported.
    """
    global _MODEL_WRAPPER
    if _MODEL_WRAPPER is not None:
        return _MODEL_WRAPPER

    if LocalModelWrapper is None:
        raise ImportError(
            "Could not import LocalModelWrapper from models.local_inference. "
            f"Original import error: {_IMPORT_ERR}"
        )

    # read config from environment
    ckpt = os.getenv("MODEL_CKPT", "models/effb3_320_curated_hardened_v6.pt")
    device = os.getenv("MODEL_DEVICE", "cpu")
    dual_policy = os.getenv("MODEL_DUAL_POLICY", "1") not in ("0", "false", "False")
    exif = os.getenv("MODEL_EXIF", "1") not in ("0", "false", "False")
    try:
        topk = int(os.getenv("MODEL_TOPK", "3"))
    except Exception:
        topk = 3
    temp = os.getenv("MODEL_TEMPERATURE", None)
    if temp is not None:
        try:
            temp = float(temp)
        except Exception:
            temp = None
    leaf_crop = os.getenv("MODEL_LEAF_CROP", "0") not in ("0", "false", "False")

    # instantiate wrapper (it lazy-loads the heavy model)
    _MODEL_WRAPPER = LocalModelWrapper(
        ckpt_path=ckpt,
        device=device,
        dual_policy=dual_policy,
        exif=exif,
        topk=topk,
        temperature=temp,
        leaf_crop=leaf_crop,
    )
    return _MODEL_WRAPPER


def classify_leaf_tool(image: Union[str, Path, bytes, bytearray]) -> dict:
    """
    Safe tool wrapper to classify an image.

    Accepts:
      - path string or Path
      - bytes / bytearray (image bytes)

    Returns:
      - dict (JSON-serializable). On error: {"error": "...", "traceback": "..."}
    """
    start = time.time()
    try:
        wrapper = _get_wrapper()
        res = wrapper.predict(image)

        # metrics: observe latency & increment count (best-effort)
        latency = time.time() - start
        try:
            if INFERENCE_LATENCY is not None:
                INFERENCE_LATENCY.observe(latency)
        except Exception:
            pass

        try:
            if CLASSIFICATION_COUNT is not None:
                lbl = str(res.get("top1_label", "unknown"))
                is_leaf = str(bool(res.get("is_leaf", False)))
                try:
                    # some metrics setups provide labeled counters
                    CLASSIFICATION_COUNT.labels(model_label=lbl, is_leaf=is_leaf).inc()
                except Exception:
                    # fallback if labeled metric not available
                    try:
                        CLASSIFICATION_COUNT.inc()
                    except Exception:
                        pass
        except Exception:
            pass

        # publish to message bus for other agents (best-effort)
        try:
            if message_bus is not None:
                event = {
                    "ts": time.time(),
                    "type": "classification",
                    "result": res,
                }
                message_bus.publish("classification", event)
        except Exception:
            pass

        return res

    except Exception as e:
        tb = traceback.format_exc()
        # record latency even on errors
        try:
            latency = time.time() - start
            if INFERENCE_LATENCY is not None:
                INFERENCE_LATENCY.observe(latency)
        except Exception:
            pass
        return {"error": str(e), "traceback": tb}


def map_label_to_disease_id(model_label: Optional[str]) -> str:
    """
    Identity mapping by default. Replace or extend this function if your model
    label naming differs from KB disease_id names.
    """
    if model_label is None:
        return ""
    return str(model_label)
