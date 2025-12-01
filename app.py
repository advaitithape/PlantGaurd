# app.py
import os
import io
import json
import time
from pathlib import Path
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# load .env once
load_dotenv()

app = Flask(__name__)

# -------- Model download / startup logic (before importing heavy modules) ----------
import requests

MODEL_URL = os.getenv("MODEL_URL", None)
# prefer explicit render path but allow user override
MODEL_CKPT = str(Path(os.getenv("MODEL_CKPT", "/opt/render/project/src/models/effb3_320_curated_hardened_v6.pt")))
DOWNLOAD_RETRIES = int(os.getenv("MODEL_DOWNLOAD_RETRIES", "3"))
DOWNLOAD_TIMEOUT = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "300"))  # seconds
CHUNK_SIZE = 1 << 20  # 1 MiB

def download_with_retries(url: str, dst: str, retries: int = 3, timeout: int = 300):
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_suffix(dst_path.suffix + ".part")
    attempt = 0
    while attempt < retries:
        try:
            attempt += 1
            print(f"[startup] Download attempt {attempt}/{retries} for {url}")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            # atomic move
            tmp.rename(dst_path)
            print(f"[startup] Downloaded model to {dst_path}")
            return str(dst_path)
        except Exception as e:
            print(f"[startup] Download attempt {attempt} failed: {e}")
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
            if attempt < retries:
                sleep = 2 ** attempt
                print(f"[startup] retrying in {sleep}s ...")
                time.sleep(sleep)
            else:
                raise
    raise RuntimeError("Failed to download model after retries")

# Only attempt download if MODEL_URL present and file missing
if MODEL_URL:
    try:
        if not Path(MODEL_CKPT).exists():
            download_with_retries(MODEL_URL, MODEL_CKPT, retries=DOWNLOAD_RETRIES, timeout=DOWNLOAD_TIMEOUT)
        else:
            print(f"[startup] Model already exists at {MODEL_CKPT}, skipping download.")
    except Exception as e:
        print("[startup] ERROR: model download failed:", e)
        # If you want a hard failure on startup, uncomment the next line:
        # raise

# Now import orchestrator and followup (these may be heavy)
from agents.orchestrator import run_pipeline
from agents.followup_agent import FollowUpAgent

# Create followup agent instance (for /followup/trigger endpoint)
memory_path = os.getenv("MEMORY_BANK_PATH", "data/memory_bank.json")
followups_path = os.getenv("FOLLOWUPS_PATH", "data/followups.json")
# short poll for demo (increase in production)
follow = FollowUpAgent(persist_path=followups_path, memory_bank=None, poll_interval=int(os.getenv("FOLLOWUP_POLL_INTERVAL", "2")))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/classify", methods=["POST"])
def classify():
    """
    Accepts multipart/form-data with 'image' file and optional 'user_id'.
    Returns the orchestrator result (classification, rag, followup).
    """
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file"}), 400

    imgfile = request.files["image"]
    user_id = request.form.get("user_id", os.getenv("DEFAULT_USER", "web_user"))

    # Save a temporary file (avoid collisions by using a unique name)
    tmp_dir = Path(os.getenv("TMP_DIR", "/tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{int(time.time() * 1000)}_{Path(imgfile.filename).name}"
    imgfile.save(tmp_path)

    # run pipeline (allow override via form)
    try:
        delay_days = float(request.form.get("followup_days", os.getenv("DEFAULT_FOLLOWUP_DAYS", "0.5")))
    except Exception:
        delay_days = 0.5

    try:
        out = run_pipeline(str(tmp_path), user_id, create_followup_days=delay_days, schedule_followup=True)
        # remove temp file
        try:
            tmp_path.unlink()
        except Exception:
            pass
        return jsonify(out)
    except Exception as e:
        # include traceback string for debugging
        import traceback as _tb
        tb = _tb.format_exc()
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/followup/trigger", methods=["POST"])
def trigger_followup():
    """
    Force-trigger a followup by id (useful for testing).
    JSON body: {"followup_id": "..."}
    """
    data = request.get_json(force=True)
    fid = data.get("followup_id")
    if not fid:
        return jsonify({"error": "missing followup_id"}), 400
    ok = follow.trigger_now(fid)
    return jsonify({"triggered": bool(ok)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    # Use threaded=True to better handle multiple short requests in dev
    app.run(host="0.0.0.0", port=port, threaded=True)
