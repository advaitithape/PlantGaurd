# app.py
import os
import io
import json
import time
from pathlib import Path
from flask import Flask, request, jsonify
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)


# ------------------------------------------------------------
# MODEL DOWNLOAD SUPPORT (HUGGINGFACE TOKEN FRIENDLY)
# ------------------------------------------------------------
import requests

MODEL_URL = os.getenv("MODEL_URL", None)
MODEL_CKPT = os.getenv("MODEL_CKPT", "/app/models/effb3_320_curated_hardened_v6.pt")
HF_TOKEN = os.getenv("HF_TOKEN", None)  # IMPORTANT: set this in Render if HF repo is private
DOWNLOAD_RETRIES = int(os.getenv("MODEL_DOWNLOAD_RETRIES", "5"))
DOWNLOAD_TIMEOUT = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "300"))
CHUNK_SIZE = 1 << 20  # 1MB chunks


def _get_auth_headers(url: str):
    """
    Adds HuggingFace Bearer token if MODEL_URL is from huggingface.co.
    """
    headers = {}
    if HF_TOKEN and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def download_with_retries(url: str, dst: str, retries: int = 5, timeout: int = 300):
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".part")

    headers = _get_auth_headers(url)

    for attempt in range(1, retries + 1):
        try:
            print(f"[startup] Download attempt {attempt}/{retries} for {url}")

            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

            tmp_path.rename(dst_path)
            print(f"[startup] Downloaded model to {dst_path}")
            return str(dst_path)

        except Exception as e:
            print(f"[startup] Download attempt {attempt} failed: {e}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except:
                    pass

            if attempt < retries:
                wait = 2 ** attempt
                print(f"[startup] retrying in {wait}s ...")
                time.sleep(wait)
            else:
                raise RuntimeError("Failed to download model after multiple retries")


if MODEL_URL and not Path(MODEL_CKPT).exists():
    try:
        download_with_retries(MODEL_URL, MODEL_CKPT, retries=DOWNLOAD_RETRIES, timeout=DOWNLOAD_TIMEOUT)
    except Exception as e:
        print("[startup] ERROR: model download failed:", e)
        # Continue startup so service still boots


# ------------------------------------------------------------
# AGENT ORCHESTRATION IMPORTS
# ------------------------------------------------------------
from agents.orchestrator import run_pipeline
from agents.followup_agent import FollowUpAgent


# Follow-up agent instance (no worker thread on Render)
memory_path = os.getenv("MEMORY_BANK_PATH", "data/memory_bank.json")
follow_path = os.getenv("FOLLOWUPS_PATH", "data/followups.json")
follow = FollowUpAgent(persist_path=follow_path, memory_bank=None, poll_interval=2)


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/classify", methods=["POST"])
def classify():
    """
    POST /classify
    Multipart form:
        image: file
        user_id: optional
        followup_days: optional float
    """
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' in form-data"}), 400

    img = request.files["image"]
    user_id = request.form.get("user_id", "web_user")

    tmp_dir = Path("/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / img.filename
    img.save(tmp_path)

    try:
        delay = float(request.form.get("followup_days", "0.5"))
    except:
        delay = 0.5

    try:
        result = run_pipeline(str(tmp_path), user_id, create_followup_days=delay, schedule_followup=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            tmp_path.unlink()
        except:
            pass

    return jsonify(result)


@app.route("/followup/trigger", methods=["POST"])
def trigger_followup():
    """
    POST /followup/trigger
    JSON:
        { "followup_id": "xxxx" }
    """
    data = request.get_json(force=True)
    fid = data.get("followup_id")
    if not fid:
        return jsonify({"error": "missing followup_id"}), 400

    ok = follow.trigger_now(fid)
    return jsonify({"triggered": bool(ok)})


# ------------------------------------------------------------
# MAIN ENTRY (Render uses gunicorn, so this is optional)
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
