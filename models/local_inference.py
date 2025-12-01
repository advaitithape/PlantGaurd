# models/local_inference.py
from pathlib import Path
import tempfile
import os
import traceback
from typing import Union
import threading

# Import functions from your infer.py (kept the same)
import infer  # assumes infer.py is importable from project root

class LocalModelWrapper:
    """
    Lazy-loading wrapper around infer.get_model_and_predictor.
    """
    _instance_lock = threading.RLock()

    def __init__(
        self,
        ckpt_path: str = None,
        device: str = "cpu",
        dual_policy: bool = True,
        exif: bool = True,
        topk: int = 3,
        temperature: float | None = None,
        leaf_crop: bool = False,
    ):
        # config from env if not passed explicitly
        env_ckpt = os.getenv("MODEL_CKPT", None)
        resolved_ckpt = ckpt_path or env_ckpt or "models/effb3_320_curated_hardened_v6.pt"
        self.ckpt_path = str(Path(resolved_ckpt))
        self.device = device or os.getenv("MODEL_DEVICE", "cpu")
        self.dual_policy = dual_policy
        self.exif = exif
        self.topk = max(1, min(3, int(topk)))
        self.temperature = temperature
        self.leaf_crop = leaf_crop

        # placeholders for actual model/predictor (lazy)
        self._model = None
        self._classes = None
        self._predictor = None

    def _ensure_loaded(self):
        with LocalModelWrapper._instance_lock:
            if self._predictor is not None:
                return
            try:
                # Use infer.get_model_and_predictor to load model & predictor
                model, classes, predictor = infer.get_model_and_predictor(
                    ckpt_path=self.ckpt_path,
                    device_str=self.device,
                    dual_policy=self.dual_policy,
                    exif=self.exif,
                    topk=self.topk,
                    temperature=self.temperature,
                    leaf_crop=self.leaf_crop,
                )
                self._model = model
                self._classes = classes
                self._predictor = predictor
                print(f"[LocalModelWrapper] Loaded model from {self.ckpt_path} on device {self.device}")
            except Exception as e:
                # surface error as string to calling code
                print("[LocalModelWrapper] Failed to load model:", e)
                raise

    def _maybe_write_bytes(self, img_bytes: bytes, suffix: str = ".jpg") -> Path:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tf.write(img_bytes)
        tf.flush()
        tf.close()
        return Path(tf.name)

    def _is_probably_leaf(self, img_path: Union[str, Path]) -> bool:
        try:
            p = Path(img_path)
            img_bgr = infer.load_rgb(str(p), exif_fix=self.exif)
            if img_bgr is None:
                return False
            cropped = infer.crop_largest_leaf_bbox(img_bgr)
            return cropped is not None
        except Exception:
            return False

    def predict(self, image: Union[str, Path, bytes]) -> dict:
        """
        Predict on an image. Lazy-load model if needed.
        """
        # ensure model loaded
        try:
            self._ensure_loaded()
        except Exception as e:
            tb = traceback.format_exc()
            return {"error": f"Model load failed: {e}", "traceback": tb}

        temp_path = None
        try:
            if isinstance(image, (bytes, bytearray)):
                temp_path = self._maybe_write_bytes(bytes(image))
                img_path = temp_path
            else:
                img_path = Path(image)

            if not Path(img_path).exists():
                return {"error": f"Image not found: {img_path}"}

            is_leaf = self._is_probably_leaf(img_path)

            if not is_leaf:
                return {
                    "is_leaf": False,
                    "top1_label": None,
                    "top1_prob": 0.0,
                    "topk": [],
                    "model_meta": {"ckpt": self.ckpt_path, "device": self.device},
                }

            preds = self._predictor(img_path)  # returns list[(label, prob)]

            if preds is None:
                raise RuntimeError("Predictor returned None")

            topk = [{"label": lbl, "prob": float(prob)} for (lbl, prob) in preds]
            top1_label = topk[0]["label"] if len(topk) > 0 else None
            top1_prob  = float(topk[0]["prob"]) if len(topk) > 0 else 0.0

            return {
                "is_leaf": True,
                "top1_label": top1_label,
                "top1_prob": top1_prob,
                "topk": topk,
                "model_meta": {"ckpt": self.ckpt_path, "device": self.device},
            }

        except Exception as e:
            tb = traceback.format_exc()
            return {"error": str(e), "traceback": tb}
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass


# convenience
def build_and_predict_example(ckpt_path: str, image_path: str):
    wrapper = LocalModelWrapper(ckpt_path=ckpt_path, device="cpu", dual_policy=True)
    return wrapper.predict(image_path)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img", required=True)
    args = ap.parse_args()
    res = build_and_predict_example(args.ckpt, args.img)
    print(json.dumps(res, indent=2))
