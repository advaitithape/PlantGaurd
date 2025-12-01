# infer.py — recursive folder support, EXIF, dual-policy (pad+crop), temperature, optional leaf crop
# Saves top-k predictions (pred1/prob1 .. pred3/prob3) to CSV when --save_csv is used.

import os, sys, argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import timm
from PIL import Image, ImageOps
import pandas as pd
from torch import nn

# ------------- Config -------------
IM_SIZE = 320
IM_MEAN = [0.485, 0.456, 0.406]
IM_STD  = [0.229, 0.224, 0.225]
DEBUG_DTYPE = False

FALLBACK_CLASSES = [
    "apple_apple_scab","apple_black_rot","apple_cedar_apple_rust",
    "bean_angular_leaf_spot","bean_rust","bell_pepper_bacterial_spot",
    "cherry_powdery_mildew","corn_cercospora_leaf_spot","corn_common_rust","corn_gray_leaf_spot",
    "corn_northern_leaf_blight","cotton_aphids","cotton_army_worm","cotton_bacterial_blight",
    "cotton_powdery_mildew","cotton_target_spot","diseased_cucumber","diseased_rice",
    "grape_black_rot","grape_esca_black_measles","grape_leaf_blight",
    "groundnut_early_leaf_spot","groundnut_late_leaf_spot","groundnut_nutrition_deficiency",
    "groundnut_rust","guava_anthracnose","guava_fruit_fly","healthy_apple","healthy_bean",
    "healthy_bell_pepper","healthy_cherry","healthy_corn","healthy_cotton","healthy_cucumber",
    "healthy_grape","healthy_groundnut","healthy_guava","healthy_lemon","healthy_peach",
    "healthy_potato","healthy_pumpkin","healthy_rice","healthy_strawberry","healthy_sugarcane",
    "healthy_tomato","healthy_wheat","lemon_anthracnose","lemon_bacterial_blight",
    "lemon_citrus_canker","lemon_curl_virus","lemon_deficiency","lemon_dry_leaf",
    "lemon_sooty_mould","lemon_spider_mites","peach_bacterial_spot","potato_early_blight",
    "potato_late_blight","pumpkin_bacterial_leaf_spot","pumpkin_downy_mildew",
    "pumpkin_mosaic_disease","pumpkin_powdery_mildew","strawberry_leaf_scorch",
    "sugarcane_bacterial_blight","sugarcane_mosaic","sugarcane_red_rot","sugarcane_rust",
    "sugarcane_yellow_leaf_disease","tomato_bacterial_spot","tomato_early_blight",
    "tomato_late_blight","tomato_septoria_leaf_spot","tomato_yellow_leaf_curl_virus",
    "wheat_aphid","wheat_black_rust","wheat_blast","wheat_brown_rust","wheat_common_root_rot",
    "wheat_fusarium_head_blight","wheat_leaf_blight","wheat_mildew","wheat_mite","wheat_septoria",
    "wheat_smut","wheat_stem_fly","wheat_tan_spot","wheat_yellow_rust"
]

# ------------- IO helpers -------------
def load_rgb(path, exif_fix=False):
    if exif_fix:
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        img = np.array(img)[:, :, ::-1]  # RGB->BGR for OpenCV ops
        return img
    else:
        return cv2.imread(str(path))  # BGR

# ------------- Preprocess -------------
def preprocess_bgr_pad(img_bgr):
    h,w = img_bgr.shape[:2]; s = IM_SIZE/max(h,w)
    nh,nw = int(h*s), int(w*s)
    img = cv2.resize(img_bgr,(nw,nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IM_SIZE,IM_SIZE,3), dtype=img.dtype)
    y0=(IM_SIZE-nh)//2; x0=(IM_SIZE-nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = img
    x = canvas[:,:,::-1].astype(np.float32)/255.0
    x = (x - np.array(IM_MEAN,np.float32))/np.array(IM_STD,np.float32)
    return torch.from_numpy(np.transpose(x,(2,0,1))).float()

def preprocess_bgr_center_crop(img_bgr):
    h,w = img_bgr.shape[:2]
    if h < w: nh,nw = IM_SIZE, int(w*IM_SIZE/h)
    else:     nh,nw = int(h*IM_SIZE/w), IM_SIZE
    img = cv2.resize(img_bgr,(nw,nh), interpolation=cv2.INTER_AREA)
    y0=(nh-IM_SIZE)//2; x0=(nw-IM_SIZE)//2
    img = img[y0:y0+IM_SIZE, x0:x0+IM_SIZE]
    x = img[:,:,::-1].astype(np.float32)/255.0
    x = (x - np.array(IM_MEAN,np.float32))/np.array(IM_STD,np.float32)
    return torch.from_numpy(np.transpose(x,(2,0,1))).float()

# ------------- Optional heuristic leaf crop -------------
def crop_largest_leaf_bbox(img_bgr, min_area_ratio=0.05, margin=0.08):
    h, w = img_bgr.shape[:2]
    if h < 32 or w < 32: return None
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 30, 30], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = max(3, int(round(min(h, w) * 0.01)))
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < (min_area_ratio * h * w): return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    mx = int(round(bw * margin)); my = int(round(bh * margin))
    x0 = max(0, x - mx); y0 = max(0, y - my)
    x1 = min(w, x + bw + mx); y1 = min(h, y + bh + my)
    if x1 - x0 < 16 or y1 - y0 < 16: return None
    return img_bgr[y0:y1, x0:x1]

# ------------- Models / loaders -------------
class MultiHeadEffB3(nn.Module):
    def __init__(self, num_species, num_diseases):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
        in_feats = self.backbone.num_features
        self.head_disease = nn.Linear(in_feats, num_diseases)
        self.head_species = nn.Linear(in_feats, num_species)
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        feats = self.backbone.global_pool(feats)
        z_d = self.head_disease(feats)
        z_s = self.head_species(feats)
        return z_d, z_s

@torch.no_grad()
def fuse_species_prior(z_d, z_s, parent_species_idx, alpha=0.8, temperature=None):
    if temperature is not None: z_d = z_d / temperature
    p_s = F.softmax(z_s, dim=1)
    parent = torch.tensor(parent_species_idx, device=z_s.device).view(1, -1)
    prior = torch.log(torch.gather(p_s, 1, parent) + 1e-8)
    return z_d + alpha * prior

def build_classes_from_ckpt(ckpt_state):
    if isinstance(ckpt_state, dict) and "label2idx" in ckpt_state:
        label2idx = ckpt_state["label2idx"]
        classes = [None] * len(label2idx)
        for lbl, idx in label2idx.items():
            classes[idx] = lbl
        if any(c is None for c in classes):
            raise RuntimeError("Corrupt label2idx mapping in checkpoint.")
        return classes
    return None

def safe_load_checkpoint(path):
    path = str(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e_weights:
        print("INFO: weights-only load failed, retrying full load (trusted ckpt assumed).")
        return torch.load(path, map_location="cpu", weights_only=False)

def load_model_and_classes(ckpt_path: str, device: torch.device):
    state = safe_load_checkpoint(ckpt_path)
    if isinstance(state, dict) and (("species2idx" in state) or ("disease_parent_species" in state)):
        sd = state["model"] if "model" in state else state
        label2idx = state.get("label2idx", {})
        if not label2idx:
            raise RuntimeError("Hierarchical checkpoint missing label2idx.")
        classes = [None]*len(label2idx)
        for lbl, idx in label2idx.items():
            classes[int(idx)] = lbl
        species2idx = state.get("species2idx", None)
        parent_species_idx = state.get("disease_parent_species", None)
        if (species2idx is None) or (parent_species_idx is None):
            species_names = sorted({lbl.split("_",1)[0] for lbl in classes})
            species2idx = {sp:i for i,sp in enumerate(species_names)}
            parent_species_idx = [species2idx[lbl.split("_",1)[0]] for lbl in classes]
        model = MultiHeadEffB3(num_species=len(species2idx), num_diseases=len(classes)).to(device).eval()
        msd = model.state_dict()
        for k, v in sd.items():
            if k in msd and msd[k].shape == v.shape:
                msd[k] = v
        model.load_state_dict(msd, strict=False)
        return model, classes, parent_species_idx
    classes = build_classes_from_ckpt(state)
    if classes is None:
        classes = FALLBACK_CLASSES
        if len(classes) == 0:
            raise RuntimeError("No classes available. Provide FALLBACK_CLASSES or label2idx in checkpoint.")
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(classes))
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.eval().to(device).float()
    return model, classes, None

def is_hierarchical(model_or_tuple):
    return hasattr(model_or_tuple, "head_species")

# ------------- Dual-policy predictor -------------
@torch.no_grad()
def predict_dual_policy(
    model, classes, parent_species_idx,
    img_path: Path, device,
    exif=False, topk=3, temperature=None, leaf_crop=False
):
    img = load_rgb(str(img_path), exif_fix=exif)  # BGR
    if img is None: raise FileNotFoundError(f"Could not read image: {img_path}")
    if leaf_crop:
        cropped = crop_largest_leaf_bbox(img)
        if cropped is not None: img = cropped

    # View 1: pad
    x1 = preprocess_bgr_pad(img).unsqueeze(0).to(device)
    if is_hierarchical(model) and parent_species_idx is not None:
        z_d1, z_s1 = model(x1)
        z1 = fuse_species_prior(z_d1, z_s1, parent_species_idx, alpha=0.8, temperature=temperature)
        probs1 = F.softmax(z1, dim=1)[0].cpu().numpy()
    else:
        logits1 = model(x1)
        if temperature is not None: logits1 = logits1 / temperature
        probs1 = F.softmax(logits1, dim=1)[0].cpu().numpy()

    # View 2: center-crop
    x2 = preprocess_bgr_center_crop(img).unsqueeze(0).to(device)
    if is_hierarchical(model) and parent_species_idx is not None:
        z_d2, z_s2 = model(x2)
        z2 = fuse_species_prior(z_d2, z_s2, parent_species_idx, alpha=0.8, temperature=temperature)
        probs2 = F.softmax(z2, dim=1)[0].cpu().numpy()
    else:
        logits2 = model(x2)
        if temperature is not None: logits2 = logits2 / temperature
        probs2 = F.softmax(logits2, dim=1)[0].cpu().numpy()

    probs = (probs1 + probs2) / 2.0
    top_idx = probs.argsort()[::-1][:topk]
    return [(classes[i], float(probs[i])) for i in top_idx]


# ------------- Agent-friendly API -------------
def get_model_and_predictor(
    ckpt_path: str,
    device_str: str = "cpu",
    dual_policy: bool = True,
    exif: bool = True,
    topk: int = 3,
    temperature: float | None = None,
    leaf_crop: bool = False,
):
    """
    Load the model once and return (model, classes, predictor_fn).

    predictor_fn can be called as: predictor_fn(Path_or_str_to_image)
    and returns: list of (class_name, prob) tuples, already top-k sorted.

    This is what we will use from the AI agent layer.
    """
    # pick device safely
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")

    # reuse your existing loader
    model, classes, parent_species_idx = load_model_and_classes(ckpt_path, device)

    # clamp topk between 1 and 3 (your script limit)
    topk = max(1, min(3, int(topk)))

    if dual_policy:
        # Use your existing dual-policy predictor
        def predictor(pth):
            pth = Path(pth)
            return predict_dual_policy(
                model, classes, parent_species_idx,
                pth, device,
                exif=exif,
                topk=topk,
                temperature=temperature,
                leaf_crop=leaf_crop,
            )
    else:
        # Single pad-only pass (no dual-policy)
        def predictor(pth):
            pth = Path(pth)
            img = load_rgb(str(pth), exif_fix=exif)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {pth}")
            if leaf_crop:
                cropped = crop_largest_leaf_bbox(img)
                if cropped is not None:
                    img = cropped
            x = preprocess_bgr_pad(img).unsqueeze(0).to(device)
            if is_hierarchical(model) and parent_species_idx is not None:
                z_d, z_s = model(x)
                z = fuse_species_prior(z_d, z_s, parent_species_idx, alpha=0.8, temperature=temperature)
                probs = F.softmax(z, dim=1)[0].cpu().numpy()
            else:
                logits = model(x)
                if temperature is not None:
                    logits = logits / temperature
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            top_idx = probs.argsort()[::-1][:topk]
            return [(classes[i], float(probs[i])) for i in top_idx]

    return model, classes, predictor


def classify_image_for_agent(predictor, img_path):
    """
    Convenience wrapper for agents.

    predictor: the function returned by get_model_and_predictor(...)
    img_path:  path to an image (str or Path)

    Returns a dict:
    {
        "top1_label": str,
        "top1_prob": float,
        "topk": [
            {"label": str, "prob": float},
            ...
        ]
    }
    """
    img_path = Path(img_path)
    preds = predictor(img_path)  # list[(label, prob)]

    if len(preds) == 0:
        raise RuntimeError(f"No predictions returned for image: {img_path}")

    top1_label, top1_prob = preds[0]

    return {
        "top1_label": top1_label,
        "top1_prob": float(top1_prob),
        "topk": [
            {"label": lbl, "prob": float(prob)}
            for (lbl, prob) in preds
        ],
    }


# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--exif", action="store_true", help="Fix EXIF orientation (recommended for phone images)")
    ap.add_argument("--dual_policy", action="store_true", help="Run both pad and crop; average probs")
    ap.add_argument("--temperature", type=float, default=None, help="Temperature for calibration (divide logits by T)")
    ap.add_argument("--save_csv", action="store_true", help="Write predictions to CSV when input is a folder")
    ap.add_argument("--output", type=str, default="inference_folder_results.csv", help="CSV path to save results (when --save_csv used)")
    ap.add_argument("--leaf_crop", action="store_true", help="Apply heuristic leaf bounding box cropping before resize")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, classes, parent_species_idx = load_model_and_classes(args.model, device)

    temperature = args.temperature
    topk = max(1, min(3, int(args.topk)))

    # Force dual-policy if requested, else default to pad-only single pass
    if args.dual_policy:
        predictor = lambda pth: predict_dual_policy(
            model, classes, parent_species_idx,
            pth, device,
            exif=args.exif,
            topk=topk,
            temperature=temperature,
            leaf_crop=args.leaf_crop
        )
    else:
        # Single pad-only pass (no TTA)
        def predictor(pth):
            img = load_rgb(str(pth), exif_fix=args.exif)
            if img is None: raise FileNotFoundError(f"Could not read image: {pth}")
            if args.leaf_crop:
                cropped = crop_largest_leaf_bbox(img)
                if cropped is not None: img = cropped
            x = preprocess_bgr_pad(img).unsqueeze(0).to(device)
            if is_hierarchical(model) and parent_species_idx is not None:
                z_d, z_s = model(x)
                z = fuse_species_prior(z_d, z_s, parent_species_idx, alpha=0.8, temperature=temperature)
                probs = F.softmax(z, dim=1)[0].cpu().numpy()
            else:
                logits = model(x)
                if temperature is not None: logits = logits / temperature
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            top_idx = probs.argsort()[::-1][:topk]
            return [(classes[i], float(probs[i])) for i in top_idx]

    in_path = Path(args.input)
    if in_path.is_file():
        preds = predictor(in_path)
        print(f"{in_path} -> {preds}")
    elif in_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        files = [p for p in in_path.rglob("*") if p.suffix.lower() in exts]
        if len(files) == 0:
            print("No images found under the provided folder.", file=sys.stderr)
            sys.exit(1)
        rows = []
        for p in sorted(files):
            try:
                preds = predictor(p)
                if len(preds) < topk:
                    preds = preds + [("", 0.0)] * (topk - len(preds))
                row = {"filepath": str(p)}
                for i in range(3):
                    if i < topk:
                        lbl, prob = preds[i]
                        row[f"pred{i+1}"] = lbl
                        row[f"prob{i+1}"] = float(prob)
                    else:
                        row[f"pred{i+1}"] = ""
                        row[f"prob{i+1}"] = ""
                row["pred"] = preds[0][0]
                row["prob"] = float(preds[0][1])
                print(f"{p} -> top1: {preds[0][0]} ({preds[0][1]:.4f}) ; top2: {preds[1][0] if len(preds)>1 else ''}")
                rows.append(row)
            except Exception as e:
                print(f"{p} -> ERROR: {e}", file=sys.stderr)
                rows.append({
                    "filepath": str(p), "pred": "ERROR", "prob": 0.0,
                    "pred1":"", "prob1":0.0, "pred2":"", "prob2":0.0, "pred3":"", "prob3":0.0
                })
        if args.save_csv:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"Saved predictions to: {out_path} (rows: {len(rows)})")
    else:
        print("Input path does not exist.", file=sys.stderr)

if __name__ == "__main__":
    main()
