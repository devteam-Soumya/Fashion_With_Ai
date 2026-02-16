# backend_gemini_overlay.py
# Virtual Try-On – Hardened (Refactored + Scores)
# Railway-friendly: lazy-load heavy libs + quick health endpoints

import os
import io
import uuid
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

# Railway-safe writable cache for rembg model downloads
os.environ.setdefault("U2NET_HOME", "/tmp/.u2net")

import anyio
import requests
import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware

from dotenv import load_dotenv
load_dotenv()

# ✅ NEW Gemini SDK (google-genai)
from google import genai
from google.genai import types

import replicate


# ─────────────────────────────
# CONFIG
# ─────────────────────────────
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "1") == "1"
ENABLE_IDM_VTON = os.getenv("ENABLE_IDM_VTON", "1") == "1"

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")

GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

app = FastAPI(title="Virtual Try-On – Railway Hardened")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
# ─────────────────────────────
# Request ID middleware
# ─────────────────────────────
class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

app.add_middleware(RequestIdMiddleware)


# ─────────────────────────────
# Gemini setup (optional) - NEW SDK
# ─────────────────────────────
gemini_client: Optional[genai.Client] = None
if ENABLE_GEMINI and GEMINI_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_KEY)
        print("✅ Gemini enabled (google-genai)")
    except Exception as e:
        gemini_client = None
        print(f"⚠️ Gemini init failed ({type(e).__name__}). Disabling Gemini.")
else:
    print("ℹ️ Gemini disabled")


# ─────────────────────────────
# Replicate setup (optional)
# ─────────────────────────────
idm_vton_ready = False
if ENABLE_IDM_VTON:
    if not REPLICATE_TOKEN:
        print("⚠️ IDM-VTON requested but REPLICATE_API_TOKEN missing. Disabling IDM-VTON.")
    else:
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_TOKEN
        idm_vton_ready = True
        print("✅ IDM-VTON enabled via Replicate")
else:
    print("ℹ️ IDM-VTON disabled (overlay only)")


# ─────────────────────────────
# Models
# ─────────────────────────────
yolo = YOLO("yolov8n-seg.pt")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False)

mp_face = mp.solutions.face_detection
face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)


# ─────────────────────────────
# Utils
# ─────────────────────────────
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def bytes_to_bgr(data: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def save_bgr_jpg(path: str, bgr: np.ndarray):
    bgr_to_pil(bgr).save(path, "JPEG", quality=92, optimize=True)

def build_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/outputs/{filename}"

def safe_output_path(filename: str) -> str:
    # Prevent path traversal (e.g., ../../secret)
    fname = os.path.basename(filename)
    return os.path.join(OUTPUT_DIR, fname)

def build_download_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/download/{filename}"

def build_view_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/view/{filename}"


# ─────────────────────────────
# Download + View endpoints
# ─────────────────────────────
@app.get("/download/{filename}")
def download_output(request: Request, filename: str):
    path = safe_output_path(filename)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"ok": False, "error": "file_not_found"})

    # Forces download (Content-Disposition: attachment)
    return FileResponse(
        path=path,
        media_type="image/jpeg",
        filename=os.path.basename(filename),
        headers={"Cache-Control": "no-store"},
    )

@app.get("/view/{filename}")
def view_output(request: Request, filename: str):
    path = safe_output_path(filename)
    if not os.path.exists(path):
        return HTMLResponse("<h3>File not found</h3>", status_code=404)

    img_url = build_url(request, os.path.basename(filename))
    dl_url = build_download_url(request, os.path.basename(filename))

    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width,initial-scale=1"/>
        <title>Try-On Result</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          .wrap {{ max-width: 900px; margin: 0 auto; }}
          img {{ width: 100%; height: auto; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.15); }}
          .btns {{ margin-top: 16px; display: flex; gap: 12px; flex-wrap: wrap; }}
          a.button {{
            display: inline-block; padding: 10px 14px; border-radius: 10px;
            text-decoration: none; border: 1px solid #ddd; color: #111;
          }}
          a.primary {{ background: #111; color: #fff; border-color: #111; }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <h2>Try-On Output</h2>
          <img src="{img_url}" alt="Try-On Result"/>
          <div class="btns">
            <a class="button primary" href="{dl_url}">Download Image</a>
            <a class="button" href="{img_url}" target="_blank">Open Raw Image</a>
          </div>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)


# ─────────────────────────────
# Detection helpers
# ─────────────────────────────
def get_person_bbox(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    try:
        r = yolo.predict(source=bgr, verbose=False)[0]
    except Exception:
        return None

    if r.boxes is None or len(r.boxes) == 0:
        return None

    best, best_area = None, 0
    for i in range(len(r.boxes)):
        try:
            if int(r.boxes.cls[i]) != 0:  # 0 = person
                continue
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
        except Exception:
            continue

        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)

    return best

def crop_person(bgr: np.ndarray) -> np.ndarray:
    bb = get_person_bbox(bgr)
    if not bb:
        return bgr

    x1, y1, x2, y2 = bb
    h, w = bgr.shape[:2]
    pad = int(0.08 * max(h, w))

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = bgr[y1:y2, x1:x2].copy()
    return crop if crop.size else bgr


# ─────────────────────────────
# Input diagnostics (safe)
# ─────────────────────────────
def detect_face(bgr: np.ndarray) -> Dict[str, Any]:
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = face_det.process(rgb)
        found = bool(res.detections)
        return {"face_detected": found, "face_count": len(res.detections) if found else 0}
    except Exception:
        return {"face_detected": False, "face_count": 0}

def estimate_body_coverage(bgr: np.ndarray) -> Dict[str, Any]:
    h, w = bgr.shape[:2]
    bb = get_person_bbox(bgr)
    if not bb:
        return {"person_detected": False, "body_coverage": "none", "bbox_area_ratio": 0.0}

    x1, y1, x2, y2 = bb
    bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
    ratio = float(bbox_area / max(1, h * w))

    if ratio < 0.08:
        cov = "tiny_person"
    elif ratio < 0.20:
        cov = "upper_body_or_far"
    elif ratio < 0.45:
        cov = "torso"
    else:
        cov = "full_or_close"

    return {"person_detected": True, "body_coverage": cov, "bbox_area_ratio": round(ratio, 3)}

def estimate_orientation_from_pose(bgr: np.ndarray) -> str:
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            return "unknown"

        lm = res.pose_landmarks.landmark
        ls = float(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility)
        rs = float(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility)
        lh = float(lm[mp_pose.PoseLandmark.LEFT_HIP].visibility)
        rh = float(lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility)

        diff = abs(ls - rs) + abs(lh - rh)
        return "profile_or_turned" if diff > 0.9 else "frontish"
    except Exception:
        return "unknown"

def build_input_diagnostics(bgr: np.ndarray) -> Dict[str, Any]:
    h, w = bgr.shape[:2]
    cov = estimate_body_coverage(bgr)
    face = detect_face(bgr)

    short_side = min(h, w)
    low_res = short_side < 512

    return {
        "image_size": {"width": w, "height": h},
        "low_resolution": low_res,
        "short_side": short_side,
        "orientation": estimate_orientation_from_pose(bgr),
        **cov,
        **face,
    }


# ─────────────────────────────
# Pose assessment (graded; NEVER throws)
# ─────────────────────────────
@dataclass
class PoseAssessment:
    pose_detected: bool
    ok_for_idm: bool
    ok_for_overlay: bool
    score: float
    reasons: List[str]

def assess_pose(user_bgr: np.ndarray) -> PoseAssessment:
    try:
        rgb = cv2.cvtColor(user_bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
    except Exception:
        return PoseAssessment(False, False, False, 0.15, ["Pose estimation failed internally."])

    if not res.pose_landmarks:
        return PoseAssessment(False, False, False, 0.20, ["No pose detected (use clearer, front-facing photo)."])

    lm = res.pose_landmarks.landmark
    shoulders = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hips = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]

    sh_vis = [float(lm[i].visibility) for i in shoulders]
    hip_vis = [float(lm[i].visibility) for i in hips]

    shoulders_ok = all(v >= 0.55 for v in sh_vis)
    hips_ok = all(v >= 0.55 for v in hip_vis)

    reasons: List[str] = []
    if not shoulders_ok:
        reasons.append("Shoulders not clearly visible.")
    if not hips_ok:
        reasons.append("Hips not clearly visible.")

    score = 0.35 + 0.25 * (sum(sh_vis) / 2.0) + 0.25 * (sum(hip_vis) / 2.0)
    h, w = user_bgr.shape[:2]
    if min(h, w) < 512:
        score -= 0.10
        reasons.append("Low resolution (short side < 512px).")

    score = float(max(0.0, min(score, 0.98)))
    return PoseAssessment(True, shoulders_ok and hips_ok, shoulders_ok, score, reasons)


# ─────────────────────────────
# Garment cutout (safe)
# ─────────────────────────────
def garment_cutout_rgba_safe(garment_bgr: np.ndarray) -> Tuple[Image.Image, List[str]]:
    warnings: List[str] = []
    try:
        pil = bgr_to_pil(garment_bgr)
        return remove(pil).convert("RGBA"), warnings
    except Exception as e:
        warnings.append(f"Garment background removal failed ({type(e).__name__}); using opaque fallback.")
        return bgr_to_pil(garment_bgr).convert("RGBA"), warnings

def compute_garment_score(garment_rgba: Image.Image) -> float:
    """Simple quality proxy: alpha coverage not too tiny, not too full."""
    try:
        g = np.array(garment_rgba)
        if g.ndim != 3 or g.shape[2] != 4:
            return 0.25
        alpha = g[:, :, 3].astype(np.float32) / 255.0
        cov = float(np.mean(alpha > 0.15))  # % of pixels that are garment
        # ideal-ish ~ 0.20–0.55 for product cutouts; penalize extremes
        score = 1.0 - min(abs(cov - 0.35) / 0.35, 1.0)
        return clamp01(0.15 + 0.85 * score)
    except Exception:
        return 0.25


# ─────────────────────────────
# Gemini description (optional; safe) - NEW SDK
# ─────────────────────────────
def gemini_describe(garment_rgba: Image.Image) -> str:
    if not gemini_client:
        return "dress"
    try:
        prompt = "Describe this garment in 1 sentence: type, color, pattern, style."

        buf = io.BytesIO()
        garment_rgba.save(buf, format="PNG")
        img_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

        # Note: order [prompt, image] or [image, prompt] both typically work.
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=[prompt, img_part],
        )

        txt = (getattr(resp, "text", None) or "").strip()
        return txt if txt else "dress"
    except Exception:
        return "dress"


# ─────────────────────────────
# Overlay fallback (preview)
# ─────────────────────────────
def overlay(user_bgr: np.ndarray, garment_rgba: Image.Image) -> np.ndarray:
    out = user_bgr.copy()
    h, w = out.shape[:2]
    g_w, g_h = garment_rgba.size

    target_w = int(w * 0.55)
    scale = target_w / max(1, g_w)
    new_w = max(1, int(g_w * scale))
    new_h = max(1, int(g_h * scale))

    g = garment_rgba.resize((new_w, new_h), Image.LANCZOS)

    x0 = (w - new_w) // 2
    y0 = int(h * 0.18)
    x1 = min(w, x0 + new_w)
    y1 = min(h, y0 + new_h)

    if x1 <= x0 or y1 <= y0:
        return out

    g_np = np.array(g.crop((0, 0, x1 - x0, y1 - y0)))  # RGBA
    alpha = (g_np[:, :, 3:4].astype(np.float32)) / 255.0
    g_bgr = g_np[:, :, :3][:, :, ::-1].astype(np.float32)

    region = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (region * (1 - alpha) + g_bgr * alpha).astype(np.uint8)
    return out


# ─────────────────────────────
# IDM-VTON generation (Replicate) + safe wrapper
# ─────────────────────────────
def idm_vton_generate(person_pil: Image.Image, garment_pil: Image.Image, desc: str) -> Image.Image:
    if not idm_vton_ready:
        raise RuntimeError("IDM-VTON not enabled")

    person_bytes = io.BytesIO()
    garment_bytes = io.BytesIO()
    person_pil.save(person_bytes, format="PNG")
    garment_pil.save(garment_bytes, format="PNG")
    person_bytes.seek(0)
    garment_bytes.seek(0)

    out = replicate.run(
        "cuuupid/idm-vton:c871bb9b046607b680449ecbae55fd8c6d945e0a1948644bf2361b3d021d3ff4",
        input={
            "human_img": person_bytes,
            "garm_img": garment_bytes,
            "garment_des": desc,
            "is_checked": True,
            "denoise_steps": 30,
            "seed": 42,
        },
    )

    if hasattr(out, "read"):
        return Image.open(io.BytesIO(out.read())).convert("RGB")

    if isinstance(out, str):
        img_data = requests.get(out, timeout=180).content
        return Image.open(io.BytesIO(img_data)).convert("RGB")

    if isinstance(out, list) and out:
        item = out[0]
        if hasattr(item, "read"):
            return Image.open(io.BytesIO(item.read())).convert("RGB")
        if isinstance(item, str):
            img_data = requests.get(item, timeout=180).content
            return Image.open(io.BytesIO(img_data)).convert("RGB")

    raise RuntimeError(f"Unexpected Replicate output: {type(out)}")

def safe_idm_vton_generate(person_pil: Image.Image, garment_pil: Image.Image, desc: str, attempts: int = 2) -> Image.Image:
    last_err: Optional[Exception] = None
    for _ in range(max(1, attempts)):
        try:
            return idm_vton_generate(person_pil, garment_pil, desc)
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("IDM-VTON failed")


# ─────────────────────────────
# Scoring (swap + overall confidence)
# ─────────────────────────────
def compute_swap_score(mode_used: str, pose_score: float, garment_score: float, diagnostics: Dict[str, Any]) -> float:
    score = 0.60 * pose_score + 0.40 * garment_score

    # Penalize overlay vs IDM (overlay is preview)
    if mode_used.startswith("overlay"):
        score *= 0.85

    # Penalize profile / low-res
    if diagnostics.get("orientation") == "profile_or_turned":
        score *= 0.80
    if diagnostics.get("low_resolution"):
        score *= 0.85

    return clamp01(score)

def compute_overall_confidence(can_tryon: bool, pose_score: float, garment_score: float, swap_score: float) -> float:
    if not can_tryon:
        return clamp01(0.10 + 0.55 * pose_score)
    return clamp01(0.50 * swap_score + 0.30 * pose_score + 0.20 * garment_score)


# ─────────────────────────────
# Unified response builder
# ─────────────────────────────
def make_response(
    request: Request,
    request_id: str,
    can_tryon: bool,
    mode_used: str,
    garment_description: str,
    pose_score: float,
    garment_score: float,
    swap_score: float,
    overall_confidence: float,
    warnings: List[str],
    diagnostics: Dict[str, Any],
    out_bgr: Optional[np.ndarray] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    output_urls: List[str] = []
    output_download_urls: List[str] = []
    output_view_urls: List[str] = []

    if out_bgr is not None:
        fname = f"tryon_{uuid.uuid4().hex[:10]}.jpg"
        save_bgr_jpg(os.path.join(OUTPUT_DIR, fname), out_bgr)

        output_urls = [build_url(request, fname)]
        output_download_urls = [build_download_url(request, fname)]
        output_view_urls = [build_view_url(request, fname)]

    return {
        "request_id": request_id,
        "can_tryon": can_tryon,
        "mode_used": mode_used,
        "garment_description": garment_description,
        "scores": {
            "pose_score": round(float(pose_score), 3),
            "garment_score": round(float(garment_score), 3),
            "swap_score": round(float(swap_score), 3),
            "overall_confidence": round(float(overall_confidence), 3),
        },
        "input_diagnostics": diagnostics,
        "warnings": warnings,
        "error": error,
        "output_urls": output_urls,
        "output_download_urls": output_download_urls,
        "output_view_urls": output_view_urls,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "gemini_enabled": bool(gemini_client),
        "idm_vton_enabled": bool(idm_vton_ready),
        "apis": ["/v1/tryon/actress-to-user", "/v1/tryon/garment-to-user"],
    }


# ─────────────────────────────
# Core pipeline (single entry)
# ─────────────────────────────
def run_pipeline(
    request: Request,
    user_bgr: np.ndarray,
    garment_bgr: np.ndarray,
    garment_des: str,
    prefer_idm: bool,
    mode_hint: str,  # "garment" or "actress"
) -> Dict[str, Any]:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    diagnostics = build_input_diagnostics(user_bgr)

    try:
        # Gating
        if not diagnostics.get("person_detected", False):
            warnings = ["No person detected clearly. Upload a clear upper/full-body photo."]
            if diagnostics.get("face_detected", False):
                warnings = ["Face-only/selfie detected. Upload upper/full-body front-facing photo."]
            return make_response(
                request, request_id, False, "input_invalid_user", garment_des.strip() or "garment",
                0.0, 0.0, 0.0, 0.0, warnings, diagnostics
            )

        if diagnostics.get("body_coverage") == "tiny_person":
            return make_response(
                request, request_id, False, "input_person_too_small", garment_des.strip() or "garment",
                0.0, 0.0, 0.0, 0.0,
                ["Person appears very small/far. Upload a closer upper/full-body photo."],
                diagnostics,
            )

        if diagnostics.get("orientation") == "profile_or_turned":
            prefer_idm = False

        # Crop user for consistent processing
        user_crop = crop_person(user_bgr)

        # Pose assessment
        pa = assess_pose(user_crop)
        pose_score = pa.score

        # Garment cutout
        garment_rgba, cutout_warnings = garment_cutout_rgba_safe(garment_bgr)
        garment_score = compute_garment_score(garment_rgba)

        desc = garment_des.strip() or (gemini_describe(garment_rgba) if gemini_client else "dress")
        warnings = list(pa.reasons) + cutout_warnings

        # If pose not detected: overlay-only if bbox exists
        if not pa.pose_detected:
            if get_person_bbox(user_crop) is not None:
                out_bgr = overlay(user_crop, garment_rgba)
                warnings.append("Pose not detected; generated overlay preview only.")
                mode_used = "overlay_no_pose"
                swap_score = compute_swap_score(mode_used, pose_score, garment_score, diagnostics)
                overall = compute_overall_confidence(True, pose_score, garment_score, swap_score)
                return make_response(
                    request, request_id, True, mode_used, desc,
                    pose_score, garment_score, swap_score, overall,
                    warnings, diagnostics, out_bgr=out_bgr
                )

            warnings.append("No person detected clearly after crop.")
            return make_response(
                request, request_id, False, "no_pose_no_person", desc,
                pose_score, garment_score, 0.0, 0.0,
                warnings, diagnostics
            )

        # Low-res: avoid IDM
        if diagnostics.get("low_resolution", False):
            prefer_idm = False
            warnings.append("Low resolution detected; using overlay preview.")

        # Decide IDM
        idm_allowed = bool(prefer_idm) and idm_vton_ready and pa.ok_for_idm

        if idm_allowed:
            try:
                white = Image.new("RGBA", garment_rgba.size, (255, 255, 255, 255))
                garment_rgb = Image.alpha_composite(white, garment_rgba).convert("RGB")

                out_pil = safe_idm_vton_generate(bgr_to_pil(user_crop), garment_rgb, desc, attempts=2)
                out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)

                mode_used = "idm-vton"
                swap_score = compute_swap_score(mode_used, pose_score, garment_score, diagnostics)
                overall = compute_overall_confidence(True, pose_score, garment_score, swap_score)

                return make_response(
                    request, request_id, True, mode_used, desc,
                    pose_score, garment_score, swap_score, overall,
                    warnings, diagnostics, out_bgr=out_bgr
                )

            except Exception as e:
                warnings.append(f"IDM-VTON failed; using overlay fallback: {type(e).__name__}")

        # Overlay path
        if pa.ok_for_overlay:
            mode_used = "overlay"
            if prefer_idm and idm_vton_ready and not pa.ok_for_idm:
                mode_used = "overlay_pose_weak"
            if diagnostics.get("orientation") == "profile_or_turned":
                mode_used = "overlay_profile"
                warnings.append("Profile/turned detected; overlay may be less accurate. Use front-facing photo.")

            out_bgr = overlay(user_crop, garment_rgba)
            swap_score = compute_swap_score(mode_used, pose_score, garment_score, diagnostics)
            overall = compute_overall_confidence(True, pose_score, garment_score, swap_score)

            return make_response(
                request, request_id, True, mode_used, desc,
                pose_score, garment_score, swap_score, overall,
                warnings, diagnostics, out_bgr=out_bgr
            )

        warnings.append("Pose detected but too weak for overlay; use front-facing photo with shoulders visible.")
        return make_response(
            request, request_id, False, "pose_too_weak", desc,
            pose_score, garment_score, 0.0, 0.0,
            warnings, diagnostics
        )

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{request_id}] Pipeline crash: {type(e).__name__}: {e}\n{tb}")
        return make_response(
            request, request_id, False, "internal_error_safe", garment_des.strip() or "garment",
            0.0, 0.0, 0.0, 0.0,
            ["Processing failed unexpectedly. Retry with clearer full-body, front-facing image."],
            diagnostics,
            error={"type": type(e).__name__, "message": "Unhandled pipeline error"},
        )


# ─────────────────────────────
# Endpoints
# ─────────────────────────────
@app.post("/v1/tryon/garment-to-user")
async def garment_to_user(
    request: Request,
    garment_image: UploadFile = File(...),
    user_image: UploadFile = File(...),
    garment_des: str = Form(""),
    prefer_idm: int = Form(1),
):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    try:
        garment_bytes = await garment_image.read()
        user_bytes = await user_image.read()
        garment_bgr = bytes_to_bgr(garment_bytes)
        user_bgr = bytes_to_bgr(user_bytes)
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content=make_response(
                request, request_id, False, "input_decode_error", garment_des.strip() or "garment",
                0.0, 0.0, 0.0, 0.0,
                ["Failed to read/parse images. Upload valid JPG/PNG images."],
                {},
                error={"type": type(e).__name__, "message": "Image decode failed"},
            ),
        )

    return await anyio.to_thread.run_sync(
        run_pipeline,
        request,
        user_bgr,
        garment_bgr,
        garment_des,
        bool(prefer_idm),
        "garment",
    )


@app.post("/v1/tryon/actress-to-user")
async def actress_to_user(
    request: Request,
    actress_image: UploadFile = File(...),
    user_image: UploadFile = File(...),
    garment_des: str = Form(""),
    prefer_idm: int = Form(1),
):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    try:
        actress_bytes = await actress_image.read()
        user_bytes = await user_image.read()
        actress_bgr = bytes_to_bgr(actress_bytes)
        user_bgr = bytes_to_bgr(user_bytes)
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content=make_response(
                request, request_id, False, "input_decode_error", garment_des.strip() or "garment",
                0.0, 0.0, 0.0, 0.0,
                ["Failed to read/parse images. Upload valid JPG/PNG images."],
                {},
                error={"type": type(e).__name__, "message": "Image decode failed"},
            ),
        )

    actress_crop = crop_person(actress_bgr)

    return await anyio.to_thread.run_sync(
        run_pipeline,
        request,
        user_bgr,
        actress_crop,
        garment_des,
        bool(prefer_idm),
        "actress",
    )


# ─────────────────────────────
# Global exception handler (stable JSON)
# ─────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    tb = traceback.format_exc()
    print(f"[{request_id}] Unhandled exception: {type(exc).__name__}: {exc}\n{tb}")
    return JSONResponse(
        status_code=200,
        content=make_response(
            request, request_id, False, "unhandled_exception", "garment",
            0.0, 0.0, 0.0, 0.0,
            ["Unexpected server error. Please retry with a clearer full-body, front-facing image."],
            {},
            error={"type": type(exc).__name__, "message": "Unhandled server exception"},
        ),
    )


if __name__ == "__main__":
    print("✅ Starting FastAPI on http://127.0.0.1:8000")
    uvicorn.run("backend_gemini_overlay:app", host="0.0.0.0", port=8000, reload=True, log_level="info")



# ─────────────────────────────
# Start
# ─────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("backend_gemini_overlay:app", host="0.0.0.0", port=port, log_level="info")

