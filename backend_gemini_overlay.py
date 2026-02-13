# backend_gemini_overlay.py
# Virtual Try-On – Hardened (Refactored + Scores) — Render-safe
# ✅ Fix: do NOT load YOLO/MediaPipe at import time (bind port first)
# ✅ Fix: MediaPipe safety — handle wrong/invalid mediapipe module gracefully

import os
import io
import sys
import uuid
import traceback
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import uvicorn
import anyio
import requests
import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ─────────────────────────────
# STARTUP LOGGING
# ─────────────────────────────
print("=" * 60, file=sys.stderr, flush=True)
print("🚀 INITIALIZING VIRTUAL TRY-ON SERVICE", file=sys.stderr, flush=True)
print("=" * 60, file=sys.stderr, flush=True)
print(f"Python: {sys.version}", file=sys.stderr, flush=True)
print(f"Working Dir: {os.getcwd()}", file=sys.stderr, flush=True)
print(f"PORT env: {os.getenv('PORT', 'NOT SET')}", file=sys.stderr, flush=True)

# ─────────────────────────────
# LOAD ENV VARS
# ─────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment loaded", file=sys.stderr, flush=True)
except Exception as e:
    print(f"⚠️ dotenv load failed (continuing): {e}", file=sys.stderr, flush=True)

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

# Set YOLO config to writable location
YOLO_CONFIG_DIR = os.getenv("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = YOLO_CONFIG_DIR
print(f"✅ YOLO config dir: {YOLO_CONFIG_DIR}", file=sys.stderr, flush=True)

# ─────────────────────────────
# INITIALIZE FASTAPI APP
# ─────────────────────────────
app = FastAPI(title="Virtual Try-On – Hardened (Refactored + Scores) — Render Safe")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
print("✅ FastAPI app created", file=sys.stderr, flush=True)

# ─────────────────────────────
# REQUEST ID MIDDLEWARE
# ─────────────────────────────
class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

app.add_middleware(RequestIdMiddleware)

# ─────────────────────────────
# MediaPipe VALIDATION HELPERS (NEW)
# ─────────────────────────────
def _is_real_mediapipe(mp_module: Any) -> bool:
    # Real mediapipe exposes mp.solutions
    return bool(getattr(mp_module, "solutions", None))

def _mp_debug(mp_module: Any) -> str:
    return f"file={getattr(mp_module,'__file__',None)} version={getattr(mp_module,'__version__',None)} has_solutions={hasattr(mp_module,'solutions')}"

# ─────────────────────────────
# LAZY MODEL STORE (Render-safe)
# ─────────────────────────────
class ModelStore:
    """
    Lazily loads heavy ML deps so Uvicorn can bind port immediately.
    Thread-safe for first-load.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._yolo = None
        self._mp = None
        self._mp_pose = None
        self._pose = None
        self._mp_face = None
        self._face_det = None
        self._remove = None
        self._mp_validated = False
        self._mp_ok = False

    def yolo(self):
        with self._lock:
            if self._yolo is None:
                print("📥 Lazy-loading YOLO model...", file=sys.stderr, flush=True)
                from ultralytics import YOLO
                self._yolo = YOLO("yolov8n-seg.pt")
                print("✅ YOLO loaded", file=sys.stderr, flush=True)
            return self._yolo

    def _load_mediapipe(self):
        """
        Safely load mediapipe. If a wrong/stub module is installed or shadowed,
        keep service running and raise a friendly error only when pose/face is used.
        """
        if self._mp_validated:
            if not self._mp_ok:
                raise RuntimeError("MediaPipe unavailable (invalid module loaded).")
            return self._mp

        import mediapipe as mp  # lazy import
        self._mp_validated = True

        if not _is_real_mediapipe(mp):
            # This is your current error: AttributeError: module 'mediapipe' has no attribute 'solutions'
            # We do NOT crash the server; just mark mediapipe unusable.
            self._mp_ok = False
            self._mp = None
            msg = f"⚠️ Invalid mediapipe module loaded. {_mp_debug(mp)}"
            print(msg, file=sys.stderr, flush=True)
            raise RuntimeError("MediaPipe unavailable (invalid module).")

        self._mp_ok = True
        self._mp = mp
        print(f"✅ MediaPipe validated. {_mp_debug(mp)}", file=sys.stderr, flush=True)
        return self._mp

    def mp_pose(self):
        with self._lock:
            mp = self._load_mediapipe()

            if self._pose is None or self._mp_pose is None:
                print("📥 Lazy-loading MediaPipe Pose...", file=sys.stderr, flush=True)
                self._mp_pose = mp.solutions.pose
                self._pose = self._mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=False,
                )
                print("✅ MediaPipe Pose loaded", file=sys.stderr, flush=True)

            return self._mp_pose, self._pose

    def face_det(self):
        with self._lock:
            mp = self._load_mediapipe()

            if self._face_det is None or self._mp_face is None:
                print("📥 Lazy-loading MediaPipe Face Detection...", file=sys.stderr, flush=True)
                self._mp_face = mp.solutions.face_detection
                self._face_det = self._mp_face.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=0.5,
                )
                print("✅ MediaPipe Face Detection loaded", file=sys.stderr, flush=True)

            return self._face_det

    def rembg_remove(self):
        with self._lock:
            if self._remove is None:
                print("📥 Lazy-loading rembg...", file=sys.stderr, flush=True)
                from rembg import remove
                self._remove = remove
                print("✅ rembg loaded", file=sys.stderr, flush=True)
            return self._remove

models = ModelStore()

# ─────────────────────────────
# OPTIONAL: background warmup (non-blocking)
# ─────────────────────────────
@app.on_event("startup")
async def warmup_models():
    """
    Kick off model downloads in a background thread.
    DOES NOT block port binding.
    """
    def _warm():
        # Warmup should never fail the service.
        try:
            models.yolo()
        except Exception as e:
            print(f"⚠️ YOLO warmup skipped: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

        try:
            models.mp_pose()
            models.face_det()
        except Exception as e:
            # This is where mediapipe error will land; we skip it safely.
            print(f"⚠️ MediaPipe warmup skipped: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

        try:
            models.rembg_remove()
        except Exception as e:
            print(f"⚠️ rembg warmup skipped: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

        print("✅ Warmup thread finished (service runs).", file=sys.stderr, flush=True)

    threading.Thread(target=_warm, daemon=True).start()

# ─────────────────────────────
# GEMINI SETUP (optional)
# ─────────────────────────────
gemini_client: Optional[Any] = None
if ENABLE_GEMINI and GEMINI_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_KEY)
        print("✅ Gemini enabled (google-genai)", file=sys.stderr, flush=True)
    except Exception as e:
        gemini_client = None
        print(f"⚠️ Gemini init failed ({type(e).__name__}). Disabling Gemini.", file=sys.stderr, flush=True)
else:
    print("ℹ️ Gemini disabled", file=sys.stderr, flush=True)

# ─────────────────────────────
# REPLICATE SETUP (optional)
# ─────────────────────────────
idm_vton_ready = False
if ENABLE_IDM_VTON:
    if not REPLICATE_TOKEN:
        print("⚠️ IDM-VTON requested but REPLICATE_API_TOKEN missing. Disabling IDM-VTON.", file=sys.stderr, flush=True)
    else:
        try:
            import replicate
            os.environ["REPLICATE_API_TOKEN"] = REPLICATE_TOKEN
            idm_vton_ready = True
            print("✅ IDM-VTON enabled via Replicate", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"⚠️ Replicate import failed: {e}", file=sys.stderr, flush=True)
else:
    print("ℹ️ IDM-VTON disabled (overlay only)", file=sys.stderr, flush=True)

# ─────────────────────────────
# UTILITY FUNCTIONS
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
    fname = os.path.basename(filename)
    return os.path.join(OUTPUT_DIR, fname)

def build_download_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/download/{filename}"

def build_view_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/view/{filename}"

# ─────────────────────────────
# ENDPOINTS: DOWNLOAD & VIEW
# ─────────────────────────────
@app.get("/download/{filename}")
def download_output(request: Request, filename: str):
    path = safe_output_path(filename)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"ok": False, "error": "file_not_found"})
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
      </head>
      <body style="font-family:Arial;margin:24px">
        <h2>Try-On Output</h2>
        <img src="{img_url}" style="width:100%;max-width:900px;border-radius:12px" />
        <div style="margin-top:16px;display:flex;gap:12px;flex-wrap:wrap">
          <a href="{dl_url}" style="padding:10px 14px;border:1px solid #111;border-radius:10px;background:#111;color:#fff;text-decoration:none">Download Image</a>
          <a href="{img_url}" target="_blank" style="padding:10px 14px;border:1px solid #ddd;border-radius:10px;color:#111;text-decoration:none">Open Raw Image</a>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)

# ─────────────────────────────
# DETECTION HELPERS (use lazy models)
# ─────────────────────────────
def get_person_bbox(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    try:
        yolo = models.yolo()
        r = yolo.predict(source=bgr, verbose=False)[0]
    except Exception:
        return None

    if r.boxes is None or len(r.boxes) == 0:
        return None

    best, best_area = None, 0
    for i in range(len(r.boxes)):
        try:
            if int(r.boxes.cls[i]) != 0:
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
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    crop = bgr[y1:y2, x1:x2].copy()
    return crop if crop.size else bgr

def detect_face(bgr: np.ndarray) -> Dict[str, Any]:
    try:
        face_det = models.face_det()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = face_det.process(rgb)
        found = bool(res.detections)
        return {"face_detected": found, "face_count": len(res.detections) if found else 0}
    except Exception:
        # if mediapipe unavailable, gracefully return "no face"
        return {"face_detected": False, "face_count": 0}

def estimate_orientation_from_pose(bgr: np.ndarray) -> str:
    try:
        mp_pose, pose = models.mp_pose()
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
        # if mediapipe unavailable, don't block pipeline
        return "unknown"

def estimate_body_coverage(bgr: np.ndarray) -> Dict[str, Any]:
    h, w = bgr.shape[:2]
    bb = get_person_bbox(bgr)
    if not bb:
        return {"person_detected": False, "body_coverage": "none", "bbox_area_ratio": 0.0}

    x1, y1, x2, y2 = bb
    bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
    ratio = float(bbox_area / max(1, h * w))

    if ratio < 0.08: cov = "tiny_person"
    elif ratio < 0.20: cov = "upper_body_or_far"
    elif ratio < 0.45: cov = "torso"
    else: cov = "full_or_close"

    return {"person_detected": True, "body_coverage": cov, "bbox_area_ratio": round(ratio, 3)}

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
# POSE ASSESSMENT (lazy pose)
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
        mp_pose, pose = models.mp_pose()
        rgb = cv2.cvtColor(user_bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
    except Exception as e:
        # If mediapipe is invalid/unavailable, don't crash service; just degrade
        return PoseAssessment(False, False, False, 0.15, [f"Pose estimation unavailable: {type(e).__name__}."])

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
    if not shoulders_ok: reasons.append("Shoulders not clearly visible.")
    if not hips_ok: reasons.append("Hips not clearly visible.")

    score = 0.35 + 0.25 * (sum(sh_vis) / 2.0) + 0.25 * (sum(hip_vis) / 2.0)
    h, w = user_bgr.shape[:2]
    if min(h, w) < 512:
        score -= 0.10
        reasons.append("Low resolution (short side < 512px).")

    score = float(max(0.0, min(score, 0.98)))
    return PoseAssessment(True, shoulders_ok and hips_ok, shoulders_ok, score, reasons)

# ─────────────────────────────
# GARMENT PROCESSING (lazy rembg)
# ─────────────────────────────
def garment_cutout_rgba_safe(garment_bgr: np.ndarray) -> Tuple[Image.Image, List[str]]:
    warnings: List[str] = []
    try:
        pil = bgr_to_pil(garment_bgr)
        remove_fn = models.rembg_remove()
        return remove_fn(pil).convert("RGBA"), warnings
    except Exception as e:
        warnings.append(f"Garment background removal failed ({type(e).__name__}); using opaque fallback.")
        return bgr_to_pil(garment_bgr).convert("RGBA"), warnings

def compute_garment_score(garment_rgba: Image.Image) -> float:
    try:
        g = np.array(garment_rgba)
        if g.ndim != 3 or g.shape[2] != 4:
            return 0.25
        alpha = g[:, :, 3].astype(np.float32) / 255.0
        cov = float(np.mean(alpha > 0.15))
        score = 1.0 - min(abs(cov - 0.35) / 0.35, 1.0)
        return clamp01(0.15 + 0.85 * score)
    except Exception:
        return 0.25

# ─────────────────────────────
# GEMINI DESCRIPTION (unchanged)
# ─────────────────────────────
def gemini_describe(garment_rgba: Image.Image) -> str:
    if not gemini_client:
        return "dress"
    try:
        from google.genai import types
        prompt = "Describe this garment in 1 sentence: type, color, pattern, style."
        buf = io.BytesIO()
        garment_rgba.save(buf, format="PNG")
        img_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
        resp = gemini_client.models.generate_content(model=GEMINI_MODEL_ID, contents=[prompt, img_part])
        txt = (getattr(resp, "text", None) or "").strip()
        return txt if txt else "dress"
    except Exception:
        return "dress"

# ─────────────────────────────
# OVERLAY (unchanged)
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
    g_np = np.array(g.crop((0, 0, x1 - x0, y1 - y0)))
    alpha = (g_np[:, :, 3:4].astype(np.float32)) / 255.0
    g_bgr = g_np[:, :, :3][:, :, ::-1].astype(np.float32)
    region = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (region * (1 - alpha) + g_bgr * alpha).astype(np.uint8)
    return out

# ─────────────────────────────
# IDM-VTON GENERATION (unchanged)
# ─────────────────────────────
def idm_vton_generate(person_pil: Image.Image, garment_pil: Image.Image, desc: str) -> Image.Image:
    if not idm_vton_ready:
        raise RuntimeError("IDM-VTON not enabled")

    import replicate

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
# SCORING + RESPONSE (unchanged)
# ─────────────────────────────
def compute_swap_score(mode_used: str, pose_score: float, garment_score: float, diagnostics: Dict[str, Any]) -> float:
    score = 0.60 * pose_score + 0.40 * garment_score
    if mode_used.startswith("overlay"):
        score *= 0.85
    if diagnostics.get("orientation") == "profile_or_turned":
        score *= 0.80
    if diagnostics.get("low_resolution"):
        score *= 0.85
    return clamp01(score)

def compute_overall_confidence(can_tryon: bool, pose_score: float, garment_score: float, swap_score: float) -> float:
    if not can_tryon:
        return clamp01(0.10 + 0.55 * pose_score)
    return clamp01(0.50 * swap_score + 0.30 * pose_score + 0.20 * garment_score)

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

# ─────────────────────────────
# HEALTH CHECK
# ─────────────────────────────
@app.get("/")
@app.get("/health")
def health():
    return {
        "ok": True,
        "status": "healthy",
        "gemini_enabled": bool(gemini_client),
        "idm_vton_enabled": bool(idm_vton_ready),
        "note": "Render-safe: models are lazy-loaded; first request may be slower.",
        "apis": ["/v1/tryon/actress-to-user", "/v1/tryon/garment-to-user"],
    }

# ─────────────────────────────
# CORE PIPELINE
# ─────────────────────────────
def run_pipeline(
    request: Request,
    user_bgr: np.ndarray,
    garment_bgr: np.ndarray,
    garment_des: str,
    prefer_idm: bool,
    mode_hint: str,
) -> Dict[str, Any]:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    diagnostics = build_input_diagnostics(user_bgr)

    try:
        if not diagnostics.get("person_detected", False):
            warnings = ["No person detected clearly. Upload a clear upper/full-body photo."]
            if diagnostics.get("face_detected", False):
                warnings = ["Face-only/selfie detected. Upload upper/full-body front-facing photo."]
            return make_response(request, request_id, False, "input_invalid_user",
                                 garment_des.strip() or "garment",
                                 0.0, 0.0, 0.0, 0.0, warnings, diagnostics)

        if diagnostics.get("body_coverage") == "tiny_person":
            return make_response(request, request_id, False, "input_person_too_small",
                                 garment_des.strip() or "garment",
                                 0.0, 0.0, 0.0, 0.0,
                                 ["Person appears very small/far. Upload a closer upper/full-body photo."],
                                 diagnostics)

        if diagnostics.get("orientation") == "profile_or_turned":
            prefer_idm = False

        user_crop = crop_person(user_bgr)
        pa = assess_pose(user_crop)
        pose_score = pa.score

        garment_rgba, cutout_warnings = garment_cutout_rgba_safe(garment_bgr)
        garment_score = compute_garment_score(garment_rgba)

        desc = garment_des.strip() or (gemini_describe(garment_rgba) if gemini_client else "dress")
        warnings = list(pa.reasons) + cutout_warnings

        if not pa.pose_detected:
            if get_person_bbox(user_crop) is not None:
                out_bgr = overlay(user_crop, garment_rgba)
                warnings.append("Pose not detected; generated overlay preview only.")
                mode_used = "overlay_no_pose"
                swap_score = compute_swap_score(mode_used, pose_score, garment_score, diagnostics)
                overall = compute_overall_confidence(True, pose_score, garment_score, swap_score)
                return make_response(request, request_id, True, mode_used, desc,
                                     pose_score, garment_score, swap_score, overall,
                                     warnings, diagnostics, out_bgr=out_bgr)

            warnings.append("No person detected clearly after crop.")
            return make_response(request, request_id, False, "no_pose_no_person", desc,
                                 pose_score, garment_score, 0.0, 0.0,
                                 warnings, diagnostics)

        if diagnostics.get("low_resolution", False):
            prefer_idm = False
            warnings.append("Low resolution detected; using overlay preview.")

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

                return make_response(request, request_id, True, mode_used, desc,
                                     pose_score, garment_score, swap_score, overall,
                                     warnings, diagnostics, out_bgr=out_bgr)

            except Exception as e:
                warnings.append(f"IDM-VTON failed; using overlay fallback: {type(e).__name__}")

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

            return make_response(request, request_id, True, mode_used, desc,
                                 pose_score, garment_score, swap_score, overall,
                                 warnings, diagnostics, out_bgr=out_bgr)

        warnings.append("Pose detected but too weak for overlay; use front-facing photo with shoulders visible.")
        return make_response(request, request_id, False, "pose_too_weak", desc,
                             pose_score, garment_score, 0.0, 0.0,
                             warnings, diagnostics)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{request_id}] Pipeline crash: {type(e).__name__}: {e}\n{tb}", file=sys.stderr, flush=True)
        return make_response(
            request, request_id, False, "internal_error_safe", garment_des.strip() or "garment",
            0.0, 0.0, 0.0, 0.0,
            ["Processing failed unexpectedly. Retry with clearer full-body, front-facing image."],
            diagnostics,
            error={"type": type(e).__name__, "message": "Unhandled pipeline error"},
        )

# ─────────────────────────────
# API ENDPOINTS
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
                error={"type": type(e).__name__", "message": "Image decode failed"},
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
# GLOBAL EXCEPTION HANDLER
# ─────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    tb = traceback.format_exc()
    print(f"[{request_id}] Unhandled exception: {type(exc).__name__}: {exc}\n{tb}", file=sys.stderr, flush=True)
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

# IMPORTANT:
# On Render, DO NOT run uvicorn here. Render already runs:
# uvicorn backend_gemini_overlay:app --host 0.0.0.0 --port $PORT
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("backend_gemini_overlay:app", host="0.0.0.0", port=port, log_level="info")
