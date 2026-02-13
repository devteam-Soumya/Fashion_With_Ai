# backend_gemini_overlay.py
# Virtual Try-On – Hardened v4 — Render-safe + MediaPipe Tasks API
#
# ✅ Fix 1: MediaPipe >=0.10.31 removed mp.solutions — migrated to MediaPipe Tasks
#            (mediapipe.tasks.python.vision) for Pose + Face detection
# ✅ Fix 2: anyio.to_thread.run_sync() wrapped in lambda (multi-arg safe)
# ✅ Fix 3: CapacityLimiter to prevent OOM → 502 on Render free tier
# ✅ Fix 4: asyncio.wait_for() timeout (25s) to avoid Render 30s hard kill → 502
# ✅ Fix 5: DISABLE_WARMUP env var for debugging

import os
import io
import sys
import uuid
import traceback
import threading
import asyncio
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import uvicorn
import anyio
import anyio.to_thread
import requests
import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ─────────────────────────────
# ENV SAFETY
# ─────────────────────────────
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENABLE_GEMINI        = os.getenv("ENABLE_GEMINI", "1") == "1"
ENABLE_IDM_VTON      = os.getenv("ENABLE_IDM_VTON", "1") == "1"
DISABLE_WARMUP       = os.getenv("DISABLE_WARMUP", "0") == "1"
GEMINI_KEY           = os.getenv("GEMINI_API_KEY", "")
REPLICATE_TOKEN      = os.getenv("REPLICATE_API_TOKEN", "")
GEMINI_MODEL_ID      = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
PIPELINE_TIMEOUT     = int(os.getenv("PIPELINE_TIMEOUT_SECS", "25"))
MAX_PIPELINE_THREADS = int(os.getenv("MAX_PIPELINE_THREADS", "2"))

YOLO_CONFIG_DIR = os.getenv("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = YOLO_CONFIG_DIR

# MediaPipe Tasks model download dir
MP_MODELS_DIR = os.getenv("MP_MODELS_DIR", "/tmp/mp_models")
os.makedirs(MP_MODELS_DIR, exist_ok=True)

print("=" * 60, file=sys.stderr, flush=True)
print("🚀 VIRTUAL TRY-ON SERVICE v4 (MediaPipe Tasks)", file=sys.stderr, flush=True)
print("=" * 60, file=sys.stderr, flush=True)
print(f"Python: {sys.version}", file=sys.stderr, flush=True)
print(f"Working Dir: {os.getcwd()}", file=sys.stderr, flush=True)
print(f"PORT: {os.getenv('PORT', 'NOT SET')}", file=sys.stderr, flush=True)

# ─────────────────────────────
# FASTAPI
# ─────────────────────────────
app = FastAPI(title="Virtual Try-On v4 — MediaPipe Tasks + Render Safe")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

_limiter: Optional[anyio.CapacityLimiter] = None

def get_limiter() -> anyio.CapacityLimiter:
    global _limiter
    if _limiter is None:
        _limiter = anyio.CapacityLimiter(MAX_PIPELINE_THREADS)
    return _limiter


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

app.add_middleware(RequestIdMiddleware)

# ─────────────────────────────
# GEMINI (optional)
# ─────────────────────────────
gemini_client: Optional[Any] = None
if ENABLE_GEMINI and GEMINI_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_KEY)
        print("✅ Gemini enabled", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"⚠️ Gemini init failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
else:
    print("ℹ️ Gemini disabled", file=sys.stderr, flush=True)

# ─────────────────────────────
# REPLICATE / IDM-VTON (optional)
# ─────────────────────────────
idm_vton_ready = False
if ENABLE_IDM_VTON:
    if not REPLICATE_TOKEN:
        print("⚠️ IDM-VTON: REPLICATE_API_TOKEN missing. Disabling.", file=sys.stderr, flush=True)
    else:
        try:
            import replicate  # noqa: F401
            os.environ["REPLICATE_API_TOKEN"] = REPLICATE_TOKEN
            idm_vton_ready = True
            print("✅ IDM-VTON enabled via Replicate", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"⚠️ Replicate import failed: {e}", file=sys.stderr, flush=True)
else:
    print("ℹ️ IDM-VTON disabled", file=sys.stderr, flush=True)

# ─────────────────────────────
# MEDIAPIPE TASKS — model download helpers
# ─────────────────────────────
# MediaPipe Tasks needs .task model files downloaded at runtime.
# We cache them in MP_MODELS_DIR (/tmp/mp_models) which persists for the process.

_MP_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_MP_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

def _download_model(url: str, dest: str) -> str:
    """Download model file if not already cached. Returns local path."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return dest
    print(f"📥 Downloading MediaPipe model: {os.path.basename(dest)} ...", file=sys.stderr, flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"✅ Model downloaded: {dest}", file=sys.stderr, flush=True)
    return dest

# ─────────────────────────────
# LAZY MODEL STORE
# ─────────────────────────────
class ModelStore:
    """
    Lazily loads all heavy ML models.
    Uses MediaPipe Tasks API (works with mediapipe>=0.10.0).
    Falls back gracefully if MediaPipe is unavailable.
    """
    def __init__(self):
        self._lock = threading.Lock()

        # YOLO
        self._yolo = None

        # MediaPipe Tasks
        self._mp_tasks_ok: Optional[bool] = None   # None = untested
        self._pose_landmarker = None
        self._face_detector   = None

        # rembg
        self._remove = None

    # ── YOLO ──────────────────────────────────────────────────────────────
    def yolo(self):
        with self._lock:
            if self._yolo is None:
                print("📥 Lazy-loading YOLO...", file=sys.stderr, flush=True)
                from ultralytics import YOLO
                self._yolo = YOLO("yolov8n-seg.pt")
                print("✅ YOLO loaded", file=sys.stderr, flush=True)
            return self._yolo

    # ── MediaPipe Tasks: shared import check ──────────────────────────────
    def _check_mp_tasks(self):
        """
        Verify mediapipe.tasks is importable. Sets self._mp_tasks_ok.
        Safe to call multiple times (cached after first call).
        """
        if self._mp_tasks_ok is not None:
            return self._mp_tasks_ok
        try:
            import mediapipe as mp
            # Try Tasks import
            from mediapipe.tasks.python import vision  # noqa: F401
            self._mp_tasks_ok = True
            print(
                f"✅ MediaPipe Tasks available (version={getattr(mp,'__version__','?')})",
                file=sys.stderr, flush=True,
            )
        except Exception as e:
            self._mp_tasks_ok = False
            print(
                f"⚠️ MediaPipe Tasks unavailable: {type(e).__name__}: {e}",
                file=sys.stderr, flush=True,
            )
        return self._mp_tasks_ok

    # ── Pose Landmarker ───────────────────────────────────────────────────
    def pose_landmarker(self):
        with self._lock:
            if not self._check_mp_tasks():
                raise RuntimeError("MediaPipe Tasks not available.")
            if self._pose_landmarker is None:
                print("📥 Lazy-loading MediaPipe Pose Landmarker...", file=sys.stderr, flush=True)
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python.core.base_options import BaseOptions

                model_path = _download_model(
                    _MP_POSE_MODEL_URL,
                    os.path.join(MP_MODELS_DIR, "pose_landmarker_lite.task"),
                )
                options = vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                )
                self._pose_landmarker = vision.PoseLandmarker.create_from_options(options)
                print("✅ Pose Landmarker loaded", file=sys.stderr, flush=True)
            return self._pose_landmarker

    # ── Face Detector ─────────────────────────────────────────────────────
    def face_detector(self):
        with self._lock:
            if not self._check_mp_tasks():
                raise RuntimeError("MediaPipe Tasks not available.")
            if self._face_detector is None:
                print("📥 Lazy-loading MediaPipe Face Detector...", file=sys.stderr, flush=True)
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python.core.base_options import BaseOptions

                model_path = _download_model(
                    _MP_FACE_MODEL_URL,
                    os.path.join(MP_MODELS_DIR, "blaze_face_short_range.tflite"),
                )
                options = vision.FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=vision.RunningMode.IMAGE,
                )
                self._face_detector = vision.FaceDetector.create_from_options(options)
                print("✅ Face Detector loaded", file=sys.stderr, flush=True)
            return self._face_detector

    # ── rembg ─────────────────────────────────────────────────────────────
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
# BACKGROUND WARMUP
# ─────────────────────────────
@app.on_event("startup")
async def warmup_models():
    if DISABLE_WARMUP:
        print("ℹ️ Warmup disabled (DISABLE_WARMUP=1)", file=sys.stderr, flush=True)
        return

    def _warm():
        for name, fn in [
            ("YOLO",             models.yolo),
            ("MP Pose",          models.pose_landmarker),
            ("MP Face",          models.face_detector),
            ("rembg",            models.rembg_remove),
        ]:
            try:
                fn()
            except Exception as e:
                print(f"⚠️ {name} warmup skipped: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print("✅ Warmup finished", file=sys.stderr, flush=True)

    threading.Thread(target=_warm, daemon=True).start()

# ─────────────────────────────
# UTILITIES
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
    return f"{str(request.base_url).rstrip('/')}/outputs/{filename}"

def safe_output_path(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, os.path.basename(filename))

def build_download_url(request: Request, filename: str) -> str:
    return f"{str(request.base_url).rstrip('/')}/download/{filename}"

def build_view_url(request: Request, filename: str) -> str:
    return f"{str(request.base_url).rstrip('/')}/view/{filename}"

# ─────────────────────────────
# DOWNLOAD / VIEW ENDPOINTS
# ─────────────────────────────
@app.get("/download/{filename}")
def download_output(request: Request, filename: str):
    path = safe_output_path(filename)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"ok": False, "error": "file_not_found"})
    return FileResponse(path=path, media_type="image/jpeg",
                        filename=os.path.basename(filename),
                        headers={"Cache-Control": "no-store"})

@app.get("/view/{filename}")
def view_output(request: Request, filename: str):
    path = safe_output_path(filename)
    if not os.path.exists(path):
        return HTMLResponse("<h3>File not found</h3>", status_code=404)
    img_url = build_url(request, os.path.basename(filename))
    dl_url  = build_download_url(request, os.path.basename(filename))
    html = f"""<!doctype html><html>
<head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Try-On Result</title></head>
<body style="font-family:Arial;margin:24px">
<h2>Try-On Output</h2>
<img src="{img_url}" style="width:100%;max-width:900px;border-radius:12px"/>
<div style="margin-top:16px;display:flex;gap:12px;flex-wrap:wrap">
<a href="{dl_url}" style="padding:10px 14px;border:1px solid #111;border-radius:10px;background:#111;color:#fff;text-decoration:none">Download</a>
<a href="{img_url}" target="_blank" style="padding:10px 14px;border:1px solid #ddd;border-radius:10px;color:#111;text-decoration:none">Open Raw</a>
</div></body></html>"""
    return HTMLResponse(html)

# ─────────────────────────────
# DETECTION HELPERS
# ─────────────────────────────
def get_person_bbox(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    try:
        r = models.yolo().predict(source=bgr, verbose=False)[0]
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
    x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad);  y2 = min(h, y2 + pad)
    crop = bgr[y1:y2, x1:x2].copy()
    return crop if crop.size else bgr

def _bgr_to_mp_image(bgr: np.ndarray):
    """Convert BGR numpy array to mediapipe.Image (RGB)."""
    import mediapipe as mp
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

def detect_face(bgr: np.ndarray) -> Dict[str, Any]:
    """Returns face detection info. Degrades gracefully if MediaPipe unavailable."""
    try:
        detector = models.face_detector()
        mp_image = _bgr_to_mp_image(bgr)
        result   = detector.detect(mp_image)
        found    = bool(result.detections)
        return {"face_detected": found, "face_count": len(result.detections) if found else 0}
    except Exception:
        return {"face_detected": False, "face_count": 0}

def estimate_orientation_from_pose(bgr: np.ndarray) -> str:
    """Estimate if person is front-facing or turned. Degrades gracefully."""
    try:
        landmarker = models.pose_landmarker()
        mp_image   = _bgr_to_mp_image(bgr)
        result     = landmarker.detect(mp_image)
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return "unknown"

        lm = result.pose_landmarks[0]  # first pose, list of NormalizedLandmark
        # Indices: 11=left_shoulder, 12=right_shoulder, 23=left_hip, 24=right_hip
        ls_vis = float(lm[11].visibility) if hasattr(lm[11], "visibility") else 0.5
        rs_vis = float(lm[12].visibility) if hasattr(lm[12], "visibility") else 0.5
        lh_vis = float(lm[23].visibility) if hasattr(lm[23], "visibility") else 0.5
        rh_vis = float(lm[24].visibility) if hasattr(lm[24], "visibility") else 0.5

        diff = abs(ls_vis - rs_vis) + abs(lh_vis - rh_vis)
        return "profile_or_turned" if diff > 0.9 else "frontish"
    except Exception:
        return "unknown"

def estimate_body_coverage(bgr: np.ndarray) -> Dict[str, Any]:
    h, w = bgr.shape[:2]
    bb   = get_person_bbox(bgr)
    if not bb:
        return {"person_detected": False, "body_coverage": "none", "bbox_area_ratio": 0.0}
    x1, y1, x2, y2 = bb
    ratio = float(max(0, x2-x1) * max(0, y2-y1) / max(1, h * w))
    if   ratio < 0.08: cov = "tiny_person"
    elif ratio < 0.20: cov = "upper_body_or_far"
    elif ratio < 0.45: cov = "torso"
    else:              cov = "full_or_close"
    return {"person_detected": True, "body_coverage": cov, "bbox_area_ratio": round(ratio, 3)}

def build_input_diagnostics(bgr: np.ndarray) -> Dict[str, Any]:
    h, w       = bgr.shape[:2]
    cov        = estimate_body_coverage(bgr)
    face       = detect_face(bgr)
    short_side = min(h, w)
    return {
        "image_size":     {"width": w, "height": h},
        "low_resolution": short_side < 512,
        "short_side":     short_side,
        "orientation":    estimate_orientation_from_pose(bgr),
        **cov,
        **face,
    }

# ─────────────────────────────
# POSE ASSESSMENT (MediaPipe Tasks)
# ─────────────────────────────
@dataclass
class PoseAssessment:
    pose_detected:  bool
    ok_for_idm:     bool
    ok_for_overlay: bool
    score:          float
    reasons:        List[str]

# Pose landmark indices in MediaPipe Tasks (same numbering as old solutions API)
_IDX_LEFT_SHOULDER  = 11
_IDX_RIGHT_SHOULDER = 12
_IDX_LEFT_HIP       = 23
_IDX_RIGHT_HIP      = 24

def assess_pose(user_bgr: np.ndarray) -> PoseAssessment:
    try:
        landmarker = models.pose_landmarker()
        mp_image   = _bgr_to_mp_image(user_bgr)
        result     = landmarker.detect(mp_image)
    except Exception as e:
        # MediaPipe fully unavailable → degrade gracefully, allow overlay
        return PoseAssessment(False, False, True, 0.30,
                              [f"Pose unavailable ({type(e).__name__}). Using overlay-only."])

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return PoseAssessment(False, False, False, 0.20,
                              ["No pose detected (use clearer, front-facing photo)."])

    lm = result.pose_landmarks[0]

    def _vis(idx: int) -> float:
        try:
            v = lm[idx].visibility
            return float(v) if v is not None else 0.5
        except Exception:
            return 0.5

    sh_vis  = [_vis(_IDX_LEFT_SHOULDER),  _vis(_IDX_RIGHT_SHOULDER)]
    hip_vis = [_vis(_IDX_LEFT_HIP),       _vis(_IDX_RIGHT_HIP)]

    shoulders_ok = all(v >= 0.55 for v in sh_vis)
    hips_ok      = all(v >= 0.55 for v in hip_vis)

    reasons: List[str] = []
    if not shoulders_ok: reasons.append("Shoulders not clearly visible.")
    if not hips_ok:      reasons.append("Hips not clearly visible.")

    score = 0.35 + 0.25 * (sum(sh_vis) / 2.0) + 0.25 * (sum(hip_vis) / 2.0)
    h, w  = user_bgr.shape[:2]
    if min(h, w) < 512:
        score -= 0.10
        reasons.append("Low resolution (short side < 512px).")

    score = float(max(0.0, min(score, 0.98)))
    return PoseAssessment(True, shoulders_ok and hips_ok, shoulders_ok, score, reasons)

# ─────────────────────────────
# GARMENT PROCESSING
# ─────────────────────────────
def garment_cutout_rgba_safe(garment_bgr: np.ndarray) -> Tuple[Image.Image, List[str]]:
    warnings_list: List[str] = []
    try:
        remove_fn = models.rembg_remove()
        return remove_fn(bgr_to_pil(garment_bgr)).convert("RGBA"), warnings_list
    except Exception as e:
        warnings_list.append(f"Background removal failed ({type(e).__name__}); using opaque fallback.")
        return bgr_to_pil(garment_bgr).convert("RGBA"), warnings_list

def compute_garment_score(garment_rgba: Image.Image) -> float:
    try:
        g = np.array(garment_rgba)
        if g.ndim != 3 or g.shape[2] != 4:
            return 0.25
        alpha = g[:, :, 3].astype(np.float32) / 255.0
        cov   = float(np.mean(alpha > 0.15))
        return clamp01(0.15 + 0.85 * (1.0 - min(abs(cov - 0.35) / 0.35, 1.0)))
    except Exception:
        return 0.25

# ─────────────────────────────
# GEMINI DESCRIPTION
# ─────────────────────────────
def gemini_describe(garment_rgba: Image.Image) -> str:
    if not gemini_client:
        return "dress"
    try:
        from google.genai import types
        buf = io.BytesIO()
        garment_rgba.save(buf, format="PNG")
        img_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=["Describe this garment in 1 sentence: type, color, pattern, style.", img_part],
        )
        txt = (getattr(resp, "text", None) or "").strip()
        return txt if txt else "dress"
    except Exception:
        return "dress"

# ─────────────────────────────
# OVERLAY
# ─────────────────────────────
def overlay(user_bgr: np.ndarray, garment_rgba: Image.Image) -> np.ndarray:
    out      = user_bgr.copy()
    h, w     = out.shape[:2]
    g_w, g_h = garment_rgba.size
    scale    = int(w * 0.55) / max(1, g_w)
    new_w    = max(1, int(g_w * scale))
    new_h    = max(1, int(g_h * scale))
    g        = garment_rgba.resize((new_w, new_h), Image.LANCZOS)
    x0       = (w - new_w) // 2
    y0       = int(h * 0.18)
    x1       = min(w, x0 + new_w)
    y1       = min(h, y0 + new_h)
    if x1 <= x0 or y1 <= y0:
        return out
    g_np   = np.array(g.crop((0, 0, x1 - x0, y1 - y0)))
    alpha  = g_np[:, :, 3:4].astype(np.float32) / 255.0
    g_bgr  = g_np[:, :, :3][:, :, ::-1].astype(np.float32)
    region = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (region * (1 - alpha) + g_bgr * alpha).astype(np.uint8)
    return out

# ─────────────────────────────
# IDM-VTON
# ─────────────────────────────
def idm_vton_generate(person_pil: Image.Image, garment_pil: Image.Image, desc: str) -> Image.Image:
    if not idm_vton_ready:
        raise RuntimeError("IDM-VTON not enabled")
    import replicate
    pb, gb = io.BytesIO(), io.BytesIO()
    person_pil.save(pb, format="PNG");  pb.seek(0)
    garment_pil.save(gb, format="PNG"); gb.seek(0)
    out = replicate.run(
        "cuuupid/idm-vton:c871bb9b046607b680449ecbae55fd8c6d945e0a1948644bf2361b3d021d3ff4",
        input={"human_img": pb, "garm_img": gb, "garment_des": desc,
               "is_checked": True, "denoise_steps": 30, "seed": 42},
    )
    def _to_pil(x) -> Image.Image:
        if hasattr(x, "read"):
            return Image.open(io.BytesIO(x.read())).convert("RGB")
        if isinstance(x, str):
            return Image.open(io.BytesIO(requests.get(x, timeout=180).content)).convert("RGB")
        raise RuntimeError(f"Unknown output type: {type(x)}")
    if isinstance(out, list) and out:
        return _to_pil(out[0])
    return _to_pil(out)

def safe_idm_vton_generate(person_pil, garment_pil, desc, attempts=2):
    last_err = None
    for _ in range(max(1, attempts)):
        try:
            return idm_vton_generate(person_pil, garment_pil, desc)
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("IDM-VTON failed")

# ─────────────────────────────
# SCORING
# ─────────────────────────────
def compute_swap_score(mode: str, ps: float, gs: float, diag: Dict[str, Any]) -> float:
    s = 0.60 * ps + 0.40 * gs
    if mode.startswith("overlay"):                         s *= 0.85
    if diag.get("orientation") == "profile_or_turned":    s *= 0.80
    if diag.get("low_resolution"):                        s *= 0.85
    return clamp01(s)

def compute_overall_confidence(can: bool, ps: float, gs: float, ss: float) -> float:
    if not can:
        return clamp01(0.10 + 0.55 * ps)
    return clamp01(0.50 * ss + 0.30 * ps + 0.20 * gs)

# ─────────────────────────────
# RESPONSE BUILDER
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
    output_urls, dl_urls, view_urls = [], [], []
    if out_bgr is not None:
        fname = f"tryon_{uuid.uuid4().hex[:10]}.jpg"
        save_bgr_jpg(os.path.join(OUTPUT_DIR, fname), out_bgr)
        output_urls = [build_url(request, fname)]
        dl_urls     = [build_download_url(request, fname)]
        view_urls   = [build_view_url(request, fname)]
    return {
        "request_id":          request_id,
        "can_tryon":           can_tryon,
        "mode_used":           mode_used,
        "garment_description": garment_description,
        "scores": {
            "pose_score":         round(float(pose_score), 3),
            "garment_score":      round(float(garment_score), 3),
            "swap_score":         round(float(swap_score), 3),
            "overall_confidence": round(float(overall_confidence), 3),
        },
        "input_diagnostics":    diagnostics,
        "warnings":             warnings,
        "error":                error,
        "output_urls":          output_urls,
        "output_download_urls": dl_urls,
        "output_view_urls":     view_urls,
    }

# ─────────────────────────────
# HEALTH
# ─────────────────────────────
@app.get("/")
@app.get("/health")
def health():
    return {
        "ok":               True,
        "status":           "healthy",
        "gemini_enabled":   bool(gemini_client),
        "idm_vton_enabled": bool(idm_vton_ready),
        "mp_tasks_ok":      models._mp_tasks_ok,
        "note":             "v4: MediaPipe Tasks API. Models lazy-loaded.",
        "apis":             ["/v1/tryon/actress-to-user", "/v1/tryon/garment-to-user"],
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
    request_id  = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    diagnostics = build_input_diagnostics(user_bgr)
    try:
        if not diagnostics.get("person_detected", False):
            msg = ["No person detected. Upload a clear upper/full-body photo."]
            if diagnostics.get("face_detected", False):
                msg = ["Face-only/selfie detected. Upload full-body front-facing photo."]
            return make_response(request, request_id, False, "input_invalid_user",
                                 garment_des.strip() or "garment",
                                 0.0, 0.0, 0.0, 0.0, msg, diagnostics)

        if diagnostics.get("body_coverage") == "tiny_person":
            return make_response(request, request_id, False, "input_person_too_small",
                                 garment_des.strip() or "garment", 0.0, 0.0, 0.0, 0.0,
                                 ["Person too small/far. Upload a closer photo."], diagnostics)

        if diagnostics.get("orientation") == "profile_or_turned":
            prefer_idm = False

        user_crop     = crop_person(user_bgr)
        pa            = assess_pose(user_crop)
        pose_score    = pa.score

        garment_rgba, cutout_warnings = garment_cutout_rgba_safe(garment_bgr)
        garment_score = compute_garment_score(garment_rgba)
        desc          = garment_des.strip() or (gemini_describe(garment_rgba) if gemini_client else "dress")
        warnings      = list(pa.reasons) + cutout_warnings

        if not pa.pose_detected:
            if get_person_bbox(user_crop) is not None:
                out_bgr   = overlay(user_crop, garment_rgba)
                mode_used = "overlay_no_pose"
                warnings.append("Pose not detected; overlay preview generated.")
                ss = compute_swap_score(mode_used, pose_score, garment_score, diagnostics)
                oc = compute_overall_confidence(True, pose_score, garment_score, ss)
                return make_response(request, request_id, True, mode_used, desc,
                                     pose_score, garment_score, ss, oc,
                                     warnings, diagnostics, out_bgr=out_bgr)
            warnings.append("No person detected after crop.")
            return make_response(request, request_id, False, "no_pose_no_person", desc,
                                 pose_score, garment_score, 0.0, 0.0, warnings, diagnostics)

        if diagnostics.get("low_resolution", False):
            prefer_idm = False
            warnings.append("Low resolution — using overlay.")

        if bool(prefer_idm) and idm_vton_ready and pa.ok_for_idm:
            try:
                white       = Image.new("RGBA", garment_rgba.size, (255, 255, 255, 255))
                garment_rgb = Image.alpha_composite(white, garment_rgba).convert("RGB")
                out_pil     = safe_idm_vton_generate(bgr_to_pil(user_crop), garment_rgb, desc, attempts=2)
                out_bgr     = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
                mode_used   = "idm-vton"
                ss = compute_swap_score(mode_used, pose_score, garment_score, diagnostics)
                oc = compute_overall_confidence(True, pose_score, garment_score, ss)
                return make_response(request, request_id, True, mode_used, desc,
                                     pose_score, garment_score, ss, oc,
                                     warnings, diagnostics, out_bgr=out_bgr)
            except Exception as e:
                warnings.append(f"IDM-VTON failed; falling back to overlay: {type(e).__name__}")

        if pa.ok_for_overlay:
            mode_used = "overlay"
            if prefer_idm and idm_vton_ready and not pa.ok_for_idm:
                mode_used = "overlay_pose_weak"
            if diagnostics.get("orientation") == "profile_or_turned":
                mode_used = "overlay_profile"
                warnings.append("Profile photo; overlay may be less accurate.")
            out_bgr = overlay(user_crop, garment_rgba)
            ss = compute_swap_score(mode_used, pose_score, garment_score, diagnostics)
            oc = compute_overall_confidence(True, pose_score, garment_score, ss)
            return make_response(request, request_id, True, mode_used, desc,
                                 pose_score, garment_score, ss, oc,
                                 warnings, diagnostics, out_bgr=out_bgr)

        warnings.append("Pose too weak for overlay; upload front-facing photo with shoulders visible.")
        return make_response(request, request_id, False, "pose_too_weak", desc,
                             pose_score, garment_score, 0.0, 0.0, warnings, diagnostics)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{request_id}] Pipeline crash: {type(e).__name__}: {e}\n{tb}", file=sys.stderr, flush=True)
        return make_response(request, request_id, False, "internal_error_safe",
                             garment_des.strip() or "garment", 0.0, 0.0, 0.0, 0.0,
                             ["Processing failed. Retry with a clear full-body front-facing image."],
                             diagnostics,
                             error={"type": type(e).__name__, "message": "Unhandled pipeline error"})

# ─────────────────────────────────────────────────────────────────────────────
# ASYNC PIPELINE RUNNER
# ✅ Lambda wrapper: anyio.to_thread.run_sync takes ONE callable — wrap multi-
#    arg calls in a lambda so all args are captured in the closure.
# ✅ CapacityLimiter: prevents OOM from concurrent threads on Render free tier.
# ✅ asyncio.wait_for: returns clean JSON on timeout instead of Render 502.
# ─────────────────────────────────────────────────────────────────────────────
async def run_pipeline_async(
    request: Request,
    user_bgr: np.ndarray,
    garment_bgr: np.ndarray,
    garment_des: str,
    prefer_idm: bool,
    mode_hint: str,
) -> Dict[str, Any]:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])

    async def _run():
        return await anyio.to_thread.run_sync(
            lambda: run_pipeline(request, user_bgr, garment_bgr, garment_des, prefer_idm, mode_hint),
            limiter=get_limiter(),
        )

    try:
        return await asyncio.wait_for(_run(), timeout=PIPELINE_TIMEOUT)
    except asyncio.TimeoutError:
        print(f"[{request_id}] Timed out after {PIPELINE_TIMEOUT}s", file=sys.stderr, flush=True)
        return make_response(
            request, request_id, False, "timeout",
            garment_des.strip() or "garment", 0.0, 0.0, 0.0, 0.0,
            [f"Request timed out ({PIPELINE_TIMEOUT}s). Try a smaller image or disable IDM-VTON."],
            {},
            error={"type": "TimeoutError", "message": "Pipeline exceeded time limit"},
        )

# ─────────────────────────────
# API ENDPOINTS
# ─────────────────────────────
@app.post("/v1/tryon/garment-to-user")
async def garment_to_user(
    request: Request,
    garment_image: UploadFile = File(...),
    user_image:    UploadFile = File(...),
    garment_des:   str        = Form(""),
    prefer_idm:    int        = Form(1),
):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    try:
        garment_bgr = bytes_to_bgr(await garment_image.read())
        user_bgr    = bytes_to_bgr(await user_image.read())
    except Exception as e:
        return JSONResponse(status_code=200, content=make_response(
            request, request_id, False, "input_decode_error",
            garment_des.strip() or "garment", 0.0, 0.0, 0.0, 0.0,
            ["Failed to parse images. Upload valid JPG/PNG."], {},
            error={"type": type(e).__name__, "message": "Image decode failed"},
        ))
    return await run_pipeline_async(request, user_bgr, garment_bgr, garment_des, bool(prefer_idm), "garment")


@app.post("/v1/tryon/actress-to-user")
async def actress_to_user(
    request: Request,
    actress_image: UploadFile = File(...),
    user_image:    UploadFile = File(...),
    garment_des:   str        = Form(""),
    prefer_idm:    int        = Form(1),
):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    try:
        actress_bgr = bytes_to_bgr(await actress_image.read())
        user_bgr    = bytes_to_bgr(await user_image.read())
    except Exception as e:
        return JSONResponse(status_code=200, content=make_response(
            request, request_id, False, "input_decode_error",
            garment_des.strip() or "garment", 0.0, 0.0, 0.0, 0.0,
            ["Failed to parse images. Upload valid JPG/PNG."], {},
            error={"type": type(e).__name__, "message": "Image decode failed"},
        ))
    actress_crop = crop_person(actress_bgr)
    return await run_pipeline_async(request, user_bgr, actress_crop, garment_des, bool(prefer_idm), "actress")


# ─────────────────────────────
# GLOBAL EXCEPTION HANDLER
# ─────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    print(f"[{request_id}] Unhandled: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
          file=sys.stderr, flush=True)
    return JSONResponse(status_code=200, content=make_response(
        request, request_id, False, "unhandled_exception", "garment",
        0.0, 0.0, 0.0, 0.0,
        ["Unexpected server error. Retry with a clear full-body front-facing image."], {},
        error={"type": type(exc).__name__, "message": "Unhandled server exception"},
    ))

# ─────────────────────────────
# ENTRYPOINT
# On Render use: uvicorn backend_gemini_overlay:app --host 0.0.0.0 --port $PORT
# ─────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("backend_gemini_overlay:app", host="0.0.0.0", port=port,
                log_level="info", timeout_keep_alive=30)
