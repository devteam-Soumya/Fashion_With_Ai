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
# Quick endpoints for Railway health checks
# ─────────────────────────────
@app.get("/")
def root():
    return {"ok": True, "service": "fashion-with-ai"}

@app.get("/healthz")
def healthz():
    return {"ok": True}


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
# Lazy-load heavy dependencies
# ─────────────────────────────
_yolo = None
_pose = None
_face_det = None

def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO("yolov8n-seg.pt")
    return _yolo

def get_mediapipe_pose_and_face():
    global _pose, _face_det
    if _pose is None or _face_det is None:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        _pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False)

        mp_face = mp.solutions.face_detection
        _face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    return _pose, _face_det

def rembg_remove(pil_img: Image.Image) -> Image.Image:
    from rembg import remove
    return remove(pil_img)


# ─────────────────────────────
# Gemini setup (optional)
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
      <head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/></head>
      <body style="font-family: Arial; margin: 24px;">
        <h2>Try-On Output</h2>
        <img src="{img_url}" style="width:100%;max-width:900px;border-radius:12px;"/>
        <div style="margin-top: 16px;">
          <a href="{dl_url}">Download</a> |
          <a href="{img_url}" target="_blank">Open Raw</a>
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
        r = get_yolo().predict(source=bgr, verbose=False)[0]
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


# ─────────────────────────────
# Garment cutout (safe)
# ─────────────────────────────
def garment_cutout_rgba_safe(garment_bgr: np.ndarray) -> Tuple[Image.Image, List[str]]:
    warnings: List[str] = []
    try:
        pil = bgr_to_pil(garment_bgr)
        return rembg_remove(pil).convert("RGBA"), warnings
    except Exception as e:
        warnings.append(f"Garment background removal failed ({type(e).__name__}); using opaque fallback.")
        return bgr_to_pil(garment_bgr).convert("RGBA"), warnings


# ─────────────────────────────
# Start
# ─────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("backend_gemini_overlay:app", host="0.0.0.0", port=port, log_level="info")
