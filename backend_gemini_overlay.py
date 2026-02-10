# backend_gemini_overlay.py
# Virtual Try-On – Hardened (Refactored + Scores + Stable Deployment)

import os
import io
import uuid
import traceback
from typing import Dict, Any, List, Tuple, Optional
import requests
import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
from rembg import remove
import replicate
import anyio

# ── Google GenAI SDK (safe import) ────────────────────────────────
try:
    from google import genai
    from google.genai import types
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False

load_dotenv()

# ── CONFIG ──────────────────────────────────────────
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "").strip()
ENABLE_GEMINI       = os.getenv("ENABLE_GEMINI", "true").lower() in ("true", "1", "yes")
GEMINI_MODEL        = os.getenv("GEMINI_MODEL_ID", "gemini-1.5-flash")  # safe default

# Use active IDM-VTON model (Feb 2026)
ENABLE_IDM_VTON     = os.getenv("ENABLE_IDM_VTON", "true").lower() in ("true", "1", "yes")
IDM_MODEL           = "yisol/IDM-VTON"  # current public & working version

if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# ── APP ─────────────────────────────────────────────
app = FastAPI(title="Virtual Try-On – Hardened (Stable Deployment)")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

app.add_middleware(RequestIdMiddleware)

# ── Gemini client (safe) ───────────────────────────────────
gemini_client: Optional[genai.Client] = None
if ENABLE_GEMINI and GEMINI_SDK_AVAILABLE and GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"Gemini enabled: {GEMINI_MODEL}")
    except Exception as e:
        gemini_client = None
        print(f"Gemini init failed: {e}")
else:
    print("Gemini disabled")

print(f"IDM-VTON enabled: {ENABLE_IDM_VTON}")

# ── UTILS ───────────────────────────────────────────
def bytes_to_rgb(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def save_jpeg(bgr: np.ndarray, path: str):
    bgr_to_pil(bgr).save(path, "JPEG", quality=92, optimize=True)

def get_url(request: Request, fname: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/outputs/{fname}"

def safe_output_path(filename: str) -> str:
    fname = os.path.basename(filename)
    return os.path.join(OUTPUT_DIR, fname)

def build_download_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/download/{filename}"

def build_view_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/view/{filename}"

# ── Download + View endpoints ───────────────────────
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
    img_url = get_url(request, os.path.basename(filename))
    dl_url = build_download_url(request, os.path.basename(filename))
    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width,initial-scale=1"/>
        <title>Try-On Result</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; background: #f9f9f9; }}
          .wrap {{ max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
          img {{ width: 100%; height: auto; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.15); }}
          .btns {{ margin-top: 20px; display: flex; gap: 12px; flex-wrap: wrap; }}
          a.button {{ display: inline-block; padding: 12px 18px; border-radius: 10px; text-decoration: none; font-weight: bold; }}
          a.primary {{ background: #007bff; color: white; }}
          a.secondary {{ background: #6c757d; color: white; }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <h2>Virtual Try-On Result</h2>
          <img src="{img_url}" alt="Try-On Result"/>
          <div class="btns">
            <a class="button primary" href="{dl_url}">Download Image</a>
            <a class="button secondary" href="{img_url}" target="_blank">View Full Size</a>
          </div>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)

# ── Gemini description (safe) ───────────────────────
def gemini_describe(garment_rgba: Image.Image) -> str:
    if not gemini_client:
        return "garment"
    try:
        prompt = "Describe this garment in 1 sentence: type, color, pattern, style, length."
        buf = io.BytesIO()
        garment_rgba.save(buf, format="PNG")
        image_part = types.Part.from_data(
            data=buf.getvalue(),
            mime_type="image/png"
        )
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, image_part]
        )
        return (response.text or "garment").strip()
    except Exception as e:
        print(f"Gemini describe failed: {e}")
        return "garment"

# ── Overlay fallback ───────────────────────────────
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
    g_crop = g.crop((0, 0, x1 - x0, y1 - y0))
    g_np = np.array(g_crop)
    alpha = g_np[:, :, 3:4].astype(np.float32) / 255.0
    g_bgr = g_np[:, :, :3][:, :, ::-1].astype(np.float32)
    region = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (region * (1 - alpha) + g_bgr * alpha).astype(np.uint8)
    return out

# ── IDM-VTON generation (safe) ─────────────────────
def idm_vton_generate(person_pil: Image.Image, garment_pil: Image.Image, desc: str) -> Image.Image:
    if not ENABLE_IDM_VTON:
        raise RuntimeError("IDM-VTON disabled")

    person_bytes = io.BytesIO()
    garment_bytes = io.BytesIO()
    person_pil.save(person_bytes, format="PNG")
    garment_pil.save(garment_bytes, format="PNG")
    person_bytes.seek(0)
    garment_bytes.seek(0)

    out = replicate.run(
        IDM_MODEL,
        input={
            "human_img": person_bytes,
            "garm_img": garment_bytes,
            "garment_des": desc,
            "is_checked": True,
            "denoise_steps": 30,
            "seed": 42,
        },
    )

    if isinstance(out, replicate.helpers.FileOutput):
        return Image.open(io.BytesIO(out.read())).convert("RGB")
    if isinstance(out, str):
        img_data = requests.get(out, timeout=180).content
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    if isinstance(out, list) and out:
        item = out[0]
        if isinstance(item, replicate.helpers.FileOutput):
            return Image.open(io.BytesIO(item.read())).convert("RGB")
        if isinstance(item, str):
            img_data = requests.get(item, timeout=180).content
            return Image.open(io.BytesIO(img_data)).convert("RGB")
    raise RuntimeError(f"Unexpected Replicate output: {type(out)}")

# ── Unified response builder ───────────────────────
def make_response(
    request: Request,
    request_id: str,
    success: bool,
    mode_used: str,
    garment_description: str,
    warnings: List[str],
    diagnostics: Dict[str, Any],
    out_bgr: Optional[np.ndarray] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    output_urls = []
    output_download_urls = []
    output_view_urls = []
    if out_bgr is not None:
        fname = f"tryon_{uuid.uuid4().hex[:10]}.jpg"
        save_jpeg(out_bgr, os.path.join(OUTPUT_DIR, fname))
        output_urls = [get_url(request, fname)]
        output_download_urls = [build_download_url(request, fname)]
        output_view_urls = [build_view_url(request, fname)]

    return {
        "request_id": request_id,
        "ok": success,
        "mode_used": mode_used,
        "garment_description": garment_description,
        "warnings": warnings,
        "input_diagnostics": diagnostics,
        "error": error,
        "output_urls": output_urls,
        "output_download_urls": output_download_urls,
        "output_view_urls": output_view_urls,
    }

# ── Core pipeline ───────────────────────────────────
def run_pipeline(
    request: Request,
    user_bgr: np.ndarray,
    garment_bgr: np.ndarray,
    garment_des: str,
    prefer_idm: bool,
    mode_hint: str,
) -> Dict[str, Any]:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    diagnostics = {"image_decoded": True}
    warnings: List[str] = []

    try:
        # Garment processing
        garment_rgba = bgr_to_pil(garment_bgr).convert("RGBA")
        try:
            garment_rgba = remove(garment_rgba).convert("RGBA")
        except Exception as e:
            warnings.append(f"Background removal failed: {type(e).__name__}")
            garment_rgba = bgr_to_pil(garment_bgr).convert("RGBA")

        desc = garment_des.strip() or (gemini_describe(garment_rgba) if gemini_client else "garment")

        # User processing
        user_crop = user_bgr  # simple for now

        # Try IDM-VTON if enabled and preferred
        if prefer_idm and ENABLE_IDM_VTON:
            try:
                white = Image.new("RGBA", garment_rgba.size, (255, 255, 255, 255))
                garment_rgb = Image.alpha_composite(white, garment_rgba).convert("RGB")
                out_pil = idm_vton_generate(bgr_to_pil(user_crop), garment_rgb, desc)
                out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
                return make_response(
                    request, request_id, True, "idm-vton", desc,
                    warnings, diagnostics, out_bgr=out_bgr
                )
            except Exception as e:
                warnings.append(f"IDM-VTON failed: {type(e).__name__} – using overlay fallback")

        # Fallback to overlay
        out_bgr = overlay(user_crop, garment_rgba)
        return make_response(
            request, request_id, True, "overlay", desc,
            warnings, diagnostics, out_bgr=out_bgr
        )

    except Exception as e:
        print(f"[{request_id}] Pipeline error: {type(e).__name__} – {e}")
        return make_response(
            request, request_id, False, "error", garment_des or "garment",
            warnings + [f"Processing failed: {str(e)[:100]}"],
            diagnostics,
            error={"type": type(e).__name__, "message": str(e)}
        )

# ── ENDPOINTS ───────────────────────────────────────
@app.post("/v1/tryon/garment-to-user")
async def garment_to_user(
    request: Request,
    garment_image: UploadFile = File(...),
    user_image: UploadFile = File(...),
    garment_des: str = Form(""),
    prefer_idm: int = Form(1),
):
    try:
        garment_bytes = await garment_image.read()
        user_bytes = await user_image.read()
        garment_bgr = bytes_to_rgb(garment_bytes)
        user_bgr = bytes_to_rgb(user_bytes)
    except Exception as e:
        return JSONResponse(status_code=200, content={
            "ok": False,
            "error": "Invalid image upload",
            "warnings": [str(e)]
        })

    return await anyio.to_thread.run_sync(
        run_pipeline,
        request,
        user_bgr,
        garment_bgr,
        garment_des,
        bool(prefer_idm),
        "garment"
    )

@app.post("/v1/tryon/actress-to-user")
async def actress_to_user(
    request: Request,
    actress_image: UploadFile = File(...),
    user_image: UploadFile = File(...),
    garment_des: str = Form(""),
    prefer_idm: int = Form(1),
):
    try:
        actress_bytes = await actress_image.read()
        user_bytes = await user_image.read()
        actress_bgr = bytes_to_rgb(actress_bytes)
        user_bgr = bytes_to_rgb(user_bytes)
    except Exception as e:
        return JSONResponse(status_code=200, content={
            "ok": False,
            "error": "Invalid image upload",
            "warnings": [str(e)]
        })

    return await anyio.to_thread.run_sync(
        run_pipeline,
        request,
        user_bgr,
        actress_bgr,
        garment_des,
        bool(prefer_idm),
        "actress"
    )

@app.get("/health")
def health():
    return {
        "ok": True,
        "gemini_enabled": bool(gemini_client),
        "idm_vton_enabled": ENABLE_IDM_VTON,
        "endpoints": ["/v1/tryon/garment-to-user", "/v1/tryon/actress-to-user"]
    }

# ── Global exception handler ───────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    print(f"[{request_id}] Unhandled exception: {exc}")
    return JSONResponse(
        status_code=200,
        content={
            "ok": False,
            "error": "Internal server error",
            "request_id": request_id,
            "warnings": ["Please try again with valid images"]
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_gemini_overlay:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
