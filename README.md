# Fashion_With_Ai

# Virtual Try-On Backend (Gemini Overlay + IDM-VTON) — FastAPI

A Railway-friendly FastAPI backend for virtual try-on:
- Detects person + crops (YOLOv8 segmentation model)
- Runs pose diagnostics (MediaPipe)
- Cuts out garment background (rembg)
- Optional garment description via Gemini (google-genai)
- Optional higher quality try-on via Replicate (IDM-VTON)
- Fallback “overlay preview” mode always available
- Returns scores + diagnostics + output URLs

---

## Features

- **Two endpoints**
  - `POST /v1/tryon/garment-to-user` (flat-lay garment + user)
  - `POST /v1/tryon/actress-to-user` (actress image used as garment + user)
- **Output hosting**
  - Saves outputs in `./outputs`
  - Serves files via `/outputs/*`, `/download/*`, `/view/*`
- **Health check**
  - `GET /health`

---

## Tech Stack

- FastAPI + Uvicorn
- OpenCV, NumPy, Pillow
- YOLOv8 (ultralytics)
- MediaPipe (pose + face detection)
- rembg (garment background removal)
- Gemini (google-genai) optional
- Replicate IDM-VTON optional

---

.
├── backend_gemini_overlay.py
├── requirements.txt
├── outputs/ # generated at runtime
└── README.md


---

## Requirements

- Python 3.10+ (recommended)
- If using `rembg` on Linux: ensure system packages exist (see notes below)
- Environment variables for Gemini/Replicate optional

---

## Environment Variables

| Variable | Default | Description |
|---|---:|---|
| `PORT` | `8080` | HTTP port |
| `ENABLE_GEMINI` | `1` | `1` enables garment description via Gemini |
| `ENABLE_IDM_VTON` | `1` | `1` enables IDM-VTON via Replicate |
| `GEMINI_API_KEY` | `""` | Gemini API key |
| `GEMINI_MODEL_ID` | `gemini-2.5-flash` | Gemini model |
| `REPLICATE_API_TOKEN` | `""` | Replicate token |
| `U2NET_HOME` | `/tmp/.u2net` | rembg model cache dir (writable path) |

> Note: If `REPLICATE_API_TOKEN` is missing, IDM-VTON auto-disables and the API falls back to overlay.

---

## Install + Run Locally

### 1) Create venv
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

2) Install deps

pip install -r requirements.txt

3) Create .env

ENABLE_GEMINI=1
GEMINI_API_KEY=YOUR_KEY
GEMINI_MODEL_ID=gemini-2.5-flash

ENABLE_IDM_VTON=1
REPLICATE_API_TOKEN=YOUR_TOKEN

PORT=8080

4) Start server

uvicorn backend_gemini_overlay:app --host 0.0.0.0 --port 8080 --log-level info

5) Verify

Health: GET http://localhost:8080/health

Docs: GET http://localhost:8080/docs

API Documentation
Health Check

GET /health

Response example:
{
  "ok": true,
  "gemini_enabled": true,
  "idm_vton_enabled": true,
  "apis": ["/v1/tryon/actress-to-user", "/v1/tryon/garment-to-user"]
}
1) Garment-to-User

POST /v1/tryon/garment-to-user

Form-data:

garment_image (file, required)

user_image (file, required)

garment_des (text, optional)

prefer_idm (int 0/1, default 1)
curl -X POST "http://localhost:8080/v1/tryon/garment-to-user" \
  -F "garment_image=@garment.jpg" \
  -F "user_image=@user.jpg" \
  -F "garment_des=camel wrap midi dress" \
  -F "prefer_idm=1"
2) Actress-to-User

POST /v1/tryon/actress-to-user

Form-data:

actress_image (file, required) # person wearing garment

user_image (file, required)

garment_des (text, optional)

prefer_idm (int 0/1, default 1)
curl -X POST "http://localhost:8080/v1/tryon/actress-to-user" \
  -F "actress_image=@actress.jpg" \
  -F "user_image=@user.jpg" \
  -F "garment_des=black gown" \
  -F "prefer_idm=1"
Response Format (Common)
{
  "request_id": "a1b2c3d4e5f6",
  "can_tryon": true,
  "mode_used": "idm-vton",
  "garment_description": "black gown with long sleeves...",
  "scores": {
    "pose_score": 0.82,
    "garment_score": 0.74,
    "swap_score": 0.79,
    "overall_confidence": 0.81
  },
  "input_diagnostics": {
    "image_size": { "width": 1080, "height": 1350 },
    "low_resolution": false,
    "short_side": 1080,
    "orientation": "frontish",
    "person_detected": true,
    "body_coverage": "torso",
    "bbox_area_ratio": 0.32,
    "face_detected": true,
    "face_count": 1
  },
  "warnings": [],
  "error": null,
  "output_urls": ["https://.../outputs/tryon_xxx.jpg"],
  "output_download_urls": ["https://.../download/tryon_xxx.jpg"],
  "output_view_urls": ["https://.../view/tryon_xxx.jpg"]
}
Notes / Common Issues
1) rembg downloads model on first run

The service writes the model under U2NET_HOME. Ensure it points to a writable path:

Linux servers: /tmp/.u2net is good

Docker: mount a volume if you want it cached persistently

2) Heavy dependencies

ultralytics, mediapipe, and rembg can increase build time and memory usage. For production:

Prefer Docker deployment

Use a machine with enough RAM (2–4GB minimum recommended)
Deployment on AWS

You have two common options:

Option A (Simplest): EC2 + Docker (Recommended)
1) Create an EC2 instance

Ubuntu 22.04 (recommended)

Instance: t3.medium or higher (for mediapipe/ultralytics)

Open inbound port 80 (or 8080) in Security Group
Open inbound port 80 (or 8080) in Security Group

2) Install Docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker ubuntu
newgrp docker
3) Add Dockerfile (in repo root)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV U2NET_HOME=/tmp/.u2net

WORKDIR /app

# System deps for OpenCV/MediaPipe on slim
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["uvicorn", "backend_gemini_overlay:app", "--host", "0.0.0.0", "--port", "8080"]
4) Build & run on EC2
docker build -t vton-api .
docker run -d \
  -p 80:8080 \
  -e ENABLE_GEMINI=1 \
  -e GEMINI_API_KEY="YOUR_KEY" \
  -e ENABLE_IDM_VTON=1 \
  -e REPLICATE_API_TOKEN="YOUR_TOKEN" \
  --name vton-api \
  vton-api
5) Test

http://EC2_PUBLIC_IP/health

http://EC2_PUBLIC_IP/docs

If you mapped -p 8080:8080, use port 8080 in the URL.


## Folder Structure
