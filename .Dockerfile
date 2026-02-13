# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim avoids bloated full image while keeping build fast on Railway
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
# libgl1 + libglib2.0-0 → required by OpenCV (cv2)
# libgomp1             → required by ultralytics/YOLO
# curl                 → health checks / debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
# Copy requirements first so Docker layer-caches the pip install
# (only re-runs when requirements.txt changes, not on every code change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── App code ──────────────────────────────────────────────────────────────────
COPY . .

# ── Output dir ────────────────────────────────────────────────────────────────
RUN mkdir -p ./outputs /tmp/Ultralytics /tmp/mp_models /tmp/matplotlib

# ── Expose port (Railway reads $PORT at runtime, this is just documentation) ──
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────────────────────
CMD uvicorn backend_gemini_overlay:app --host 0.0.0.0 --port ${PORT:-8000}
