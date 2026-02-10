# Use slim Python image (much smaller than full python:3.11)
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies needed for opencv, rembg, mediapipe, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY backend_gemini_overlay.py .

# Final stage – copy only what's needed
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY backend_gemini_overlay.py .

# Create outputs folder
RUN mkdir -p /app/outputs

# Expose FastAPI port
EXPOSE 8000

# Run with uvicorn (production settings)
CMD ["uvicorn", "backend_gemini_overlay:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
