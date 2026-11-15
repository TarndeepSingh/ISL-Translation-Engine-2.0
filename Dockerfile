# ==============================================================================
# STAGE 1: Build the React + TypeScript Frontend (ISL/web)
# ==============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /build/web
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY web/ ./
RUN npm run build

# ==============================================================================
# STAGE 2: Production Python 3.12 Runtime (FastAPI + ONNX Runtime + MediaPipe)
# ==============================================================================
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MPLBACKEND=Agg \
    HOME=/home/appuser \
    PORT=7860 \
    ISL_MODEL_DIR=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    python-multipart>=0.0.9 \
    pydantic>=2.6.0 \
    numpy>=1.26.0 \
    scipy>=1.12.0 \
    onnxruntime>=1.18.0 \
    mediapipe==0.10.14 \
    opencv-python-headless>=4.9.0 \
    pillow>=10.2.0 \
    httpx>=0.27.0

# Copy runtime production packages and exported models only
COPY isl/ ./isl/
COPY backend/ ./backend/
COPY models/ ./models/

# Copy the compiled React production bundle from Stage 1
COPY --from=frontend-builder /build/web/dist ./web/dist

RUN python -c "import mediapipe as mp; print('MediaPipe version verified:', mp.__version__)" && \
    mkdir -p /app/tmp /home/appuser/.cache /home/appuser/.config/matplotlib && \
    chown -R 1000:1000 /app /home/appuser && \
    chmod -R a+rX /usr/local/lib/python3.12/site-packages/mediapipe

USER 1000

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/api/v1/health || curl -f http://localhost:${PORT:-7860}/health || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]