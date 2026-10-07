# ==========================================
# FastAPI Predictive Maintenance Backend
# ==========================================
FROM python:3.11-slim

# Set Python runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    MODEL_PATH=/app/models/lightgbm_model.txt \
    THRESHOLD=0.5

# Install system dependencies (libgomp1 is mandatory for LightGBM OpenMP support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python package dependencies
COPY [aA][pP][iI]/requirements-api.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy source code and model weights (using glob to support both api and API casing)
COPY [aA][pP][iI] /app/API
COPY models /app/models

# Create lowercase symlink for case-insensitive compatibility across platforms
RUN ln -sf /app/API /app/api

# Create and switch to non-privileged user for enhanced production security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose default API port
EXPOSE 8000

# Container healthcheck using FastAPI /health endpoint
HEALTHCHECK --interval=20s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server
CMD ["uvicorn", "API.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
