# ==============================================================================
# StockMind-AI — Backend Dockerfile
# Python 3.11 Institutional Fast Inference & REST/WebSocket Gateway
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal system dependencies required for scientific wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application and AI engine modules
COPY backend /app/backend
COPY ai_engine /app/ai_engine
COPY data /app/data
COPY database /app/database
COPY models /app/models

# Create runtime directories if missing
RUN mkdir -p /app/logs /app/data/cache

# Expose FastAPI HTTP & WebSocket port
EXPOSE 8000

# Docker healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn gateway
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
