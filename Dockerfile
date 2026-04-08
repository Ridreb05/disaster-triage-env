# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
#
# Build from repo root:
#   docker build -t disaster-triage-env:latest .
#   docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... disaster-triage-env:latest
#
# Hugging Face Spaces automatically builds this Dockerfile.
# Port 7860 is required by HF Spaces Docker SDK.

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces best practice)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy and install dependencies first (layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=appuser:appuser . .

# Switch to non-root
USER appuser

# Ensure the app directory is on PYTHONPATH
ENV PYTHONPATH=/app
ENV PORT=7860
ENV ENV=production

# Expose port required by HF Spaces Docker SDK
EXPOSE 7860

# Health check — ensures HF Space ping returns 200
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
