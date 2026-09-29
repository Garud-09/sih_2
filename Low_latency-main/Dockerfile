# ISRO PS 26172 - Low Latency Voice Activator Server
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libsndfile for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY server/requirements.txt ./server/

# Install Python dependencies
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy all project files
COPY . .

# Cloud deployment environment settings
ENV PORT=8000
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8765

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

# Start the ASR & Telemetry server
CMD ["python", "server/asr_server.py"]
