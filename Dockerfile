# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV MODEL_CACHE_DIR=/app/models

# Render provides $PORT, default to 5000 locally
ENV PORT=5000

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (includes gunicorn)
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Create necessary directories
RUN mkdir -p models/binary models/classifiers

# Copy application files
COPY app.py .
COPY models/binary/* models/binary/
COPY models/classifiers/* models/classifiers/

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port (Render overrides but good for local runs)
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start Gunicorn server (production WSGI)
CMD ["gunicorn", "-b", "0.0.0.0:${PORT}", "app:app"]
