# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for T1D Analytics Suite

# Stage 1: Build Web Frontend Assets
FROM node:20-alpine AS frontend-builder
WORKDIR /app/t1d-analytics-web
COPY t1d-analytics-web/package*.json ./
RUN npm ci
COPY t1d-analytics-web/ ./
RUN npm run build

# Stage 2: Python Runtime and Backend
FROM python:3.12-slim AS runner

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-root application user
RUN groupadd -g 1000 t1d && \
    useradd -u 1000 -g t1d -m -s /bin/bash t1d

# Copy Python requirements and install
COPY t1d-analytics-backend/requirements.txt t1d-analytics-backend/requirements-dev.txt t1d-analytics-backend/pyproject.toml t1d-analytics-backend/README.md ./t1d-analytics-backend/
RUN pip install --no-cache-dir -r t1d-analytics-backend/requirements.txt -r t1d-analytics-backend/requirements-dev.txt

# Copy source code and install package in editable/production mode
COPY t1d-analytics-backend/src/ ./t1d-analytics-backend/src/
COPY t1d-analytics-backend/tests/ ./t1d-analytics-backend/tests/
COPY t1d-analytics-backend/docs_src/ ./t1d-analytics-backend/docs_src/
COPY scripts/ ./scripts/
COPY USAGE.md README.md ./
RUN pip install --no-cache-dir -e ./t1d-analytics-backend

# Copy compiled frontend assets from Stage 1
COPY --from=frontend-builder /app/t1d-analytics-web/dist ./t1d-analytics-web/dist

# Create persistent data directory with proper permissions
RUN mkdir -p /app/data && chown -R t1d:t1d /app

USER t1d
EXPOSE 8000

ENV T1D_DB_PATH=/app/data/t1d.duckdb
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "t1d_analytics.api:app", "--host", "0.0.0.0", "--port", "8000"]
