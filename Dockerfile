# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for T1D Analytics Suite

# Stage 1: Build Web Frontend Assets
FROM node:20-alpine AS frontend-builder
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: Python Runtime and Backend
FROM python:3.12-slim AS runner

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends 
    curl 
    ca-certificates 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-root application user
RUN groupadd -g 1000 t1d && 
    useradd -u 1000 -g t1d -m -s /bin/bash t1d

# Copy Python requirements and install
COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy source code and install package in editable/production mode
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY USAGE.md README.md ./
RUN pip install --no-cache-dir -e .

# Copy compiled frontend assets from Stage 1
COPY --from=frontend-builder /app/web/dist ./web/dist

# Create persistent data directory with proper permissions
RUN mkdir -p /app/data && chown -R t1d:t1d /app

USER t1d
EXPOSE 8000

ENV T1D_DB_PATH=/app/data/t1d.duckdb
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "t1d_analytics.api:app", "--host", "0.0.0.0", "--port", "8000"]
