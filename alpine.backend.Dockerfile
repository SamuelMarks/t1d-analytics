FROM python:3.12-alpine

WORKDIR /app

# Install build dependencies for DuckDB, etc.
RUN apk add --no-cache build-base linux-headers

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "t1d_analytics.api:app", "--host", "0.0.0.0", "--port", "8000"]