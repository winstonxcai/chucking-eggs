FROM python:3.11-slim

WORKDIR /app

# Install system deps for potential Rust/numpy builds
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY web/backend/requirements.txt /app/web/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/web/backend/requirements.txt

# Copy source (volumes override in dev)
COPY src/ /app/src/
COPY web/backend/ /app/web/backend/

ENV PYTHONPATH=/app/src:/app/web/backend

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/web/backend/app", "--reload-dir", "/app/src"]
