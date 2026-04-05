FROM python:3.11-slim

WORKDIR /app

COPY web/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Build context is repo root — use full paths
COPY ml/src/ /app/src/
COPY web/backend/app/ /app/app/

ENV PYTHONPATH=/app/src:/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
