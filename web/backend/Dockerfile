FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Build context is web/backend/ (Fly.io working dir) — src/ and app/ are relative to it
COPY src/ /app/src/
COPY app/ /app/app/

ENV PYTHONPATH=/app/src:/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
