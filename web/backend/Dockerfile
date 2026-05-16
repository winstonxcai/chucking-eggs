FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN python3 -c "\
import tomllib, subprocess, sys; \
deps = tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['web']; \
subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir',*deps],check=True)"

# Build context is repo root — use full paths
COPY ml/src/ /app/src/
COPY web/backend/app/ /app/app/

ENV PYTHONPATH=/app/src:/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
