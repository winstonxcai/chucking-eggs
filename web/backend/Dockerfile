FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim@sha256:4f5d923c9dcea037f57bda425dd209f3ec643da2f0b74227f68d09dab0b3bb36

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:/app/web/backend"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --only-group web-server --no-install-project

# Build context is repo root: keep runtime paths aligned with docker-compose.
COPY ml/src/ /app/src/
COPY web/backend/app/ /app/web/backend/app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
