FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_DOWNLOADS=automatic

COPY pyproject.toml uv.lock .python-version README.md ./
COPY src/ ./src/
COPY data/ ./data/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

CMD ["uv", "run", "avito-data-science-bootcamp-2026"]
