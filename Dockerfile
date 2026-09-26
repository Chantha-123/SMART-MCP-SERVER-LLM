# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage "app": lean web image. Used by docker-compose, where Ollama runs in its
# own container (no Ollama/CUDA libraries baked in here).
#   docker compose up -d --build
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS app

# System dependencies: curl (web reader tool) and Node.js (npx-based MCP servers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    procps \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy uv package manager binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files first so code changes don't invalidate this layer
COPY pyproject.toml requirements.txt ./

# The cache mount keeps downloaded wheels between rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Pre-install the stdio MCP servers so agents don't download them (uv run --with /
# npx @latest) on every new container. Rebuild the image to update them.
# Installed outside /root so a non-root runtime user (Hugging Face runs as UID 1000)
# can execute them.
ENV UV_TOOL_DIR=/opt/uv-tools \
    UV_TOOL_BIN_DIR=/usr/local/bin
RUN --mount=type=cache,target=/root/.cache/uv \
    uv tool install mcp-atlassian \
    && npm install -g --no-fund --no-audit slack-mcp-server \
    && npm cache clean --force

# Copy application files (see .dockerignore)
COPY . .

ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OLLAMA_BASE_URL=http://ollama:11434 \
    OLLAMA_MODEL=llama3.2

EXPOSE 8000

CMD ["sh", "-c", "uvicorn web_server:app --host 0.0.0.0 --port ${PORT:-8000}"]

# ---------------------------------------------------------------------------
# Stage "standalone": single container with Ollama embedded (Hugging Face
# Spaces builds the last stage by default). Not used by docker-compose.
# ---------------------------------------------------------------------------
FROM app AS standalone

COPY --from=ollama/ollama:latest /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama/ollama:latest /usr/lib/ollama /usr/lib/ollama

# Local model defaults for the Space (override with Space variables). qwen2.5:3b
# was the fastest model with reliable tool calling in our CPU tests.
ENV PORT=7860 \
    LLM_PROVIDER=ollama \
    OLLAMA_MODEL=qwen2.5:3b \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    OLLAMA_KEEP_ALIVE=24h \
    OLLAMA_CONTEXT_LENGTH=8192 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    HOME=/tmp/app-home \
    OLLAMA_MODELS=/tmp/app-home/.ollama/models \
    MCP_STATE_DIR=/tmp/app-home/.mcp

EXPOSE 7860

# Wait until Ollama answers (instead of a fixed sleep), pull the model, start the app
CMD ["sh", "-c", "mkdir -p \"$HOME\" && (ollama serve &) && until ollama list >/dev/null 2>&1; do sleep 1; done && ollama pull \"$OLLAMA_MODEL\" && exec uvicorn web_server:app --host 0.0.0.0 --port ${PORT:-7860}"]
