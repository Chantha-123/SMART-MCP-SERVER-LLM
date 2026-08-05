# Use a Python base image with Debian slim
FROM python:3.12-slim

# Install system dependencies and Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    build-essential \
    procps \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy uv package manager binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy official Ollama binary and llama-server libraries directly from official image
COPY --from=ollama/ollama:latest /usr/bin/ollama /usr/local/bin/ollama
COPY --from=ollama/ollama:latest /usr/lib/ollama /usr/local/lib/ollama

# Set working directory
WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml requirements.txt ./

# Install python dependencies using uv
RUN uv pip install --system -r requirements.txt

# Copy application files
COPY . .

# Expose Hugging Face default port 7860
EXPOSE 7860

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_BASE_URL=http://127.0.0.1:11434
ENV OLLAMA_MODEL=llama3.2:1b

# Start Ollama service in background, wait for it to boot, pre-pull llama3.2:1b, then launch web server
CMD ["sh", "-c", "ollama serve & sleep 5 && ollama pull llama3.2:1b && uvicorn web_server:app --host 0.0.0.0 --port ${PORT:-7860}"]
