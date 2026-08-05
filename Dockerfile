# Use a Python base image with Debian slim
FROM python:3.12-slim

# Install system dependencies, including Node.js (required for npx and npm MCP servers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    # Clean up apt caches to minimize image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (required for Python-based MCP servers using uv run)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set the working directory
WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml requirements.txt ./

# Install python dependencies using uv (faster and cleaner)
RUN uv pip install --system -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Run the FastAPI server directly with uvicorn
CMD ["uvicorn", "web_server:app", "--host", "0.0.0.0", "--port", "8000"]
