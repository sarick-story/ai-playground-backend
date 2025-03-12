FROM python:3.12-slim

WORKDIR /app

# Install git and required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /bin/uv

# Set UV to use system Python
ENV UV_SYSTEM_PYTHON=1

# Clone both repositories
RUN git clone https://github.com/piplabs/story-mcp-hub.git && \
    git clone https://github.com/sarick-story/ai-playground-backend.git

# Set working directory to ai-playground-backend
WORKDIR /app/ai-playground-backend

# Install dependencies using UV
RUN uv pip install --system -e .

# Set environment variables
ENV MCP_SERVER_PATH=/app/story-mcp-hub/storyscan-mcp/server.py
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the application directly with system Python
CMD ["python", "-m", "api.index"]