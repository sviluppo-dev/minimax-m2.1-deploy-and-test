# Dockerfile for minimax-deploy API service
# This runs the FastAPI service that connects to a vLLM server

FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install dependencies
RUN uv pip install --system -e ".[api]"

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run API server
CMD ["python", "-m", "uvicorn", "minimax_deploy.api:app", "--host", "0.0.0.0", "--port", "8080"]
