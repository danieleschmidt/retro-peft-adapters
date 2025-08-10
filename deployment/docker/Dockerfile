# Production Dockerfile for Retro-PEFT-Adapters

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd --gid 1000 retropeft && \
    useradd --uid 1000 --gid retropeft --shell /bin/bash --create-home retropeft

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy and install Python dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel

# Production stage
FROM base as production

# Install production dependencies only
RUN pip install --no-dev .

# Copy application code
COPY src/ ./src/
COPY README.md LICENSE ./

# Create data directories
RUN mkdir -p /app/data /app/logs /app/models && \
    chown -R retropeft:retropeft /app

# Switch to non-root user
USER retropeft

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import retro_peft; print('Health check passed')"

# Default command
CMD ["python", "-m", "retro_peft.api.gateway"]

# Development stage
FROM base as development

# Install all dependencies including dev tools
RUN pip install -e ".[dev,test,all-backends]"

# Copy everything for development
COPY . .

# Switch to non-root user
USER retropeft

# Default command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]