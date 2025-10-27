# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ✅ Correct PATH for uv
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY src/ ./src/
COPY main.py ./

# Create directory for ChromaDB
RUN mkdir -p ./chroma/db

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the application
CMD ["uv", "run", "streamlit", "run", "main.py"]
