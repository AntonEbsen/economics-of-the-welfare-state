# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for building some scientific packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata and source, then install via pyproject.toml.
# docs/README.md is the README referenced by pyproject.toml.
COPY pyproject.toml ./
COPY docs/README.md ./docs/README.md
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Copy the rest of the application
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Default command (run dashboard)
CMD ["streamlit", "run", "src/dashboard.py"]
