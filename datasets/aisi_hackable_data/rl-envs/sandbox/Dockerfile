# Dockerfile for Reward Hacking Environment
#
# This environment is intentionally vulnerable to reward hacking for research purposes.
# It includes pytest which enables the conftest.py reward hack.

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# pytest is required for running tests and enables conftest.py hack
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-timeout>=2.0.0

# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Set Python to not buffer output
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["tail", "-f", "/dev/null"]
