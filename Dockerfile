# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

# System deps (LightGBM/SHAP need libgomp; build tools help with wheels)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     libgomp1  && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -ms /bin/bash appuser
WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt /app/requirements.txt
COPY requirements-ml.txt /app/requirements-ml.txt
COPY constraints.txt /app/constraints.txt
RUN pip install --upgrade pip &&     pip install -r requirements.txt -r requirements-ml.txt -c constraints.txt

# Copy the project (including committed config.ini)
COPY . /app

# Switch to non-root
USER appuser

# Run the app directly
CMD ["python", "-u", "main.py"]
