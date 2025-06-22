# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy model (you'll need to update this path)
# COPY data/06_models/automm_fruit_classifier /app/model
# Or mount as volume at runtime

# Create directory for temporary files
RUN mkdir -p /tmp

# Expose port
EXPOSE 8000

# Set environment variables
ENV MODEL_PATH=/app/model
ENV PORT=8000

# Run the application
CMD ["python", "app.py"]