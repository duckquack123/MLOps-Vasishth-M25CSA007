# Use PyTorch official image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ ./src/
COPY model/ ./model/

# Create data directories
RUN mkdir -p data/train data/test

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - evaluate the pretrained model
CMD ["python", "src/evaluate.py"]

# To run evaluation, use:
# docker run --gpus all -v $(pwd)/data/test:/app/data/test your-image-name
#
# Note: Mount only the test subdirectory to /app/data/test
# Your data structure should be:
#   data/test/airplane/*.jpg
#   data/test/bird/*.jpg
#   data/test/car/*.jpg
#   ... (one subfolder per class)

# For CPU-only evaluation:
# docker run -v $(pwd)/data/test:/app/data/test your-image-name
