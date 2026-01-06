FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and model download script
COPY backend/requirements.txt .
COPY backend/download_models.py .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CPU version
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install Hugging Face Transformers and related libraries
RUN pip install --no-cache-dir \
    transformers==4.35.2 \
    tokenizers==0.15.0 \
    huggingface-hub==0.19.4 \
    safetensors==0.4.1

# Install specific version of sentence-transformers
RUN pip install --no-cache-dir sentence-transformers==2.2.2

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Verify installation
RUN python -c "import sentence_transformers; print('✓ sentence-transformers instalado')"

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Pre-download models
RUN python download_models.py

# Copy backend
COPY backend/ ./backend/

# Copy frontend
COPY frontend/ ./frontend/

# Copy data
COPY data/ ./data/

# Create data directory
RUN mkdir -p data

WORKDIR /app/backend

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]