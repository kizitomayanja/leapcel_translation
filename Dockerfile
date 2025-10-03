# Use a slim Python 3.11 image
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Create a writable cache directory for Hugging Face
RUN mkdir -p /tmp/huggingface_cache && chmod -R 777 /tmp/huggingface_cache

# Copy requirements and app files
COPY requirements.txt .
COPY app.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Set environment variable for Hugging Face cache
ENV HF_HOME=/tmp/huggingface_cache

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
# Note: Ensure to pass the HF_TOKEN environment variable when running the container