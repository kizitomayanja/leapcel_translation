```
# Use a slim Python 3.11 image
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Copy requirements and app files
COPY requirements.txt .
COPY app.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```