FROM python:3.12-bookworm

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.128.0 \
    uvicorn==0.40.0 \
    transformers==5.0.0rc3 \
    torch==2.10.0 \
    sentencepiece==0.2.1

# Copy application code
COPY sentiment/api.py .

# Expose port
EXPOSE 8001

# Run the service
CMD ["python", "api.py"]
