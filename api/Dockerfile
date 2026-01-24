FROM python:3.12-bookworm

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.128.0 \
    uvicorn==0.40.0 \
    httpx>=0.27.0 \
    pandas>=3.0.0 \
    sqlalchemy>=2.0.46 \
    psycopg2-binary>=2.9.11 \
    pyarrow>=23.0.0

# Copy application code
COPY api/ ./api/
COPY main/main.py ./main/
COPY main/__init__.py ./main/

# Expose port
EXPOSE 8000

# Run the gateway
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
