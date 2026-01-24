# Quick Start Guide - API Gateway

## Overview

The API Gateway orchestrates multi-model analysis for French TV ecological content using a microservices architecture.

## Architecture

```
Frontend/Client → API Gateway (port 8000) → Model Microservices
                        ↓
                  PostgreSQL Database
```

## Quick Start

### 1. Start the Services

```bash
# Start API Gateway with all dependencies
docker compose up --build api-gateway sentiment-service postgres

# Or start all services including metabase
docker compose up --build
```

### 2. Check Health

```bash
curl http://localhost:8000/health
```

### 3. Analyze a Segment

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "segment_id": "test_123",
    "channel_title": "France 2",
    "channel_name": "france2",
    "segment_start": "2024-01-15T20:00:00",
    "segment_end": "2024-01-15T20:05:00",
    "duration_seconds": "300",
    "report_text": "Un excellent reportage sur l agriculture biologique.",
    "llm_category": "agriculture_alimentation"
  }'
```

### 4. Run Test Suite

```bash
# Install requests if needed
pip install requests

# Run tests
python api/test_api.py
```

## Available Endpoints

- `GET /` - Root information
- `GET /health` - Health check for all services
- `POST /analyze` - Analyze single segment
- `POST /analyze/batch` - Analyze multiple segments
- `GET /results/{segment_id}` - Get stored results

## Workflow

1. **Submit data**: Send segment data (CSV row) to `/analyze` or `/analyze/batch`
2. **Model orchestration**: API Gateway queries all available model microservices in parallel
3. **Result aggregation**: Combines responses from all models
4. **Database storage**: Stores enriched data in PostgreSQL `category_classification` table
5. **Return response**: Returns aggregated results to client

## Current Models

- **Sentiment Analysis** (port 8001): French sentiment classification
  - Model: `ac0hik/Sentiment_Analysis_French`
  - Returns: sentiment label + confidence score

## Adding New Models

See detailed guide in `api/README.md`. The sentiment service serves as a template:

1. Create `your_model/api.py` with FastAPI
2. Create `your_model/Dockerfile`
3. Add service URL to `api/config.py`
4. Update `docker-compose.yml`
5. Add database columns to `inference/models.py`
6. Update gateway logic in `api/main.py`

## Database

Results are stored in PostgreSQL:
- **Host**: localhost:5432
- **Database**: ome_hackathon
- **User**: user
- **Password**: ilovedataforgood
- **Table**: `category_classification`

View results in Metabase at http://localhost:3000

## Documentation

- Full API documentation: `api/README.md`
- Interactive API docs: http://localhost:8000/docs (when running)
- OpenAPI schema: http://localhost:8000/openapi.json

## Development

Run gateway locally without Docker:

```bash
# Start dependencies
docker compose up postgres sentiment-service

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install httpx

# Run gateway
uvicorn api.main:app --reload --port 8000
```

## Troubleshooting

**Connection refused errors**: Make sure all services are running
```bash
docker compose ps
```

**Database errors**: Check PostgreSQL is healthy
```bash
docker compose logs postgres
```

**Model service errors**: Check service logs
```bash
docker compose logs sentiment-service
```

## Next Steps

1. Add classification service for category prediction
2. Add keyword extraction service
3. Connect frontend UI to the API
4. Add authentication and rate limiting for production
