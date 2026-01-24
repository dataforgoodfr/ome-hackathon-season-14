# Inference Service (Legacy)

## Overview

This is the **standalone batch inference service** for SetFit category classification. It's separate from the API Gateway and runs as a one-off batch job.

## Purpose

- Loads the trained SetFit model from `models/setfit-ome`
- Processes the entire test dataset in batches
- Stores classification results and metrics in PostgreSQL
- Used for model evaluation and bulk processing

## Files

- `predict.py` - Main batch prediction script
- `Dockerfile` - Container for batch inference
- `database_connection.py` - Database utilities (also used by API)
- `models.py` - SQLAlchemy models (also used by API)
- `logs.py` - Colored logging utilities (also used by API)

## Usage

```bash
# Run batch inference
docker compose up --build inference

# Or manually
python inference/predict.py
```

## Note

The shared utilities (`database_connection.py`, `models.py`, `logs.py`) are also copied to `api/database/` for use by the API Gateway. The inference service continues to use its local copies for backward compatibility.

## Workflow

1. Loads test dataset from Hugging Face
2. Loads SetFit model from `models/setfit-ome`
3. Predicts categories in batches (batch_size=8)
4. Generates classification report
5. Stores results in `category_classification` table
6. Stores metrics in `classification_metrics` table

## Difference from API Gateway

| Feature | Inference Service | API Gateway |
|---------|------------------|-------------|
| Purpose | Batch model evaluation | Real-time orchestration |
| Input | Full test dataset | Individual segments via API |
| Models | SetFit classification only | All models (sentiment, future additions) |
| Output | Database + metrics | API response + database |
| Usage | One-off batch job | Continuous service |

The API Gateway is the recommended way to process new data. This inference service is primarily for model evaluation and historical batch processing.
