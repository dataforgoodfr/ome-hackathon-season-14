"""
API Gateway for OME Hackathon
Orchestrates multi-model analysis and stores results in PostgreSQL
"""

import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import (
    SERVICE_URLS,
    DB_CONFIG,
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    SERVICE_TIMEOUT,
)
from api.database.database_connection import connect_to_db, get_db_session
from api.database.models import (
    CategoryClassifications,
    create_tables,
    upsert_data_optimized,
    create_hash_id,
)

# Global database engine
db_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global db_engine
    print("Starting API Gateway...")

    try:
        db_engine = connect_to_db(
            database=str(DB_CONFIG["database"]),
            user=str(DB_CONFIG["user"]),
            password=str(DB_CONFIG["password"]),
            host=str(DB_CONFIG["host"]),
            port=int(DB_CONFIG["port"]),
        )
        create_tables(db_engine)
        print("Database connection established and tables created")
    except Exception as e:
        print(f"Warning: Database connection failed: {e}")
        db_engine = None

    yield

    # Shutdown
    if db_engine:
        db_engine.dispose()
        print("Database connection closed")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION, lifespan=lifespan
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class SegmentData(BaseModel):
    """Input model for segment analysis"""

    segment_id: str
    channel_title: str
    channel_name: str
    segment_start: datetime
    segment_end: datetime
    duration_seconds: str
    report_text: str
    llm_category: Optional[str] = None
    predicted_category: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "segment_id": "12345",
                "channel_title": "France 2",
                "channel_name": "france2",
                "segment_start": "2024-01-15T20:00:00",
                "segment_end": "2024-01-15T20:05:00",
                "duration_seconds": "300",
                "report_text": "Un reportage sur l'agriculture durable en France.",
                "llm_category": "agriculture_alimentation",
            }
        }


class AnalysisResult(BaseModel):
    """Response model for analysis results"""

    id: str
    segment_id: str
    sentiment: Optional[str] = None
    sentiment_confidence: Optional[float] = None
    # Future fields can be added here easily
    # classification: Optional[str] = None
    # keywords: Optional[List[str]] = None
    status: str = "success"
    message: Optional[str] = None


class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis"""

    segments: List[SegmentData]


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis"""

    processed: int
    successful: int
    failed: int
    results: List[AnalysisResult]


class ServiceHealth(BaseModel):
    """Health status for a service"""

    name: str
    status: str
    url: str


class HealthResponse(BaseModel):
    """Overall health check response"""

    status: str
    database: str
    services: List[ServiceHealth]


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "OME Analysis API Gateway",
        "version": API_VERSION,
        "endpoints": {
            "analyze": "/analyze",
            "batch": "/analyze/batch",
            "results": "/results/{segment_id}",
            "health": "/health",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check health of API gateway and all downstream services
    """
    # Check database
    db_status = "healthy" if db_engine else "unavailable"

    # Check all model services
    service_health = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, service_url in SERVICE_URLS.items():
            try:
                response = await client.get(f"{service_url}/health")
                if response.status_code == 200:
                    status = "healthy"
                else:
                    status = "unhealthy"
            except Exception:
                status = "unavailable"

            service_health.append(
                ServiceHealth(name=service_name, status=status, url=service_url)
            )

    # Overall status
    all_healthy = db_status == "healthy" and all(
        s.status == "healthy" for s in service_health
    )
    overall_status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=overall_status, database=db_status, services=service_health
    )


async def call_sentiment_service(text: str) -> tuple[Optional[str], Optional[float]]:
    """
    Call sentiment analysis service
    Returns (sentiment, confidence) tuple
    """
    if "sentiment" not in SERVICE_URLS:
        return None, None

    try:
        async with httpx.AsyncClient(timeout=SERVICE_TIMEOUT) as client:
            response = await client.post(
                f"{SERVICE_URLS['sentiment']}/predict", json={"text": text}
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("sentiment"), result.get("confidence")
            else:
                print(f"Sentiment service returned status {response.status_code}")
                return None, None

    except Exception as e:
        print(f"Error calling sentiment service: {e}")
        return None, None


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_segment(segment: SegmentData):
    """
    Analyze a single segment by querying all available model services
    and storing the enriched results in PostgreSQL
    """
    if not db_engine:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        # Call sentiment service
        sentiment, sentiment_confidence = await call_sentiment_service(
            segment.report_text
        )

        # Future service calls can be added here in parallel:
        # classification, keywords = await asyncio.gather(
        #     call_classification_service(segment.report_text),
        #     call_keyword_service(segment.report_text)
        # )

        # Prepare data for database
        segment_dict = segment.model_dump()
        segment_dict["sentiment"] = sentiment
        segment_dict["sentiment_confidence"] = sentiment_confidence

        # Create DataFrame and add hash ID
        df = pd.DataFrame([segment_dict])
        df = create_hash_id(df, column_name="id", id_column="segment_id", position=0)

        # Upsert to database
        session = get_db_session(db_engine)
        try:
            upsert_data_optimized(
                session=session,
                df=df,
                table_class=CategoryClassifications,
                primary_key="id",
            )
        finally:
            session.close()

        return AnalysisResult(
            id=df["id"].iloc[0],
            segment_id=segment.segment_id,
            sentiment=sentiment,
            sentiment_confidence=sentiment_confidence,
            status="success",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")


@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """
    Analyze multiple segments in parallel using async processing
    """
    if not db_engine:
        raise HTTPException(status_code=503, detail="Database unavailable")

    results = []
    successful = 0
    failed = 0

    # Process all segments concurrently
    async def process_segment(segment: SegmentData) -> AnalysisResult:
        try:
            result = await analyze_segment(segment)
            return result
        except Exception as e:
            return AnalysisResult(
                id="", segment_id=segment.segment_id, status="failed", message=str(e)
            )

    # Use asyncio.gather to process in parallel
    results = await asyncio.gather(
        *[process_segment(seg) for seg in request.segments], return_exceptions=False
    )

    # Count successes and failures
    for result in results:
        if result.status == "success":
            successful += 1
        else:
            failed += 1

    return BatchAnalysisResponse(
        processed=len(request.segments),
        successful=successful,
        failed=failed,
        results=results,
    )


@app.get("/results/{segment_id}")
async def get_results(segment_id: str):
    """
    Retrieve analysis results for a specific segment from PostgreSQL
    """
    if not db_engine:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        session = get_db_session(db_engine)
        try:
            # Query by segment_id
            result = (
                session.query(CategoryClassifications)
                .filter(CategoryClassifications.segment_id == segment_id)
                .first()
            )

            if not result:
                raise HTTPException(status_code=404, detail="Segment not found")

            # Convert to dict
            result_dict = {
                "id": result.id,
                "segment_id": result.segment_id,
                "channel_title": result.channel_title,
                "channel_name": result.channel_name,
                "segment_start": result.segment_start,
                "segment_end": result.segment_end,
                "duration_seconds": result.duration_seconds,
                "report_text": result.report_text,
                "llm_category": result.llm_category,
                "predicted_category": result.predicted_category,
                "sentiment": result.sentiment,
                "sentiment_confidence": result.sentiment_confidence,
            }

            return result_dict

        finally:
            session.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving results: {str(e)}"
        )
