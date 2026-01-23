from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Column,
    DateTime,
    Double,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base

if TYPE_CHECKING:
    from sqlalchemy.orm.decl_api import DeclarativeMeta


Base: type[DeclarativeMeta] = declarative_base()


class CategoryClassifications(Base):
    __tablename__ = "category_classification"

    id = Column(Text, primary_key=True)
    segment_id = Column(Text)
    channel_title = Column(String, nullable=False)
    channel_name = Column(Text, nullable=False)
    segment_start = Column(DateTime(), default=datetime.now)
    segment_end = Column(DateTime(), default=datetime.now)
    duration_seconds = Column(Text)
    report_text = Column(Text)
    llm_category = Column(Text)
    predicted_category = Column(Text)


class ClassificationMetrics(Base):
    __tablename__ = "classification_metrics"

    run_id = Column(Text, primary_key=True)
    model_name = Column(String, nullable=False)
    mobility_transport_precision = Column(Double, nullable=True)
    agriculture_alimentation_precision = Column(Double, nullable=True)  # Was Text ?
    energy_precision = Column(Double, nullable=True)
    other_precision = Column(Double, nullable=True)
    macro_precision = Column(Double, nullable=True)
    weighted_precision = Column(Double, nullable=True)

    mobility_transport_recall = Column(Double, nullable=True)
    agriculture_alimentation_recall = Column(Double, nullable=True)  # Was Text ?
    energy_recall = Column(Double, nullable=True)
    other_recall = Column(Double, nullable=True)
    macro_recall = Column(Double, nullable=True)
    weighted_recall = Column(Double, nullable=True)

    mobility_transport_f1 = Column(Double, nullable=True)
    agriculture_alimentation_f1 = Column(Double, nullable=True)  # Was Text ?
    energy_f1 = Column(Double, nullable=True)
    other_f1 = Column(Double, nullable=True)
    macro_f1 = Column(Double, nullable=True)
    weighted_f1 = Column(Double, nullable=True)
