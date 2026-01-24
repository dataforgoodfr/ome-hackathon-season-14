"""
Preprocessing module for OME Hackathon data.

Provides utilities for cleaning and deduplicating text data,
particularly for removing repetitive loops in report_text fields.
"""

from .deduplication import (
    find_loop_period,
    remove_text_loops,
    preprocess_dataframe,
)

__all__ = [
    "find_loop_period",
    "remove_text_loops",
    "preprocess_dataframe",
]
