"""
Main data loader module for OME Hackathon Season 14

This module provides a centralized way to load and cache the dataset,
avoiding repeated downloads. Other modules can import and use the data.

Usage:
    from main.main import load_data, get_agriculture_data
    
    # Load full dataset
    df = load_data()
    
    # Get filtered agriculture data
    df_agri = get_agriculture_data()
"""

import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Cache file paths
TRAIN_CACHE = DATA_DIR / "train_cached.parquet"
TEST_CACHE = DATA_DIR / "test_cached.parquet"


def download_and_cache_data(split="train", force_download=False):
    """
    Download dataset from Hugging Face and cache it locally.
    
    Args:
        split (str): Dataset split to download ('train' or 'test')
        force_download (bool): Force re-download even if cache exists
        
    Returns:
        pd.DataFrame: The requested dataset
    """
    cache_file = TRAIN_CACHE if split == "train" else TEST_CACHE
    
    # Check if cached file exists and is not forced to re-download
    if cache_file.exists() and not force_download:
        print(f"Loading cached {split} data from {cache_file}")
        return pd.read_parquet(cache_file)
    
    # Download from Hugging Face
    print(f"Downloading {split} data from Hugging Face...")
    dataset = load_dataset("DataForGood/ome-hackathon-season-14", split=split)
    df = dataset.to_pandas()
    
    # Cache to disk
    print(f"Caching {split} data to {cache_file}")
    df.to_parquet(cache_file, index=False)
    
    return df


def load_data(split="train", force_download=False):
    """
    Load the dataset (from cache if available, otherwise download).
    
    Args:
        split (str): Dataset split to load ('train' or 'test')
        force_download (bool): Force re-download even if cache exists
        
    Returns:
        pd.DataFrame: The requested dataset
    """
    return download_and_cache_data(split=split, force_download=force_download)


def get_agriculture_data(split="train", force_download=False):
    """
    Get filtered data for agriculture_alimentation category.
    
    Args:
        split (str): Dataset split to load ('train' or 'test')
        force_download (bool): Force re-download even if cache exists
        
    Returns:
        pd.DataFrame: Filtered dataset for agriculture_alimentation
    """
    df = load_data(split=split, force_download=force_download)
    df_filtered = df[df["category"].str.contains("agriculture_alimentation", case=False, na=False)]
    print(f"Filtered {len(df_filtered)} agriculture_alimentation records from {len(df)} total records")
    return df_filtered


def get_category_data(category, split="train", force_download=False):
    """
    Get filtered data for a specific category.
    
    Args:
        category (str): Category to filter (e.g., 'mobility_transport', 'energy', 'other')
        split (str): Dataset split to load ('train' or 'test')
        force_download (bool): Force re-download even if cache exists
        
    Returns:
        pd.DataFrame: Filtered dataset for the specified category
    """
    df = load_data(split=split, force_download=force_download)
    df_filtered = df[df["category"].str.contains(category, case=False, na=False)]
    print(f"Filtered {len(df_filtered)} {category} records from {len(df)} total records")
    return df_filtered


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    # Load full dataset
    df = load_data()
    print(f"\nTotal records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Get agriculture data
    print("\n" + "=" * 80)
    print("AGRICULTURE/ALIMENTATION DATA")
    print("=" * 80)
    df_agri = get_agriculture_data()
    print(f"Sample:\n{df_agri[['channel_title', 'category', 'report_text']].head(3)}")
