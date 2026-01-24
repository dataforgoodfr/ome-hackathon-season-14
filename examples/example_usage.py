"""
Example: How to use the centralized data loader in your own modules

This example shows how to import and use the data loading utilities
from the main module in your own analysis scripts.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.main import load_data, get_agriculture_data, get_category_data

# Example 1: Load full dataset
print("=" * 80)
print("EXAMPLE 1: Load Full Dataset")
print("=" * 80)
df_full = load_data(split="train")
print(f"Total records: {len(df_full)}")
print(f"Categories: {df_full['category'].value_counts()}\n")

# Example 2: Get agriculture data
print("=" * 80)
print("EXAMPLE 2: Get Agriculture Data")
print("=" * 80)
df_agri = get_agriculture_data()
print(f"Agriculture records: {len(df_agri)}")
print(f"Sample channels: {df_agri['channel_title'].value_counts().head()}\n")

# Example 3: Get other category data
print("=" * 80)
print("EXAMPLE 3: Get Energy Data")
print("=" * 80)
df_energy = get_category_data("energy")
print(f"Energy records: {len(df_energy)}")

# Example 4: Perform custom analysis
print("\n" + "=" * 80)
print("EXAMPLE 4: Custom Analysis")
print("=" * 80)
print("Average text length by category:")
df_full['text_length'] = df_full['report_text'].str.len()
avg_length = df_full.groupby('category')['text_length'].mean().sort_values(ascending=False)
print(avg_length)
