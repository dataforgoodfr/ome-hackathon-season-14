import pandas as pd

splits = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
df = pd.read_parquet(
    "hf://datasets/DataForGood/ome-hackathon-season-14/" + splits["train"]
)

# Basic dataset information
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nColumn names and types:")
print(df.dtypes)

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({"Missing Count": missing, "Percentage": missing_pct})
print(
    missing_df[missing_df["Missing Count"] > 0].sort_values(
        "Missing Count", ascending=False
    )
)

print("\n" + "=" * 80)
print("FIRST FEW ROWS")
print("=" * 80)
print(df.head())

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(df.describe(include="all"))

print("\n" + "=" * 80)
print("UNIQUE VALUES PER COLUMN")
print("=" * 80)
for col in df.columns:
    try:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
    except TypeError:
        # Handle columns with unhashable types (like lists/arrays)
        print(f"{col}: (contains lists/arrays)")

# If there are categorical columns with few unique values, show value counts
print("\n" + "=" * 80)
print("VALUE COUNTS FOR KEY COLUMNS")
print("=" * 80)
for col in df.columns:
    try:
        unique_count = df[col].nunique()
        if (
            unique_count < 20 and unique_count > 1
        ):  # Show value counts for columns with 2-19 unique values
            print(f"\n{col}:")
            print(df[col].value_counts())
    except TypeError:
        # Skip columns with unhashable types
        pass

# Additional insights for text data
print("\n" + "=" * 80)
print("TEXT LENGTH STATISTICS")
print("=" * 80)
if "report_text" in df.columns:
    df["text_length"] = df["report_text"].str.len()
    print("Report text length stats:")
    print(df["text_length"].describe())

print("\n" + "=" * 80)
print("SAMPLE RECORDS")
print("=" * 80)
print("\nSample from different categories:")
for category in df["category"].unique():
    print(f"\n--- {category} ---")
    sample = df[df["category"] == category].iloc[0]
    print(f"Channel: {sample['channel_title']}")
    print(f"Duration: {sample['duration_seconds']}s")
    print(f"Keywords: {sample['num_keywords']}")
    print(f"Text preview: {sample['report_text'][:200]}...")
    if len(sample["report_text"]) > 200:
        print(f"... (total {len(sample['report_text'])} characters)")
