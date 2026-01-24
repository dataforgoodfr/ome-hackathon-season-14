import pandas as pd
from pysentimiento import create_analyzer

# Load sentiment analysis model (Multilingual support including French)
print("Loading sentiment analysis model...")
sentiment_analyzer = create_analyzer(task="sentiment", lang="en")  # Using English as base for multilingual

# Load dataset
print("Loading dataset...")
splits = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
df = pd.read_parquet(
    "hf://datasets/DataForGood/ome-hackathon-season-14/" + splits["train"]
)

print(f"Total records: {len(df)}")

# Filter for agriculture_alimentation category (equivalent to SQL WHERE category LIKE '%agriculture_alimentation%')
df_filtered = df[df["category"].str.contains("agriculture_alimentation", case=False, na=False)]
print(f"Filtered records (agriculture_alimentation): {len(df_filtered)}")

# Take a small subset for testing (first 10 rows)
df_filtered = df_filtered.head(10)
print(f"Using subset of {len(df_filtered)} records for testing")

# Function to analyze sentiment with text truncation (models have max token limits)
def analyze_sentiment(text):
    if pd.isna(text) or text == "":
        return "neutral"
    try:
        # Truncate text to avoid token limit issues (max ~512 tokens for most models)
        text_truncated = text[:2000]
        result = sentiment_analyzer.predict(text_truncated)
        # pysentimiento returns NEG, NEU, or POS
        return result.output.lower()  # Convert to lowercase: neg, neu, pos
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return "neutral"

# Apply sentiment analysis to the report_text column
print("Performing sentiment analysis...")
df_filtered["sentiment_analysis"] = df_filtered["report_text"].apply(analyze_sentiment)

# Display results
print("\n" + "="*80)
print("SENTIMENT ANALYSIS RESULTS")
print("="*80)
print(f"\nSentiment distribution:")
print(df_filtered["sentiment_analysis"].value_counts())

print(f"\nSample results:")
print(df_filtered[["channel_title", "report_text", "sentiment_analysis"]].head(10))

# Save the results
output_path = "dataset/train_agriculture_with_sentiment.parquet"
df_filtered.to_parquet(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# Optional: Save as CSV for easy viewing
csv_output_path = "dataset/train_agriculture_with_sentiment.csv"
df_filtered.to_csv(csv_output_path, index=False)
print(f"Results also saved as CSV to: {csv_output_path}")