import sys
from pathlib import Path
import pandas as pd
from pysentimiento import create_analyzer

# Add project root to path to import main module
sys.path.insert(0, str(Path(__file__).parent.parent))
from main.main import get_agriculture_data, DATA_DIR

# Load sentiment analysis model (Multilingual support including French)
print("Loading sentiment analysis model...")
sentiment_analyzer = create_analyzer(task="sentiment", lang="en")  # Using English as base for multilingual

# Load dataset using centralized data loader
print("\nLoading dataset...")
df_filtered = get_agriculture_data(split="train")

# Take a small subset for testing (first 10 rows)
# Remove this line to process the full dataset
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
output_dir = DATA_DIR.parent / "dataset"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "train_agriculture_with_sentiment.parquet"
df_filtered.to_parquet(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# Optional: Save as CSV for easy viewing
csv_output_path = output_dir / "train_agriculture_with_sentiment.csv"
df_filtered.to_csv(csv_output_path, index=False)
print(f"Results also saved as CSV to: {csv_output_path}")