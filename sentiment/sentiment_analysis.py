import sys
from pathlib import Path
import pandas as pd
from transformers import pipeline

# Add project root to path to import main module
sys.path.insert(0, str(Path(__file__).parent.parent))
from main.main import get_agriculture_data, DATA_DIR

# Load sentiment analysis model for French
print("Loading French sentiment analysis model...")
sentiment_analyzer = pipeline(
    "text-classification",
    model="ac0hik/Sentiment_Analysis_French"
)

# Load dataset using centralized data loader
print("\nLoading dataset...")
df_filtered = get_agriculture_data(split="train")

# Take a small subset for testing (first 10 rows)
# Remove this line to process the full dataset
df_filtered = df_filtered.head(100)
print(f"Using subset of {len(df_filtered)} records for testing")

# Function to analyze sentiment with text truncation (models have max token limits)
def analyze_sentiment(text):
    if pd.isna(text) or text == "":
        return "neutral"
    try:
        # Truncate text to avoid token limit issues (max ~512 tokens for most models)
        text_truncated = text[:512]
        result = sentiment_analyzer(text_truncated)[0]
        # Returns: POSITIVE, NEGATIVE, or NEUTRAL (depending on model)
        label = result['label'].upper()
        if 'POSITIVE' in label or 'POS' in label:
            return 'positive'
        elif 'NEGATIVE' in label or 'NEG' in label:
            return 'negative'
        else:
            return 'neutral'
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return "neutral"

# Apply sentiment analysis to the report_text column
print("Performing sentiment analysis...")
df_filtered["sentiment"] = df_filtered["report_text"].apply(analyze_sentiment)

# Display results
print("\n" + "="*80)
print("SENTIMENT & EMOTION ANALYSIS RESULTS")
# Display results
print("\n" + "="*80)
print("SENTIMENT ANALYSIS RESULTS")
print("="*80)
print(f"\nSentiment distribution:")
print(df_filtered["sentiment"].value_counts())

print(f"\nSample results:")
print(df_filtered[["channel_title", "report_text", "sentiment"]].head(30))

# Save the results to data folder (not dataset)
output_dir = DATA_DIR
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "agriculture_sentiment_analysis.parquet"
df_filtered.to_parquet(output_path, index=False)
print(f"\nResults saved to: {output_path}")