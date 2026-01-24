"""
Script to classify agriculture texts as 'productiviste', 'alternatif' or 'neutre' using Ollama.

Uses a local LLM to analyze each text and determine if it represents
a productivist, alternative, or neutral agriculture discourse.
"""

import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.main import get_agriculture_data
from preprocessing.deduplication import preprocess_dataframe

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral:7b"

SYSTEM_PROMPT = """Tu es un expert en analyse des discours médiatiques sur l'agriculture.

Tu dois classifier les textes en trois catégories:
- "productiviste": discours favorisant l'agriculture intensive, industrielle, la productivité, les rendements, l'agro-industrie, les pesticides, les OGM, la compétitivité économique
- "alternatif": discours favorisant l'agriculture biologique, durable, locale, les circuits courts, l'agroécologie, la permaculture, le bien-être animal, l'environnement
- "neutre": texte qui ne prend pas position clairement, ou qui n'est pas directement lié à l'agriculture

Réponds UNIQUEMENT par "productiviste", "alternatif" ou "neutre", sans aucune explication."""

USER_PROMPT_TEMPLATE = """Classifie ce texte sur l'agriculture:

{text}

Classification (productiviste, alternatif ou neutre):"""


def classify_text(text: str, max_text_length: int = 2000) -> str:
    """
    Classify a single text using Ollama.

    Args:
        text: The text to classify
        max_text_length: Maximum text length to send to the model

    Returns:
        'productiviste', 'alternatif', or 'error' if classification fails
    """
    # Truncate text if too long
    truncated_text = text[:max_text_length] if len(text) > max_text_length else text

    prompt = USER_PROMPT_TEMPLATE.format(text=truncated_text)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for consistent classification
            "num_predict": 20,   # We only need a short response
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip().lower()

        # Parse the response
        if "productiviste" in answer:
            return "productiviste"
        elif "alternatif" in answer:
            return "alternatif"
        elif "neutre" in answer:
            return "neutre"
        else:
            # Try to handle unexpected responses
            return f"unclear:{answer[:50]}"
    except requests.exceptions.RequestException as e:
        return f"error:{str(e)[:50]}"


def classify_dataframe(df: pd.DataFrame, text_column: str = "report_text") -> pd.DataFrame:
    """
    Classify all texts in a dataframe.

    Args:
        df: DataFrame with text data
        text_column: Name of the column containing text

    Returns:
        DataFrame with added 'prediction' column
    """
    df = df.copy()
    predictions = []

    for text in tqdm(df[text_column], desc="Classifying texts"):
        pred = classify_text(text)
        predictions.append(pred)

    df["prediction"] = predictions
    return df


def test_classification(n_samples: int = 10):
    """
    Test the classification on a small sample.

    Args:
        n_samples: Number of samples to test
    """
    print("=" * 80)
    print(f"TESTING CLASSIFICATION ON {n_samples} SAMPLES")
    print("=" * 80)

    # Load and deduplicate data
    print("\nLoading agriculture data...")
    df_agri = get_agriculture_data(split="train")
    df_deduplicated = preprocess_dataframe(df_agri, text_column="report_text")

    # Take sample
    df_sample = df_deduplicated.head(n_samples).copy()
    print(f"Testing on {len(df_sample)} samples\n")

    # Classify
    df_result = classify_dataframe(df_sample)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for i, row in df_result.iterrows():
        text_preview = row["report_text"][:300].replace("\n", " ")
        print(f"\n[{i}] Prediction: {row['prediction']}")
        print(f"    Text: {text_preview}...")
        print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(df_result["prediction"].value_counts())

    # Save test results
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "agri_data_test.csv"
    df_result.to_csv(output_path, index=False)
    print(f"\nSaved test results to: {output_path}")

    return df_result


def run_full_classification():
    """
    Run classification on the full dataset and save results.
    """
    print("=" * 80)
    print("FULL CLASSIFICATION")
    print("=" * 80)

    # Load and deduplicate data
    print("\nLoading agriculture data...")
    df_agri = get_agriculture_data(split="train")
    df_deduplicated = preprocess_dataframe(df_agri, text_column="report_text")
    print(f"Total records: {len(df_deduplicated)}")

    # Classify all
    print("\nClassifying all texts (this may take a while)...")
    df_result = classify_dataframe(df_deduplicated)

    # Save results
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "agri_data_classified.csv"
    df_result.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(df_result["prediction"].value_counts())

    return df_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify agriculture texts")
    parser.add_argument("--test", action="store_true", help="Run test on 10 samples")
    parser.add_argument("--full", action="store_true", help="Run full classification")
    parser.add_argument("-n", type=int, default=10, help="Number of test samples")

    args = parser.parse_args()

    if args.test:
        test_classification(n_samples=args.n)
    elif args.full:
        run_full_classification()
    else:
        # Default: run test
        test_classification(n_samples=args.n)
