"""
Script to prepare deduplicated agriculture data for annotation.

Loads agriculture training data, removes text loops/duplications,
and saves the result to a CSV file.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.main import get_agriculture_data
from preprocessing.deduplication import preprocess_dataframe, get_deduplication_stats


def main():
    print("=" * 80)
    print("PREPARING DEDUPLICATED AGRICULTURE DATA")
    print("=" * 80)

    # Load agriculture training data
    print("\nLoading agriculture data (train split)...")
    df_agri = get_agriculture_data(split="train")
    print(f"Loaded {len(df_agri)} records")

    # Get deduplication stats before processing
    print("\nComputing deduplication statistics...")
    stats = get_deduplication_stats(df_agri)
    print(f"  Records with loops: {stats['records_with_loops']:,}")
    print(f"  Original total chars: {stats['original_total_chars']:,}")
    print(f"  Expected deduplicated chars: {stats['deduplicated_total_chars']:,}")
    print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")

    # Apply deduplication
    print("\nApplying deduplication...")
    df_deduplicated = preprocess_dataframe(df_agri, text_column="report_text")
    print(f"Deduplication complete. Records: {len(df_deduplicated)}")

    # Create output directory
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Save to CSV (text column only)
    output_path = output_dir / "agri_data.csv"
    df_deduplicated[["report_text"]].to_csv(output_path, index=False)
    print(f"\nSaved deduplicated data to: {output_path}")
    print(f"Output file size: {output_path.stat().st_size / 1024:.1f} KB")

    # Show sample of deduplicated data
    print("\n" + "=" * 80)
    print("SAMPLE OF DEDUPLICATED DATA")
    print("=" * 80)
    for i in range(min(3, len(df_deduplicated))):
        text = df_deduplicated.iloc[i]["report_text"]
        print(f"\nRecord {i}: {len(text)} chars")
        print(f"  Preview: {text[:200]}...")


if __name__ == "__main__":
    main()
