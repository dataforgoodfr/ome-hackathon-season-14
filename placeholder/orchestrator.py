from datasets import load_dataset

from placeholder.preprocessing import count_duplicates
from placeholder.category_scoring import default_scorer
from placeholder.preprocessing import clean_input_text
from placeholder.interface import TextAnalysis, Segment
from placeholder.segment import segment_by_embedding, merge_chunks_with_boundaries

scorer = default_scorer()


def segment_and_categorize(input_text: str) -> TextAnalysis:
    """Segment text and predict categories of generated segments."""
    cleaned_text = clean_input_text(input_text)
    chunks_with_boundaries = segment_by_embedding(cleaned_text)
    segments = merge_chunks_with_boundaries(chunks_with_boundaries)

    segments_with_prediction = [scorer.score(segment, return_details=True) for segment in segments]

    return TextAnalysis(
        input_text=input_text,
        initial_word_count=len(input_text.split()),
        nb_duplicates=count_duplicates(input_text),
        final_word_count=len(cleaned_text.split()),
        segments=[
            Segment(
                content=segment,
                word_count=len(segment.split()),
                category=segment_with_prediction[1]["best_theme"],
                score=segment_with_prediction[1]["best_score"]
            ) for segment, segment_with_prediction in zip(segments, segments_with_prediction)
        ]
    )


def _segment_dataset_row(row) -> dict:
    """Segment row from OME dataset using the report text."""
    prediction =  segment_and_categorize(row["report_text"])
    return {
        "segments_content": [segment.content for segment in prediction.segments],
        "segments_category": [segment.category for segment in prediction.segments],
    }


def segment_dataset() -> None:
    """Segment the dataset OME."""
    dataset = load_dataset("DataForGood/ome-hackathon-season-14", split="test")
    dataset_with_segments = dataset.map(_segment_dataset_row)

    dataset_with_segments.to_parquet("ome-dataset-with-segements.parquet")


if __name__ == "__main__":
    segment_dataset()
