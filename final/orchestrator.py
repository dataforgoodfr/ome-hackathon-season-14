from processing.cleaning import clean_input_text
from processing.segment import segment_by_embedding, merge_chunks_with_boundaries


def preprocess_text(input_text: str) -> list[str]:
    """Preprocess input text to get a list of segments"""
    chunks_with_boundaries = segment_by_embedding(
        clean_input_text(input_text),
    )

    return merge_chunks_with_boundaries(chunks_with_boundaries)
