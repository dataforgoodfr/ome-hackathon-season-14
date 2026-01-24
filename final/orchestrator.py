from final.cleaning import count_duplicates
from final.scoring import default_scorer
from final.cleaning import clean_input_text
from final.interface import TextAnalysis, Segment
from final.segment import segment_by_embedding, merge_chunks_with_boundaries

scorer = default_scorer()

def predict_text(input_text: str) -> TextAnalysis:
    """Preprocess input text to get a list of segments"""
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

