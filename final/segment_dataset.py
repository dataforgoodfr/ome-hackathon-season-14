from datasets import load_dataset

from final.orchestrator import predict_text

def segment_text(row) -> dict:
    prediction =  predict_text(row["report_text"])
    return {
        "segments_content": [segment.content for segment in prediction.segments],
        "segments_category": [segment.category for segment in prediction.segments],
    }

dataset = load_dataset("DataForGood/ome-hackathon-season-14", split="test")
dataset_with_duplicates = dataset.map(segment_text)
print(dataset_with_duplicates)
