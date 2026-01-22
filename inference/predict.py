import hashlib
import os

import pandas as pd
from datasets import load_dataset
from logs import get_logger
from setfit import SetFitModel
from sklearn.metrics import classification_report
from sqlalchemy.orm import sessionmaker

from models import (
    CategoryClassifications,
    ClassificationMetrics,
    connect_to_db,
    create_hash_id,
    create_tables,
    upsert_data_optimized,
)

logger = get_logger()
model_name = "setfit-ome"

dataset = load_dataset("DataForGood/ome-hackathon-season-14", split="test")
model = SetFitModel.from_pretrained("models/" + model_name)

BATCH_SIZE = 8


label_map = {
    "mobility_transport": 0,
    "agriculture_alimentation": 1,
    "energy": 2,
    "other": 3,
}


def get_numerical_labels(labels):
    return list(map(lambda x: label_map.get(x), labels))


# Classification
results = {
    "segment_id": [],
    "channel_title": [],
    "channel_name": [],
    "segment_start": [],
    "segment_end": [],
    "duration_seconds": [],
    "report_text": [],
    "llm_category": [],
    "predicted_category": [],
}

for batch in dataset.batch(batch_size=BATCH_SIZE):
    preds = get_numerical_labels(model.predict(batch["report_text"]))
    labels = get_numerical_labels(batch["category"])
    results["segment_id"].extend(batch["segment_id"])
    results["channel_title"].extend(batch["channel_title"])
    results["channel_name"].extend(batch["channel_name"])
    results["segment_start"].extend(batch["segment_start"])
    results["segment_end"].extend(batch["segment_end"])
    results["duration_seconds"].extend(batch["duration_seconds"])
    results["report_text"].extend(batch["report_text"])
    results["llm_category"].extend(labels)
    results["predicted_category"].extend(preds)
results_df = pd.DataFrame(results)
results_df = create_hash_id(results_df, "id", "segment_id")

# Run summary
run_id = hashlib.sha256().hexdigest()
report = classification_report(
    results["llm_category"],
    results["predicted_category"],
    output_dict=True,
    labels=list(label_map.keys()),
)
run_results = {
    "run_id": [run_id],
    "model_name": [model_name],
}

for key in report:
    if key not in ("accuracy", "micro avg"):
        label_name = key.split(" ")[0]
        for second_key in report[key]:
            if second_key != "support":
                second_key_label = second_key.split("-")[0]
                run_results.update(
                    {label_name + "_" + second_key_label: [report[key][second_key]]}
                )

run_df = pd.DataFrame(run_results)


conn_kwargs = dict(
    user=os.getenv("POSTGRES_USER", "user"),
    host=os.getenv("POSTGRES_HOST", "localhost"),
    database=os.getenv("POSTGRES_DB", "barometre"),
    port=int(os.getenv("POSTGRES_PORT", 5432)),
    password=os.getenv("POSTGRES_PASSWORD", "password"),
)
target_connection = connect_to_db(**conn_kwargs)
target_session = sessionmaker(bind=target_connection)()
create_tables(target_connection)
try:
    upsert_data_optimized(
        session=target_session,
        df=results_df,
        table_class=CategoryClassifications,
        primary_key="id",
    )
    upsert_data_optimized(
        session=target_session,
        df=run_df,
        table_class=ClassificationMetrics,
        primary_key="run_id",
    )
    logger.info("Data upsert completed successfully")

except Exception as e:
    logger.error(f"Error during data upsert: {e}")
    raise
finally:
    target_session.close()
