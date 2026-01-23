from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import cyclopts
import pandas as pd
from datasets import Dataset, load_dataset
from setfit import SetFitModel
from sklearn.metrics import classification_report
from sqlalchemy.orm import sessionmaker

from inference.models import (
    CategoryClassifications,
    ClassificationMetrics,
    connect_to_db,
    create_hash_id,
    create_tables,
    upsert_data_optimized,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

from common.logs import configure_logging

configure_logging("inference", "common", "models")

_logger = logging.getLogger(__name__)


Label = Literal["mobility_transport", "agriculture_alimentation", "energy", "other"]
LABEL_TO_NUM = {
    "mobility_transport": 0,
    "agriculture_alimentation": 1,
    "energy": 2,
    "other": 3,
}


def get_numerical_labels(labels: Iterable[Label]) -> list[int]:
    return list(map(LABEL_TO_NUM.__getitem__, labels))


# Classification
def run_batched_inference(
    dataset: Dataset,
    predict: Callable[[list[str]], list[Label]],
    *,
    batch_size: int = 8,
) -> pd.DataFrame:
    results: dict[str, list[Any]] = {
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

    for batch in dataset.batch(batch_size=batch_size):
        batch = cast(Mapping[str, list[Any]], batch)
        preds = get_numerical_labels(predict(batch["report_text"]))
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
    return create_hash_id(results_df, "id", "segment_id")


def evaluate_results(
    results_df: pd.DataFrame,
    *,
    run_id: str,
    model_name: str,
) -> pd.DataFrame:
    # Run summary
    report = classification_report(
        results_df["llm_category"],
        results_df["predicted_category"],
        output_dict=True,
        labels=list(LABEL_TO_NUM.keys()),
    )
    run_results = {
        "run_id": [run_id],
        "model_name": [model_name],
    }

    for key in report:
        if key not in {"accuracy", "micro avg"}:
            label_name = key.split(" ")[0]
            for second_key in report[key]:
                if second_key != "support":
                    second_key_label = second_key.split("-")[0]
                    run_results.update(
                        {label_name + "_" + second_key_label: [report[key][second_key]]}
                    )
    return pd.DataFrame(run_results)


def save_results_to_db(
    results_df: pd.DataFrame,
    run_df: pd.DataFrame,
) -> None:
    conn_kwargs = {
        "user": os.getenv("POSTGRES_USER", "user"),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "database": os.getenv("POSTGRES_DB", "barometre"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "password": os.getenv("POSTGRES_PASSWORD", "password"),
    }
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
        _logger.info("Data upsert completed successfully")

    except Exception as error:
        _logger.exception("Error during data upsert: %r", error)  # noqa: TRY401
        raise
    finally:
        target_session.close()


def main(
    model_name: str = "setfit-ome",
    *,
    split: Literal["train", "test", "validation"] = "test",
    run_id: str | None = None,
    batch_size: int = 8,
    save_to_db: bool = False,
) -> None:
    if run_id is None:
        run_id = secrets.token_hex(8)

    dataset = load_dataset("DataForGood/ome-hackathon-season-14", split=split)
    model = SetFitModel.from_pretrained(f"models/{model_name}")

    results_df = run_batched_inference(
        dataset,
        model.predict,
        batch_size=batch_size,
    )
    run_df = evaluate_results(results_df, run_id=run_id, model_name=model_name)
    if save_to_db:
        save_results_to_db(results_df, run_df)


if __name__ == "__main__":
    cyclopts.run(main)
