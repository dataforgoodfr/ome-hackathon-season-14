from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from sklearn.metrics import classification_report

from reporto.db import (
    CategoryClassifications,
    ClassificationMetrics,
    connect_to_db,
    create_hash_id,
    create_tables,
    get_db_session,
    upsert_data_optimized,
)
from reporto.labels import (
    LABELS,
    Label,
    get_numerical_labels,
    get_numerical_labels_task1,
)

if TYPE_CHECKING:
    from datasets import Dataset


_logger = logging.getLogger(__name__)


def run_batched_inference(
    dataset: Dataset,
    predict: Callable[[list[str]], list[Label]],
    *,
    batch_size: int = 8,
    task1: bool = False,
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
        if task1:
            preds = get_numerical_labels_task1(predict(batch["report_text"]))
            labels = get_numerical_labels_task1(batch["text_type"])
        else:
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
        labels=LABELS,
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

    engine = connect_to_db()
    session = get_db_session(engine)
    create_tables(engine)
    try:
        upsert_data_optimized(
            session=session,
            df=results_df,
            table_class=CategoryClassifications,
            primary_key="id",
        )
        upsert_data_optimized(
            session=session,
            df=run_df,
            table_class=ClassificationMetrics,
            primary_key="run_id",
        )
        _logger.info("Data upsert completed successfully")

    except Exception as error:
        _logger.exception("Error during data upsert: %r", error)  # noqa: TRY401
        raise
    finally:
        session.close()
