from __future__ import annotations

import secrets
from typing import Literal

from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

from reporto.evaluation import (
    evaluate_results,
    run_batched_inference,
    save_results_to_db,
)
from reporto.labels import LABELS

# TODO: Group


def train(
    output_name: str,
    base_model: str = "dangvantuan/sentence-camembert-base",
    *,
    split: str = "train",
    batch_size: int = 8,
    num_epochs: int = 1,
    test_size: int = 100,
) -> None:
    """Train a SetFit model on the dataset."""
    # Initializing a new SetFit model
    model = SetFitModel.from_pretrained(base_model, labels=LABELS)

    # Preparing the dataset
    dataset = load_dataset("DataForGood/ome-hackathon-season-14", split=split)
    dataset = dataset.map(
        lambda example: {"text": example["report_text"], "label": example["category"]}
    )
    train_dataset = dataset.train_test_split(test_size=test_size)
    eval_dataset = train_dataset["test"]
    train_dataset = sample_dataset(
        train_dataset["train"], label_column="label", num_samples=8
    )

    # Preparing the training arguments
    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # Preparing the trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )
    trainer.train()

    # Evaluating
    metrics = trainer.evaluate(eval_dataset)
    print(metrics)

    model.save_pretrained(f"models/{output_name}")
    print(f"Model saved to models/{output_name}")


def predict(
    model_name: str = "setfit-ome",
    *,
    split: Literal["train", "test", "validation"] = "test",
    run_id: str | None = None,
    batch_size: int = 8,
    save_to_db: bool = False,
) -> None:
    """Run inference with a SetFit model on the dataset."""
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
