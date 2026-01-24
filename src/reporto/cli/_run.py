from __future__ import annotations

import secrets
from pathlib import Path
from typing import Literal

from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import classification_report

from reporto.evaluation import (
    evaluate_results,
    run_batched_inference,
    save_results_to_db,
)
from reporto.labels import LABELS

# TODO: Group


MODELS_FOLDER = Path("models/")


def train(
    *,
    base_model: str = "dangvantuan/sentence-camembert-base",
    output_name: str | None = None,
    # Dataset
    dataset_name: str = "DataForGood/ome-hackathon-season-14",
    num_samples: int = 8,
    # Evaluation
    test_size: int = 100,
    evaluate: bool = False,
    # Training
    batch_size: int = 8,
    num_epochs: int = 1,
    device: str = "cpu",
) -> None:
    """Train a SetFit model on the dataset."""
    if output_name is None:
        output_name = f"{base_model.rsplit('/', 1)[-1]}-{secrets.token_hex(4)}"
        print(f"Using output name: {output_name}")

    # Preparing the dataset
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.map(
        lambda example: {"text": example["report_text"], "label": example["category"]}
    )
    train_dataset = dataset.train_test_split(test_size=test_size)
    eval_dataset = train_dataset["test"]
    train_dataset = sample_dataset(
        train_dataset["train"],
        label_column="label",
        num_samples=num_samples,
    )

    # Preparing the training arguments
    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    model = SetFitModel.from_pretrained(base_model, labels=LABELS, device=device)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )
    trainer.train()

    # Evaluating
    metrics = trainer.evaluate(eval_dataset)
    print("Eval metrics:")
    print(metrics)

    output_path = MODELS_FOLDER / output_name
    model.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

    if evaluate:
        results_df = run_batched_inference(
            dataset,
            model.predict,
            batch_size=batch_size,
        )
        report = classification_report(
            results_df["llm_category"],
            results_df["predicted_category"],
            labels=LABELS,
        )
        print("Classification report:")
        print(report)


def predict(
    model_path: str,
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
    if not Path(model_path).exists():
        print(f"Path {model_path} is not local, looking for model on HF")
    model = SetFitModel.from_pretrained(model_path)

    results_df = run_batched_inference(
        dataset,
        model.predict,
        batch_size=batch_size,
    )
    run_df = evaluate_results(results_df, run_id=run_id, model_name=model_path)
    if save_to_db:
        save_results_to_db(results_df, run_df)
