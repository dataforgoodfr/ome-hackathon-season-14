import os

from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

# Initializing a new SetFit model
model = SetFitModel.from_pretrained(
    "dangvantuan/sentence-camembert-base",
    labels=["agriculture_alimentation", "mobility_transport", "energy", "other"],
)

# Preparing the dataset
dataset = load_dataset("DataForGood/ome-hackathon-season-14", split="train")
dataset = dataset.map(
    lambda example: {"text": example["report_text"], "label": example["category"]}
)
train_dataset = dataset.train_test_split(test_size=100)
eval_dataset = train_dataset["test"]
train_dataset = sample_dataset(
    train_dataset["train"], label_column="label", num_samples=8
)

# Preparing the training arguments
args = TrainingArguments(
    batch_size=8,
    num_epochs=1,
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


# os.mkdir("./models", exist_ok=True)
# Saving the trained model
model.save_pretrained("models/setfit-ome")
