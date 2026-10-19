"""Fine-tuning and inference for a contextual Hugging Face sequence classifier."""

from __future__ import annotations

# JSON is used to save and reload the model's training metadata and label mapping.
import json
from pathlib import Path
from typing import Any

# NumPy handles selecting the highest-scoring predicted class.
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Shared project utilities: the fixed set of target classes and the text formatter.
from .data import DESTINATION_LABELS, build_model_text


# Adapts a pandas DataFrame to the format expected by PyTorch / Hugging Face Trainer.
class ClauseDataset(Dataset):
    """Tokenize clauses without padding; the batch collator pads dynamically."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        label2id: dict[str, int],
        max_length: int = 256,
        with_labels: bool = True,
    ):
        # Reset indices so positional lookups in __getitem__ are consistent.
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.with_labels = with_labels
        # Combine each row's relevant fields into the exact text representation
        # the classifier will see (defined in the shared data helper).
        self.texts = [build_model_text(row) for _, row in self.frame.iterrows()]

    def __len__(self):
        # PyTorch uses this to determine the number of examples in the dataset.
        return len(self.frame)

    def __getitem__(self, idx):
        # Padding is deferred to DataCollatorWithPadding so each batch is padded
        # only to its longest sequence. max_length still controls truncation.
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        # Hugging Face tokenizers return lists of token IDs / attention masks;
        # convert them to integer tensors for the model.
        item = {
            key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()
        }
        # During training/evaluation, attach the gold class ID as "labels".
        # In inference datasets this can be disabled because labels are unknown.
        if self.with_labels:
            item["labels"] = torch.tensor(
                self.label2id[str(self.frame.iloc[idx]["destination_label"])],
                dtype=torch.long,
            )
        return item


def _compute_metrics(eval_pred):
    """Compute validation metrics over the complete, fixed destination taxonomy."""
    from sklearn.metrics import accuracy_score, f1_score

    # logits are the model's raw class scores; argmax chooses the top-scoring class.
    # labels are the known target IDs for the validation examples.
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # Include every class in macro-F1, even if a class is absent in this split.
    all_label_ids = list(range(len(DESTINATION_LABELS)))
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(
            f1_score(
                labels,
                preds,
                labels=all_label_ids,
                average="macro",
                zero_division=0,
            )
        ),
    }


# Check labels against the project-wide taxonomy and count each class consistently.
def _validate_partition_labels(
    frame: pd.DataFrame, partition_name: str
) -> dict[str, int]:
    # Convert labels to strings so comparisons and dictionary keys are consistent.
    observed = set(frame["destination_label"].astype(str))
    unexpected = sorted(observed - set(DESTINATION_LABELS))
    if unexpected:
        raise ValueError(
            f"Unexpected destination labels in {partition_name} data: {unexpected}"
        )
    # Count observed examples, then return counts in the fixed taxonomy order.
    counts = frame["destination_label"].astype(str).value_counts().to_dict()
    missing = [label for label in DESTINATION_LABELS if counts.get(label, 0) == 0]
    if missing:
        # A missing class is allowed, but explicitly surfaced in metadata and logs.
        print(
            f"WARNING: {partition_name} partition has no examples for: {', '.join(missing)}"
        )
    return {label: int(counts.get(label, 0)) for label in DESTINATION_LABELS}


def read_training_metadata(model_dir: str | Path) -> dict[str, Any]:
    """Read metadata saved beside the model; model_dir is normally ``.../model``."""
    model_dir = Path(model_dir)
    # Training writes metadata next to the model directory, not inside it.
    metadata_path = model_dir.parent / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Training metadata not found at {metadata_path}. Retrain the model or provide max_length explicitly."
        )
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata


# Keep inference token truncation consistent with training unless overridden explicitly.
def resolve_max_length(model_dir: str | Path, max_length: int | None = None) -> int:
    """Use the training max length unless the caller deliberately overrides it."""
    if max_length is None:
        max_length = int(read_training_metadata(model_dir)["max_length"])
    if max_length < 1:
        raise ValueError("max_length must be a positive integer")
    return int(max_length)


# Supported options for handling uneven numbers of examples across target classes.
CLASS_WEIGHTING_OPTIONS = ("none", "balanced")


def compute_class_weights(
    train_counts: dict[str, int], scheme: str = "balanced"
) -> list[float]:
    """Per-class loss weights in ``DESTINATION_LABELS`` order.

    ``balanced`` uses n_samples / (n_classes * class_count), the same formula as
    scikit-learn's ``class_weight="balanced"``, so rare classes count more in the
    training loss. A class with no training examples gets weight 1.0 (it never
    appears in a training batch, so its weight has no effect). ``none`` returns
    all ones, which reproduces the standard unweighted loss.
    """
    if scheme not in CLASS_WEIGHTING_OPTIONS:
        raise ValueError(f"class_weighting must be one of {CLASS_WEIGHTING_OPTIONS}")
    if scheme == "none":
        return [1.0] * len(DESTINATION_LABELS)
    # Arrange counts in the same order used to encode labels for the model.
    counts = [int(train_counts.get(label, 0)) for label in DESTINATION_LABELS]
    total, present = sum(counts), sum(1 for c in counts if c > 0)
    if total == 0:
        raise ValueError(
            "Cannot compute class weights from an empty training partition"
        )
    return [total / (present * c) if c > 0 else 1.0 for c in counts]


# Fine-tune a pretrained transformer and save the best validation-selected model.
def train_bert(
    frame: pd.DataFrame,
    output_dir: str | Path,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 256,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    seed: int = 42,
    overwrite: bool = False,
    class_weighting: str = "none",
) -> Path:
    """Train on train rows and select the checkpoint by validation macro-F1.

    Requires internet/model cache for the initial pretrained model download.
    The test partition is never used for checkpoint selection. Confidence values
    from softmax are uncalibrated probabilities unless a separate calibration step
    has been fitted and evaluated on validation data.

    ``class_weighting="balanced"`` weights the cross-entropy loss by inverse
    training-class frequency (see ``compute_class_weights``). The default,
    ``"none"``, keeps the standard unweighted loss. Compare the two on
    validation macro-F1, never on the test split.
    """
    if max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if batch_size < 1 or epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    if class_weighting not in CLASS_WEIGHTING_OPTIONS:
        raise ValueError(f"class_weighting must be one of {CLASS_WEIGHTING_OPTIONS}")

    # Avoid accidentally mixing a new run with files from an earlier run.
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{out} already contains files; pass --overwrite to replace it"
        )
    out.mkdir(parents=True, exist_ok=True)

    # Only the training split updates model weights; validation is used for selection.
    # The test split is deliberately not used in this function.
    train = frame[frame["split"] == "train"].copy()
    val = frame[frame["split"] == "validation"].copy()
    if train.empty or val.empty:
        raise ValueError("Train and validation partitions must both be non-empty")
    train_counts = _validate_partition_labels(train, "training")
    val_counts = _validate_partition_labels(val, "validation")

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    # Create stable mappings between human-readable labels and integer class IDs.
    label2id = {label: i for i, label in enumerate(DESTINATION_LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    # Load the tokenizer and pretrained sequence-classification model. The model
    # classification head is configured for this project's destination categories.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    train_ds = ClauseDataset(train, tokenizer, label2id, max_length)
    val_ds = ClauseDataset(val, tokenizer, label2id, max_length)
    # Pad examples dynamically within each batch rather than padding every example
    # to max_length in advance; this usually avoids unnecessary computation.
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Configure training, validation, logging, and checkpoint saving.
    # Evaluation and saving happen each epoch so the best validation checkpoint can
    # be restored at the end. Macro-F1 gives each class equal weight in selection.
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        seed=seed,
        data_seed=seed,
        eval_strategy="epoch",  # Evaluate validation metrics after each epoch.
        save_strategy="epoch",  # Save a checkpoint after each epoch.
        logging_strategy="epoch",
        load_best_model_at_end=True,  # Restore the checkpoint with best validation score.
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    # Compute weights in label-ID order. The standard Trainer is used by default;
    # a small subclass is only needed when weighted cross-entropy is requested.
    class_weights = compute_class_weights(train_counts, class_weighting)
    trainer_cls = Trainer
    if class_weighting != "none":
        weight_tensor = torch.tensor(class_weights, dtype=torch.float)

        class WeightedTrainer(Trainer):
            """Trainer whose cross-entropy loss is weighted by class."""

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                # Separate targets from model inputs before running the forward pass.
                labels = inputs.pop("labels")
                # The model produces logits; cross-entropy compares them with true labels.
                outputs = model(**inputs)
                loss_fn = torch.nn.CrossEntropyLoss(
                    weight=weight_tensor.to(outputs.logits.device)
                )
                loss = loss_fn(
                    outputs.logits.view(-1, outputs.logits.shape[-1]), labels.view(-1)
                )
                return (loss, outputs) if return_outputs else loss

        trainer_cls = WeightedTrainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=_compute_metrics,
        processing_class=tokenizer,
    )
    # Fine-tune the model using the training set and monitor the validation set.
    trainer.train()

    # With load_best_model_at_end=True, trainer.model is the selected checkpoint.
    model_dir = out / "model"
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    # Save the label mapping so class IDs can be interpreted consistently later.
    with (model_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump({"label2id": label2id, "id2label": id2label}, handle, indent=2)

    # Record which checkpoint was selected using validation macro-F1.
    best_checkpoint = trainer.state.best_model_checkpoint
    best_epoch = None
    if best_checkpoint:
        checkpoint_name = Path(best_checkpoint).name
        if checkpoint_name.startswith("checkpoint-"):
            try:
                best_step = int(checkpoint_name.split("-")[-1])
                for record in trainer.state.log_history:
                    if "epoch" in record and record.get("step") == best_step:
                        best_epoch = float(record["epoch"])
                        break
            except ValueError:
                pass
    # Save settings, class counts, evaluation history, software versions, and
    # confidence-score caveats to make the experiment easier to reproduce.
    metadata = {
        "model_name": model_name,
        "max_length": max_length,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "train_rows": len(train),
        "validation_rows": len(val),
        "train_class_counts": train_counts,
        "validation_class_counts": val_counts,
        "label_order": DESTINATION_LABELS,
        "selection_metric": "eval_macro_f1",
        "best_validation_metric": trainer.state.best_metric,
        "best_checkpoint_path": best_checkpoint,
        "best_epoch": best_epoch,
        "global_step": trainer.state.global_step,
        "evaluation_history": [
            record
            for record in trainer.state.log_history
            if any(key.startswith("eval_") for key in record)
        ],
        "confidence_semantics": "Maximum softmax score; not a calibrated probability unless separately calibrated on validation data.",
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "dynamic_padding": True,
        "class_weighting": class_weighting,
        "class_weights": dict(zip(DESTINATION_LABELS, class_weights)),
    }
    with (out / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return model_dir


# Load a saved tokenizer/model pair and prepare the model for inference.
def load_bert(model_dir: str | Path, device: str | None = None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Disable training behavior (such as dropout) for deterministic inference behavior.
    model.eval()
    return tokenizer, model, device


# Run inference in batches and return one prediction plus scores for every class.
def predict_bert(
    frame: pd.DataFrame,
    model_dir: str | Path,
    max_length: int | None = None,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Predict labels and return raw softmax scores (not calibrated confidence)."""
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    max_length = resolve_max_length(model_dir, max_length)
    tokenizer, model, device = load_bert(model_dir)
    # Recover readable class names from the model configuration in class-ID order.
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    records = []
    # Batch predictions to limit memory use on large input frames.
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        texts = [build_model_text(row) for _, row in batch.iterrows()]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        # Gradients are unnecessary for prediction, so disable gradient tracking.
        with torch.no_grad():
            logits = model(**enc).logits
        # Convert logits into class scores that sum to 1 for each example.
        # These softmax scores are not necessarily calibrated probabilities.
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for (_, row), prob in zip(batch.iterrows(), probs):
            # Select the class with the largest softmax score.
            best = int(np.argmax(prob))
            item = {
                "clause_id": str(row.clause_id),
                "review_id": str(row.review_id),
                "bert_predicted_label": labels[best],
                "bert_confidence": float(prob[best]),
            }
            # Keep scores for all classes, not just the winning prediction, so
            # downstream analysis can inspect uncertainty or compare alternatives.
            item.update(
                {f"prob_{labels[i]}": float(prob[i]) for i in range(len(labels))}
            )
            records.append(item)
    return pd.DataFrame(records)
