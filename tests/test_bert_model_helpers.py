"""Unit tests for the small, model-free helpers in ``cruise_xai.bert_model``.

None of these tests download or train a real model. They use a stub
tokenizer and hand-built inputs to check data preparation, metric
computation, label validation, max-length lookup, and class weighting.
"""

import pytest

# Requires PyTorch (requirements-bert.txt); skipped in the lightweight CI install.
# importorskip skips this whole module (rather than failing) if torch is missing.
# It must run before the cruise_xai.bert_model import below, which needs torch.
pytest.importorskip("torch")

import json

import numpy as np
import pandas as pd
import pytest  # duplicate of the import above; harmless

from cruise_xai.bert_model import (
    ClauseDataset,  # wraps clauses as a PyTorch dataset
    _compute_metrics,  # metric hook used during training/validation
    compute_class_weights,  # per-class loss weights
    _validate_partition_labels,  # label check + per-class counts for one split
    resolve_max_length,  # finds the max_length a model was trained with
)
from cruise_xai.data import DESTINATION_LABELS


class TinyTokenizer:
    """Stand-in for a Hugging Face tokenizer that returns a fixed encoding.

    Every input becomes the same three token IDs (101 and 102 are BERT's
    [CLS] and [SEP] IDs), so tests can check exact outputs without loading
    a real tokenizer.
    """

    def __call__(self, text, truncation=True, max_length=256):
        # Mimics the unpadded list-of-token-ids shape returned by HF tokenizers.
        return {"input_ids": [101, 200, 102], "attention_mask": [1, 1, 1]}


# --- ClauseDataset -----------------------------------------------------------


def test_clause_dataset_defers_padding_and_keeps_labels():
    # Padding should happen later, per batch (dynamic padding by the data
    # collator), not inside the dataset; and labels must map to the right index.
    frame = pd.DataFrame(
        [{"clause_text": "We visited Roatan", "destination_label": "Roatan"}]
    )
    # Label-to-index mapping in DESTINATION_LABELS order, as used in training.
    ds = ClauseDataset(
        frame, TinyTokenizer(), {label: i for i, label in enumerate(DESTINATION_LABELS)}
    )
    item = ds[0]
    # Token IDs are passed through unchanged (as tensors).
    assert item["input_ids"].tolist() == [101, 200, 102]
    # The text label is converted to its integer class index.
    assert item["labels"].item() == DESTINATION_LABELS.index("Roatan")
    assert len(item["input_ids"]) == 3  # not padded to max_length


# --- Metrics -----------------------------------------------------------------


def test_macro_f1_explicitly_uses_all_destination_labels():
    # Only Roatan appears in truth and predictions; absent classes count as zero.
    # Two clauses, five classes: column 0 has the highest logit, so both are
    # predicted as class 0 (assumed to be Roatan, the first entry of
    # DESTINATION_LABELS), and both gold labels are class 0.
    logits = np.array([[5, 0, 0, 0, 0], [5, 0, 0, 0, 0]], dtype=float)
    labels = np.array([0, 0])
    metrics = _compute_metrics((logits, labels))
    # Class 0 has F1 = 1 and the other four classes have F1 = 0, so macro-F1
    # over all five labels is 1/5. If only present classes were averaged, it
    # would wrongly be 1.0.
    assert metrics["macro_f1"] == pytest.approx(0.2)


# --- Label validation --------------------------------------------------------


def test_validate_partition_labels_returns_zero_counts_for_missing_classes():
    # A split may lack some classes entirely; they should be reported as 0
    # rather than missing from the counts.
    frame = pd.DataFrame({"destination_label": ["Roatan", "Roatan"]})
    counts = _validate_partition_labels(frame, "test")
    assert counts["Roatan"] == 2
    assert counts["Cozumel"] == 0


# --- max_length lookup -------------------------------------------------------


def test_resolve_max_length_reads_training_metadata(tmp_path):
    # Layout mirrors a real run: <run_dir>/model/ plus
    # <run_dir>/training_metadata.json one level above it.
    # (tmp_path is a fresh temporary folder provided by pytest.)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (tmp_path / "training_metadata.json").write_text(
        json.dumps({"max_length": 128}), encoding="utf-8"
    )
    # Without an override, the training value is used so inference truncates
    # inputs the same way training did.
    assert resolve_max_length(model_dir) == 128
    # An explicit value takes priority over the metadata.
    assert resolve_max_length(model_dir, 64) == 64


def test_resolve_max_length_requires_metadata_if_not_overridden(tmp_path):
    # With no metadata file and no override, there is no safe value to guess,
    # so a clear error is expected instead of a silent default.
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_max_length(model_dir)


# --- Class weighting ---------------------------------------------------------


def test_class_weights_default_is_unweighted():
    # "none" gives every class weight 1.0, i.e. the ordinary unweighted loss.
    counts = {label: 10 for label in DESTINATION_LABELS}
    assert compute_class_weights(counts, "none") == [1.0] * len(DESTINATION_LABELS)


def test_balanced_class_weights_upweight_rare_classes():
    # Imbalanced counts: Multiple Destinations is rarest (10), General Cruise
    # next (30), and the three ports are common (80 each). Total = 280.
    counts = {
        "Roatan": 80,
        "Costa Maya": 80,
        "Cozumel": 80,
        "Multiple Destinations": 10,
        "General Cruise": 30,
    }
    # Weights come back as a list in DESTINATION_LABELS order; zip them back to names.
    weights = dict(zip(DESTINATION_LABELS, compute_class_weights(counts, "balanced")))
    # Rarer classes must get larger weights.
    assert (
        weights["Multiple Destinations"] > weights["General Cruise"] > weights["Roatan"]
    )
    # Same formula as scikit-learn's class_weight="balanced": n / (k * count).
    # n = 280 clauses, k = 5 classes, so Roatan's weight is 280 / 400 = 0.7.
    assert weights["Roatan"] == pytest.approx(280 / (5 * 80))


def test_class_weights_reject_unknown_scheme():
    # Only "none" and "balanced" are supported; anything else should fail loudly.
    with pytest.raises(ValueError):
        compute_class_weights({"Roatan": 1}, "inverse")
