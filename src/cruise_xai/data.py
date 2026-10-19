"""Data loading, validation, normalization, and review-level splitting.

This module centralizes the data-handling rules shared by training and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Canonical destination classes used throughout the project. Keep this order
# consistent because model label IDs and evaluation metrics depend on it.
DESTINATION_LABELS = [
    "Roatan",
    "Costa Maya",
    "Cozumel",
    "Multiple Destinations",
    "General Cruise",
]

# Prediction may abstain when the destination is not clear enough to assign.
PREDICTION_LABELS = DESTINATION_LABELS + ["Unclear"]

# Sentiment labels are defined separately from destination labels.
SENTIMENT_LABELS = ["Positive", "Negative", "Neutral", "Mixed"]

# Minimum schema expected in the annotation file.
REQUIRED_COLUMNS = ["review_id", "clause_id", "clause_text", "destination_label"]


def normalize_label(value: object) -> str:
    """Standardize known label variants while leaving other values unchanged."""
    # Convert missing values to an empty string so they can be checked downstream.
    if pd.isna(value):
        return ""
    raw = str(value).strip()

    # Map known spelling/casing/format variants to the project's canonical labels.
    # casefold() provides case-insensitive matching; unknown labels pass through.
    aliases = {
        "multiple_destinations": "Multiple Destinations",
        "multiple destinations": "Multiple Destinations",
        "general_cruise": "General Cruise",
        "general cruise": "General Cruise",
        "positve": "Positive",
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "mixed": "Mixed",
    }
    return aliases.get(raw.casefold(), raw)


def load_annotations(path: str | Path) -> pd.DataFrame:
    """Load and validate the annotation CSV without silently dropping rows.

    Known sentiment spelling errors are normalized. This function does not
    independently verify the correctness of the human/source annotations.
    """
    # Read identifier columns as strings to preserve IDs (including leading zeros).
    df = pd.read_csv(path, dtype={"review_id": str, "clause_id": str})

    # Fail early if the input does not have the columns required by the pipeline.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Annotation file is missing required columns: {missing}")

    # Standardize core text and ID columns; do not drop rows during cleanup.
    df["review_id"] = df["review_id"].astype(str)
    df["clause_id"] = df["clause_id"].astype(str)
    df["clause_text"] = df["clause_text"].fillna("").astype(str)

    # Normalize labels to canonical forms before validating them.
    df["destination_label"] = df["destination_label"].map(normalize_label)
    if "sentiment_label" in df:
        df["sentiment_label"] = df["sentiment_label"].map(normalize_label)

    # Clause IDs are expected to uniquely identify rows in this dataset.
    if df["clause_id"].duplicated().any():
        duplicated = df.loc[df["clause_id"].duplicated(), "clause_id"].head().tolist()
        raise ValueError(f"Duplicate clause_id values: {duplicated}")

    # Reject unexpected destination classes instead of silently accepting typos.
    invalid = sorted(set(df["destination_label"]) - set(DESTINATION_LABELS))
    if invalid:
        raise ValueError(f"Unexpected destination labels: {invalid}")
    return df


def build_model_text(row: pd.Series | dict) -> str:
    """Construct the target-plus-context text passed to the BERT model."""
    # A row may come from a pandas DataFrame or be provided as a dictionary.
    get = row.get

    def text_value(key: str) -> str:
        # Convert missing context fields to empty strings and trim extra whitespace.
        value = get(key, "")
        return "" if pd.isna(value) else str(value).strip()

    # The clause being classified is always included as the primary target.
    target = text_value("clause_text")

    # Optional surrounding context can help resolve references that the clause
    # alone may not make clear (for example, "the excursion was amazing").
    sentence = text_value("sentence_text")
    previous = text_value("previous_sentence")
    following = text_value("next_sentence")

    parts = [f"TARGET: {target}"]

    # Include the larger sentence only when it differs from the target clause.
    if sentence and sentence.casefold() != target.casefold():
        parts.append(f"SENTENCE: {sentence}")
    if previous:
        parts.append(f"PREVIOUS: {previous}")
    if following:
        parts.append(f"NEXT: {following}")

    # The separator marks the different text fields for the tokenizer/model.
    return " [SEP] ".join(parts)


def make_group_splits(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.15,
    validation_size: float = 0.15,
) -> pd.DataFrame:
    """Create train/validation/test splits grouped by review.

    Grouping prevents clauses from the same review appearing in more than one
    partition. The split is reproducible for the same data and seed, but it is
    not guaranteed to preserve each destination class's proportions.
    """
    # Validate the requested proportions; validation_size is the final fraction
    # of the full dataset, not a fraction of the train-plus-validation remainder.
    if (
        not 0 < test_size < 1
        or not 0 < validation_size < 1
        or test_size + validation_size >= 1
    ):
        raise ValueError(
            "test_size and validation_size must be in (0,1) and sum to less than 1"
        )

    # GroupShuffleSplit assigns whole review IDs together, keeping their clauses
    # in a single partition.
    groups = df["review_id"].astype(str).to_numpy()
    dummy = df.index.to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(first.split(dummy, groups=groups))
    trainval = df.iloc[trainval_idx]

    # GroupShuffleSplit's second test_size is relative to trainval, so convert
    # the desired overall validation fraction to that relative fraction.
    relative_val = validation_size / (1.0 - test_size)
    second = GroupShuffleSplit(
        n_splits=1, test_size=relative_val, random_state=seed + 1
    )
    train_local, val_local = next(
        second.split(trainval, groups=trainval["review_id"].astype(str))
    )

    # Build one row per unique review, then assign each review its partition.
    split = pd.DataFrame({"review_id": df["review_id"].astype(str).drop_duplicates()})
    train_reviews = set(trainval.iloc[train_local]["review_id"].astype(str))
    val_reviews = set(trainval.iloc[val_local]["review_id"].astype(str))
    test_reviews = set(df.iloc[test_idx]["review_id"].astype(str))
    split["split"] = split["review_id"].map(
        lambda x: (
            "train"
            if x in train_reviews
            else "validation" if x in val_reviews else "test"
        )
    )

    # Defensive assertion: no review should appear in two partitions.
    if (
        train_reviews & val_reviews
        or train_reviews & test_reviews
        or val_reviews & test_reviews
    ):
        raise AssertionError("Review-level split leakage detected")
    return split


def attach_splits(
    df: pd.DataFrame, split_path: str | Path, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an existing split file or create it once, then attach it to each clause."""
    path = Path(split_path)

    if path.exists():
        # Reuse saved splits so training and evaluation operate on the same reviews.
        splits = pd.read_csv(path, dtype={"review_id": str})
        if not {"review_id", "split"}.issubset(splits.columns):
            raise ValueError("Split file must contain review_id and split columns")
        if splits["review_id"].duplicated().any():
            raise ValueError("Split file must contain one row per review_id")
        if set(splits["split"]) - {"train", "validation", "test"}:
            raise ValueError("split values must be train, validation, or test")

        # Every review in the current data must have an assigned partition.
        missing = set(df["review_id"]) - set(splits["review_id"])
        if missing:
            raise ValueError(f"Split file is missing {len(missing)} reviews")
    else:
        # First run: create and persist the review-level split for later reuse.
        splits = make_group_splits(df, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(path, index=False)

    # many_to_one verifies that each review maps to at most one split row.
    out = df.merge(splits, on="review_id", how="left", validate="many_to_one")
    if out["split"].isna().any():
        raise ValueError("Some rows did not receive a split")
    return out, splits


def require_existing_splits(split_path: str | Path) -> Path:
    """Return the split path, or fail if the frozen split has not been created."""
    path = Path(split_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run scripts/prepare_data.py first to create the frozen split."
        )
    return path


def load_jsonl_reviews(path: str | Path) -> pd.DataFrame:
    """Read JSON Lines reviews into a DataFrame, validating each non-empty line."""
    rows = []
    with open(path, encoding="utf-8") as f:
        # Each non-empty line should contain one independent JSON object.
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    # Include the line number to make malformed input easier to locate.
                    raise ValueError(f"Invalid JSON on line {line_no}") from exc

                # Require the expected review field before adding the record.
                if not isinstance(item, dict) or not isinstance(
                    item.get("review"), str
                ):
                    raise ValueError(
                        f"Line {line_no} must be an object with a review string"
                    )
                rows.append(item)

    # Convert parsed records to a tabular structure for downstream processing.
    return pd.DataFrame(rows)
