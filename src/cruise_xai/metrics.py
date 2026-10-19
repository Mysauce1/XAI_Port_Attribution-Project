"""Common metrics and plots for the two attribution methods."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from .data import DESTINATION_LABELS, PREDICTION_LABELS


def evaluate_predictions(
    gold,
    predicted,
    labels=None,
    min_support: int = 10,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Evaluate destination predictions with explicit abstention semantics.

    Gold labels must be one of the five destination classes. Predictions may
    additionally be ``Unclear``. The confusion matrix has five gold rows and
    six prediction columns, so abstentions remain visible without inventing a
    gold ``Unclear`` class.

    Classes with fewer than ``min_support`` gold examples are listed in
    ``low_support_classes`` and flagged in the per-class table: their per-class
    scores rest on very few clauses and should be reported as uncertain.
    """
    # Use the six prediction labels by default so the expected prediction
    # classes, including the optional abstention label, are explicit.
    labels = list(PREDICTION_LABELS if labels is None else labels)

    # Catch accidental repeated or missing/extra labels before computing metrics.
    if len(labels) != len(set(labels)):
        raise ValueError("labels must not contain duplicates")
    if set(labels) != set(PREDICTION_LABELS):
        raise ValueError(f"labels must contain exactly {PREDICTION_LABELS}")

    # Convert missing values to empty strings and normalize values to strings.
    # Empty strings will then be caught by the label validation below.
    y_true = ["" if pd.isna(x) else str(x).strip() for x in gold]
    y_pred = ["" if pd.isna(x) else str(x).strip() for x in predicted]

    # The two inputs must describe the same non-empty set of clause predictions.
    if not y_true or not y_pred:
        raise ValueError("gold and predicted inputs must be non-empty")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"gold and predicted lengths differ: {len(y_true)} != {len(y_pred)}"
        )

    # Gold labels represent the five true destination classes only; Unclear is
    # permitted as a model prediction, not as a ground-truth class.
    invalid_gold = sorted(set(y_true) - set(DESTINATION_LABELS))
    if invalid_gold:
        raise ValueError(f"Invalid gold destination labels: {invalid_gold}")
    invalid_pred = sorted(set(y_pred) - set(PREDICTION_LABELS))
    if invalid_pred:
        raise ValueError(f"Invalid prediction labels: {invalid_pred}")

    # Identify gold classes represented in this evaluation subset. This list is
    # used for balanced accuracy so absent classes do not artificially count
    # as zero-recall classes.
    present = [label for label in DESTINATION_LABELS if label in set(y_true)]

    # Build a report across all five destination classes. With zero_division=0,
    # undefined precision/recall/F1 values are returned as zero rather than
    # raising warnings.
    report = classification_report(
        y_true,
        y_pred,
        labels=DESTINATION_LABELS,
        output_dict=True,
        zero_division=0,
    )

    # Create a compact per-class table and flag classes with limited gold support.
    per_class = pd.DataFrame(
        [
            {
                "label": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1-score": report[label]["f1-score"],
                "support": int(report[label]["support"]),
                "low_support": int(report[label]["support"]) < min_support,
            }
            for label in DESTINATION_LABELS
        ]
    )

    # Compute recall over only classes present in the gold labels. The mean of
    # these recalls is the balanced-accuracy value reported below.
    present_report = classification_report(
        y_true,
        y_pred,
        labels=present,
        output_dict=True,
        zero_division=0,
    )

    # Selective accuracy measures correctness among non-abstained predictions.
    # Coverage is the share of all predictions that are not "Unclear".
    accepted = [p == t for t, p in zip(y_true, y_pred) if p != "Unclear"]

    metrics = {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            np.mean([present_report[label]["recall"] for label in present])
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=DESTINATION_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=DESTINATION_LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
        "coverage": float(np.mean(np.array(y_pred) != "Unclear")),
        "selective_accuracy": float(np.mean(accepted)) if accepted else None,
        "abstention_count": int(np.sum(np.array(y_pred) == "Unclear")),
        "gold_class_support": {
            label: int(sum(x == label for x in y_true)) for label in DESTINATION_LABELS
        },
        "evaluation_classes_present": present,
        "low_support_classes": [
            label
            for label in DESTINATION_LABELS
            if report[label]["support"] < min_support
        ],
        "min_support_threshold": int(min_support),
        "per_class": {label: report[label] for label in DESTINATION_LABELS},
    }
    # Keep five ground-truth rows but include an extra prediction column for
    # "Unclear", making abstentions visible without adding a gold Unclear class.
    cm = confusion_matrix(y_true, y_pred, labels=DESTINATION_LABELS + ["Unclear"])
    return metrics, per_class, cm


def align_predictions(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    id_columns: tuple[str, ...] = ("review_id", "clause_id"),
) -> pd.DataFrame:
    """Validate exact clause-key equality and return candidate in reference order."""
    # Ensure both tables have the identifiers needed to uniquely identify clauses.
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = set(id_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} data missing ID columns: {sorted(missing)}")
        if frame[list(id_columns)].isna().any().any():
            raise ValueError(f"{name} data contains missing ID values")

        # String normalization makes values such as numeric and string IDs
        # compare consistently; duplicated composite keys are not allowed.
        keys = frame[list(id_columns)].astype(str)
        if keys.duplicated().any():
            raise ValueError(f"{name} data contains duplicate ID keys")

    # Compare the complete sets of (review_id, clause_id) keys. Matching row
    # counts alone would not detect a missing clause paired with an extra one.
    ref_keys = set(map(tuple, reference[list(id_columns)].astype(str).to_numpy()))
    cand_keys = set(map(tuple, candidate[list(id_columns)].astype(str).to_numpy()))
    if ref_keys != cand_keys:
        # Show a few examples of mismatches to help diagnose alignment problems.
        ref_only = list(ref_keys - cand_keys)[:5]
        cand_only = list(cand_keys - ref_keys)[:5]
        raise ValueError(
            f"Clause keys differ (reference-only examples={ref_only}; candidate-only examples={cand_only})"
        )

    # Return a copy, normalize identifier types, and reorder candidate rows to
    # exactly match the reference table's clause order.
    out = candidate.copy()
    for col in id_columns:
        out[col] = out[col].astype(str)
    order = pd.MultiIndex.from_frame(reference[list(id_columns)].astype(str))
    candidate_index = pd.MultiIndex.from_frame(out[list(id_columns)])
    out = out.set_index(candidate_index).loc[order].reset_index(drop=True)
    return out


def save_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output: str | Path,
    title: str,
    *,
    normalize: bool = False,
) -> None:
    """Plot and save a confusion matrix, optionally normalized by gold row."""
    # Keep the supplied label order because it determines the matrix axes.
    labels = list(labels)
    cm = np.asarray(cm)

    # A full matrix includes a row for every label; the abstention-aware layout
    # instead omits the "Unclear" gold row while retaining its prediction column.
    expected_rows = len(labels) - (1 if "Unclear" in labels else 0)
    if cm.shape != (expected_rows, len(labels)) and cm.shape != (
        len(labels),
        len(labels),
    ):
        raise ValueError(
            f"Confusion matrix shape {cm.shape} is incompatible with {len(labels)} labels"
        )

    # If given a full square matrix with an Unclear label, remove its gold row:
    # "Unclear" is a valid prediction but not a valid gold destination class.
    if cm.shape[0] == len(labels) and "Unclear" in labels:
        keep_rows = [i for i, label in enumerate(labels) if label != "Unclear"]
        cm = cm[keep_rows, :]
        row_labels = [labels[i] for i in keep_rows]
    else:
        # Otherwise, the caller has already supplied the prediction-only layout.
        row_labels = (
            [label for label in labels if label != "Unclear"]
            if "Unclear" in labels
            else labels
        )

    # Convert counts to floats for optional row-wise normalization.
    display = cm.astype(float)
    if normalize:
        # Normalize each gold-label row independently. A row with no examples
        # stays all zeros instead of producing division-by-zero values.
        totals = display.sum(axis=1, keepdims=True)
        display = np.divide(
            display, totals, out=np.zeros_like(display), where=totals != 0
        )

    # Create the destination folder if needed before saving the figure.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Draw the heatmap. Normalized values are bounded to [0, 1].
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        display,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1 if normalize else None,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(row_labels)),
        xticklabels=labels,
        yticklabels=row_labels,
        ylabel="Gold label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(
        ax.get_xticklabels(),
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )

    # Choose a contrasting text color based on the displayed cell intensity.
    threshold = (
        0.5
        if normalize
        else (display.max() / 2 if display.size and display.max() else 0)
    )
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            value = f"{display[i, j]:.1%}" if normalize else str(int(cm[i, j]))
            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if display[i, j] > threshold else "black",
            )

    # Tighten spacing, write the image at print-friendly resolution, then close
    # the figure to release resources when generating many plots in one run.
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_metrics(
    metrics: dict,
    per_class: pd.DataFrame,
    prefix: str | Path,
) -> None:
    """Save aggregate metrics as JSON and per-class metrics as CSV."""
    # Treat prefix as a filename stem, not a directory. Ensure its parent folder
    # exists, then write two companion outputs sharing the same stem.
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # JSON is useful for programmatic comparison; CSV is convenient for inspecting
    # the per-class table in spreadsheet software.
    with open(prefix.with_name(prefix.name + ".json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)
    per_class.to_csv(prefix.with_name(prefix.name + "_per_class.csv"), index=False)
