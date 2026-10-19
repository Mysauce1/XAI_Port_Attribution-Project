"""Review-level paired bootstrap confidence intervals for attribution metrics.

Reviews, not clauses, are the resampling units. The same sampled review IDs are
used for both methods so confidence intervals for paired metric differences
reflect the comparison on identical resamples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .data import DESTINATION_LABELS

# Metrics reported for each method and for the paired BERT-minus-rule difference.
# "Unclear" predictions count as abstentions for coverage-related metrics.
DEFAULT_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "balanced_accuracy",
    "coverage",
    "selective_accuracy",
)


def _scores(gold: list[str], pred: list[str]) -> dict[str, float]:
    # Arrays make metric calculations and filtering out abstentions straightforward.
    y = np.asarray(gold, dtype=object)
    p = np.asarray(pred, dtype=object)
    # A prediction is covered when the method commits to a destination label.
    covered = p != "Unclear"
    # Balanced accuracy is averaged over gold labels present in this sample.
    observed = [label for label in DESTINATION_LABELS if label in set(gold)]
    return {
        # Exact-match accuracy across all clauses, including "Unclear" predictions.
        "accuracy": float(accuracy_score(y, p)),
        "macro_precision": float(
            precision_score(
                y, p, labels=DESTINATION_LABELS, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y, p, labels=DESTINATION_LABELS, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(y, p, labels=DESTINATION_LABELS, average="macro", zero_division=0)
        ),
        # Balanced accuracy is defined over gold classes present in this sample.
        "balanced_accuracy": (
            float(recall_score(y, p, labels=observed, average="macro", zero_division=0))
            if observed
            else float("nan")
        ),
        # Fraction of clauses receiving a non-"Unclear" prediction.
        "coverage": float(np.mean(covered)),
        # Accuracy among only the clauses for which a destination was predicted.
        "selective_accuracy": (
            float(accuracy_score(y[covered], p[covered]))
            if covered.any()
            else float("nan")
        ),
    }


def paired_review_bootstrap(
    frame: pd.DataFrame,
    *,
    gold_col: str = "destination_label",
    review_col: str = "review_id",
    rule_col: str = "rule_prediction",
    bert_col: str = "bert_prediction",
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Return percentile CIs for both methods and paired BERT-minus-rule deltas.

    Each bootstrap replicate samples the unique review IDs with replacement and
    includes all clauses for each selected review. If a review is sampled twice,
    its clauses appear twice, as required by cluster bootstrap sampling.
    Macro precision/recall/F1 use the fixed five-class taxonomy, with absent
    classes contributing zero via ``zero_division=0``. Balanced accuracy uses
    only gold classes present in that replicate and records this rule in metadata.
    Replicates with no covered predictions have undefined selective accuracy and
    are omitted only from that metric's interval.
    """
    # Validate settings and input columns before starting the bootstrap.
    if n_resamples < 100:
        raise ValueError(
            "n_resamples must be at least 100 for useful percentile intervals"
        )
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    required = [review_col, gold_col, rule_col, bert_col]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required bootstrap columns: {missing}")
    # Keep only the needed columns and avoid modifying the caller's DataFrame.
    data = frame[required].copy()
    if data[required].isna().any().any():
        raise ValueError(
            "Bootstrap inputs contain missing review IDs, gold labels, or predictions"
        )
    data[review_col] = data[review_col].astype(str)
    if data[review_col].eq("").any():
        raise ValueError("review_id must not be empty")
    invalid_gold = sorted(set(data[gold_col].astype(str)) - set(DESTINATION_LABELS))
    allowed_predictions = set(DESTINATION_LABELS) | {"Unclear"}
    invalid_pred = sorted(
        (set(data[rule_col].astype(str)) | set(data[bert_col].astype(str)))
        - allowed_predictions
    )
    if invalid_gold or invalid_pred:
        raise ValueError(
            f"Invalid labels; gold={invalid_gold}, predictions={invalid_pred}"
        )
    if data.empty:
        raise ValueError("Cannot bootstrap an empty evaluation set")

    # Alignment between gold labels and both prediction columns must be
    # established by the caller when the comparison table is built. Here we
    # validate row identity only; comparing a table to itself would not prove
    # that independently generated method outputs were correctly aligned.
    if "clause_id" in frame.columns:
        keys = frame[[review_col, "clause_id"]].copy()
        if keys.isna().any().any() or keys.astype(str).eq("").any().any():
            raise ValueError("review_id and clause_id must not be missing or empty")
        keys = keys.astype(str)
        if keys.duplicated().any():
            raise ValueError(
                "Bootstrap input contains duplicate (review_id, clause_id) keys"
            )

    # If split metadata is present, enforce the test-only contract. If it is
    # absent, callers (for example, externally prepared evaluation tables) are
    # responsible for supplying held-out test rows only.
    if "split" in frame.columns:
        splits = frame["split"].astype(str).str.strip().str.lower()
        if not splits.eq("test").all():
            found = sorted(splits.unique().tolist())
            raise ValueError(
                f"Bootstrap input must contain only test-split rows; found: {found}"
            )

    # Resample whole reviews because clauses within one review are related.
    # Treating clauses as independent observations could understate uncertainty.
    review_ids = data[review_col].drop_duplicates().to_numpy()
    if len(review_ids) < 2:
        raise ValueError(
            "At least two distinct reviews are required for review-level bootstrap"
        )
    # Cache row positions once; repeated review draws duplicate the same row
    # positions in the replicate, preserving cluster-bootstrap semantics.
    # Cache row positions so each sampled review brings along all its clauses.
    review_positions = {
        rid: np.flatnonzero(data[review_col].to_numpy() == rid) for rid in review_ids
    }
    # Store gold labels and both methods' predictions together for each review.
    review_arrays = {
        rid: {
            "gold": data[gold_col].astype(str).to_numpy()[positions],
            "rule": data[rule_col].astype(str).to_numpy()[positions],
            "bert": data[bert_col].astype(str).to_numpy()[positions],
        }
        for rid, positions in review_positions.items()
    }
    # A fixed seed makes the bootstrap draws reproducible.
    rng = np.random.default_rng(random_state)
    method_names = {"rule": rule_col, "bert": bert_col}
    samples: dict[str, dict[str, list[float]]] = {
        method: {metric: [] for metric in DEFAULT_METRICS} for method in method_names
    }
    deltas = {metric: [] for metric in DEFAULT_METRICS}
    missing_class_counts = {label: 0 for label in DESTINATION_LABELS}
    # Both methods use the same sampled review IDs, making the comparison paired.
    for _ in range(n_resamples):
        draw = rng.choice(review_ids, size=len(review_ids), replace=True)
        gold_parts = [review_arrays[rid]["gold"] for rid in draw]
        gold_array = np.concatenate(gold_parts)
        gold = gold_array.tolist()
        gold_present = set(gold)
        for label in DESTINATION_LABELS:
            if label not in gold_present:
                missing_class_counts[label] += 1
        predictions = {
            "rule": np.concatenate(
                [review_arrays[rid]["rule"] for rid in draw]
            ).tolist(),
            "bert": np.concatenate(
                [review_arrays[rid]["bert"] for rid in draw]
            ).tolist(),
        }
        # Score both methods on identical resampled clauses and gold labels.
        current = {name: _scores(gold, predictions[name]) for name in method_names}
        for method in method_names:
            for metric in DEFAULT_METRICS:
                value = current[method][metric]
                if np.isfinite(value):
                    samples[method][metric].append(value)
        # Positive differences favor BERT; negative differences favor the rule-based method.
        for metric in DEFAULT_METRICS:
            delta = current["bert"][metric] - current["rule"][metric]
            if np.isfinite(delta):
                deltas[metric].append(float(delta))

    # Percentile intervals use the lower and upper quantiles of bootstrap scores.
    alpha = (1 - confidence_level) / 2
    rows = []
    # Point estimates use the full evaluation table; intervals use bootstrap resamples.
    full_scores = {
        m: _scores(data[gold_col].astype(str).tolist(), data[c].astype(str).tolist())
        for m, c in method_names.items()
    }
    for metric in DEFAULT_METRICS:
        for method in ("rule", "bert"):
            values = np.asarray(samples[method][metric], dtype=float)
            rows.append(
                {
                    "metric": metric,
                    "estimate": full_scores[method][metric],
                    "comparison": method,
                    "ci_lower": (
                        float(np.quantile(values, alpha)) if len(values) else None
                    ),
                    "ci_upper": (
                        float(np.quantile(values, 1 - alpha)) if len(values) else None
                    ),
                    "valid_resamples": int(len(values)),
                    "requested_resamples": n_resamples,
                }
            )
        values = np.asarray(deltas[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "estimate": full_scores["bert"][metric] - full_scores["rule"][metric],
                "comparison": "bert_minus_rule",
                "ci_lower": float(np.quantile(values, alpha)) if len(values) else None,
                "ci_upper": (
                    float(np.quantile(values, 1 - alpha)) if len(values) else None
                ),
                "valid_resamples": int(len(values)),
                "requested_resamples": n_resamples,
            }
        )
    # Store settings and diagnostics so the results can be interpreted and reproduced.
    summary = {
        "method": "paired_review_level_cluster_bootstrap_percentile_ci",
        "confidence_level": confidence_level,
        "n_resamples_requested": n_resamples,
        "n_reviews": int(len(review_ids)),
        "n_clauses": int(len(data)),
        "random_state": random_state,
        "resampling_unit": review_col,
        "paired_difference": "BERT minus rule-based",
        "macro_metrics_taxonomy": list(DESTINATION_LABELS),
        "macro_precision_recall_f1_absent_class_policy": "fixed five classes; zero_division=0",
        "balanced_accuracy_absent_class_policy": "average recall over gold classes present in each replicate",
        "missing_gold_class_resamples": missing_class_counts,
        "missing_gold_class_resample_fraction": {
            label: count / n_resamples for label, count in missing_class_counts.items()
        },
        "undefined_selective_accuracy_resamples": {
            method: n_resamples - len(samples[method]["selective_accuracy"])
            for method in method_names
        },
        "test_split_validation": "Enforced when a split column is present; otherwise the caller must provide test-only rows.",
        "alignment_contract": "Caller must build the comparison table by key-aligning gold labels and both methods before calling this function; this function validates unique clause keys when clause_id is present.",
        "note": "Percentile intervals quantify review-sampling uncertainty; they do not include model-training-seed variability. Missing-class resample counts should be reported as a test-set limitation when substantial.",
    }
    return pd.DataFrame(rows), summary


def save_bootstrap_results(
    table: pd.DataFrame, summary: dict, output_prefix: str | Path
) -> None:
    # Treat output_prefix as a filename stem for the CSV table and JSON summary.
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(prefix.with_name(prefix.name + ".csv"), index=False)
    with prefix.with_name(prefix.name + ".json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
