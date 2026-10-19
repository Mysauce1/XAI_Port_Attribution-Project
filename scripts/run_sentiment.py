"""VADER sentiment analysis on top of the rule-based and BERT destination attributions.

Kept separate from the attribution pipeline: this script only reads the saved
prediction files and the gold annotations. Run it after ``run_rules.py`` and,
optionally, ``evaluate_bert.py``. It never retrains or changes either method.
"""

# Note: this docstring is also used as the --help description (see main()).
#
# What it produces (under --output-dir, default outputs/sentiment):
#   1. VADER vs gold sentiment: per-class scores and confusion matrices.
#   2. Destination-by-sentiment profiles for gold, rule, and BERT attributions,
#      their errors against the gold profile, and a net-sentiment figure.
#   3. Joint (destination AND sentiment) accuracy, review-level bootstrap
#      confidence intervals, a per-clause table, and sentiment_summary.json.

# --- Path setup (must run before importing cruise_xai) -----------------------
import sys

# Imported under a private alias so it is clearly only for the path setup below.
from pathlib import Path as _Path

# Make the ``src`` directory importable when this file is run directly as a
# script (``python scripts/<name>.py``) without installing the package.
# parents[1] is the project root, assuming this file lives one level below it.
# Inserting at position 0 makes the local package take priority over any
# installed copy of the same name.
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

# --- Imports -----------------------------------------------------------------
import argparse  # command-line options
from pathlib import Path  # output and prediction paths

import pandas as pd

# require_existing_splits: stops with an error if the split file is missing,
# so this script can never create a new split.
from cruise_xai.data import (
    SENTIMENT_LABELS,
    attach_splits,
    load_annotations,
    require_existing_splits,
)

# align_predictions: reorders saved predictions to match a given set of clause keys.
from cruise_xai.metrics import align_predictions, save_confusion_matrix

# All the analysis logic lives in cruise_xai.sentiment; this script just
# wires it together and writes the outputs.
from cruise_xai.sentiment import (
    DEFAULT_NEGATIVE_THRESHOLD,
    DEFAULT_POSITIVE_THRESHOLD,
    add_vader_columns,
    all_destination_profiles,
    default_bootstrap_statistics,
    evaluate_sentiment,
    joint_accuracy,
    profile_errors,
    review_bootstrap,
    save_json,
    save_net_sentiment_figure,
    summarize_profile_errors,
)

# Where each method's saved predictions live (relative to --predictions-dir)
# and which column in that file holds the predicted destination.
PREDICTION_FILES = {
    "rule": ("rule_predictions_all.csv", "predicted_label"),
    "bert": ("bert_predictions_all.csv", "bert_predicted_label"),
}
# A clause is identified by its review and clause ID together.
KEYS = ["review_id", "clause_id"]


def attach_method_predictions(
    df: pd.DataFrame, predictions_dir: Path, methods: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Add rule_prediction / bert_prediction columns, aligned by clause key.

    A method whose prediction file is missing is skipped with a note rather
    than an error, so the script also works when only the rule baseline has
    been run. Returns the updated frame and the methods actually attached.
    The new columns are added to ``df`` in place.
    """
    available = []
    for method in methods:
        filename, column = PREDICTION_FILES[method]
        path = predictions_dir / filename
        if not path.is_file():
            print(f"NOTE: {path} not found; skipping the {method} method.")
            continue
        # Read IDs as strings so values like "007" match the annotation keys.
        saved = pd.read_csv(path, dtype={"review_id": str, "clause_id": str})
        if column not in saved.columns:
            raise ValueError(f"{path} has no '{column}' column")
        # Reorder the saved rows to match ``df`` clause by clause.
        aligned = align_predictions(df[KEYS], saved)
        # Guard against stale prediction files built from different annotations.
        # After alignment, the gold label saved with each prediction must equal
        # the current gold label for the same clause.
        if (
            "destination_label" in aligned.columns
            and not (
                aligned["destination_label"].astype(str).to_numpy()
                == df["destination_label"].astype(str).to_numpy()
            ).all()
        ):
            raise ValueError(
                f"{path} was built from different gold labels; rerun that method first."
            )
        # Assigned positionally (.to_numpy()) because ``aligned`` is already in
        # ``df`` row order; this avoids index-label mismatches. The column name
        # (rule_prediction / bert_prediction) is what cruise_xai.sentiment expects.
        df[f"{method}_prediction"] = aligned[column].to_numpy()
        available.append(method)
    return df, available


def main():
    """Score VADER, build destination sentiment profiles, and write all outputs."""
    # --- Command-line arguments ---------------------------------------------
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data", default="data/processed/annotated_reviews.csv"
    )  # annotated clauses
    p.add_argument(
        "--splits", default="data/processed/review_splits.csv"
    )  # existing split file
    p.add_argument(
        "--predictions-dir",
        default="outputs",
        help="Folder holding rule_predictions_all.csv and bert_predictions_all.csv",
    )
    p.add_argument("--output-dir", default="outputs/sentiment")
    # Which clauses to analyse. "test" is the default so results sit beside the
    # held-out attribution metrics; "all" combines every split.
    p.add_argument(
        "--split", choices=["test", "validation", "train", "all"], default="test"
    )
    # Attribution methods to compare against gold (gold is always included).
    p.add_argument(
        "--methods", nargs="+", choices=["rule", "bert"], default=["rule", "bert"]
    )
    # VADER compound cut-offs (standard values: +0.05 / -0.05).
    p.add_argument(
        "--positive-threshold", type=float, default=DEFAULT_POSITIVE_THRESHOLD
    )
    p.add_argument(
        "--negative-threshold", type=float, default=DEFAULT_NEGATIVE_THRESHOLD
    )
    p.add_argument(
        "--mixed-min",
        type=float,
        default=None,
        help="Optional: label Mixed when VADER pos and neg proportions both reach this value. "
        "Choose it on train/validation only.",
    )
    p.add_argument(
        "--min-clauses",
        type=int,
        default=5,
        help="Destinations with fewer attributed clauses are flagged and left out of mean profile errors",
    )
    p.add_argument(
        "--min-support",
        type=int,
        default=10,
        help="Sentiment classes with fewer gold examples are flagged as low-support",
    )
    p.add_argument("--bootstrap-resamples", type=int, default=2000)
    # Bootstrap random seed only (it does not affect the split).
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    # Fail early if the split file is missing, before any work is done.
    split_path = require_existing_splits(a.splits)
    out = Path(a.output_dir)
    # Creating figures/ also creates the output folder itself (parents=True).
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # --- Load and validate data ---------------------------------------------
    df = load_annotations(a.data)
    if "sentiment_label" not in df.columns:
        raise ValueError(
            "The annotation file has no sentiment_label column; sentiment cannot be evaluated."
        )
    # astype(str) turns missing values into "nan", so blanks are caught here too.
    invalid = sorted(set(df["sentiment_label"].astype(str)) - set(SENTIMENT_LABELS))
    if invalid:
        raise ValueError(f"Unexpected sentiment labels: {invalid}")
    df, _ = attach_splits(df, split_path)
    # ``methods`` may be shorter than requested if a prediction file is missing.
    df, methods = attach_method_predictions(df, Path(a.predictions_dir), a.methods)

    # Restrict to the chosen split. Predictions were attached first, on the
    # full frame, so alignment covers every clause.
    if a.split != "all":
        df = df[df["split"] == a.split].copy()
    if df.empty:
        raise ValueError(f"No clauses in split '{a.split}'")
    # BERT has seen the training clauses, so scores there look better than they are.
    if a.split in {"train", "all"} and "bert" in methods:
        print(
            "WARNING: BERT was trained on the train split; its attributions there are in-sample."
        )

    # Add VADER scores (compound/pos/neu/neg) and the mapped ``vader_label``.
    scored = add_vader_columns(
        df,
        positive_threshold=a.positive_threshold,
        negative_threshold=a.negative_threshold,
        mixed_min=a.mixed_min,
    )

    # 1. VADER vs gold sentiment.
    metrics, per_class, cm = evaluate_sentiment(
        scored["sentiment_label"], scored["vader_label"], a.min_support
    )
    per_class.to_csv(out / "vader_sentiment_per_class.csv", index=False)
    # Raw counts, then a row-normalised version where each gold class sums to 100%.
    save_confusion_matrix(
        cm,
        SENTIMENT_LABELS,
        out / "figures" / "vader_sentiment_confusion_matrix.png",
        f"VADER vs gold sentiment ({a.split})",
    )
    save_confusion_matrix(
        cm,
        SENTIMENT_LABELS,
        out / "figures" / "vader_sentiment_confusion_matrix_normalized.png",
        f"VADER vs gold sentiment ({a.split}, row-normalized)",
        normalize=True,
    )

    # 2. Destination-by-sentiment profiles under each attribution source.
    # One row per (attribution source, sentiment source, destination).
    profiles = all_destination_profiles(scored, methods, min_clauses=a.min_clauses)
    profiles.to_csv(out / "destination_sentiment_profiles.csv", index=False)
    # How far each profile is from the gold-destination, gold-sentiment reference.
    errors = profile_errors(profiles, methods)
    errors.to_csv(out / "destination_sentiment_profile_errors.csv", index=False)
    save_net_sentiment_figure(
        profiles, methods, out / "figures" / "destination_net_sentiment.png"
    )

    # 3. End-to-end correctness and uncertainty.
    # Share of clauses where both the destination and the VADER sentiment are right.
    joint = {method: joint_accuracy(scored, method) for method in methods}
    # Confidence intervals from resampling whole reviews; every statistic is
    # computed on the same resamples, so BERT-minus-rule differences are paired.
    intervals = review_bootstrap(
        scored,
        default_bootstrap_statistics(methods, a.min_clauses),
        n_resamples=a.bootstrap_resamples,
        random_state=a.seed,
    )
    intervals.to_csv(out / "sentiment_bootstrap_intervals.csv", index=False)

    # --- Per-clause table for error analysis --------------------------------
    # Only prediction columns for methods that were actually attached are kept.
    keep = [
        "review_id",
        "clause_id",
        "split",
        "clause_text",
        "destination_label",
        *[f"{m}_prediction" for m in methods],
        "sentiment_label",
        "vader_label",
        "vader_compound",
        "vader_pos",
        "vader_neu",
        "vader_neg",
    ]
    clause_table = scored[keep].copy()
    clause_table["vader_correct"] = (
        clause_table["vader_label"] == clause_table["sentiment_label"]
    )
    clause_table.to_csv(out / f"clause_sentiment_{a.split}.csv", index=False)

    # --- Summary JSON -------------------------------------------------------
    # Records the headline results together with every setting needed to
    # reproduce them. save_json converts NaN to null so the file stays valid JSON.
    save_json(
        {
            "split": a.split,
            "n_clauses": int(len(scored)),
            "n_reviews": int(scored["review_id"].nunique()),
            "methods": methods,
            "vader_settings": {
                "positive_threshold": a.positive_threshold,
                "negative_threshold": a.negative_threshold,
                "mixed_min": a.mixed_min,
                "text_column": "clause_text",
            },
            "vader_vs_gold": metrics,
            "joint_destination_and_sentiment": joint,
            "profile_error_summary": summarize_profile_errors(errors),
            "min_clauses_for_profile": a.min_clauses,
            "bootstrap": {
                "file": "sentiment_bootstrap_intervals.csv",
                "resamples": a.bootstrap_resamples,
                "unit": "review_id",
                "seed": a.seed,
            },
            # Interpretation caveats, stored with the results so they travel together.
            "notes": [
                "Profile errors are measured against the gold-destination, gold-sentiment profile.",
                "Abstained (Unclear) rule clauses are excluded from rule profiles; see coverage.",
                "Without --mixed-min, VADER cannot output Mixed, so Mixed recall is 0 by construction.",
                "Low-support destinations and sentiment classes are flagged; interpret them cautiously.",
            ],
        },
        out / "sentiment_summary.json",
    )

    # --- Terminal summary ---------------------------------------------------
    print(
        f"VADER vs gold sentiment on {len(scored)} {a.split} clauses: "
        f"accuracy {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}"
    )
    if metrics["low_support_classes"]:
        print(
            f"WARNING: low-support sentiment classes (< {a.min_support}): {', '.join(metrics['low_support_classes'])}"
        )
    print(f"Wrote sentiment outputs to {out}")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
