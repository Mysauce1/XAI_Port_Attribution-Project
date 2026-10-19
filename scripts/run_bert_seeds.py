"""Train/evaluate BERT across predetermined seeds on one fixed review-level split.

The held-out test metrics are reported per seed and summarized after all runs;
they must not be used to choose a preferred seed or model.
"""

# Note: this docstring is also used as the --help description (see main()).
# The purpose of a multi-seed run is to show how much the test score varies
# with random initialisation and data order alone, with everything else held
# fixed. Mean ± std across seeds is the result to report.
#
# Outputs (under --output-dir):
#     seed_<N>/                        model and training files for each seed
#     seed_<N>/test_predictions.csv    that seed's predictions on the test split
#     test_metrics_by_seed.csv         one row of test metrics per seed
#     test_metrics_seed_summary.csv    mean and std of each metric across seeds

# --- Path setup (must run before importing cruise_xai) -----------------------
import argparse  # command-line options
import sys

# Imported under a private alias; reused below for output paths.
from pathlib import Path as _Path

# Make the ``src`` directory importable when this file is run directly as a
# script (``python scripts/<name>.py``) without installing the package.
# parents[1] is the project root, assuming this file lives one level below it.
# Inserting at position 0 makes the local package take priority over any
# installed copy of the same name.
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

# --- Imports -----------------------------------------------------------------
import pandas as pd

# train_bert:   fine-tunes a model and returns the directory it was saved to.
# predict_bert: predicts labels for the given clauses with a saved model.
from cruise_xai.bert_model import predict_bert, train_bert

# require_existing_splits: stops with an error if the split file is missing,
# so this script can never create a new split.
from cruise_xai.data import attach_splits, load_annotations, require_existing_splits

# Returns (metrics dict, per-class table, confusion matrix); only the dict is used here.
from cruise_xai.metrics import evaluate_predictions


def main():
    """Train one model per seed, score each on the test split, and summarise."""
    # --- Command-line arguments ---------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/processed/annotated_reviews.csv"
    )  # annotated clauses
    parser.add_argument(
        "--splits", default="data/processed/review_splits.csv"
    )  # existing split file
    parser.add_argument("--output-dir", default="outputs/bert_multi_seed")
    # Seeds are fixed in advance (not chosen after seeing results).
    # nargs="+" accepts one or more values, e.g. --seeds 1 2 3.
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    # Hugging Face model to fine-tune; DistilBERT is a smaller, faster BERT.
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    # Maximum input length in tokens; longer inputs are truncated.
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5
    )  # typical BERT fine-tuning rate
    # "balanced" weights the loss inversely to class frequency so rare
    # destinations count more during training; "none" uses the plain loss.
    parser.add_argument(
        "--class-weighting", choices=["none", "balanced"], default="none"
    )
    args = parser.parse_args()

    output_dir = _Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_annotations(args.data)
    # All seeds use the same frozen review-level split created by prepare_data.py.
    df, _ = attach_splits(df, require_existing_splits(args.splits))
    # Test clauses are selected once and shared by every seed.
    test = df[df["split"] == "test"].copy()
    rows = []  # one metrics dict per seed

    # --- One full training + evaluation run per seed ------------------------
    for seed in args.seeds:
        # Each seed gets its own folder so runs do not overwrite each other.
        run_dir = output_dir / f"seed_{seed}"
        # The full frame (all splits) is passed; train_bert is expected to
        # train on the train split and use validation internally, never test.
        # Only ``seed`` changes between runs; every hyperparameter is the same.
        model_dir = train_bert(
            df,
            run_dir,
            model_name=args.model_name,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=seed,
            class_weighting=args.class_weighting,
        )
        # Predict the held-out test clauses with this seed's model.
        predictions = predict_bert(test, model_dir)
        # Gold labels and predictions are paired by row position, so this
        # relies on predict_bert returning rows in the same order as ``test``.
        metrics, _, _ = evaluate_predictions(
            test["destination_label"], predictions["bert_predicted_label"]
        )
        # Record which run produced these numbers.
        metrics["seed"] = seed
        metrics["model_dir"] = str(model_dir)
        metrics["class_weighting"] = args.class_weighting
        rows.append(metrics)
        predictions.to_csv(run_dir / "test_predictions.csv", index=False)

    # --- Results across seeds -----------------------------------------------
    # One row per seed. Nested values in the metrics dicts (e.g. per-class
    # scores) end up as dict/list cells in the CSV.
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "test_metrics_by_seed.csv", index=False)
    # Headline metrics to summarise, keeping only those actually present.
    numeric_metrics = [
        c
        for c in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")
        if c in results
    ]
    # ddof=1 gives the sample standard deviation across seeds. With a single
    # seed it is undefined, so std will be NaN.
    summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": results[metric].mean(),
                "std": results[metric].std(ddof=1),
            }
            for metric in numeric_metrics
        ]
    )
    summary.to_csv(output_dir / "test_metrics_seed_summary.csv", index=False)
    print(f"Saved per-seed test metrics and summary to {output_dir}")
    # Picking the best-scoring seed on test data would overstate performance.
    print(
        "Do not select a seed using these test results; use validation data for model selection."
    )


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
