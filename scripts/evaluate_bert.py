"""Predict with the fine-tuned BERT model, then compare it with the rule baseline.

Run order: this is the step after the rule baseline and BERT training.
    1. Rule baseline   - writes rule_predictions_all.csv and the split file.
    2. BERT training   - writes the model to --model-dir.
    3. This script     - writes bert_predictions_all.csv, BERT metrics, and the
                         rule-vs-BERT comparison tables and figures.

Outputs go to --output-dir (see ``evaluate_bert`` and ``make_comparison`` in
``cruise_xai.pipeline`` for the full list).
"""

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

# Checks that the split file already exists (and stops with an error if it
# does not), so BERT is scored on the same test reviews as the rule baseline.
from cruise_xai.data import require_existing_splits

# evaluate_bert:   predicts every clause with the saved model and scores the test split.
# make_comparison: aligns rule and BERT predictions and writes comparison outputs.
from cruise_xai.pipeline import evaluate_bert, make_comparison


def main():
    """Run BERT inference and evaluation, then build the method comparison."""
    # --- Command-line arguments ---------------------------------------------
    # Defaults match the standard project layout, so the script can be run
    # with no arguments from the project root.
    p = argparse.ArgumentParser(
        description="Predict with fine-tuned BERT and compare both methods"
    )
    p.add_argument(
        "--data", default="data/processed/annotated_reviews.csv"
    )  # annotated clauses
    p.add_argument(
        "--splits", default="data/processed/review_splits.csv"
    )  # existing split file
    p.add_argument(
        "--model-dir", default="outputs/bert/model"
    )  # fine-tuned BERT weights
    p.add_argument("--output-dir", default="outputs")  # must also hold the rule outputs
    a = p.parse_args()

    # Step 1: BERT predictions and held-out metrics. The split check runs
    # first, so a missing split file fails before the model is loaded.
    evaluate_bert(a.data, require_existing_splits(a.splits), a.model_dir, a.output_dir)

    # Step 2: rule-vs-BERT comparison. Reads both prediction files from
    # --output-dir, so the rule baseline must have been run with the same
    # output directory; otherwise this raises FileNotFoundError.
    make_comparison(a.output_dir)
    print("Saved BERT predictions, held-out metrics, and method comparison.")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
