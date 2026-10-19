"""Run the auditable forward-window rule baseline and evaluate it on the test split.

Requires the review-level split file to exist already (created by
prepare_data.py); this script never creates or changes the split.

Outputs go to --output-dir (see ``run_rules`` in ``cruise_xai.pipeline``):
rule_predictions_all.csv, rule metrics, confusion-matrix figures, and
rule_test_cases.csv with the decision trace for each test clause.
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

# Stops with an error if the split file is missing, so run_rules can never
# fall back to creating a new split.
from cruise_xai.data import require_existing_splits

# Applies the rule baseline to every clause and scores the test split.
from cruise_xai.pipeline import run_rules


def main():
    """Run the rule baseline on the fixed split and print a short summary."""
    # --- Command-line arguments ---------------------------------------------
    # Defaults match the standard project layout, so the script can be run
    # with no arguments from the project root.
    p = argparse.ArgumentParser(
        description="Run auditable forward-window baseline and held-out evaluation"
    )
    p.add_argument(
        "--data", default="data/processed/annotated_reviews.csv"
    )  # annotated clauses
    p.add_argument(
        "--splits", default="data/processed/review_splits.csv"
    )  # existing split file
    p.add_argument(
        "--output-dir", default="outputs"
    )  # BERT evaluation must use the same folder
    # Number of sentences after an explicit port mention that inherit that port.
    p.add_argument("--window-size", type=int, default=2)
    a = p.parse_args()

    # The split check runs first. No seed is passed: run_rules only uses a seed
    # when creating a split, which cannot happen here.
    result = run_rules(
        a.data,
        require_existing_splits(a.splits),
        a.output_dir,
        window_size=a.window_size,
    )

    # ``result`` holds every clause (all splits); only test rows are scored.
    print(
        f"Wrote rule predictions for {len(result)} clauses; test rows: {(result.split == 'test').sum()}"
    )


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
