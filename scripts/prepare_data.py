"""Validate the annotations and create (or reuse) the fixed review-level splits.

This is the designated split-creation step. Run it once, before any modelling:
    - If the split file does not exist, reviews are assigned to
      train/validation/test using --seed and the assignment is saved.
    - If it already exists, it is loaded and validated, and --seed is ignored.

Splits are made at the review level, so all clauses from one review stay in
the same split (no leakage of a review's context between train and test).

Prints a summary of clause counts per split, destination and sentiment
labels per split, and any low-support split/class cells.
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
import pandas as pd  # crosstab summaries
from pathlib import Path  # checking whether the split file exists

# DESTINATION_LABELS / SENTIMENT_LABELS: canonical label lists (reporting order).
# load_annotations: reads and validates the annotated clause file.
# attach_splits:    loads the split file, or creates it if missing, and adds
#                   each clause's split as a ``split`` column.
from cruise_xai.data import (
    DESTINATION_LABELS,
    SENTIMENT_LABELS,
    load_annotations,
    attach_splits,
)


def main():
    """Validate annotations, create or reuse the split file, and print a summary."""
    # --- Command-line arguments ---------------------------------------------
    # Defaults match the standard project layout, so the script can be run
    # with no arguments from the project root.
    p = argparse.ArgumentParser(
        description="Validate annotations and create fixed review-level train/validation/test splits"
    )
    p.add_argument(
        "--data", default="data/processed/annotated_reviews.csv"
    )  # annotated clauses
    p.add_argument(
        "--splits", default="data/processed/review_splits.csv"
    )  # split file to create or reuse
    p.add_argument(
        "--seed", type=int, default=42
    )  # only used when creating a new split
    p.add_argument(
        "--min-support",
        type=int,
        default=10,
        help="Flag split/class cells with fewer gold clauses than this",
    )
    a = p.parse_args()

    # Checked BEFORE attach_splits runs, because attach_splits creates the file
    # when it is missing. This records whether the split was reused or new.
    existed = Path(a.splits).is_file()

    # Load and validate the annotations, then attach splits (creating the split
    # file with ``a.seed`` if it did not exist). ``splits`` is the review-level
    # table: one row per review with its assigned split.
    df = load_annotations(a.data)
    df, splits = attach_splits(df, a.splits, a.seed)

    # --- Summary ------------------------------------------------------------
    print(f"Validated {len(df)} clauses across {df.review_id.nunique()} reviews")
    # Clause counts (not review counts) per split.
    print("Rows by split:\n", df.groupby("split").size().to_string())

    # Fixed display order, keeping only the splits that are actually present.
    order = [s for s in ("train", "validation", "test") if s in set(df.split)]

    # Clauses per split × destination label. Reindexing fixes the row/column
    # order and shows absent labels as 0. Gold values outside
    # DESTINATION_LABELS are not shown.
    by_destination = pd.crosstab(df.split, df.destination_label).reindex(
        index=order, columns=DESTINATION_LABELS, fill_value=0
    )
    print("Destination labels by split:\n", by_destination.to_string())

    # Sentiment labels are optional; summarise them only if the column exists.
    if "sentiment_label" in df:
        by_sentiment = pd.crosstab(df.split, df.sentiment_label).reindex(
            index=order, columns=SENTIMENT_LABELS, fill_value=0
        )
        print("Sentiment labels by split:\n", by_sentiment.to_string())

    # --- Low-support check --------------------------------------------------
    # List every split/label cell with fewer than --min-support clauses,
    # formatted as "test/Roatan (n=4)". Per-class scores for these cells will
    # be unstable.
    low = [
        f"{split}/{label} (n={by_destination.loc[split, label]})"
        for split in order
        for label in DESTINATION_LABELS
        if by_destination.loc[split, label] < a.min_support
    ]
    if low:
        # Re-splitting until the numbers look better would bias the evaluation,
        # so the advice is to report the weakness rather than "fix" it.
        print(
            f"WARNING: destination classes with fewer than {a.min_support} clauses: {', '.join(low)}. "
            "Do not re-split after seeing results; report these classes as low-support instead."
        )

    # --- Report whether the split was reused or created ---------------------
    if existed:
        # An existing split always wins, so a different --seed has no effect.
        print(
            f"Reused existing split file {a.splits} (--seed ignored; delete the file only to start a new experiment)"
        )
    else:
        print(
            f"Created {len(splits)} review-level assignments (seed {a.seed}) in {a.splits}"
        )


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
