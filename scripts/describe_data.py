"""Write dataset quality checks and label-distribution figure.

This script is descriptive only: it requires an existing review-level split file
and never creates or changes the experimental split.

Outputs (under --output-dir):
    tables/destination_label_counts.csv     clause count per gold destination label
    tables/destination_label_by_split.csv   the same counts broken down by split
    tables/data_quality_summary.json        sizes, data-quality checks, low-support cells
    figures/destination_label_distribution.png
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
from pathlib import Path  # filesystem paths
import pandas as pd  # tables and counting
import matplotlib.pyplot as plt  # the label-distribution figure

# DESTINATION_LABELS: canonical label list, also used as the reporting order.
# load_annotations:   reads the annotated clause file.
# attach_splits:      adds each clause's train/validation/test split.
from cruise_xai.data import DESTINATION_LABELS, load_annotations, attach_splits


def main():
    """Read the annotations and split, then write the summary tables, JSON, and figure."""
    # --- Command-line arguments ---------------------------------------------
    # Defaults match the standard project layout, so the script can be run
    # with no arguments from the project root.
    p = argparse.ArgumentParser(
        description="Summarize annotation labels, missingness, and review-level split support"
    )
    p.add_argument(
        "--data", default="data/processed/annotated_reviews.csv"
    )  # annotated clauses
    p.add_argument(
        "--splits", default="data/processed/review_splits.csv"
    )  # existing split file
    p.add_argument("--output-dir", default="outputs")  # root for tables/ and figures/
    p.add_argument(
        "--min-support",
        type=int,
        default=10,
        help="Flag split/class cells with fewer gold clauses than this",
    )
    a = p.parse_args()

    # Fail early rather than letting attach_splits generate a new split: the
    # split must come from the designated pipeline step so all results share it.
    split_path = Path(a.splits)
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. Create the review-level splits "
            "using the designated pipeline step before running this descriptive script."
        )

    # Create the output folders if needed; exist_ok avoids an error on re-runs.
    out = Path(a.output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # One row per annotated clause.
    df = load_annotations(a.data)
    # The split file must already exist. attach_splits loads it and validates it;
    # the guard above prevents this script from invoking its split-creation path.
    # ``df`` gains a ``split`` column; ``splits`` is the review-level table
    # (one row per review with its split).
    df, splits = attach_splits(df, split_path)

    # --- Overall label distribution -----------------------------------------
    # Reindexing fixes the row order to DESTINATION_LABELS and shows labels
    # with no clauses as 0. Gold values outside DESTINATION_LABELS are not
    # included in these counts.
    label_counts = (
        df["destination_label"]
        .value_counts()  # clauses per label, largest first
        .reindex(
            DESTINATION_LABELS, fill_value=0
        )  # canonical order, zeros for absent labels
        .astype(int)  # reindex can produce floats; restore ints
    )
    # Save as a two-column table: destination_label, n.
    label_counts.rename_axis("destination_label").reset_index(name="n").to_csv(
        out / "tables" / "destination_label_counts.csv", index=False
    )

    # --- Label distribution per split ---------------------------------------
    # Rows = splits, columns = destination labels, cells = clause counts.
    by_split = pd.crosstab(df["split"], df["destination_label"])
    # Put splits in a fixed order (keeping only those present) and add any
    # label missing from the crosstab as an all-zero column.
    by_split = by_split.reindex(
        index=[
            split
            for split in ("train", "validation", "test")
            if split in by_split.index
        ],
        columns=DESTINATION_LABELS,
        fill_value=0,
    )
    # Index (split names) is kept so each row stays labelled in the CSV.
    by_split.to_csv(out / "tables" / "destination_label_by_split.csv")

    # --- Data-quality summary -----------------------------------------------
    # Every value is cast to a plain Python int/str so it serialises cleanly.
    quality = {
        # Dataset size.
        "n_clauses": int(len(df)),
        "n_reviews": int(df.review_id.nunique()),
        # Counts clause IDs that repeat anywhere in the file (not per review).
        # duplicated() marks every occurrence after the first.
        "duplicate_clause_ids": int(df.clause_id.duplicated().sum()),
        # Missing or whitespace-only clause text.
        "empty_clause_text": int(df.clause_text.fillna("").str.strip().eq("").sum()),
        # Overall clause count per label (same numbers as the CSV above).
        "labels": {label: int(label_counts[label]) for label in DESTINATION_LABELS},
        # Counted from the review-level table, so these are reviews, not clauses.
        # Splits with no reviews are reported as 0 rather than omitted.
        "reviews_by_split": {
            str(split): int(count)
            for split, count in splits["split"]
            .value_counts()
            .reindex(["train", "validation", "test"], fill_value=0)
            .items()
        },
        # Nested {split: {label: clause count}}, same numbers as the by-split CSV.
        "label_counts_by_split": {
            split: {
                label: int(by_split.loc[split, label]) for label in DESTINATION_LABELS
            }
            for split in by_split.index
        },
        # Class-imbalance safeguard: split/class cells too small for stable per-class scores.
        "low_support_threshold": int(a.min_support),
        # For each split, the labels whose clause count is below the threshold.
        # An empty list means every label has enough support in that split.
        "low_support_by_split": {
            split: [
                label
                for label in DESTINATION_LABELS
                if by_split.loc[split, label] < a.min_support
            ]
            for split in by_split.index
        },
    }
    # dtype=object keeps the nested dicts/lists intact when written as JSON.
    pd.Series(quality, dtype=object).to_json(
        out / "tables" / "data_quality_summary.json", indent=2
    )

    # --- Figure -------------------------------------------------------------
    # Bar chart of the overall gold label distribution (all splits combined).
    ax = label_counts.plot(kind="bar", figsize=(8, 5))  # size in inches
    ax.set_title("Annotated destination-label distribution")
    ax.set_xlabel("Gold destination label")
    ax.set_ylabel("Clause count")
    ax.tick_params(axis="x", rotation=25)  # angle labels so long names don't overlap
    ax.figure.tight_layout()  # stop rotated labels being clipped at the edge
    ax.figure.savefig(out / "figures" / "destination_label_distribution.png", dpi=180)
    plt.close(ax.figure)  # release the figure's memory
    # Short confirmation for the terminal.
    print(
        f"Wrote data summary and label distribution for {len(df)} clauses / "
        f"{df.review_id.nunique()} reviews"
    )


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
