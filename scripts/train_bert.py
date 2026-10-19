"""Fine-tune the contextual BERT destination classifier on the training split.

Trains on the train split and selects the best checkpoint on the validation
split (the test split is never used here). Requires the split file created by
prepare_data.py.

Outputs (under --output-dir, default outputs/bert):
    the saved model (by default outputs/bert/model, which evaluate_bert.py reads)
    training_metadata.json, written by train_bert and extended here with
    start/finish times and the exact command line
"""

# --- Path setup (must run before importing cruise_xai) -----------------------
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
import argparse  # command-line options
import json  # reading/updating training metadata
import shutil  # deleting a previous run with --overwrite
from datetime import datetime, timezone  # UTC timestamps for the metadata
from cruise_xai.data import load_annotations, attach_splits

# Fine-tunes the model and returns the path it was saved to.
from cruise_xai.bert_model import train_bert


def main():
    """Fine-tune BERT on the fixed split and record when and how it was trained."""
    # --- Command-line arguments ---------------------------------------------
    # Defaults match the standard project layout, so the script can be run
    # with no arguments from the project root.
    p = argparse.ArgumentParser(
        description="Fine-tune contextual BERT on review-level training split"
    )
    p.add_argument(
        "--data", default="data/processed/annotated_reviews.csv"
    )  # annotated clauses
    p.add_argument(
        "--splits", default="data/processed/review_splits.csv"
    )  # existing split file
    p.add_argument("--output-dir", default="outputs/bert")
    # Hugging Face model to fine-tune; DistilBERT is a smaller, faster BERT.
    p.add_argument("--model-name", default="distilbert-base-uncased")
    # Maximum input length in tokens; longer inputs are truncated.
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--learning-rate", type=float, default=2e-5
    )  # typical BERT fine-tuning rate
    # Training seed (initialisation, data order). It does not affect the split.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default="none",
        help="Weight the training loss by inverse class frequency (compare on validation only)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the previous training run in --output-dir before training",
    )
    a = p.parse_args()

    # Fail early rather than letting attach_splits create a new split: the
    # split must come from prepare_data.py so every method shares it.
    if not _Path(a.splits).is_file():
        raise FileNotFoundError(
            f"Split file not found: {a.splits}. Run scripts/prepare_data.py first."
        )

    # --- Optional clean-up of a previous run --------------------------------
    out = _Path(a.output_dir)
    if a.overwrite and out.exists():
        # Safety check: only delete a folder that holds a previous training run.
        # This stops a mistyped --output-dir (e.g. "outputs") from wiping
        # unrelated results.
        if (
            not (out / "training_metadata.json").exists()
            and not (out / "checkpoints").exists()
        ):
            raise SystemExit(
                f"Refusing to delete {out}: it doesn't look like a BERT training folder."
            )
        # Note: the old run is deleted before training starts, so if training
        # then fails, the previous model is gone too.
        shutil.rmtree(out)

    # --- Training -----------------------------------------------------------
    # Timestamp taken before loading data, so the duration covers the whole run.
    started = datetime.now(timezone.utc)
    df = load_annotations(a.data)
    df, _ = attach_splits(df, a.splits)
    # The full frame (all splits) is passed; train_bert trains on the train
    # split and selects the best epoch on validation.
    path = train_bert(
        df,
        out,
        model_name=a.model_name,
        max_length=a.max_length,
        epochs=a.epochs,
        batch_size=a.batch_size,
        learning_rate=a.learning_rate,
        seed=a.seed,
        overwrite=a.overwrite,
        class_weighting=a.class_weighting,
    )

    # --- Metadata -----------------------------------------------------------
    # Record when this (current) model was trained.
    # train_bert writes training_metadata.json; this adds to it rather than
    # replacing it, so the training settings it recorded are kept.
    meta_path = out / "training_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        started_at_utc=started.isoformat(timespec="seconds"),
        finished_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Exact command used, so the run can be reproduced.
        command_line=" ".join(sys.argv),
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved best validation-selected model to {path}")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
