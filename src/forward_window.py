"""Compatibility entry point for the project's auditable forward-window baseline.

The implementation lives in `cruise_xai.rules` so the method can be imported,
tested, and reused by the evaluation pipeline. This file preserves the familiar
`forward_window.py` entry point for project workflows.

Usage:
    python forward_window.py                     # run with the defaults below
    python forward_window.py --window-size 3     # carry port context 3 sentences

Older code that did ``from forward_window import ForwardWindowAttributor``
keeps working, because the classes are re-exported here.
"""

# Re-export the public classes so existing imports of this module still work.
from cruise_xai.rules import ForwardWindowAttributor, RuleResult

# Limits ``from forward_window import *`` to the two re-exported names.
__all__ = ["ForwardWindowAttributor", "RuleResult"]

# Command-line entry point: only runs when the file is executed directly,
# not when it is imported.
if __name__ == "__main__":
    # Imported here so importing this module for the classes alone does not
    # pull in the pipeline (pandas, metrics, plotting, etc.).
    import argparse
    from cruise_xai.pipeline import run_rules

    parser = argparse.ArgumentParser(
        description="Run the forward-window destination-attribution baseline"
    )
    # Annotated clause-level data, with gold destination labels.
    parser.add_argument("--data", default="data/processed/annotated_reviews.csv")
    # Review-level train/dev/test split. Created on first run if missing, then
    # reused so the BERT evaluation scores on exactly the same test reviews.
    parser.add_argument("--splits", default="data/processed/review_splits.csv")
    # Where predictions, metrics, and figures are written.
    parser.add_argument("--output-dir", default="outputs")
    # Number of sentences after an explicit port mention that inherit that port.
    parser.add_argument("--window-size", type=int, default=2)
    # Only used when a new split file has to be generated.
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Arguments are passed positionally, so their order must match
    # run_rules(data_path, split_path, output_dir, window_size, seed).
    run_rules(args.data, args.splits, args.output_dir, args.window_size, args.seed)
