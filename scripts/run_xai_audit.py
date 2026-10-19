"""Generate BERT explanation, faithfulness, and stability audits on held-out clauses."""

# Three checks are run on a random sample of test clauses:
#   1. Integrated Gradients (IG): token-level attributions explaining each
#      prediction. Clauses whose IG approximation looks unreliable
#      (convergence flag) are rerun with more integration steps.
#   2. Perturbation faithfulness: delete the most-attributed tokens and
#      measure the drop in predicted-class probability, compared with deleting
#      random tokens (matched) and low-attribution tokens. If the attributions
#      are faithful, removing top tokens should hurt the prediction more.
#   3. Stability: make small punctuation edits and check whether the
#      predicted label stays the same.
#
# Outputs (under --output-dir, default outputs/explanations):
#   integrated_gradients_tokens.csv               IG attributions per token
#   integrated_gradients_reliability_reruns.csv   higher-step reruns (only if any)
#   faithfulness_perturbation.csv                 one row per clause
#   stability_punctuation_probe.csv               one row per clause
#   xai_audit_summary.json                        headline numbers and caveats

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
import json  # writing the summary file
from pathlib import Path  # output paths
import numpy as np
import pandas as pd

# require_existing_splits: stops with an error if the split file is missing,
# so this script can never create a new split.
from cruise_xai.data import load_annotations, attach_splits, require_existing_splits

# integrated_gradients_for_row: token attributions for one clause.
# perturbation_faithfulness:    deletion test for one clause, given its attributions.
# stability_test:               punctuation-edit test for one clause.
from cruise_xai.explainability import (
    integrated_gradients_for_row,
    perturbation_faithfulness,
    stability_test,
)

# Loads the tokenizer, model, and device from a saved model directory.
from cruise_xai.bert_model import load_bert


def _review_bootstrap_difference(
    frame: pd.DataFrame, n_boot: int = 2000, seed: int = 42
) -> dict:
    """Review-cluster bootstrap CI for paired top-positive minus random probability drop.

    For each eligible clause, the paired difference is (probability drop after
    deleting the top-attributed tokens) minus (drop after deleting the same
    number of random tokens). A positive mean means the attributions point at
    tokens that matter more than chance. Reviews, not clauses, are resampled
    because clauses from the same review are correlated.

    Returns a JSON-ready dict; ``status`` is "unavailable" (with a reason)
    when there is nothing to compute.
    """
    needed = {"review_id", "top_positive_probability_drop", "random_probability_drop"}
    eligible = frame.copy()
    if not needed.issubset(eligible.columns):
        return {"status": "unavailable", "reason": "required columns missing"}
    # Keep clauses that had top-positive tokens to delete and valid values for
    # both drops. Missing availability flags count as "not available".
    eligible = eligible[
        eligible["top_positive_tokens_available"].fillna(False).astype(bool)
        & eligible["top_positive_probability_drop"].notna()
        & eligible["random_probability_drop"].notna()
    ].copy()
    if eligible.empty:
        return {
            "status": "unavailable",
            "reason": "no eligible matched clauses",
            "n_clauses": 0,
            "n_reviews": 0,
        }
    eligible["paired_difference"] = (
        eligible["top_positive_probability_drop"] - eligible["random_probability_drop"]
    )
    # Point estimate: mean paired difference over all eligible clauses.
    point = float(eligible["paired_difference"].mean())
    # Pre-group the differences by review so each replicate is a fast lookup.
    groups = {
        str(k): g["paired_difference"].to_numpy(dtype=float)
        for k, g in eligible.groupby("review_id", dropna=False)
    }
    review_ids = np.array(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        # Draw as many reviews as there are, with replacement, and pool all
        # their clauses. The replicate mean is clause-weighted, matching the
        # point estimate.
        sampled = rng.choice(review_ids, size=len(review_ids), replace=True)
        values = [groups[rid] for rid in sampled]
        estimates[b] = np.concatenate(values).mean()
    # 95% percentile interval.
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {
        "status": "ok",
        "estimand": "mean(top_positive_probability_drop - random_probability_drop)",
        "point_estimate": point,
        "ci_95_percentile": [float(lo), float(hi)],
        "n_clauses": int(len(eligible)),
        "n_reviews": int(len(groups)),
        "bootstrap_replicates": int(n_boot),
        "note": "Reviews are resampled as clusters; clauses within a sampled review stay together.",
    }


def main():
    """Run the IG, faithfulness, and stability audits and write all outputs."""
    # --- Command-line arguments ---------------------------------------------
    p = argparse.ArgumentParser(
        description="Run IG, perturbation faithfulness, and punctuation stability probes"
    )
    p.add_argument(
        "--data", default="data/processed/annotated_reviews.csv"
    )  # annotated clauses
    p.add_argument(
        "--splits", default="data/processed/review_splits.csv"
    )  # existing split file
    p.add_argument(
        "--model-dir", default="outputs/bert/model"
    )  # fine-tuned BERT to explain
    p.add_argument("--output-dir", default="outputs/explanations")
    p.add_argument(
        "--n-examples", type=int, default=24
    )  # number of test clauses to audit
    p.add_argument(
        "--ig-steps", type=int, default=32
    )  # IG integration steps (first pass)
    p.add_argument(
        "--reliability-steps", type=int, default=64
    )  # IG steps for reruns of flagged clauses
    p.add_argument(
        "--random-draws", type=int, default=5
    )  # random deletions averaged per clause
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    # Seeds the clause sample, the random deletions, and the bootstrap.
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    # p.error prints the usage message and exits instead of raising a traceback.
    if (
        a.n_examples < 1
        or a.ig_steps < 1
        or a.reliability_steps < 1
        or a.random_draws < 1
        or a.bootstrap_replicates < 1
    ):
        p.error(
            "example count, IG steps, random draws, and bootstrap replicates must be positive"
        )

    # --- Select clauses -----------------------------------------------------
    df = load_annotations(a.data)
    df, _ = attach_splits(df, require_existing_splits(a.splits))
    # Shuffle the test clauses reproducibly and take the first N, giving a
    # random sample (all test clauses if N exceeds the test size).
    test = df[df.split == "test"].sample(frac=1, random_state=a.seed).head(a.n_examples)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    components = load_bert(
        a.model_dir
    )  # Load tokenizer/model/device once for this audit.
    ig_rows, rerun_rows, faith_rows, stability_rows = [], [], [], []

    # --- Per-clause audits --------------------------------------------------
    for _, row in test.iterrows():
        # 1. IG: returns one record per token, each with a convergence flag.
        attrs = integrated_gradients_for_row(
            row, a.model_dir, n_steps=a.ig_steps, components=components
        )
        ig_rows.extend(attrs)
        # The flag is per token but describes the clause's IG run; collect the
        # distinct values. Missing flags count as "ok".
        clause_flags = {r.get("ig_convergence_flag", "ok") for r in attrs}
        selected_attrs = attrs
        if any(flag != "ok" for flag in clause_flags):
            # More steps give a more accurate IG approximation.
            rerun = integrated_gradients_for_row(
                row, a.model_dir, n_steps=a.reliability_steps, components=components
            )
            for rec in rerun:
                rec["ig_rerun_n_steps"] = a.reliability_steps
            rerun_rows.extend(rerun)
            # Use the higher-step result for the perturbation probe, while retaining both outputs.
            selected_attrs = rerun
        # 2. Faithfulness: deletion test driven by the selected attributions.
        faith_rows.append(
            perturbation_faithfulness(
                row,
                a.model_dir,
                selected_attrs,
                random_state=a.seed,
                random_draws=a.random_draws,
                components=components,
            )
        )
        # 3. Stability: punctuation edits; independent of the attributions.
        stability_rows.append(stability_test(row, a.model_dir, components=components))

    # --- Save detailed outputs ----------------------------------------------
    ig_df = pd.DataFrame(ig_rows)
    faith_df = pd.DataFrame(faith_rows)
    stability_df = pd.DataFrame(stability_rows)
    ig_df.to_csv(out / "integrated_gradients_tokens.csv", index=False)
    # The rerun file is only written when at least one clause was rerun.
    if rerun_rows:
        pd.DataFrame(rerun_rows).to_csv(
            out / "integrated_gradients_reliability_reruns.csv", index=False
        )
    faith_df.to_csv(out / "faithfulness_perturbation.csv", index=False)
    stability_df.to_csv(out / "stability_punctuation_probe.csv", index=False)

    # --- IG reliability summary ---------------------------------------------
    # Clause-level reliability count (not token-level, since each clause has many tokens).
    # A clause counts as flagged if any of its token records is not "ok".
    clause_flagged = 0
    for _, group in ig_df.groupby("clause_id", dropna=False):
        if (
            group.get("ig_convergence_flag", pd.Series(dtype=str))
            .fillna("ok")
            .ne("ok")
            .any()
        ):
            clause_flagged += 1
    # IDs of the flagged clauses, i.e. the ones that were rerun.
    flagged_ids = (
        set(
            ig_df.loc[
                ig_df["ig_convergence_flag"].fillna("ok").ne("ok"), "clause_id"
            ].astype(str)
        )
        if not ig_df.empty
        else set()
    )
    # How many rerun clauses were still flagged at the higher step count.
    rerun_df = pd.DataFrame(rerun_rows)
    if not rerun_df.empty:
        rerun_clause_flags = rerun_df.groupby("clause_id")["ig_convergence_flag"].apply(
            lambda x: bool(x.fillna("ok").ne("ok").any())
        )
        rerun_still_flagged = int(rerun_clause_flags.sum())
    else:
        rerun_still_flagged = 0

    # --- Stability summary --------------------------------------------------
    # Only clauses where a meaningful edit could be made are eligible. The
    # .get(...) fallbacks keep this working if a column is absent (all
    # clauses then count as ineligible).
    eligible_stability = stability_df[
        stability_df.get(
            "eligible_for_stability_aggregate",
            pd.Series(False, index=stability_df.index),
        )
        .fillna(False)
        .astype(bool)
    ]
    stability_summary = {
        "eligible_n": int(len(eligible_stability)),
        "total_n": int(len(stability_df)),
        # Eligible clauses whose predicted label did not change after the edit.
        "agreement_n": int(
            eligible_stability.get("prediction_agreement", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "agreement_rate": (
            float(eligible_stability["prediction_agreement"].astype(bool).mean())
            if len(eligible_stability)
            else None
        ),
        "excluded_n": int(len(stability_df) - len(eligible_stability)),
        # Count of each reason a clause was excluded, e.g. {"no punctuation to edit": 3}.
        "exclusion_reasons": (
            stability_df.loc[
                ~stability_df.index.isin(eligible_stability.index),
                "stability_eligibility_reason",
            ]
            .value_counts(dropna=False)
            .to_dict()
            if "stability_eligibility_reason" in stability_df
            else {}
        ),
        "interpretation": "Prediction agreement is summarized only over eligible edited inputs; it is not attribution stability.",
    }

    # --- Faithfulness summary -----------------------------------------------
    # Means start as None and are filled in below only if eligible clauses exist.
    faith_summary = {
        "eligible_top_positive_n": int(
            faith_df.get("top_positive_tokens_available", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "total_n": int(len(faith_df)),
        "mean_top_positive_probability_drop": None,
        "mean_random_probability_drop_matched": None,
        "mean_low_attribution_probability_drop": None,
        "paired_review_bootstrap_top_minus_random": _review_bootstrap_difference(
            faith_df, a.bootstrap_replicates, a.seed
        ),
        "note": "Primary faithfulness comparison uses matched eligible clauses; low-attribution deletion is secondary.",
    }
    if not faith_df.empty:
        # All three means use the same eligible clauses, so they are comparable.
        eligible_faith = faith_df[
            faith_df["top_positive_tokens_available"].fillna(False).astype(bool)
        ].copy()
        if not eligible_faith.empty:
            faith_summary["mean_top_positive_probability_drop"] = float(
                eligible_faith["top_positive_probability_drop"].mean()
            )
            faith_summary["mean_random_probability_drop_matched"] = float(
                eligible_faith["random_probability_drop"].mean()
            )
            faith_summary["mean_low_attribution_probability_drop"] = float(
                eligible_faith["low_attribution_probability_drop"].mean()
            )

    # --- Overall summary JSON -----------------------------------------------
    summary = {
        "test_clauses_evaluated": int(len(test)),
        "ig": {
            "initial_steps": int(a.ig_steps),
            "flagged_clauses_initial": int(clause_flagged),
            "flagged_clause_fraction_initial": (
                float(clause_flagged / len(test)) if len(test) else None
            ),
            "flagged_clauses_rerun_at_higher_steps": int(len(flagged_ids)),
            "rerun_steps": int(a.reliability_steps),
            "still_flagged_after_rerun": int(rerun_still_flagged),
            "rerun_output_file": (
                "integrated_gradients_reliability_reruns.csv" if rerun_rows else None
            ),
            "note": "Initial attributions are preserved; flagged clauses are separately rerun at higher n_steps. Faithfulness uses rerun attributions for those clauses.",
        },
        "faithfulness": faith_summary,
        "stability": stability_summary,
    }
    # default=str converts any value json can't handle (e.g. NumPy types) to text.
    (out / "xai_audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote XAI audit for {len(test)} held-out clauses to {out}. Inspect xai_audit_summary.json and manually review edits before interpretation."
    )


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
