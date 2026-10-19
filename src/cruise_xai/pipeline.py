"""Run rule-based and contextual BERT inference, then compare held-out predictions.

Pipeline overview
-----------------
1. ``run_rules``        - apply the forward-window rule baseline to every clause,
                          evaluate on the held-out test reviews, and (if needed)
                          create the review-level split file.
2. ``evaluate_bert``    - run a trained BERT model over the same clauses, reusing
                          the split file from step 1 so both methods are scored
                          on identical test reviews.
3. ``make_comparison``  - align the two prediction files clause by clause,
                          categorise agreement/disagreement, run a paired
                          review-level bootstrap, and write summary tables/plots.

Steps must be run in this order: step 2 requires the split file written by
step 1, and step 3 requires the prediction CSVs written by steps 1 and 2.
"""

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from .data import load_annotations, attach_splits, PREDICTION_LABELS
from .rules import ForwardWindowAttributor
from .metrics import evaluate_predictions, save_confusion_matrix, save_metrics
from .bootstrap import paired_review_bootstrap, save_bootstrap_results


def _warn_low_support(metrics: dict, method: str) -> None:
    """Print a visible warning when held-out classes have too few gold examples."""
    # ``evaluate_predictions`` flags classes whose gold count in the test set is
    # below ``min_support_threshold``; per-class precision/recall/F1 for those
    # classes rest on a handful of examples and should not be over-interpreted.
    if metrics.get("low_support_classes"):
        support = metrics["gold_class_support"]
        details = ", ".join(
            f"{label} (n={support[label]})" for label in metrics["low_support_classes"]
        )
        print(
            f"WARNING [{method}]: low-support test classes (< {metrics['min_support_threshold']}): {details}. "
            "Treat their per-class scores as highly uncertain."
        )


def run_rules(
    data_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    window_size: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the rule-based baseline on all clauses and evaluate it on the test split.

    Args:
        data_path: Annotated clause data (one row per clause, with gold
            ``destination_label``).
        split_path: Review-level split file. If it does not exist yet,
            ``attach_splits`` creates it (using ``seed``); later runs, including
            BERT evaluation, reuse it so every method sees the same split.
        output_dir: Directory for predictions, metrics, and figures.
        window_size: How far (in clauses) the rule looks ahead for a port
            mention when attributing a clause.
        seed: Random seed used only when a new split has to be generated.

    Returns:
        All clauses (every split) merged with the rule predictions and their
        decision traces.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load gold annotations and tag each clause with train/dev/test based on
    # its review, so all clauses from one review stay in the same split.
    df = load_annotations(data_path)
    df, _ = attach_splits(df, split_path, seed=seed)

    # The rule predictor needs no training, so it is applied to every clause.
    # ``predict`` returns one trace dict per clause: the prediction plus the
    # evidence behind it (detected ports, context distance, abstention, ...).
    predictor = ForwardWindowAttributor(window_size=window_size)
    traces = pd.DataFrame(predictor.predict(df))

    # Drop columns already present in ``df`` (keys and clause text) so the merge
    # does not create duplicated ``_x``/``_y`` columns, then re-attach the two
    # key columns so the trace rows can be joined back. ``validate`` guarantees
    # exactly one trace per clause.
    trace_details = traces.drop(
        columns=["review_id", "clause_id", "clause_text"], errors="ignore"
    )
    merged = df.merge(
        trace_details.assign(
            clause_id=traces["clause_id"], review_id=traces["review_id"]
        ),
        on=["clause_id", "review_id"],
        how="left",
        validate="one_to_one",
    )
    merged.to_csv(output_dir / "rule_predictions_all.csv", index=False)

    # Score only on held-out reviews.
    test = merged[merged["split"] == "test"].copy()
    metrics, per_class, cm = evaluate_predictions(
        test.destination_label, test.predicted_label
    )

    # Record run settings alongside the scores so results are traceable.
    metrics["method"] = "forward_window_rule_based"
    metrics["window_size"] = window_size
    metrics["split_file"] = str(Path(split_path))
    save_metrics(metrics, per_class, output_dir / "rule_metrics")
    _warn_low_support(metrics, "rules")

    # Raw-count confusion matrix: shows where errors fall in absolute terms.
    save_confusion_matrix(
        cm,
        PREDICTION_LABELS,
        output_dir / "figures" / "rule_confusion_matrix.png",
        "Rule-based attribution: held-out reviews",
    )
    # Row-normalized view: each gold class sums to 100%, so rare classes are as readable as common ones.
    save_confusion_matrix(
        cm,
        PREDICTION_LABELS,
        output_dir / "figures" / "rule_confusion_matrix_normalized.png",
        "Rule-based attribution: held-out reviews (row-normalized)",
        normalize=True,
    )

    # Per-clause test cases with the full decision trace, for error analysis.
    test[
        [
            "review_id",
            "clause_id",
            "clause_text",
            "destination_label",
            "predicted_label",
            "attribution_method",
            "decision_reason",
            "detected_ports",
            "active_port_before",
            "active_port_after",
            "context_distance",
            "topic_transition",
            "abstained",
        ]
    ].to_csv(output_dir / "rule_test_cases.csv", index=False)
    return merged


def evaluate_bert(
    data_path: str | Path,
    split_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a trained contextual BERT model on all clauses and evaluate on the test split.

    Unlike ``run_rules``, this function refuses to create a new split: the split
    file must already exist so BERT is scored on exactly the same held-out
    reviews as the rule baseline.

    Returns:
        All clauses merged with ``bert_predicted_label`` and ``bert_confidence``.
    """
    # Imported lazily so the rule pipeline can run without torch/transformers installed.
    from .bert_model import predict_bert

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_annotations(data_path)

    # Guard against silently generating a different split than the rules used.
    split_file = Path(split_path)
    if not split_file.exists():
        raise FileNotFoundError(
            f"Expected an existing review-level split file at {split_file}. Run the rule pipeline first so both methods use identical splits."
        )
    df, _ = attach_splits(df, split_file, seed=seed)

    # Predict for every clause (all splits) and join on the clause key;
    # ``validate`` ensures one prediction per clause.
    preds = predict_bert(df, model_dir)
    merged = df.merge(
        preds, on=["review_id", "clause_id"], how="left", validate="one_to_one"
    )
    merged.to_csv(output_dir / "bert_predictions_all.csv", index=False)

    # Score only on held-out reviews.
    test = merged[merged.split == "test"].copy()
    metrics, per_class, cm = evaluate_predictions(
        test.destination_label, test.bert_predicted_label
    )

    # Record run settings alongside the scores so results are traceable.
    metrics["method"] = "contextual_bert"
    metrics["model_dir"] = str(Path(model_dir))
    metrics["split_file"] = str(Path(split_path))
    save_metrics(metrics, per_class, output_dir / "bert_metrics")
    _warn_low_support(metrics, "BERT")

    # Raw-count confusion matrix.
    save_confusion_matrix(
        cm,
        PREDICTION_LABELS,
        output_dir / "figures" / "bert_confusion_matrix.png",
        "Contextual BERT: held-out reviews",
    )
    # Row-normalized view: each gold class sums to 100%, so rare classes are as readable as common ones.
    save_confusion_matrix(
        cm,
        PREDICTION_LABELS,
        output_dir / "figures" / "bert_confusion_matrix_normalized.png",
        "Contextual BERT: held-out reviews (row-normalized)",
        normalize=True,
    )

    # Per-clause test cases for error analysis.
    test.to_csv(output_dir / "bert_test_cases.csv", index=False)
    return merged


def make_comparison(output_dir: str | Path) -> pd.DataFrame:
    """Align and compare saved rule/BERT predictions for identical clauses.

    Reads ``rule_predictions_all.csv`` and ``bert_predictions_all.csv`` from
    ``output_dir``, checks that they describe exactly the same clauses with the
    same gold labels, splits and text, then writes:

    - ``method_comparison_all.csv`` / ``method_comparison_test.csv``:
      side-by-side predictions with a ``comparison_group`` per clause.
    - ``tables/comparison_group_counts.csv``: how many test clauses fall in
      each agreement/disagreement group.
    - ``tables/paired_review_bootstrap*``: paired bootstrap over test reviews.
    - ``tables/method_metrics_comparison.csv`` and
      ``figures/method_metrics_comparison.png``: headline metrics side by side.

    Returns:
        The aligned comparison table for all clauses.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rule_path = output_dir / "rule_predictions_all.csv"
    bert_path = output_dir / "bert_predictions_all.csv"
    if not rule_path.exists() or not bert_path.exists():
        raise FileNotFoundError(
            "Run the rule baseline and train/evaluate BERT before making the comparison"
        )

    # Read IDs as strings so values like "007" are not turned into integers,
    # which would break key matching between the two files.
    rule = pd.read_csv(rule_path, dtype={"review_id": str, "clause_id": str})
    bert = pd.read_csv(bert_path, dtype={"review_id": str, "clause_id": str})

    # Keep the shared source columns plus each method's prediction columns,
    # renaming rule outputs so they are clearly labelled after the merge.
    cols = [
        "review_id",
        "clause_id",
        "destination_label",
        "split",
        "clause_text",
        "predicted_label",
        "attribution_method",
        "decision_reason",
        "abstained",
    ]
    r = rule[cols].rename(
        columns={
            "predicted_label": "rule_prediction",
            "attribution_method": "rule_method",
            "decision_reason": "rule_reason",
            "abstained": "rule_abstained",
        }
    )
    bcols = ["review_id", "clause_id", "bert_predicted_label", "bert_confidence"]
    b = bert[
        [
            "review_id",
            "clause_id",
            "destination_label",
            "split",
            "clause_text",
            *bcols[2:],
        ]
    ].copy()

    # Sanity check 1: both files must cover exactly the same set of clauses.
    keys = ["review_id", "clause_id"]
    rule_keys, bert_keys = set(map(tuple, r[keys].astype(str).to_numpy())), set(
        map(tuple, b[keys].astype(str).to_numpy())
    )
    if rule_keys != bert_keys:
        raise ValueError(
            f"Rule/BERT clause sets differ: {len(rule_keys - bert_keys)} rule-only, {len(bert_keys - rule_keys)} BERT-only clauses"
        )

    # Outer merge with an indicator so any unmatched row would be caught, and
    # ``validate`` to catch duplicate keys in either file.
    aligned = r.merge(
        b,
        on=keys,
        how="outer",
        suffixes=("_rule", "_bert"),
        validate="one_to_one",
        indicator=True,
    )
    if not (aligned["_merge"] == "both").all():
        raise ValueError("Rule/BERT predictions could not be fully aligned")

    # Sanity check 2: the source data each method was run on must be identical
    # (same gold label, same split assignment, same text) for every clause.
    for col in ["destination_label", "split", "clause_text"]:
        if not (
            aligned[f"{col}_rule"].fillna("").astype(str)
            == aligned[f"{col}_bert"].fillna("").astype(str)
        ).all():
            raise ValueError(f"Rule/BERT source data disagree on {col}")

    # Having verified the duplicated columns match, keep a single copy of each.
    out = aligned.drop(
        columns=["_merge", "destination_label_bert", "split_bert", "clause_text_bert"]
    ).rename(
        columns={
            "destination_label_rule": "destination_label",
            "split_rule": "split",
            "clause_text_rule": "clause_text",
            "bert_predicted_label": "bert_prediction",
        }
    )
    out["rule_correct"] = out.rule_prediction == out.destination_label
    out["bert_correct"] = out.bert_prediction == out.destination_label

    # Assign each clause to exactly one group (conditions are checked in order):
    #   both_correct                - both methods right
    #   rule_only_correct           - rule right, BERT wrong
    #   bert_correct_rule_abstained - BERT right, rule declined to predict
    #   bert_correct_rule_wrong     - BERT right, rule made a wrong prediction
    #   both_wrong_rule_abstained   - both wrong, and the rule abstained
    #   both_wrong_rule_incorrect   - both wrong, and the rule made a wrong prediction
    out["comparison_group"] = out.apply(
        lambda x: (
            "both_correct"
            if x.rule_correct and x.bert_correct
            else (
                "rule_only_correct"
                if x.rule_correct
                else (
                    "bert_correct_rule_abstained"
                    if x.bert_correct and bool(x.rule_abstained)
                    else (
                        "bert_correct_rule_wrong"
                        if x.bert_correct
                        else (
                            "both_wrong_rule_abstained"
                            if bool(x.rule_abstained)
                            else "both_wrong_rule_incorrect"
                        )
                    )
                )
            )
        ),
        axis=1,
    )
    out.to_csv(output_dir / "method_comparison_all.csv", index=False)

    # All remaining analysis uses held-out clauses only.
    test = out[out.split == "test"].copy()
    test.to_csv(output_dir / "method_comparison_test.csv", index=False)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Counts per agreement/disagreement group on the test set.
    group_counts = (
        test.groupby("comparison_group", dropna=False).size().rename("n").reset_index()
    )
    group_counts.to_csv(tables_dir / "comparison_group_counts.csv", index=False)

    # Paired bootstrap that resamples whole reviews (not individual clauses),
    # because clauses within a review are correlated. Both methods are scored
    # on the same resample each time, giving confidence intervals for the
    # metric differences. Fixed seed for reproducibility.
    bootstrap_table, bootstrap_summary = paired_review_bootstrap(
        test, n_resamples=2000, random_state=42
    )
    save_bootstrap_results(
        bootstrap_table, bootstrap_summary, tables_dir / "paired_review_bootstrap"
    )

    # A compact comparison summary is written only after both methods have run.
    # Pull the headline metrics each method saved, skipping any missing file.
    metric_rows = []
    for method, filename in [
        ("Rule-based", "rule_metrics.json"),
        ("Contextual BERT", "bert_metrics.json"),
    ]:
        metric_path = output_dir / filename
        if metric_path.exists():
            with open(metric_path, encoding="utf-8") as handle:
                values = json.load(handle)
            # ``coverage`` / ``selective_accuracy`` describe the effect of
            # abstention (share of clauses predicted, and accuracy on those);
            # run settings are carried along for traceability.
            metric_rows.append(
                {
                    "method": method,
                    "accuracy": values.get("accuracy"),
                    "balanced_accuracy": values.get("balanced_accuracy"),
                    "macro_f1": values.get("macro_f1"),
                    "weighted_f1": values.get("weighted_f1"),
                    "coverage": values.get("coverage"),
                    "selective_accuracy": values.get("selective_accuracy"),
                    "window_size": values.get("window_size"),
                    "model_dir": values.get("model_dir"),
                    "split_file": values.get("split_file"),
                    "n": values.get("n"),
                    "low_support_classes": "; ".join(
                        values.get("low_support_classes", [])
                    ),
                }
            )
    if metric_rows:
        summary = pd.DataFrame(metric_rows)
        summary.to_csv(tables_dir / "method_metrics_comparison.csv", index=False)

        # Grouped bar chart of the class-balance-aware metrics. matplotlib is
        # imported here so the module loads without it when no plot is needed.
        import matplotlib.pyplot as plt

        plot_metrics = [
            c
            for c in ["macro_f1", "weighted_f1", "balanced_accuracy"]
            if c in summary and summary[c].notna().any()
        ]
        if plot_metrics:
            ax = summary.set_index("method")[plot_metrics].plot(
                kind="bar", figsize=(8, 5), ylim=(0, 1)
            )
            ax.set_ylabel("Score")
            ax.set_title("Held-out destination attribution metrics")
            ax.legend(loc="lower right")
            ax.tick_params(axis="x", rotation=0)
            ax.figure.tight_layout()
            ax.figure.savefig(figures_dir / "method_metrics_comparison.png", dpi=180)
            plt.close(ax.figure)
    return out
