"""Unit tests for ``cruise_xai.bootstrap`` (paired review-level bootstrap).

The bootstrap resamples whole reviews and scores the rule baseline and BERT
on the same resampled reviews, giving confidence intervals for each method
and for the BERT-minus-rule difference.
"""

import pandas as pd
import pytest

from cruise_xai.bootstrap import paired_review_bootstrap, save_bootstrap_results


def _comparison():
    """Build a tiny method-comparison table: 4 reviews × 2 clauses, all in test.

    Shaped like ``method_comparison_test.csv`` from the pipeline. By design:
      - BERT is always correct.
      - The rule baseline is correct on each review's first clause and
        abstains ("Unclear") on the second.
    So on the full data the rule has accuracy 0.5, coverage 0.5, and
    selective accuracy 1.0, while BERT scores 1.0 throughout.
    """
    rows = []
    # Gold destination labels for the two clauses of each review. Every
    # destination class appears at least once.
    for review_id, golds in {
        "r1": ["Roatan", "Costa Maya"],
        "r2": ["Cozumel", "General Cruise"],
        "r3": ["Multiple Destinations", "Roatan"],
        "r4": ["Costa Maya", "Cozumel"],
    }.items():
        for i, gold in enumerate(golds):
            rows.append(
                {
                    "review_id": review_id,
                    "clause_id": f"{review_id}-{i}",  # unique across reviews, e.g. "r1-0"
                    "destination_label": gold,
                    "rule_prediction": (
                        gold if i == 0 else "Unclear"
                    ),  # abstain on second clause
                    "bert_prediction": gold,  # always correct
                    "split": "test",
                }
            )
    return pd.DataFrame(rows)


def test_review_bootstrap_is_reproducible_and_returns_all_metrics():
    frame = _comparison()
    # Same data and same random_state twice should give identical results.
    # (n_resamples=100 keeps the test fast.)
    table1, summary1 = paired_review_bootstrap(frame, n_resamples=100, random_state=17)
    table2, summary2 = paired_review_bootstrap(frame, n_resamples=100, random_state=17)
    pd.testing.assert_frame_equal(table1, table2)
    assert summary1 == summary2
    # Every expected metric is reported. coverage and selective_accuracy
    # capture the rule baseline's abstentions.
    assert set(table1["metric"]) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
        "coverage",
        "selective_accuracy",
    }
    # Each metric is reported for each method and for the paired difference.
    assert set(table1["comparison"]) == {"rule", "bert", "bert_minus_rule"}
    # The resampling unit is the review: 4 reviews, not 8 clauses.
    assert summary1["n_reviews"] == 4


def test_review_bootstrap_rejects_non_test_rows():
    # The bootstrap is only meant for held-out results, so a single non-test
    # row should be rejected rather than silently included.
    frame = _comparison()
    frame.loc[0, "split"] = "train"
    # ``match`` checks that the error message mentions the test split.
    with pytest.raises(ValueError, match="test-split"):
        paired_review_bootstrap(frame, n_resamples=100)


def test_save_bootstrap_results_writes_csv_and_json(tmp_path):
    # tmp_path is a fresh temporary folder provided by pytest.
    table, summary = paired_review_bootstrap(
        _comparison(), n_resamples=100, random_state=4
    )
    # The path is a stem: the function adds .csv (table) and .json (summary).
    save_bootstrap_results(table, summary, tmp_path / "bootstrap")
    assert (tmp_path / "bootstrap.csv").exists()
    assert (tmp_path / "bootstrap.json").exists()
