"""Unit tests for ``cruise_xai.metrics.evaluate_predictions``.

Checks how abstentions ("Unclear" predictions) are scored and that classes
with too few gold examples are flagged as low-support.
"""

from cruise_xai.metrics import evaluate_predictions


def test_metrics_handles_abstentions():
    # Three clauses: two predicted correctly, one abstained (gold Cozumel,
    # predicted "Unclear").
    metrics, per_class, cm = evaluate_predictions(
        ["Roatan", "Cozumel", "General Cruise"], ["Roatan", "Unclear", "General Cruise"]
    )
    assert metrics["n"] == 3
    # Coverage = share of clauses that received a real prediction (2 of 3).
    assert metrics["coverage"] == 2 / 3
    # The confusion matrix uses all prediction labels: the 5 destination
    # labels plus "Unclear", so abstentions appear as their own column.
    assert cm.shape == (6, 6)
    assert "Roatan" in per_class.label.values
    # "Unclear" is never a gold label, so it gets no per-class row; precision
    # and recall for it would be meaningless.
    assert (
        "Unclear" not in per_class.label.values
    )  # prediction-only abstention is reported via coverage


def test_metrics_flag_low_support_classes():
    # 12 Roatan and 3 Cozumel gold clauses, all predicted correctly. With
    # min_support=10, Cozumel (3) is below the threshold and Roatan (12) is not.
    metrics, per_class, _ = evaluate_predictions(
        ["Roatan"] * 12 + ["Cozumel"] * 3,
        ["Roatan"] * 12 + ["Cozumel"] * 3,
        min_support=10,
    )
    # Flagged in the summary metrics...
    assert "Cozumel" in metrics["low_support_classes"]
    assert "Roatan" not in metrics["low_support_classes"]
    # ...and in the per-class table. Perfect scores do not remove the flag,
    # since 3 examples are too few to trust a per-class score either way.
    assert bool(per_class.set_index("label").loc["Cozumel", "low_support"])
