"""Tests for the separate VADER sentiment layer. They run without vaderSentiment installed."""

# They run without the package because every test uses FakeVader (below)
# instead of the real analyzer, so load_vader() is never called.
import json

import numpy as np
import pandas as pd
import pytest

from cruise_xai.sentiment import (
    _net_sentiment_by_destination,  # fast NumPy profiles used inside the bootstrap
    add_vader_columns,
    all_destination_profiles,
    common_destinations,
    default_bootstrap_statistics,
    destination_sentiment_profile,
    evaluate_sentiment,
    joint_accuracy,
    profile_errors,
    review_bootstrap,
    save_json,
    vader_label,
)


class FakeVader:
    """Stand-in for VADER: scores are looked up from the text itself.

    Each clause text is a key in TABLE, so the scores, and therefore the
    VADER labels, are known exactly:
        "great"           -> Positive  (compound  0.8)
        "awful"           -> Negative  (compound -0.7)
        "fine"            -> Neutral   (compound  0.0)
        "great but awful" -> Positive by default (compound 0.1), or Mixed when
                             mixed_min <= 0.35 (pos and neg are both 0.35)
    """

    TABLE = {
        "great": {"compound": 0.8, "pos": 0.6, "neu": 0.4, "neg": 0.0},
        "awful": {"compound": -0.7, "pos": 0.0, "neu": 0.4, "neg": 0.6},
        "fine": {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0},
        "great but awful": {"compound": 0.1, "pos": 0.35, "neu": 0.3, "neg": 0.35},
    }

    def polarity_scores(self, text):
        # Same method name and return shape as the real SentimentIntensityAnalyzer.
        return self.TABLE[text]


def _frame():
    """Six scored clauses across three reviews, with known right and wrong answers.

    Columns per clause: text, gold destination, rule prediction, BERT
    prediction, gold sentiment. Deliberate differences from gold:
      - r1-1: rule abstains ("Unclear"); gold is Roatan.
      - r2-0: BERT predicts Roatan; gold is Cozumel.
      - r3-1: gold sentiment is Mixed, but VADER (no mixed_min) says Positive.
    Everything else is correct.
    """
    rows = []
    spec = {
        #      text      gold        rule        BERT        gold sentiment
        "r1": [
            ("great", "Roatan", "Roatan", "Roatan", "Positive"),
            ("awful", "Roatan", "Unclear", "Roatan", "Negative"),
        ],
        "r2": [
            ("fine", "Cozumel", "Cozumel", "Roatan", "Neutral"),
            ("great", "Cozumel", "Cozumel", "Cozumel", "Positive"),
        ],
        "r3": [
            ("awful", "Costa Maya", "Costa Maya", "Costa Maya", "Negative"),
            (
                "great but awful",
                "General Cruise",
                "General Cruise",
                "General Cruise",
                "Mixed",
            ),
        ],
    }
    for review, clauses in spec.items():
        for i, (text, gold, rule, bert, sentiment) in enumerate(clauses):
            rows.append(
                {
                    "review_id": review,
                    "clause_id": f"{review}-{i}",
                    "clause_text": text,
                    "destination_label": gold,
                    "rule_prediction": rule,
                    "bert_prediction": bert,
                    "sentiment_label": sentiment,
                }
            )
    # Add vader_compound/pos/neu/neg and vader_label using the fake analyzer.
    return add_vader_columns(pd.DataFrame(rows), analyzer=FakeVader())


# --- Mapping VADER scores to labels ------------------------------------------


def test_vader_label_uses_standard_cutoffs_and_optional_mixed():
    # The cut-offs are inclusive: exactly +0.05 is Positive, exactly -0.05 Negative.
    assert vader_label(0.05) == "Positive"
    assert vader_label(-0.05) == "Negative"
    # Anything strictly between them is Neutral.
    assert vader_label(0.04) == "Neutral"
    assert vader_label(0.1, pos=0.35, neg=0.35) == "Positive"  # Mixed is off by default
    # With mixed_min set, substantial positive AND negative content gives Mixed,
    # even though the compound score alone would say Positive.
    assert vader_label(0.1, pos=0.35, neg=0.35, mixed_min=0.3) == "Mixed"
    # Thresholds in the wrong order are rejected.
    with pytest.raises(ValueError):
        vader_label(0.0, positive_threshold=-0.1, negative_threshold=0.1)


def test_add_vader_columns_scores_every_clause():
    frame = _frame()
    # One label per clause, in row order. The last clause ("great but awful")
    # is Positive because mixed_min is off.
    assert frame["vader_label"].tolist() == [
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
        "Positive",
    ]
    # All four raw score columns are added (<= means "is a subset of").
    assert {"vader_compound", "vader_pos", "vader_neu", "vader_neg"} <= set(
        frame.columns
    )


# --- Sentiment metrics -------------------------------------------------------


def test_evaluate_sentiment_uses_fixed_four_class_taxonomy_and_flags_low_support():
    # Four clauses, one per class; only the Mixed clause is wrong (predicted Positive).
    metrics, per_class, cm = evaluate_sentiment(
        ["Positive", "Negative", "Neutral", "Mixed"],
        ["Positive", "Negative", "Neutral", "Positive"],
        min_support=2,
    )
    # All four sentiment classes are always included.
    assert cm.shape == (4, 4)
    assert metrics["accuracy"] == pytest.approx(0.75)
    # Mixed is never predicted, so it contributes F1 = 0 to the four-class macro average.
    assert metrics["per_class"]["Mixed"]["f1-score"] == 0
    # Dropping Mixed from the average therefore raises macro-F1.
    assert metrics["macro_f1"] < metrics["macro_f1_three_class"]
    # Each class has 1 gold example, below min_support=2, so all are flagged.
    assert set(metrics["low_support_classes"]) == {
        "Positive",
        "Negative",
        "Neutral",
        "Mixed",
    }
    assert per_class["low_support"].all()
    # Labels outside the taxonomy are rejected rather than ignored.
    with pytest.raises(ValueError, match="Invalid sentiment"):
        evaluate_sentiment(["Positive"], ["Happy"])


# --- Destination profiles ----------------------------------------------------


def test_profile_excludes_abstentions_and_computes_net_sentiment():
    # Profile from rule attributions and gold sentiment.
    profile = destination_sentiment_profile(
        _frame(), "rule_prediction", "sentiment_label", min_clauses=1
    )
    roatan = profile.set_index("destination").loc["Roatan"]
    assert roatan["n_clauses"] == 1  # the Unclear clause is not attributed to Roatan
    # The one remaining Roatan clause (r1-0) is Positive: 100% positive - 0% negative.
    assert roatan["net_sentiment"] == pytest.approx(1.0)
    # A destination with no clauses still gets a row, with NaN sentiment.
    md = profile.set_index("destination").loc["Multiple Destinations"]
    assert md["n_clauses"] == 0 and np.isnan(md["net_sentiment"])


def test_profile_errors_are_zero_when_attribution_matches_gold():
    # Sanity check: if the rule's attributions were perfect, its profile would
    # equal the gold profile and every error would be 0.
    frame = _frame()
    frame["rule_prediction"] = frame["destination_label"]
    errors = profile_errors(
        all_destination_profiles(frame, ["rule"], min_clauses=1), ["rule"]
    )
    rule_gold = errors[
        (errors.attribution_source == "rule")
        & (errors.sentiment_source == "gold_sentiment")
    ]
    # dropna() skips destinations with no clauses, where the error is undefined.
    assert rule_gold["net_sentiment_abs_error"].dropna().eq(0).all()


def test_common_destinations_require_support_under_every_method():
    frame = _frame()
    errors = profile_errors(
        all_destination_profiles(frame, ["rule", "bert"], min_clauses=2),
        ["rule", "bert"],
    )
    # With min_clauses=2, no destination has >= 2 clauses under gold, rule,
    # and BERT attribution alike:
    #   Roatan:  gold 2, rule 1 (r1-1 abstained), BERT 3
    #   Cozumel: gold 2, rule 2,                  BERT 1 (r2-0 went to Roatan)
    #   Costa Maya and General Cruise: 1 clause each.
    # So the shared set is empty.
    assert common_destinations(errors) == []
    # With min_clauses=1, Roatan has at least one clause under every source.
    errors1 = profile_errors(
        all_destination_profiles(frame, ["rule", "bert"], min_clauses=1),
        ["rule", "bert"],
    )
    assert "Roatan" in common_destinations(errors1)


def test_fast_bootstrap_profiles_match_pandas_profiles():
    # The NumPy version used inside the bootstrap must give the same numbers
    # as the pandas version used for the reported tables.
    frame = _frame()
    methods = ["rule", "bert"]
    fast = _net_sentiment_by_destination(frame, methods, min_clauses=1)
    profiles = all_destination_profiles(frame, methods, min_clauses=1)
    # Net sentiment must match for every attribution × sentiment source.
    # equal_nan=True treats NaN == NaN (destinations with no clauses).
    for (attribution, source), values in fast["net"].items():
        slow = profiles[
            (profiles.attribution_source == attribution)
            & (profiles.sentiment_source == source)
        ]
        np.testing.assert_allclose(
            values, slow["net_sentiment"].to_numpy(), equal_nan=True
        )
    # The shared-destination mask must also match common_destinations.
    # ``slow`` from the last loop iteration is reused only for its destination
    # order, which is the same in every profile.
    errors = profile_errors(profiles, methods)
    shared = [d for d, keep in zip(slow["destination"], fast["shared"]) if keep]
    assert shared == common_destinations(errors)


# --- Joint accuracy ----------------------------------------------------------


def test_joint_accuracy_counts_abstention_as_wrong_destination():
    result = joint_accuracy(_frame(), "rule")
    # 5 of 6 clauses got a real destination (r1-1 abstained)...
    assert result["coverage"] == pytest.approx(5 / 6)
    # ...and the abstention counts as wrong, so destination accuracy is also 5/6.
    assert result["destination_accuracy"] == pytest.approx(5 / 6)
    # Joint correct: r1-0, r2-0, r2-1, r3-0 (the Mixed clause is a sentiment miss).
    assert result["joint_accuracy"] == pytest.approx(4 / 6)


# --- Bootstrap ---------------------------------------------------------------


def test_review_bootstrap_is_reproducible_and_paired():
    frame = _frame()
    stats = default_bootstrap_statistics(["rule", "bert"], min_clauses=1)
    # Same seed twice should give identical tables.
    a = review_bootstrap(frame, stats, n_resamples=200, random_state=3)
    b = review_bootstrap(frame, stats, n_resamples=200, random_state=3)
    pd.testing.assert_frame_equal(a, b)
    # The difference statistic must equal the difference of the two methods'
    # estimates, i.e. both are computed on the same data.
    table = a.set_index("statistic")
    assert table.loc["bert_minus_rule_joint_accuracy", "estimate"] == pytest.approx(
        table.loc["bert_joint_accuracy", "estimate"]
        - table.loc["rule_joint_accuracy", "estimate"]
    )
    # Too few resamples are rejected.
    with pytest.raises(ValueError, match="at least 100"):
        review_bootstrap(frame, stats, n_resamples=10)


# --- Saving ------------------------------------------------------------------


def test_save_json_writes_nan_as_null(tmp_path):
    # NaN becomes null (keeping the file valid JSON), and NumPy numbers become
    # plain Python numbers. tmp_path is a fresh temporary folder from pytest.
    save_json(
        {"a": float("nan"), "b": np.float64(1.5), "c": [np.int64(2)]},
        tmp_path / "x.json",
    )
    assert json.loads((tmp_path / "x.json").read_text()) == {
        "a": None,
        "b": 1.5,
        "c": [2],
    }
