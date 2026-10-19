"""Unit tests for the forward-window rule baseline (``cruise_xai.rules``).

Each test feeds one short review through ``predict_review`` and checks the
predicted label (and sometimes the decision trace) for each clause.
"""

import pandas as pd
from cruise_xai.rules import ForwardWindowAttributor


def frame(texts, review_id="r1"):
    """Build a one-review DataFrame with one clause per sentence.

    IDs follow the project format, e.g. sentence "r1-S001" and clause
    "r1-S001-C01". Numbers are zero-padded so string sorting keeps them in
    reading order.
    """
    rows = []
    for i, text in enumerate(texts, 1):
        rows.append(
            {
                "review_id": review_id,
                "sentence_id": f"{review_id}-S{i:03d}",
                "clause_id": f"{review_id}-S{i:03d}-C01",
                "clause_text": text,
            }
        )
    return pd.DataFrame(rows)


def test_neutral_explicit_mention_establishes_context():
    # A port mention with no opinion still sets the active port, because
    # sentiment is deliberately not used for attribution.
    out = ForwardWindowAttributor(window_size=2).predict_review(
        frame(["We stopped in Cozumel.", "The ride to town was short."])
    )
    assert out[0]["predicted_label"] == "Cozumel"
    # The second sentence names no port but is within the window, so it
    # inherits Cozumel via carry-forward (not via an explicit mention).
    assert out[1]["predicted_label"] == "Cozumel"
    assert out[1]["attribution_method"] == "bounded_forward_context"


def test_ship_topic_does_not_inherit_last_port():
    # A shipboard topic ("ship", "Wi-Fi") overrides carry-forward even though
    # Roatan is still within the window.
    out = ForwardWindowAttributor(window_size=2).predict_review(
        frame(["Roatan was beautiful.", "The ship's Wi-Fi was terrible."])
    )
    assert out[0]["predicted_label"] == "Roatan"
    assert out[1]["predicted_label"] == "General Cruise"
    # The ship topic also clears the active port, so later clauses cannot
    # inherit Roatan either.
    assert out[1]["active_port_after"] == ""


def test_comparison_gets_multiple_destinations():
    # Two ports in one clause with a comparison cue ("better ... than"):
    # the opinion is about both, so no single port is chosen.
    out = ForwardWindowAttributor().predict_review(
        frame(["Roatan had better snorkeling than Cozumel."])
    )
    assert out[0]["predicted_label"] == "Multiple Destinations"


def test_unanchored_clause_abstains():
    # No port, no ship/cruise keyword, and no context to inherit, so the
    # rule abstains instead of guessing.
    out = ForwardWindowAttributor(window_size=0).predict_review(
        frame(["That was an experience I will never forget."])
    )
    assert out[0]["predicted_label"] == "Unclear"
    # ``is True`` checks for a real bool, not just a truthy value.
    assert out[0]["abstained"] is True


def test_context_window_expires():
    # window_size=1: only the one sentence right after the mention inherits it.
    #   sentence 1: explicit mention         -> Cozumel
    #   sentence 2: inside the window        -> Cozumel (carried forward)
    #   sentence 3: beyond the window        -> Unclear (context expired)
    out = ForwardWindowAttributor(window_size=1).predict_review(
        frame(
            [
                "Cozumel was wonderful.",
                "The beach was calm.",
                "The restaurant was expensive.",
            ]
        )
    )
    assert [x["predicted_label"] for x in out] == ["Cozumel", "Cozumel", "Unclear"]
