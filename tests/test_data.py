"""Unit tests for data helpers in ``cruise_xai.data``.

Covers label cleaning, review-level splitting, the text fed to BERT, and the
guard that stops scripts from creating a split by accident.
"""

import pandas as pd
import pytest
from cruise_xai.data import (
    normalize_label,
    make_group_splits,
    build_model_text,
    require_existing_splits,
)


def test_label_alias_normalization():
    # Common annotation variants should map to the canonical label names.
    # A known misspelling is corrected...
    assert normalize_label("Positve") == "Positive"
    # ...and snake_case is converted to the canonical spaced, title-case form.
    assert normalize_label("multiple_destinations") == "Multiple Destinations"


def test_group_split_never_leaks_reviews():
    # 20 reviews with 3 clauses each (60 rows), e.g. "r0", "r0", "r0", "r1", ...
    df = pd.DataFrame({"review_id": [f"r{i}" for i in range(20) for _ in range(3)]})
    splits = make_group_splits(df, seed=7)
    # The split table has one row per review, so each review is assigned to
    # exactly one split and its clauses cannot end up in different splits.
    assert splits.review_id.is_unique
    # With 20 reviews, all three splits should receive at least one review.
    assert set(splits.split) == {"train", "validation", "test"}


def test_model_text_contains_target_and_context():
    # BERT sees the clause together with its neighbouring sentences. Each part
    # is tagged so the model can tell the clause being classified (TARGET)
    # from the surrounding context.
    text = build_model_text(
        {
            "clause_text": "The water was clear",
            "previous_sentence": "We docked in Roatan.",
            "next_sentence": "We took a taxi.",
        }
    )
    assert "TARGET: The water was clear" in text
    # The previous sentence matters here: it is the only place "Roatan" appears.
    assert "PREVIOUS: We docked in Roatan." in text
    assert "NEXT: We took a taxi." in text


def test_require_existing_splits_never_creates_a_split(tmp_path):
    # tmp_path is a fresh temporary folder provided by pytest.
    missing = tmp_path / "review_splits.csv"
    # A missing file must raise, and the message must point the user to
    # prepare_data.py (``match`` checks the message text).
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        require_existing_splits(missing)
    # The guard must not have created the file as a side effect.
    assert not missing.exists()
    # Once a split file exists, the same path is returned unchanged so it can
    # be passed straight on to attach_splits.
    missing.write_text("review_id,split\nr1,train\n", encoding="utf-8")
    assert require_existing_splits(missing) == missing
