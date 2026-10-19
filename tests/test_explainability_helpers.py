"""Unit tests for pure helper functions in ``cruise_xai.explainability``.

Covers the punctuation edit used by the stability probe and the checks that
Integrated Gradients token records belong to the right clause and tokens.
No model or real attribution is computed.
"""

import pytest

# Requires PyTorch (requirements-bert.txt); skipped in the lightweight CI install.
# importorskip skips this whole module (rather than failing) if torch is missing.
pytest.importorskip("torch")

import sys
import types

import pytest  # duplicate of the import above; harmless

# These tests exercise pure helpers and do not require Captum/model weights.
# cruise_xai.explainability imports Captum at module level, so if Captum is not
# installed, register a minimal fake ``captum.attr`` module in sys.modules.
# The import below then succeeds, and the helpers under test never touch it.
# Note: the fake stays in sys.modules for the rest of the test session.
try:
    import captum.attr  # noqa: F401
except ImportError:
    captum = types.ModuleType("captum")
    attr = types.ModuleType("captum.attr")
    # Placeholder for the only Captum name the module imports.
    attr.LayerIntegratedGradients = object
    captum.attr = attr
    sys.modules["captum"] = captum
    sys.modules["captum.attr"] = attr

# _punctuation_edit:        the small text change applied by the stability probe.
# _validate_token_records:  sanity checks on IG token records before they are used.
from cruise_xai.explainability import _punctuation_edit, _validate_token_records

# --- Punctuation edit (stability probe) --------------------------------------


def test_punctuation_edit_changes_exclamation_and_spacing():
    # Each "!" becomes "." and runs of spaces collapse to one. The words are
    # untouched, so a stable model should predict the same label.
    assert _punctuation_edit("Great!!  beach!") == "Great.. beach."


def test_punctuation_edit_marks_already_normalized_sentence_as_unchanged():
    # Nothing to edit: the text comes back identical. The stability probe can
    # detect this and exclude the clause, since an unedited input says nothing
    # about stability.
    assert _punctuation_edit("The beach was great.") == "The beach was great."


# --- IG token-record validation ----------------------------------------------
# Arguments to _validate_token_records: the clause row, its IG token records,
# the tokenised clause (including [CLS]/[SEP]), and the token positions the
# records are expected to cover.


def test_token_record_validation_rejects_wrong_clause():
    # Records from clause "c2" must not be accepted as explaining clause "c1".
    row = {"clause_id": "c1"}
    records = [
        {
            "clause_id": "c2",
            "target_label": "Roatan",
            "token_index": 1,
            "token": "beach",
        }
    ]
    with pytest.raises(ValueError, match="do not match"):
        _validate_token_records(row, records, ["[CLS]", "beach", "[SEP]"], [1])


def test_token_record_validation_rejects_token_index_mismatch():
    # The record says position 1 is "water", but the tokenised clause has
    # "beach" there, so the record does not line up with the input.
    row = {"clause_id": "c1"}
    records = [
        {
            "clause_id": "c1",
            "target_label": "Roatan",
            "token_index": 1,
            "token": "water",
        }
    ]
    with pytest.raises(ValueError, match="token indices do not match"):
        _validate_token_records(row, records, ["[CLS]", "beach", "[SEP]"], [1])


def test_token_record_validation_requires_consistent_target():
    # All records for one clause must explain the same predicted label.
    # Mixing Roatan and Cozumel attributions would make the faithfulness
    # deletion test meaningless.
    row = {"clause_id": "c1"}
    records = [
        {
            "clause_id": "c1",
            "target_label": "Roatan",
            "token_index": 1,
            "token": "beach",
        },
        {
            "clause_id": "c1",
            "target_label": "Cozumel",
            "token_index": 2,
            "token": "water",
        },
    ]
    with pytest.raises(ValueError, match="consistent"):
        _validate_token_records(
            row, records, ["[CLS]", "beach", "water", "[SEP]"], [1, 2]
        )
