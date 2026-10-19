"""Integrated Gradients token attributions and perturbation/stability audits.

This module explains which input tokens influence a selected model score and
checks how predictions respond to token deletion or small text edits.
These are diagnostic methods, not proof of causal explanations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients

from .bert_model import load_bert, resolve_max_length
from .data import build_model_text


def _load_components(model_dir: str | Path, components=None):
    """Reuse supplied model components or load the tokenizer/model/device tuple."""
    # Passing preloaded components avoids repeatedly loading BERT in batch workflows.
    # The model_dir-only calling style is retained for convenience and compatibility.
    if components is not None:
        if len(components) != 3:
            raise ValueError("components must be a (tokenizer, model, device) tuple")
        return components
    return load_bert(model_dir)


def _label_to_id(model, label: str | None, predicted_id: int) -> int:
    """Resolve an optional label name to the model's numeric class index."""
    # If no target is specified, explain the model's own top prediction.
    if label is None:
        return predicted_id

    label2id = getattr(model.config, "label2id", None) or {}
    if label not in label2id:
        raise ValueError(
            f"Unknown target_label {label!r}; expected one of {sorted(label2id)}"
        )
    idx = int(label2id[label])

    # Guard against an inconsistent label map or an invalid class index.
    if not 0 <= idx < int(model.config.num_labels):
        raise ValueError(f"target_label {label!r} maps to invalid class index {idx}")
    return idx


def _eligible_content_positions(input_ids: torch.Tensor, tokenizer) -> list[int]:
    """Return token positions that represent content, excluding special/padding tokens."""
    # Convert the single encoded sequence to ordinary Python IDs for inspection.
    ids = input_ids[0].detach().cpu().tolist()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    # Add the usual boundary and padding tokens even if the tokenizer's list omits them.
    for token_id in (
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "sep_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ):
        if token_id is not None:
            special_ids.add(int(token_id))

    # These are the positions eligible for attribution or deletion.
    return [i for i, token_id in enumerate(ids) if int(token_id) not in special_ids]


def _validate_token_records(
    row: pd.Series | dict,
    token_records: list[dict],
    tokens: list[str],
    eligible_positions: list[int],
) -> tuple[list[dict], str]:
    """Ensure previously calculated attributions match this exact clause input."""
    clause_id = str(row.get("clause_id", ""))
    if not token_records:
        raise ValueError("token_records is empty; calculate Integrated Gradients first")

    # Every record must belong to the clause currently being analyzed.
    for record in token_records:
        if str(record.get("clause_id", "")) != clause_id:
            raise ValueError("token_records do not match the supplied clause_id")

    # All records must refer to one non-empty target class.
    targets = {str(record.get("target_label", "")) for record in token_records}
    if len(targets) != 1 or not next(iter(targets)):
        raise ValueError(
            "token_records must have one consistent, non-empty target_label"
        )
    target_label = next(iter(targets))

    # Check that token indices are unique and still identify the same token strings.
    seen = set()
    for record in token_records:
        pos = int(record["token_index"])
        if pos in seen:
            raise ValueError(f"Duplicate token_index in token_records: {pos}")
        seen.add(pos)
        if pos < 0 or pos >= len(tokens) or tokens[pos] != record.get("token"):
            raise ValueError(
                "token_records token indices do not match the current model input"
            )
        # Attributions should only refer to content tokens, not CLS/SEP/padding.
        if pos not in eligible_positions:
            raise ValueError(
                "token_records contain a special-token or padding position"
            )
    return token_records, target_label


def integrated_gradients_for_row(
    row: pd.Series | dict,
    model_dir: str | Path,
    target_label: str | None = None,
    max_length: int | None = None,
    n_steps: int = 32,
    components=None,
) -> list[dict]:
    """Calculate Layer Integrated Gradients for content tokens in one clause.

    The target is the specified label, or the model's predicted class if omitted.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    max_length = resolve_max_length(model_dir, max_length)
    tokenizer, model, device = _load_components(model_dir, components)

    # Use the same input builder as training/prediction so the analyzed text matches
    # the representation the model actually receives.
    text = build_model_text(row)
    encoded = tokenizer(
        text, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    # Determine the model's ordinary prediction before choosing the attribution target.
    with torch.no_grad():
        logits = model(**encoded).logits
        pred_idx = int(logits.argmax(dim=-1).item())
    target_idx = _label_to_id(model, target_label, pred_idx)

    # Captum calls this forward function along its integration path. It accepts token IDs
    # and returns logits for all classes; the target_idx selects the score being explained.
    def forward(ids):
        kwargs = {"input_ids": ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if "token_type_ids" in encoded:
            kwargs["token_type_ids"] = encoded["token_type_ids"]
        return model(**kwargs).logits

    # Build a reference input by replacing content tokens with PAD while retaining
    # sequence boundary tokens. Integrated Gradients compares the input with this baseline.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError(
            "Tokenizer must define pad_token_id for the Integrated Gradients baseline"
        )
    baselines = input_ids.clone()
    content_positions = _eligible_content_positions(input_ids, tokenizer)
    if content_positions:
        baselines[:, content_positions] = int(pad_id)

    # LayerIntegratedGradients attributes the target score through the embedding layer.
    lig = LayerIntegratedGradients(forward, model.get_input_embeddings())
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baselines,
        target=target_idx,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # Sum embedding-dimension contributions to get one attribution score per token.
    scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # The completeness residual compares the summed attribution with the score difference
    # between the real input and baseline. Scores here are logits, not probabilities.
    with torch.no_grad():
        input_target_logit = float(forward(input_ids)[0, target_idx].item())
        baseline_target_logit = float(forward(baselines)[0, target_idx].item())
    output_difference = abs(input_target_logit - baseline_target_logit)
    convergence_delta = float(delta.detach().cpu().reshape(-1)[0])
    relative_error = (
        abs(convergence_delta) / output_difference if output_difference > 1e-8 else None
    )
    convergence_flag = (
        "unreliable_denominator"
        if relative_error is None
        else "high_error" if relative_error > 0.05 else "ok"
    )

    # Convert IDs back to readable tokens and store one result per content token.
    tokens = tokenizer.convert_ids_to_tokens(
        input_ids.squeeze(0).detach().cpu().tolist()
    )
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    special_tokens = {
        getattr(tokenizer, "cls_token", None),
        getattr(tokenizer, "sep_token", None),
        getattr(tokenizer, "pad_token", None),
    }
    records = []
    for i, (token, score, token_id) in enumerate(
        zip(tokens, scores, input_ids[0].detach().cpu().tolist())
    ):
        # Special tokens are excluded from the exported attribution table.
        if token in special_tokens or int(token_id) in special_ids:
            continue
        records.append(
            {
                "clause_id": str(row.get("clause_id", "")),
                "review_id": str(row.get("review_id", "")),
                "token_index": i,
                "token": token,
                "attribution": float(score),
                "abs_attribution": float(abs(score)),
                "target_label": model.config.id2label[int(target_idx)],
                "target_class_id": int(target_idx),
                "predicted_label": model.config.id2label[pred_idx],
                "ig_convergence_delta": convergence_delta,
                "ig_output_difference": float(output_difference),
                "ig_relative_convergence_error": relative_error,
                "ig_convergence_flag": convergence_flag,
            }
        )
    return records


def perturbation_faithfulness(
    row: pd.Series | dict,
    model_dir: str | Path,
    token_records: list[dict],
    max_length: int | None = None,
    top_k: int = 3,
    random_state: int = 42,
    random_draws: int = 5,
    components=None,
) -> dict:
    """Compare target-score changes after deleting selected groups of content tokens.

    Compares high positive-attribution tokens with low-attribution and random controls.
    Deletion is a diagnostic only; it does not establish that a token caused the prediction.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if random_draws < 1:
        raise ValueError("random_draws must be at least 1")
    max_length = resolve_max_length(model_dir, max_length)
    tokenizer, model, device = _load_components(model_dir, components)

    # Recreate the exact model input and verify the supplied attribution rows align with it.
    text = build_model_text(row)
    original = tokenizer(
        text, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    input_ids = original["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
    eligible = _eligible_content_positions(input_ids, tokenizer)
    records, target_label = _validate_token_records(
        row, token_records, tokens, eligible
    )

    # Resolve the class whose score will be monitored across all deletion conditions.
    label2id = getattr(model.config, "label2id", None) or {}
    if target_label not in label2id:
        raise ValueError(f"Unknown attribution target label {target_label!r}")
    target_idx = int(label2id[target_label])
    if not 0 <= target_idx < int(model.config.num_labels):
        raise ValueError(
            f"Attribution target {target_label!r} maps to invalid class index {target_idx}"
        )

    # Ensure there is exactly one attribution record for every eligible content token.
    record_positions = {int(r["token_index"]) for r in records}
    if record_positions != set(eligible):
        raise ValueError(
            "token_records do not cover exactly the eligible content tokens in the current input"
        )

    # Record the original target class's probability and raw logit for comparison.
    with torch.no_grad():
        orig_logits = model(**original).logits[0]
        orig_prob = torch.softmax(orig_logits, dim=-1)[target_idx].item()
        orig_logit = orig_logits[target_idx].item()

    # Only positive attributions can be selected as the top-positive group.
    positive_records = [r for r in records if float(r["attribution"]) > 0.0]
    top = sorted(
        positive_records,
        key=lambda r: (-float(r["attribution"]), int(r["token_index"])),
    )[: min(top_k, len(positive_records))]

    # Control conditions remove the same number of tokens as the top-positive group.
    random_count = len(top)
    rng = np.random.default_rng(random_state)

    def edited_scores(selected: list[dict]) -> tuple[float, float]:
        """Return target probability and logit after deleting selected token positions."""
        remove_positions = {int(r["token_index"]) for r in selected}
        keep_positions = [
            i for i in range(input_ids.shape[1]) if i not in remove_positions
        ]

        # Remove selected columns directly from the encoded tensors rather than rebuilding
        # and re-tokenizing the text. Non-deletable special tokens remain in the sequence.
        edited = {}
        for key, value in original.items():
            if (
                torch.is_tensor(value)
                and value.ndim == 2
                and value.shape[1] == input_ids.shape[1]
            ):
                edited[key] = value[:, keep_positions]
            else:
                edited[key] = value
        with torch.no_grad():
            logits = model(**edited).logits[0]
            prob = torch.softmax(logits, dim=-1)[target_idx].item()
        return float(prob), float(logits[target_idx].item())

    # Main test: delete the highest positive-attribution tokens.
    top_prob, top_logit = (
        edited_scores(top) if top else (float(orig_prob), float(orig_logit))
    )

    # Control 1: delete tokens with the smallest absolute attribution magnitudes.
    low_selected = sorted(
        records,
        key=lambda r: (abs(float(r["attribution"])), int(r["token_index"])),
    )[:random_count]
    low_prob, low_logit = (
        edited_scores(low_selected)
        if low_selected
        else (float(orig_prob), float(orig_logit))
    )

    # Control 2: repeat random deletion to estimate how much results depend on token choice.
    random_results = []
    if random_count:
        for _ in range(random_draws):
            chosen = sorted(
                rng.choice(len(records), size=random_count, replace=False).tolist()
            )
            random_selected = [records[i] for i in chosen]
            random_results.append(edited_scores(random_selected))
    random_prob = (
        float(np.mean([x[0] for x in random_results]))
        if random_results
        else float(orig_prob)
    )
    random_logit = (
        float(np.mean([x[1] for x in random_results]))
        if random_results
        else float(orig_logit)
    )

    # Positive drops mean the model's score for the target class decreased after deletion.
    return {
        "clause_id": str(row.get("clause_id", "")),
        "review_id": str(row.get("review_id", "")),
        "target_label": target_label,
        "original_target_probability": float(orig_prob),
        "original_target_logit": float(orig_logit),
        "top_positive_probability_after_deletion": float(top_prob),
        "low_attribution_probability_after_deletion": float(low_prob),
        "random_probability_after_deletion": float(random_prob),
        "random_probability_after_deletion_sd": (
            float(np.std([x[0] for x in random_results], ddof=1))
            if len(random_results) > 1
            else (0.0 if random_results else None)
        ),
        "random_draws_requested": int(random_draws),
        "random_draws_completed": int(len(random_results)),
        "top_positive_tokens_available": bool(len(top) > 0),
        "top_positive_probability_drop": float(orig_prob - top_prob),
        "low_attribution_probability_drop": float(orig_prob - low_prob),
        "random_probability_drop": float(orig_prob - random_prob),
        "top_positive_logit_drop": float(orig_logit - top_logit),
        "low_attribution_logit_drop": float(orig_logit - low_logit),
        "random_logit_drop": float(orig_logit - random_logit),
        "random_logit_drop_sd": (
            float(np.std([orig_logit - x[1] for x in random_results], ddof=1))
            if len(random_results) > 1
            else (0.0 if random_results else None)
        ),
        "top_k_requested": int(top_k),
        "tokens_removed": int(len(top)),
        "low_attribution_tokens_removed": int(len(low_selected)),
        "random_tokens_removed_per_draw": int(random_count),
        "interpretation_caveat": "Token deletion can create unnatural or incomplete text; score changes are evidence, not proof of causal faithfulness.",
    }


def _punctuation_edit(text: str) -> str:
    """Apply a deterministic punctuation and spacing normalization probe."""
    # Replace exclamation marks, collapse repeated spaces, and trim surrounding spaces.
    modified = text.replace("!", ".")
    while "  " in modified:
        modified = modified.replace("  ", " ")
    modified = modified.strip()

    # Remove a trailing period before adding one, so the output ends with one period.
    modified = modified[:-1] if modified.endswith(".") else modified
    return modified + "." if modified else modified


def stability_test(
    row: pd.Series | dict,
    model_dir: str | Path,
    max_length: int | None = None,
    components=None,
) -> dict:
    """Test prediction stability after a conservative punctuation/spacing edit.

    Unchanged inputs are reported but should not count as meaningful perturbations
    in aggregate agreement. Structural changes to the constructed input are flagged.
    """
    max_length = resolve_max_length(model_dir, max_length)
    tokenizer, model, device = _load_components(model_dir, components)

    # Create original and minimally edited copies of the row.
    original_clause = str(row.get("clause_text", "") or "")
    modified_clause = _punctuation_edit(original_clause)
    original_row = dict(row)
    modified_row = dict(row)
    modified_row["clause_text"] = modified_clause

    original_sentence = str(row.get("sentence_text", "") or "").strip()
    # If sentence_text duplicates clause_text, edit both so build_model_text doesn't
    # unexpectedly add a separate SENTENCE segment after the punctuation change.
    if (
        original_sentence
        and original_sentence.casefold() == original_clause.strip().casefold()
    ):
        modified_row["sentence_text"] = modified_clause

    # Construct both inputs using the same logic used by the BERT pipeline.
    texts = [build_model_text(original_row), build_model_text(modified_row)]
    text_changed = texts[0] != texts[1]

    # Exclude pairs where the number of SENTENCE segments changes, because that would
    # confound a punctuation stability check with a change in input structure.
    sentence_marker = "SENTENCE:"
    original_marker_count = texts[0].count(sentence_marker)
    modified_marker_count = texts[1].count(sentence_marker)
    structural_change = original_marker_count != modified_marker_count
    eligible_for_aggregate = bool(text_changed and not structural_change)

    if not text_changed:
        eligibility_reason = "unchanged_input"
    elif structural_change:
        eligibility_reason = "structural_change"
    else:
        eligibility_reason = "eligible"

    # Run both versions together; padding makes them the same tensor length for inference.
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    p1, p2 = probs
    return {
        "clause_id": str(row.get("clause_id", "")),
        "review_id": str(row.get("review_id", "")),
        "original_text": texts[0],
        "modified_text": texts[1],
        "text_changed": bool(text_changed),
        "eligible_for_stability_aggregate": eligible_for_aggregate,
        "stability_eligibility_reason": eligibility_reason,
        "original_sentence_marker_count": int(original_marker_count),
        "modified_sentence_marker_count": int(modified_marker_count),
        "structural_change": bool(structural_change),
        "original_prediction": labels[int(p1.argmax())],
        "modified_prediction": labels[int(p2.argmax())],
        "prediction_agreement": bool(p1.argmax() == p2.argmax()),
        # L1 distance sums the absolute probability change across all classes.
        "probability_l1_distance": float(np.abs(p1 - p2).sum()),
        "edit_note": "Punctuation/spacing normalization intended to preserve meaning; manually verify a sample before aggregate interpretation.",
    }
