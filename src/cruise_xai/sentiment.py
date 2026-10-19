"""VADER sentiment analysis layered on top of destination attribution.

This module is deliberately separate from the attribution code. It does not
change how the rule baseline or BERT assign destinations; it only asks two
follow-up questions:

1. **Sentiment accuracy:** how well does VADER's clause-level sentiment agree
   with the gold ``sentiment_label`` annotations?
2. **Destination-level sentiment:** when clauses are grouped by destination
   using each attribution source (gold labels, the rule baseline, BERT), how
   closely does each method's destination-by-sentiment profile match the profile
   obtained from the gold destination labels?

VADER is a fixed lexicon-and-rules scorer (Hutto & Gilbert, 2014). It is not
trained on this corpus, so it has no train/validation/test dependence of its
own; results are still reported on the held-out split by default so they sit
beside the attribution metrics.

Module layout
-------------
- VADER scoring:            score clauses and map scores to sentiment labels.
- Sentiment metrics:        VADER labels vs gold sentiment labels.
- Destination profiles:     sentiment shares per destination, per attribution
                            source, and their error against the gold profile.
- Joint accuracy:           destination AND sentiment both correct.
- Review-level bootstrap:   confidence intervals for all of the above.
- Figure / saving helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .data import DESTINATION_LABELS, SENTIMENT_LABELS

# Conventional VADER cut-offs on the compound score (VADER project documentation).
# compound >= 0.05 -> Positive, <= -0.05 -> Negative, otherwise Neutral.
DEFAULT_POSITIVE_THRESHOLD = 0.05
DEFAULT_NEGATIVE_THRESHOLD = -0.05
# Attribution columns understood by this module, in reporting order.
# Keys are short source names used throughout; values are the column holding
# that source's destination for each clause.
ATTRIBUTION_SOURCES = {
    "gold": "destination_label",
    "rule": "rule_prediction",
    "bert": "bert_prediction",
}
# Sentiment columns: the human annotation and the VADER-derived label.
SENTIMENT_SOURCES = {
    "gold_sentiment": "sentiment_label",
    "vader_sentiment": "vader_label",
}


# ---------------------------------------------------------------------------
# VADER scoring
# ---------------------------------------------------------------------------
def load_vader():
    """Return a VADER analyzer, with an actionable error if the package is missing."""
    # Imported lazily so the rest of the package works without vaderSentiment.
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only without the dependency
        raise ImportError(
            "vaderSentiment is not installed. Run: pip install -r requirements-sentiment.txt"
        ) from exc
    return SentimentIntensityAnalyzer()


def vader_scores(texts: Iterable[str], analyzer=None) -> pd.DataFrame:
    """Score each text with VADER; returns compound, pos, neu, and neg columns."""
    # Reuse a caller-supplied analyzer when given (avoids reloading the lexicon).
    analyzer = analyzer or load_vader()
    rows = []
    for text in texts:
        # Missing text is scored as an empty string (VADER returns all-neutral).
        scores = analyzer.polarity_scores("" if pd.isna(text) else str(text))
        # compound: normalised overall polarity in [-1, 1].
        # pos/neu/neg: proportions of the text in each polarity (sum to ~1).
        rows.append(
            {
                "vader_compound": float(scores["compound"]),
                "vader_pos": float(scores["pos"]),
                "vader_neu": float(scores["neu"]),
                "vader_neg": float(scores["neg"]),
            }
        )
    # Explicit columns keep the schema stable even when ``texts`` is empty.
    return pd.DataFrame(
        rows, columns=["vader_compound", "vader_pos", "vader_neu", "vader_neg"]
    )


def vader_label(
    compound: float,
    pos: float = 0.0,
    neg: float = 0.0,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
    mixed_min: float | None = None,
) -> str:
    """Map VADER scores to the annotation taxonomy.

    By default only Positive/Negative/Neutral are produced, using the standard
    compound cut-offs. VADER has no native "Mixed" output. If ``mixed_min`` is
    set, a clause whose positive *and* negative proportions both reach that value
    is labelled Mixed. Choose any such threshold on training or validation data,
    never on the test split.
    """
    if negative_threshold >= positive_threshold:
        raise ValueError("negative_threshold must be lower than positive_threshold")
    # Mixed is checked first: a clause with substantial positive and negative
    # content can still have a strongly signed compound score.
    if mixed_min is not None and pos >= mixed_min and neg >= mixed_min:
        return "Mixed"
    if compound >= positive_threshold:
        return "Positive"
    if compound <= negative_threshold:
        return "Negative"
    # Compound falls strictly between the two thresholds.
    return "Neutral"


def add_vader_columns(
    frame: pd.DataFrame,
    text_col: str = "clause_text",
    analyzer=None,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
    mixed_min: float | None = None,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with VADER scores and a mapped ``vader_label``."""
    if text_col not in frame.columns:
        raise ValueError(f"Missing text column: {text_col}")
    # Reset the index so the positional concat below lines rows up correctly
    # (the scores DataFrame has a fresh 0..n-1 index).
    out = frame.reset_index(drop=True).copy()
    scores = vader_scores(out[text_col], analyzer=analyzer)
    out = pd.concat([out, scores], axis=1)
    out["vader_label"] = [
        vader_label(c, p, n, positive_threshold, negative_threshold, mixed_min)
        for c, p, n in zip(out["vader_compound"], out["vader_pos"], out["vader_neg"])
    ]
    return out


# ---------------------------------------------------------------------------
# Sentiment metrics (VADER vs gold)
# ---------------------------------------------------------------------------
def evaluate_sentiment(
    gold, predicted, min_support: int = 10
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Score sentiment predictions against gold labels over the fixed four-class taxonomy.

    Macro-F1 averages over all four classes, so a class the predictor never
    outputs (for example Mixed under the default VADER mapping) contributes zero.
    ``macro_f1_three_class`` is reported alongside it for transparency; it is a
    secondary view, not a replacement for the four-class score.

    Returns:
        (metrics dict, per-class table, confusion matrix in SENTIMENT_LABELS order)
    """
    # Normalise to stripped strings; missing values become "" and are then
    # rejected by the label check below rather than silently ignored.
    y_true = ["" if pd.isna(x) else str(x).strip() for x in gold]
    y_pred = ["" if pd.isna(x) else str(x).strip() for x in predicted]
    if not y_true:
        raise ValueError("Cannot evaluate an empty set of sentiment labels")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"gold and predicted lengths differ: {len(y_true)} != {len(y_pred)}"
        )
    invalid = sorted((set(y_true) | set(y_pred)) - set(SENTIMENT_LABELS))
    if invalid:
        raise ValueError(f"Invalid sentiment labels: {invalid}")

    # Always evaluate over the full label set so absent classes still appear
    # (with zero scores) instead of being dropped by sklearn.
    labels = list(SENTIMENT_LABELS)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    # Balanced accuracy only averages recall over classes present in gold;
    # recall is undefined for a class with no gold examples.
    present = [label for label in labels if label in set(y_true)]
    support = {label: int(sum(t == label for t in y_true)) for label in labels}
    three_class = [label for label in labels if label != "Mixed"]
    metrics = {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": (
            float(np.mean([report[label]["recall"] for label in present]))
            if present
            else None
        ),
        "macro_f1_three_class": float(
            f1_score(
                y_true, y_pred, labels=three_class, average="macro", zero_division=0
            )
        ),
        "gold_class_support": support,
        # Shows directly whether the predictor ever outputs each class (e.g. Mixed).
        "predicted_class_counts": {
            label: int(sum(p == label for p in y_pred)) for label in labels
        },
        "low_support_classes": [
            label for label in labels if support[label] < min_support
        ],
        "min_support_threshold": int(min_support),
        "per_class": {label: report[label] for label in labels},
    }
    # Same per-class numbers as a flat table for CSV export.
    per_class = pd.DataFrame(
        [
            {
                "label": label,
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1-score": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
                "low_support": support[label] < min_support,
            }
            for label in labels
        ]
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, per_class, cm


# ---------------------------------------------------------------------------
# Destination-by-sentiment profiles
# ---------------------------------------------------------------------------
def destination_sentiment_profile(
    frame: pd.DataFrame,
    destination_col: str,
    sentiment_col: str,
    min_clauses: int = 5,
) -> pd.DataFrame:
    """Summarize sentiment for each destination under one attribution and one sentiment source.

    ``Unclear`` and any other non-destination value in ``destination_col`` are
    excluded (abstained clauses are not attributed to any destination). Net
    sentiment is the Positive share minus the Negative share. Destinations with
    fewer than ``min_clauses`` attributed clauses are flagged ``low_support``.

    Returns one row per destination in DESTINATION_LABELS order, including
    destinations with zero clauses (their shares are NaN).
    """
    rows = []
    attributed = frame[frame[destination_col].isin(DESTINATION_LABELS)]
    for destination in DESTINATION_LABELS:
        group = attributed[attributed[destination_col] == destination]
        n = int(len(group))
        row = {
            "destination": destination,
            "n_clauses": n,
            "low_support": n < min_clauses,
        }
        # Share of this destination's clauses carrying each sentiment label.
        for label in SENTIMENT_LABELS:
            row[f"share_{label.lower()}"] = (
                float((group[sentiment_col] == label).mean()) if n else np.nan
            )
        # Net sentiment ranges from -1 (all negative) to +1 (all positive).
        row["net_sentiment"] = (
            (row["share_positive"] - row["share_negative"]) if n else np.nan
        )
        # Continuous companion to the label shares; NaN if VADER was not run.
        row["mean_vader_compound"] = (
            float(group["vader_compound"].mean())
            if n and "vader_compound" in group
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def all_destination_profiles(
    frame: pd.DataFrame, methods: list[str], min_clauses: int = 5
) -> pd.DataFrame:
    """Profiles for every attribution source × sentiment source combination (long format)."""
    pieces = []
    # Gold attribution is always included because it is the reference.
    for attribution in ["gold", *methods]:
        for sentiment_source, sentiment_col in SENTIMENT_SOURCES.items():
            profile = destination_sentiment_profile(
                frame,
                ATTRIBUTION_SOURCES[attribution],
                sentiment_col,
                min_clauses=min_clauses,
            )
            # Tag rows with their source pair so the tables can be stacked.
            profile.insert(0, "sentiment_source", sentiment_source)
            profile.insert(0, "attribution_source", attribution)
            pieces.append(profile)
    return pd.concat(pieces, ignore_index=True)


def profile_errors(profiles: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Compare each method's destination profile with the gold-attribution profile.

    The reference for both sentiment sources is the profile built from gold
    destinations and **gold** sentiment, so the error combines attribution error
    and (for VADER rows) sentiment-classification error. The ``gold`` attribution
    row with VADER sentiment isolates VADER's contribution.
    """
    reference = profiles[
        (profiles["attribution_source"] == "gold")
        & (profiles["sentiment_source"] == "gold_sentiment")
    ].set_index("destination")
    share_cols = [f"share_{label.lower()}" for label in SENTIMENT_LABELS]
    rows = []
    for attribution in ["gold", *methods]:
        for sentiment_source in SENTIMENT_SOURCES:
            if attribution == "gold" and sentiment_source == "gold_sentiment":
                continue  # the reference itself
            current = profiles[
                (profiles["attribution_source"] == attribution)
                & (profiles["sentiment_source"] == sentiment_source)
            ].set_index("destination")
            for destination in DESTINATION_LABELS:
                ref, cur = reference.loc[destination], current.loc[destination]
                # Errors are only defined when both profiles have clauses for
                # this destination; otherwise the shares are NaN.
                comparable = bool(ref["n_clauses"] > 0 and cur["n_clauses"] > 0)
                rows.append(
                    {
                        "attribution_source": attribution,
                        "sentiment_source": sentiment_source,
                        "destination": destination,
                        "n_clauses_reference": int(ref["n_clauses"]),
                        "n_clauses_compared": int(cur["n_clauses"]),
                        # Low support on either side makes the comparison unreliable.
                        "low_support": bool(ref["low_support"] or cur["low_support"]),
                        "net_sentiment_reference": ref["net_sentiment"],
                        "net_sentiment_compared": cur["net_sentiment"],
                        "net_sentiment_abs_error": (
                            abs(cur["net_sentiment"] - ref["net_sentiment"])
                            if comparable
                            else np.nan
                        ),
                        # Mean absolute difference across all four sentiment shares,
                        # i.e. how far apart the whole sentiment distributions are.
                        "share_mean_abs_error": (
                            float(np.mean([abs(cur[c] - ref[c]) for c in share_cols]))
                            if comparable
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def common_destinations(errors: pd.DataFrame) -> list[str]:
    """Destinations with enough clauses under *every* compared attribution source.

    Mean errors are averaged over this shared set so that methods are compared on
    the same destinations, not on whichever destinations each happened to cover.
    """
    # A destination is usable only if no comparison row for it is low-support...
    usable = errors.groupby("destination")["low_support"].apply(lambda s: not s.any())
    # ...and every comparison row actually produced an error value.
    usable &= errors.groupby("destination")["net_sentiment_abs_error"].apply(
        lambda s: s.notna().all()
    )
    # Return in canonical DESTINATION_LABELS order.
    return [d for d in DESTINATION_LABELS if bool(usable.get(d, False))]


def summarize_profile_errors(errors: pd.DataFrame) -> dict:
    """Mean errors per attribution × sentiment source, over the shared non-low-support destinations."""
    shared = common_destinations(errors)
    # Record which destinations were used and which were dropped, for reporting.
    summary = {
        "destinations_compared": shared,
        "destinations_excluded_low_support": [
            d for d in DESTINATION_LABELS if d not in shared
        ],
    }
    for (attribution, sentiment_source), group in errors.groupby(
        ["attribution_source", "sentiment_source"], sort=False
    ):
        kept = group[group["destination"].isin(shared)]
        # Keys look like "rule|vader_sentiment".
        summary[f"{attribution}|{sentiment_source}"] = {
            "mean_net_sentiment_abs_error": (
                float(kept["net_sentiment_abs_error"].mean()) if len(kept) else None
            ),
            "mean_share_abs_error": (
                float(kept["share_mean_abs_error"].mean()) if len(kept) else None
            ),
        }
    return summary


# ---------------------------------------------------------------------------
# End-to-end (destination + sentiment) correctness
# ---------------------------------------------------------------------------
def joint_accuracy(
    frame: pd.DataFrame, method: str, sentiment_col: str = "vader_label"
) -> dict:
    """Share of clauses where the method's destination AND the sentiment both match gold.

    Abstentions (``Unclear``) count as incorrect destinations. Coverage is reported
    so abstention is not hidden.
    """
    dest_col = ATTRIBUTION_SOURCES[method]
    destination_correct = frame[dest_col] == frame["destination_label"]
    sentiment_correct = frame[sentiment_col] == frame["sentiment_label"]
    # Coverage = share of clauses assigned to a real destination (not abstained).
    covered = frame[dest_col].isin(DESTINATION_LABELS)
    return {
        "n": int(len(frame)),
        "coverage": float(covered.mean()) if len(frame) else None,
        "destination_accuracy": (
            float(destination_correct.mean()) if len(frame) else None
        ),
        "sentiment_accuracy": float(sentiment_correct.mean()) if len(frame) else None,
        "joint_accuracy": (
            float((destination_correct & sentiment_correct).mean())
            if len(frame)
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Review-level cluster bootstrap
# ---------------------------------------------------------------------------
def review_bootstrap(
    frame: pd.DataFrame,
    statistics: dict[str, Callable[[pd.DataFrame], float | None]],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    review_col: str = "review_id",
) -> pd.DataFrame:
    """Percentile intervals for arbitrary statistics, resampling whole reviews.

    Every statistic is computed on the same resampled reviews in each replicate,
    so differences between methods are paired. Replicates where a statistic is
    undefined (None/NaN) are skipped for that statistic only.

    Resampling reviews rather than clauses respects the fact that clauses in
    the same review are correlated (shared author, shared context).
    """
    if n_resamples < 100:
        raise ValueError(
            "n_resamples must be at least 100 for useful percentile intervals"
        )
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    data = frame.reset_index(drop=True)
    reviews = data[review_col].astype(str).to_numpy()
    review_ids = pd.unique(reviews)
    if len(review_ids) < 2:
        raise ValueError(
            "At least two distinct reviews are required for a review-level bootstrap"
        )
    # Precompute the row positions of each review so each replicate is a fast
    # positional gather instead of repeated boolean filtering.
    positions = {rid: np.flatnonzero(reviews == rid) for rid in review_ids}
    rng = np.random.default_rng(random_state)
    draws = {name: [] for name in statistics}
    for _ in range(n_resamples):
        # Draw as many reviews as there are, with replacement; a review drawn
        # twice contributes all its clauses twice.
        sampled = rng.choice(review_ids, size=len(review_ids), replace=True)
        replicate = data.iloc[np.concatenate([positions[rid] for rid in sampled])]
        for name, stat in statistics.items():
            value = stat(replicate)
            if value is not None and np.isfinite(value):
                draws[name].append(float(value))
    # Two-sided percentile interval, e.g. 2.5th and 97.5th percentiles for 95%.
    alpha = (1 - confidence_level) / 2
    rows = []
    for name, stat in statistics.items():
        values = np.asarray(draws[name], dtype=float)
        # Point estimate on the original (non-resampled) data.
        estimate = stat(data)
        rows.append(
            {
                "statistic": name,
                "estimate": (
                    None
                    if estimate is None or not np.isfinite(estimate)
                    else float(estimate)
                ),
                "ci_lower": float(np.quantile(values, alpha)) if len(values) else None,
                "ci_upper": (
                    float(np.quantile(values, 1 - alpha)) if len(values) else None
                ),
                # Fewer valid than requested resamples means the statistic was
                # sometimes undefined; worth checking when interpreting the interval.
                "valid_resamples": int(len(values)),
                "requested_resamples": int(n_resamples),
            }
        )
    return pd.DataFrame(rows)


def _mean_net_error(
    frame: pd.DataFrame,
    method: str,
    sentiment_source: str,
    min_clauses: int,
    methods: list[str],
    cache: dict,
) -> float | None:
    """Statistic helper: mean |net sentiment error| for one method over the shared destination set.

    The shared set is recomputed in each bootstrap replicate from all compared
    methods, so paired differences always average over the same destinations.
    Profiles are computed once per replicate and reused by every statistic.
    """
    # Cache keyed on object identity: each replicate is a new DataFrame, so the
    # first statistic called on it fills the cache and the rest reuse it. The
    # cache holds a reference to the frame, so its identity cannot be recycled.
    if cache.get("frame") is not frame:
        cache.update(
            frame=frame, **_net_sentiment_by_destination(frame, methods, min_clauses)
        )
    shared = cache["shared"]
    if not shared.any():
        return None  # no comparable destination in this replicate
    reference = cache["net"][("gold", "gold_sentiment")]
    current = cache["net"][(method, sentiment_source)]
    return float(np.abs(current[shared] - reference[shared]).mean())


def _net_sentiment_by_destination(
    frame: pd.DataFrame, methods: list[str], min_clauses: int
) -> dict:
    """Fast NumPy equivalent of the profile tables, used inside the bootstrap.

    Returns the net sentiment per destination for every attribution × sentiment
    source, and a boolean mask of the shared destinations (enough clauses under
    the gold reference and under every method). Matches ``profile_errors`` and
    ``common_destinations`` exactly; see tests/test_sentiment.py.
    """
    n_dest, n_sent = len(DESTINATION_LABELS), len(SENTIMENT_LABELS)
    dest_index = {label: i for i, label in enumerate(DESTINATION_LABELS)}
    sent_index = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
    pos, neg = sent_index["Positive"], sent_index["Negative"]
    net, counts = {}, {}
    for attribution in ["gold", *methods]:
        # Map labels to integer codes; anything outside the label set (e.g.
        # "Unclear") becomes NaN and is excluded, mirroring the profile filter.
        d = (
            frame[ATTRIBUTION_SOURCES[attribution]]
            .map(dest_index)
            .to_numpy(dtype=float)
        )
        for source, column in SENTIMENT_SOURCES.items():
            s = frame[column].map(sent_index).to_numpy(dtype=float)
            ok = ~np.isnan(d) & ~np.isnan(s)
            # Build a destination × sentiment count table in one pass by
            # encoding each (d, s) pair as a single flat index d * n_sent + s.
            table = np.bincount(
                (d[ok] * n_sent + s[ok]).astype(int), minlength=n_dest * n_sent
            ).reshape(n_dest, n_sent)
            n = table.sum(axis=1)  # clauses per destination
            # Destinations with zero clauses yield NaN (0/0) silently.
            with np.errstate(invalid="ignore", divide="ignore"):
                net[(attribution, source)] = (table[:, pos] - table[:, neg]) / n
            # Per-destination clause counts for this attribution source
            # (overwritten by the last sentiment source in the loop).
            counts[attribution] = n
    # A destination is shared only if the gold reference and every method
    # attribute at least ``min_clauses`` clauses to it.
    shared = counts["gold"] >= min_clauses
    for method in methods:
        shared &= counts[method] >= min_clauses
    return {"net": net, "shared": shared}


def default_bootstrap_statistics(
    methods: list[str], min_clauses: int = 5
) -> dict[str, Callable]:
    """The statistics reported by ``scripts/run_sentiment.py``.

    Each value is a function taking a (resampled) DataFrame and returning a
    float or None. All net-sentiment statistics share one cache so the
    destination profiles are computed only once per replicate.
    """
    cache: dict = {}
    stats: dict[str, Callable] = {
        "vader_accuracy": lambda f: float(
            (f["vader_label"] == f["sentiment_label"]).mean()
        ),
        "vader_macro_f1": lambda f: float(
            f1_score(
                f["sentiment_label"],
                f["vader_label"],
                labels=SENTIMENT_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }
    for method in methods:
        col = ATTRIBUTION_SOURCES[method]
        # Default arguments (col=col, m=method, s=source) bind the current loop
        # values; without them every lambda would see the last loop value.
        stats[f"{method}_joint_accuracy"] = lambda f, col=col: float(
            (
                (f[col] == f["destination_label"])
                & (f["vader_label"] == f["sentiment_label"])
            ).mean()
        )
        for source in SENTIMENT_SOURCES:
            stats[f"{method}_net_sentiment_error|{source}"] = (
                lambda f, m=method, s=source: _mean_net_error(
                    f, m, s, min_clauses, methods, cache
                )
            )
    # Paired differences (BERT minus rule): because both terms are computed on
    # the same replicate, their interval reflects the paired comparison.
    # Positive joint-accuracy difference favours BERT; positive error
    # difference favours the rule baseline (lower error is better).
    if {"rule", "bert"}.issubset(methods):
        stats["bert_minus_rule_joint_accuracy"] = lambda f: (
            stats["bert_joint_accuracy"](f) - stats["rule_joint_accuracy"](f)
        )
        for source in SENTIMENT_SOURCES:

            def _diff(f, s=source):
                b = stats[f"bert_net_sentiment_error|{s}"](f)
                r = stats[f"rule_net_sentiment_error|{s}"](f)
                # Propagate "undefined" if either side is undefined.
                return None if b is None or r is None else b - r

            stats[f"bert_minus_rule_net_sentiment_error|{source}"] = _diff
    return stats


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
# Categorical slots 1-3 of the reference chart palette, validated for colour-vision
# deficiency separation on a light surface. Values are also printed on the bars.
_SOURCE_COLORS = {"gold": "#2a78d6", "rule": "#eb6834", "bert": "#1baf7a"}
_SOURCE_NAMES = {
    "gold": "Gold destinations",
    "rule": "Rule baseline",
    "bert": "Contextual BERT",
}


def save_net_sentiment_figure(
    profiles: pd.DataFrame, methods: list[str], output: str | Path
) -> None:
    """Grouped bars of net sentiment per destination, one panel per sentiment source.

    Left panel uses gold sentiment labels, right panel VADER labels. Within each
    panel, every destination gets one bar per attribution source. Bar labels show
    the value, with "*" for low support and "n=0" when no clauses were attributed.
    """
    import matplotlib.pyplot as plt  # lazy import: plotting is optional

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = ["gold", *methods]
    panels = [
        ("gold_sentiment", "Gold sentiment labels"),
        ("vader_sentiment", "VADER sentiment labels"),
    ]
    # sharey keeps the two panels directly comparable.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    # Split 80% of each destination's slot evenly between the sources.
    width = 0.8 / len(sources)
    x = np.arange(len(DESTINATION_LABELS))
    for ax, (sentiment_source, title) in zip(axes, panels):
        for i, source in enumerate(sources):
            part = profiles[
                (profiles["attribution_source"] == source)
                & (profiles["sentiment_source"] == sentiment_source)
            ]
            # Reindex so bars always appear in the canonical destination order.
            part = part.set_index("destination").reindex(DESTINATION_LABELS)
            values = part["net_sentiment"].to_numpy(dtype=float)
            # Offset each source's bars symmetrically around the tick position.
            positions = x + (i - (len(sources) - 1) / 2) * width
            # NaN (no clauses) is drawn as a zero-height bar and labelled "n=0".
            bars = ax.bar(
                positions,
                np.nan_to_num(values),
                width * 0.92,
                color=_SOURCE_COLORS[source],
                label=_SOURCE_NAMES[source],
                edgecolor="#fcfcfb",
                linewidth=1,
            )
            for bar, value, n, low in zip(
                bars, values, part["n_clauses"], part["low_support"]
            ):
                text = "n=0" if n == 0 else f"{value:+.2f}" + ("*" if low else "")
                y = bar.get_height()
                # Place the label above positive bars and below negative ones.
                ax.annotate(
                    text,
                    (bar.get_x() + bar.get_width() / 2, y),
                    xytext=(0, 3 if y >= 0 else -3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if y >= 0 else "top",
                    fontsize=7,
                    color="#52514e",
                )
        # Zero line, light horizontal grid behind the bars, and no top/right spines.
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_xticks(x, DESTINATION_LABELS, rotation=20, ha="right")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # Net sentiment is bounded in [-1, 1]; extra margin leaves room for labels.
    axes[0].set_ylabel("Net sentiment (positive share − negative share)")
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(
        "Destination-level net sentiment by attribution source (* = fewer clauses than the minimum)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)  # free memory when generating many figures


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------
def save_json(payload: dict, path: str | Path) -> None:
    """Write JSON with NaN/inf converted to null, so the file is always valid JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False makes json.dumps raise if any NaN slipped past _clean.
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _clean(value):
    """Recursively convert NumPy types to plain Python and non-finite floats to None."""
    if isinstance(value, dict):
        return {
            str(k): _clean(v) for k, v in value.items()
        }  # JSON keys must be strings
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value
