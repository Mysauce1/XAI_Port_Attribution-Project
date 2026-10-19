"""Transparent forward-window destination attribution with auditable traces.

Each review clause is assigned to a cruise port, to "General Cruise"
(shipboard / whole-trip topics), to "Multiple Destinations", or to "Unclear"
(abstention). Every prediction carries a trace explaining which rule fired and
what context was active, so individual decisions can be audited.

Core idea: when a clause explicitly names a single port, that port becomes the
"active" context and is carried forward to later clauses that do not name a
port, but only for a fixed number of following sentences (``window_size``).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Any

# --- Keyword patterns -------------------------------------------------------
# All patterns are case-insensitive and anchored on word boundaries (\b) so
# that, e.g., "ship" does not match inside "relationship".

# Port name patterns. "Roatan" accepts the accented spelling (Roatán), and
# "Costa Maya" tolerates any whitespace between the two words.
PORT_PATTERNS = {
    "Roatan": re.compile(r"\broat[aá]n\b", re.I),
    "Costa Maya": re.compile(r"\bcosta\s+maya\b", re.I),
    "Cozumel": re.compile(r"\bcozumel\b", re.I),
}

# Comparative cues. When a clause names two or more ports together with one of
# these, it is treated as an explicit comparison between destinations.
COMPARISON_RE = re.compile(
    r"\b(better than|worse than|compared (?:with|to)|versus|vs\.?|while|whereas|more than|less than|unlike|both|between)\b",
    re.I,
)

# Shipboard / onboard vocabulary. A match signals the reviewer has moved from a
# port to the ship itself, so any inherited port context is dropped.
SHIP_TOPIC_RE = re.compile(
    r"\b(ship|onboard|on board|wifi|wi-fi|cabin|stateroom|buffet|dining room|pool deck|theater|theatre|entertainment|crew|room service|casino|embarkation|disembarkation)\b",
    re.I,
)

# Whole-trip vocabulary. Weaker than SHIP_TOPIC_RE: it only counts as
# "General Cruise" when no port-related vocabulary is also present.
GENERAL_CRUISE_RE = re.compile(r"\b(cruise|itinerary|boarding)\b", re.I)

# Shore / destination vocabulary. Used only to block GENERAL_CRUISE_RE (e.g.
# "the cruise excursion" stays destination-related rather than General Cruise).
PORT_TOPIC_RE = re.compile(
    r"\b(port|island|beach|snorkel|snorkeling|reef|shore|taxi|driver|tour|excursion|ruins|waterfall|town|village|market|water|shoreline|scooter|dock|pier|guide|diving|dived|swim|swimming|restaurant|local|vendor)\b",
    re.I,
)


@dataclass
class RuleResult:
    """Prediction for one clause plus the trace that explains it."""

    clause_id: str
    review_id: str
    clause_text: str
    predicted_label: (
        str  # Port name, "General Cruise", "Multiple Destinations" or "Unclear"
    )
    detected_ports: str  # Ports named in this clause, ";"-separated (empty if none)
    active_port_before: (
        str  # Context port before this clause was processed ("" if none)
    )
    active_port_after: str  # Context port after this clause was processed ("" if none)
    context_distance: (
        int | None
    )  # Sentences since the explicit port mention (0 = this sentence); None when not applicable
    context_window_status: str  # "active" or "inactive" after this clause
    attribution_method: str  # Short ID of the rule that fired
    decision_reason: str  # Human-readable explanation of the decision
    topic_transition: str  # Kind of topic change this clause caused, or "none"
    abstained: bool  # True when the label is "Unclear"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ForwardWindowAttributor:
    """Rule-based attribution with finite context and explicit decision precedence.

    Precedence: (1) explicit port mention / explicit multi-port comparison,
    (2) clear ship/general-cruise topic, (3) bounded contextual carry-forward,
    (4) abstain as Unclear. Neutral clauses can establish destination context;
    sentiment is intentionally not used to control destination attribution.
    """

    def __init__(self, window_size: int = 2):
        # window_size = number of sentences AFTER an explicit port mention that
        # may inherit that port. 0 means context never carries forward.
        if window_size < 0:
            raise ValueError("window_size must be >= 0")
        self.window_size = window_size

    @staticmethod
    def _ports(text: str) -> list[str]:
        """Return every known port named in ``text``, in PORT_PATTERNS order."""
        return [
            port
            for port, pattern in PORT_PATTERNS.items()
            if pattern.search(text or "")
        ]

    def predict_review(self, frame) -> list[dict[str, Any]]:
        """Attribute every clause of a single review, processing them in reading order.

        Context (the active port and its window) is carried from clause to
        clause, so this must only be called with clauses from one review.
        """
        # Restore reading order. The stable sort keeps the original order for
        # ties. Note: if the IDs are strings, ordering is lexicographic
        # ("10" sorts before "2"), so zero-padded or numeric IDs are assumed.
        ordered = frame.sort_values(["sentence_id", "clause_id"], kind="stable")
        # The window is measured in sentence positions, not clause positions.
        # Multiple clauses from the same sentence share one sentence index.
        # ngroup(sort=False) numbers sentences 0, 1, 2, ... in the order above.
        ordered["_sentence_position"] = ordered.groupby(
            "sentence_id", sort=False
        ).ngroup()

        # Context state carried across clauses:
        #   active_port - port most recently named on its own (None if no context)
        #   window_end  - last sentence position that may inherit active_port
        active_port: str | None = None
        window_end: int | None = None
        results: list[dict[str, Any]] = []

        for _, row in ordered.iterrows():
            sentence_position = int(row["_sentence_position"])
            text = str(row.get("clause_text", "") or "")
            ports = self._ports(text)
            before = active_port or ""

            # Defaults: abstain. Each rule below overwrites these if it fires.
            label = "Unclear"
            method = "abstention"
            reason = "No explicit destination or reliable in-window context"
            transition = "none"
            # window_end - window_size is the sentence where the port was named,
            # so this is how many sentences we are past that mention.
            distance = (
                sentence_position - (window_end - self.window_size)
                if window_end is not None
                else None
            )

            # Expire context only after the explicit-port sentence plus the
            # configured number of following sentences has been passed.
            if window_end is not None and sentence_position > window_end:
                active_port, window_end = None, None
                before = ""
                distance = None

            # Rules are checked in precedence order; the first match wins.
            # Explicit mentions update context regardless of sentiment.

            # (1a) Two or more ports in a comparative construction, e.g.
            # "Cozumel was better than Roatan". Context is cleared because the
            # following clauses could refer to either port.
            if len(ports) >= 2 and COMPARISON_RE.search(text):
                label = "Multiple Destinations"
                method = "explicit_multi_port_comparison"
                reason = "Multiple ports are named in a comparative construction"
                active_port, window_end = None, None
                transition = "multiple_destination_comparison"

            # (1b) Exactly one port named: label it and (re)start the window
            # from this sentence. Record a switch if a different port was active.
            elif len(ports) == 1:
                label = ports[0]
                distance = 0
                method = "explicit_port_mention"
                reason = f"Explicit destination mention: {ports[0]}"
                if active_port and active_port != ports[0]:
                    transition = f"destination_switch:{active_port}->{ports[0]}"
                active_port, window_end = ports[0], sentence_position + self.window_size

            # (1c) Several ports named without a comparison cue (e.g. a list of
            # stops). Treated the same way as 1a but traced separately.
            elif len(ports) >= 2:
                distance = None
                # Multiple named ports without comparison: avoid guessing which one
                # receives the opinion; keep it explicit as a multi-destination case.
                label = "Multiple Destinations"
                method = "multiple_port_mentions"
                reason = "Multiple destinations named; no single destination selected"
                active_port, window_end = None, None
                transition = "multiple_destination_mention"

            # (2a) Shipboard topic with no port named: General Cruise. This beats
            # carry-forward, so "the buffet was great" right after a Cozumel
            # sentence is not attributed to Cozumel. Context is cleared.
            elif SHIP_TOPIC_RE.search(text):
                distance = None
                label = "General Cruise"
                method = "ship_topic_transition"
                reason = "Shipboard or onboard topic overrides inherited port context"
                transition = (
                    "destination_to_ship_or_onboard_topic"
                    if active_port
                    else "ship_or_onboard_topic"
                )
                active_port, window_end = None, None

            # (2b) Whole-trip wording ("the cruise", "the itinerary") with no
            # shore vocabulary: General Cruise, and context is cleared.
            elif GENERAL_CRUISE_RE.search(text) and not PORT_TOPIC_RE.search(text):
                distance = None
                label = "General Cruise"
                method = "general_cruise_keyword"
                reason = "General cruise or itinerary topic without destination-specific evidence"
                transition = "general_cruise_topic"
                active_port, window_end = None, None

            # (3) No explicit evidence, but a port is still in its window:
            # inherit it. The window is left unchanged, so carried-forward
            # clauses do not extend it.
            elif (
                active_port is not None
                and window_end is not None
                and sentence_position <= window_end
            ):
                label = active_port
                method = "bounded_forward_context"
                distance = sentence_position - (window_end - self.window_size)
                reason = f"Inherited {active_port} context within {self.window_size}-following-sentence window"
                transition = "context_carried_forward"

            # (4) No rule matched: the Unclear/abstention defaults set above stand.

            # Record the decision together with the context state before and after.
            result = RuleResult(
                clause_id=str(row.get("clause_id", "")),
                review_id=str(row.get("review_id", "")),
                clause_text=text,
                predicted_label=label,
                detected_ports=";".join(ports),
                active_port_before=before,
                active_port_after=active_port or "",
                context_distance=distance,
                context_window_status=(
                    "active" if active_port is not None else "inactive"
                ),
                attribution_method=method,
                decision_reason=reason,
                topic_transition=transition,
                abstained=(label == "Unclear"),
            )
            results.append(result.to_dict())
        return results

    def predict(self, frame) -> list[dict[str, Any]]:
        """Attribute all clauses in ``frame``, one review at a time.

        Context never crosses review boundaries because each review is
        processed independently by ``predict_review``.
        """
        outputs = []
        for _, review in frame.groupby("review_id", sort=False):
            outputs.extend(self.predict_review(review))
        return outputs
