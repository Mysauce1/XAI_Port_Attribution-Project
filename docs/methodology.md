# Methodology and decision protocol

## Research question

How do rule-based and contextual BERT approaches compare in predictive performance, interpretability, and explainability when attributing cruise-review statements to destinations?

## Unit of analysis and labels

The unit is the annotated clause. Destination labels are Roatan, Costa Maya, Cozumel, Multiple Destinations, and General Cruise. `Unclear` is a prediction-only abstention output for the rule baseline, not a gold class in the current annotation file. The target is destination attribution. Gold sentiment labels are used only in the separate sentiment analysis described below, never as an input to either attribution method.

## Data split

Assign a single split to each review ID, then inherit that split for all clauses in the review. Use train for model fitting, validation for checkpoint selection, and test once for final reporting. The same split file is shared by both methods. A group split prevents leakage of a review's distinctive phrasing across train and test, although it does not eliminate all distributional uncertainty.

## Rule-based method

The forward-window method uses explicit port detection, a finite context window (the rest of the mention sentence plus a configured number of following sentences), topic-transition overrides, multiple-destination handling, and abstention. Each prediction records detected ports, context before/after, distance, method, topic transition, and a plain-language decision reason. Context is not dependent on sentiment polarity. No backward-context fallback is enabled.

Rules are deterministic but not automatically correct. The explicit trace tells an auditor which rule fired; it does not prove the rule's assumption was appropriate for a particular clause.

## Contextual BERT

Fine-tune `distilbert-base-uncased` for the five destination labels. The input marks the target clause and includes the clause's sentence and adjacent sentence context when available. Select checkpoints on validation macro-F1. Class-weighted training (inverse training-class frequency) is available as an opt-in option; if used, it is chosen by comparing validation macro-F1, never test results. Report test accuracy, balanced accuracy, macro-F1, weighted-F1, per-class metrics, and confusion matrix. Save predicted class probabilities for confidence/error inspection.

## Explanation methods

- **Rule trace:** intrinsic explanation of the implemented decision path.
- **Integrated Gradients:** post-hoc token attributions for the predicted class logit. Report the target class, tokenization, baseline choice, integration steps, and convergence delta where available.
- **Perturbation diagnostic:** compare score changes after removing top-attribution tokens versus low-attribution tokens. Deletion artifacts are a limitation.
- **Stability probe:** compare predictions and probability vectors after punctuation normalization. This is a limited first probe; meaning preservation requires manual review.

## Comparison and case analysis

Compare the methods on identical held-out clauses. Categorize each as both correct, rule only correct, BERT correct while the rule abstained, BERT correct while the rule was wrong, both wrong with a rule abstention, or both wrong with an incorrect rule label. Review examples with the rule trace and BERT attribution/perturbation evidence. Treat recurring patterns as descriptive findings, not proof of broad systematic bias.

## Reporting requirements

Report class supports and coverage, not only aggregate accuracy. Flag classes with fewer than 10 held-out gold clauses as low-support and present their per-class scores as highly uncertain. Include the split seed, model version, model settings, software versions, annotation provenance, and whether any examples were manually reviewed. Distinguish observed results from hypotheses and interpretation. Do not describe post-hoc attributions as a literal transcript of model reasoning.

## Class imbalance

Destination classes are imbalanced (`Multiple Destinations` is about 2% of clauses). Macro-F1 is the primary metric, alongside balanced accuracy, weighted F1, and per-class scores with support. Row-normalized confusion matrices make rare-class behavior visible. The split is a review-level group split and is not stratified; it is not regenerated to fix thin classes. Low-support classes are reported as a limitation.

## Secondary analysis: sentiment on top of attribution

This analysis asks what each attribution method means downstream for destination-level sentiment. It does not change either method.

- **Sentiment model.** VADER (Hutto & Gilbert, 2014), a fixed lexicon-and-rules scorer applied to `clause_text`. Compound score ≥ 0.05 is Positive, ≤ −0.05 is Negative, otherwise Neutral (the conventional cut-offs). VADER has no Mixed output; an optional rule labels Mixed when both the positive and negative proportions reach a threshold, which must be chosen on train/validation data.
- **Reference.** The gold `sentiment_label` annotations, over the four-class taxonomy (Positive, Negative, Neutral, Mixed).
- **Evaluation unit and split.** Clauses in the held-out test split, matching the attribution evaluation.
- **Measures.** VADER-vs-gold accuracy, macro-F1 (four-class, with a three-class view for transparency), and per-class scores; destination-by-sentiment profiles under gold, rule, and BERT attribution; the absolute error in net sentiment (positive share − negative share) relative to the gold-destination, gold-sentiment profile; and joint destination-and-sentiment accuracy.
- **Fair comparison.** Mean profile errors are averaged over destinations that have at least five clauses under the gold labels and every compared method, so methods are compared on the same destinations. Rule abstentions are excluded from rule profiles and reported through coverage.
- **Uncertainty.** A review-level cluster bootstrap with paired resamples gives percentile intervals for each measure and for BERT-minus-rule differences.
