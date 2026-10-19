# Responsible AI considerations

## Stakeholders and use context

The artifact is designed for researchers or analysts studying destination attribution in cruise-review text. It is a research prototype, not a deployed customer-feedback system. Misattribution could distort destination-level summaries or lead readers to believe a comment concerns a port when it concerns shipboard services.

## Transparency versus correctness

The rule baseline is inspectable because its decisions are explicit and traced. This does not guarantee that its rules reflect the intended meaning of every clause. BERT may use broader context, but its token attributions are approximate post-hoc explanations. Neither method should be treated as self-validating.

## Data limitations

The corpus is synthetic and relatively small. Results may not generalize to real reviews, different cruise lines, different ports, multilingual content, or changes in writing style. Annotation uncertainty is important: destination attribution can depend on context, and the supplied labels should be checked before being treated as gold truth. Author-name fields were removed from the bundled raw review corpus because they are not needed for this task; review IDs are used only for grouping and joining. One malformed/truncated raw JSONL line was omitted rather than reconstructed.

## Evaluation limitations

- Review-level split avoids clause leakage, but a single split can be noisy on a small dataset. A future study should use repeated group splits or grouped cross-validation and confidence intervals.
- The five-class task is imbalanced (6 `Multiple Destinations` clauses in the test set); per-class support and macro-F1 accompany accuracy, and low-support classes are flagged. Class-weighted training is available but off by default.
- The rule abstention label has no gold support in the current annotations; report coverage with metrics.
- Integrated Gradients depends on model, tokenization, and baseline.
- Token deletion may create out-of-distribution text; perturbation results are suggestive, not causal proof.
- Punctuation stability is a narrow proxy and does not substitute for human-validated meaning-preserving paraphrases.
- The sentiment analysis relies on VADER, a general-purpose lexicon that does not model sarcasm, domain-specific phrasing, or which entity an opinion is about, and cannot output Mixed by default. Destination-level sentiment summaries inherit both attribution errors and VADER errors; report both.
- No human-subject study is conducted. Claims about human understandability or audit utility require a future user study or independent reviewer protocol.

## Appropriate claims

State which method performed better on this particular held-out synthetic corpus and what the inspected explanations showed. Destination-level sentiment results describe this synthetic corpus and this sentiment scorer; they are not evidence about how real passengers feel about these ports. Avoid claims that one method is universally more trustworthy, that attribution maps reveal internal reasoning, or that a small sample establishes systematic bias. Keep model/data versions and analysis choices documented.
