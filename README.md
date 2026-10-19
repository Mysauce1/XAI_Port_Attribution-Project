# Which Port Is It Anyway? Rules vs BERT

**An explainability audit of how a transparent rule system and a fine-tuned BERT decide which cruise port a review clause is about.**

Myla Simmons · Explainable AI · Duke University

**Repository:** [github.com/Mysauce1/XAI_Port_Attribution-Project](https://github.com/Mysauce1/XAI_Port_Attribution-Project) · **Dataset construction:** [github.com/Mysauce1/processed_cruise_data_setup](https://github.com/Mysauce1/processed_cruise_data_setup)

> *"Roatan was stunning but we had a bad experience with a local taxi driver… **It left a bit of a sour taste in my mouth.**"*
>
> The bold sentence names no port. Only context says it is about Roatan.

Cruise reviews move between ports, the ship and the trip as a whole, often within a single paragraph. Before anyone can say how guests feel about a port, each clause has to be credited to the right place. This project compares two ways of doing that step, a transparent forward-window rule system and a fine-tuned DistilBERT classifier. It then audits whether BERT's post-hoc explanations reflect what the model actually relies on.

**Research question.** How do a transparent rule system and a fine-tuned BERT decide which destination a clause is about, and are BERT's explanations faithful to what it actually uses?

The question breaks down into four parts:

1. **Performance.** How accurate is each method on the same held-out clauses, including the rare classes and the clauses the rules decline to label?
2. **Decision process.** What evidence does each method use? The rules expose a decision trace; BERT is explained with Integrated Gradients.
3. **Explanation faithfulness and stability.** Does deleting the tokens BERT's explanations rank highest actually change its prediction more than deleting random tokens? Does a small punctuation edit change the prediction?
4. **Downstream effect (secondary).** Do attribution errors change the destination-level sentiment that a cruise line would read?

**Contribution.** This is a reproducible, side-by-side explainability audit. It pairs every rule decision trace with BERT's Integrated Gradients attributions on the same clauses, tests those attributions for faithfulness, and uses review-level bootstrap intervals to quantify uncertainty. The audit is useful even where neither method clearly wins, because it shows *why* each one fails.

> **Terminology.** This README uses **explainability** throughout. The rule system is *explainable by design*: every prediction carries a trace naming the rule that fired, the active port and how far back it was set. BERT is a black-box model explained *post hoc* with Integrated Gradients. "Transparent" refers to the rule system and "post-hoc explanation" to BERT's attributions.

## Contents

- [Why this matters](#why-this-matters)
- [Results at a glance](#results-at-a-glance)
- [What this repository contains](#what-this-repository-contains)
- [Data](#data)
- [Methods](#methods)
- [Setup and usage](#setup-and-usage)
- [Outputs](#outputs)
- [Class imbalance safeguards](#class-imbalance-safeguards)
- [Responsible AI considerations and limitations](#responsible-ai-considerations-and-limitations)
- [Future work](#future-work)
- [Repository layout](#repository-layout)
- [References](#references)

## Why this matters

Cruise lines decide which ports and itineraries to invest in partly from guest reviews, especially the sentiment in them. Clause-level attribution is the first step of that analysis, and errors there are silent. A complaint credited to the wrong port makes that port look worse and the real one look better. If the attribution step is a black box, an analyst cannot tell *why* a port looks good or bad.

The people affected include:

- **Itinerary planners and analysts** at cruise lines, who read aggregated sentiment;
- **Port communities and local businesses** (taxi drivers, excursion operators, vendors), whose reputations and income can depend on those summaries;
- **Travellers**, whose reviews are summarised on their behalf.

The project asks a question that applies beyond cruises: when a transparent system and a more flexible neural model perform similarly, what does each one's explanation let you check, and can the neural model's explanation be trusted?

## Results at a glance

These results come from the run reported on the poster. That run used the 207-clause test set (20 reviews), window size 2, and a single class-weighted DistilBERT model trained with seed 42 and selected on validation macro-F1. Re-running on different hardware can shift BERT's numbers slightly.

**Accuracy is a statistical tie; the rules win on macro-F1.**

| Test set (207 clauses) | Rule | BERT |
|---|---|---|
| Accuracy | 0.821 | 0.787 |
| Macro-F1 | **0.748** | 0.610 |
| Balanced accuracy | 0.687 | 0.624 |
| Coverage (answers given) | 0.879 | **1.000** |
| Accuracy when it answers | **0.934** | 0.787 |

BERT − Rule, 95% paired review-level bootstrap intervals:

| Metric | Difference [95% CI] |
|---|---|
| Accuracy | −0.03 [−0.08, +0.01] (interval includes 0: no detectable difference) |
| Macro-F1 | −0.14 [−0.22, −0.06] |
| Balanced accuracy | −0.06 [−0.14, −0.00] |
| Accuracy when answering | −0.15 [−0.19, −0.10] |

The rule system's precision comes from abstaining on 12% of clauses. Agreement on the 207 test clauses was 149 both correct, 21 rule only, 14 BERT only and 23 both wrong.

**Key findings**

1. **Opposite failures in the same review.** In review 242, right after "Roatan was beautiful," a shipboard sentence correctly triggers the rule's ship-topic rule. That rule clears the active port, so every later Roatan clause abstains. BERT gets those later clauses right, but it also labels the shipboard sentence as Roatan, following the nearest port name across a topic change.
2. **BERT's reach is shorter.** On clauses the rule resolved by inheritance, both methods score about 0.9 when the port was named 0–1 sentences back (n = 70). When the port was named 2 sentences back (n = 22), the rule scores 0.91 and BERT 0.41, because the port name has left BERT's ±1-sentence input. This explains 13 of the 21 clauses only the rule got right.
3. **Rule failures are traceable.** All 25 rule abstentions trace to four causes:
   - a ship topic erased the port (8);
   - the window expired (8);
   - a lead-in came before the port was named (6);
   - generic talk about "the ports" (3).

   Keyword misfires are just as easy to diagnose. In "The buffet lunch was cold chicken and rice," gold Roatan for a beach excursion, the word "buffet" fired the ship rule.
4. **BERT's attributions pass the deletion test.** Deleting the top Integrated Gradients (IG) tokens drops the predicted-class probability by 0.476, against 0.022 for random tokens and −0.002 for the lowest-IG tokens. Top − random = +0.45 [0.42, 0.49] over all 207 test clauses. The top token is usually a piece of a port name ("costa", "##mel"), and in 89 of 207 clauses it sits outside the clause itself. BERT's accuracy is 0.82 when a port name is in its input (n = 187) and 0.50 without one (n = 20).
5. **Reading ahead helps and hurts.** Unlike the rules, BERT also sees the *next* sentence, and the same habit cuts both ways:
   - **Helps.** In "Traveling as a couple, we preferred slower-paced days," it correctly predicts Cozumel (0.85) from the port named in the following sentence.
   - **Hurts.** In "The ports were a mixed bag," gold General Cruise, it predicts Costa Maya (0.73). The top attributed token is "costa" (+1.80), taken from the next sentence.

   Separately, with no port in view, BERT defaults to Costa Maya, the largest class. It predicts Costa Maya 76 times against 56 gold labels, which accounts for 22 of its 44 errors.
6. **Stability evidence is weak.** The punctuation probe changed only 46 inputs; 43 kept their label.
7. **Downstream sentiment barely moves on average.** Net sentiment is a destination's positive share minus its negative share (−1 to +1). Measured with gold sentiment labels, swapping gold destinations for either method's attributions moves a destination's net sentiment by at most 0.07. Swapping gold sentiment for VADER's labels, with gold destinations, moves it by up to 0.27. So the sentiment scorer adds more error than the attribution step. These averages cover only destinations with at least 5 attributed clauses under gold, rule and BERT, which leaves out Multiple Destinations. Two spots need care:
   - **Complaints depend more on context.** 63% of negative comments rely on context to place them, against 51% of positive ones.
   - **Rare comparisons swing wildly.** Of 6 comments comparing ports, the rule caught 1 (score +1.00) and BERT none (no score), while the gold score was +0.33.

## What this repository contains

- **Transparent rule baseline.** Explicit port detection, a bounded forward context window measured in sentences, topic-transition rules, multiple-port handling, abstention, and a per-clause decision trace (`attribution_method`, `decision_reason`, active port, context distance).
- **Contextual BERT.** Fine-tuned `distilbert-base-uncased`. The input is a marked target clause plus its sentence and one sentence before and after. Class-weighted training is optional and is chosen on validation data.
- **Comparable evaluation.** Both methods are scored on identical held-out clauses from a frozen review-level split. Reports include:
  - accuracy, balanced accuracy, and macro and weighted F1;
  - per-class precision, recall, F1 and support;
  - coverage and selective accuracy;
  - raw and row-normalised confusion matrices;
  - low-support flags;
  - a paired review-level bootstrap for seven metrics.
- **Explainability audits.** Integrated Gradients token attributions with a convergence check and reruns, a deletion-based faithfulness probe (top vs random vs lowest attribution), and a punctuation stability probe.
- **Qualitative case groups.** Six agreement/disagreement groups (`both_correct`, `rule_only_correct`, `bert_correct_rule_abstained`, `bert_correct_rule_wrong`, `both_wrong_rule_abstained`, `both_wrong_rule_incorrect`) separate rule abstentions from rule errors.
- **Secondary sentiment analysis.** VADER clause sentiment compared with gold sentiment, plus destination-level sentiment profiles under gold, rule and BERT attribution, with review-level bootstrap intervals. It reads saved predictions and never changes either attribution method.
- **Reproducibility.** A frozen split file that later scripts refuse to recreate, saved training metadata, fixed seeds, unit tests and a GitHub Actions test workflow.

## Data

### Source and preparation

The dataset was built in a separate repository: **[processed_cruise_data_setup](https://github.com/Mysauce1/processed_cruise_data_setup)**. That repository contains:

- `src/sample_reviews.py`, which draws a length-stratified random sample (seed 42) of 130 reviews from a 600-review corpus;
- `src/build_clauses.py`, which splits each review into sentences and clauses with spaCy and gives each one a stable ID (`{review_id}-S{nnn}-C{nn}`);
- `src/make_annotation_template.py`, which produces the blank annotation sheet;
- `ANNOTATION_GUIDELINES.md`, the labelling rules used to produce the gold labels, with worked examples and edge cases;
- `schemas/dataset_schemas.md`, the field definitions for every dataset file.

The reviews are **synthetic**. They were generated with GPT-3 and modelled on r/royalcaribbean posts about a Western Caribbean itinerary: Roatán (Honduras), and Costa Maya and Cozumel (Mexico). The 130 sampled reviews produced **1,233 annotated clauses**.

In this repository:

- `data/raw/cruise_reviews.jsonl` is a sanitised copy of the corpus. Author-name fields were removed, and one malformed/truncated JSONL line was omitted rather than reconstructed.
- `data/processed/annotated_reviews.csv` holds the 1,233 clause-level annotations.
- `data/processed/review_splits.csv` is the frozen review-level split: about 70/15/15 by review, with every clause from a review in one partition. The test set is 207 clauses from 20 reviews and is used once, for final reporting.

### Labels

| Field | Values used in this project | Used by |
|---|---|---|
| `destination_label` | Roatan, Costa Maya, Cozumel, Multiple Destinations, General Cruise | Both attribution methods (gold target) |
| `sentiment_label` | Positive, Negative, Neutral, Mixed | Only the separate sentiment analysis, as VADER's reference; never an input to either attribution method |
| `Unclear` | Prediction-only abstention | Rule baseline output; counted in confusion matrices and coverage |

### How the annotation guidelines map to this project

The guidelines in the data repository and the code here use slightly different conventions:

- **Label spelling.** The guidelines specify lowercase snake-case values (`multiple_destinations`, `general_cruise`, `positive`). This project uses Title Case. On load, `cruise_xai.data` normalises the guideline values to Title Case (e.g. `multiple_destinations` → `Multiple Destinations`). It also corrects two `Positve` typos, and the stored CSV is normalised too.
- **`unclear` as a gold label.** The guidelines allow `unclear` as a gold destination when context cannot settle it. The evaluation taxonomy here has five gold destination labels, and `Unclear` appears only as the rule system's abstention.
- **Port aliases.** The guidelines count aliases as explicit port mentions: West End, West Bay, Coxen Hole and Mahogany Bay for Roatan; Mahahual for Costa Maya; San Gervasio and San Miguel for Cozumel. The rule system detects only the three port names (accepting the accented spelling Roatán). An alias-only clause therefore looks unanchored to the rules, while BERT can learn aliases from training data.
- **Carry-forward.** The guidelines (§3.3) carry a port forward until the reviewer names another port, moves to the cruise in general, or changes subject. They set no distance limit. The rule system approximates this with a fixed window of 2 following sentences, which is the source of the "window expired" abstentions.
- **Comparisons.** The guidelines label any clause comparing in-scope ports as Multiple Destinations, and a comparison favouring one port as Mixed sentiment. The rule system needs two named ports to predict Multiple Destinations. VADER has no Mixed output.
- **Evidence type.** Following the guidelines, annotations record whether a port is named in the clause (`explicit`), inferred from surrounding text (`contextual`) or undeterminable (`insufficient`). This field is not an input to either method.
- **Segmentation.** Clause boundaries come from the data repository's heuristic splitter, which has two documented quirks: an "and" rule that rarely fires, and splitting on adverbial "yet". Both methods use the boundaries as given.
- **Guideline version.** The guidelines are a working draft (version 0.1) with some rules marked `[confirm]`. Record which version was used.

**Verify annotation provenance and adjudicate questionable labels before treating the annotations as definitive gold truth.** No inter-annotator agreement figures are reported in this repository. The guidelines recommend Cohen's kappa on a second annotator's sample.

## Methods

### Rule system (transparent)

For each clause, in order:

1. Detect explicit port mentions.
2. Two or more ports in a comparative construction → `Multiple Destinations`.
3. Exactly one port → that port, and start (or restart) its context window.
4. Two or more ports without a comparison → `Multiple Destinations`.
5. Clearly shipboard topic → `General Cruise`, ending any inherited port context.
6. General cruise or itinerary wording without port-activity words → `General Cruise`.
7. Otherwise, inherit the active port if the clause is within the window: the rest of the mention sentence plus the configured number of following sentences (`--window-size`, default 2).
8. Abstain as `Unclear` when none of the above applies.

Context only flows forward: a port named earlier is carried to later clauses, but the rules never look ahead, so a clause that comes before its port is named (a lead-in) abstains. Sentiment plays no part in the decision. Every prediction records which rule fired and why (`attribution_method`, `decision_reason`, `active_port_before`, `active_port_after`, `context_distance`, `topic_transition`). The implementation is in `src/cruise_xai/rules.py`. A transparent rule is not automatically correct, so read the traces. Choose the window size on the training split, not on test results.

### Contextual BERT

- **Model.** `distilbert-base-uncased`, fine-tuned on the training reviews.
- **Input.** The target clause, marked as `TARGET:`, plus its sentence and the previous and next sentences. So BERT sees ±1 sentence of context: one sentence back and one ahead. The rules see the opposite shape: a port can be carried up to 2 sentences forward from where it was named, but nothing ahead of the clause is used.
- **Selection.** The checkpoint and options are selected on **validation macro-F1 over all five destination classes**. The test set is never used for selection.
- **Reported model.** The class-weighted loss was chosen on validation macro-F1: 0.646, against 0.612 for the unweighted model. See [Choosing unweighted vs class-weighted BERT](#choosing-unweighted-vs-class-weighted-bert).
- **Confidence scores.** `bert_confidence` and the `prob_*` columns are raw softmax scores, **not calibrated probabilities**.

### Evaluation

Both methods are scored on the same 207 test clauses. Coverage and selective accuracy are reported alongside accuracy and F1, so the rule system's abstention is visible rather than hidden.

The paired bootstrap resamples whole reviews, with all their clauses, and scores both methods on the same resample, using 2,000 resamples. It reports intervals for each method and for the BERT-minus-rule difference. These intervals reflect review-sampling uncertainty, not variation across BERT training seeds.

### Explainability audits (BERT)

- **Integrated Gradients** (via Captum) gives token-level attributions toward the predicted class. Clauses flagged by a relative convergence check are rerun with more integration steps (64 by default). The initial results are kept, and the rerun is used for that clause's deletion probe.
- **Deletion faithfulness** deletes the top-positive IG tokens and compares the drop in predicted-class probability with deleting the same number of random tokens (averaged over 5 draws) and the lowest-attribution tokens. The primary comparison uses only clauses that have a positive-attribution token. Its interval comes from a review-cluster bootstrap.
- **Punctuation stability** normalises punctuation and spacing and checks whether the predicted label stays the same. Agreement is reported only over clauses where the edit actually changed the text, along with the exclusion reasons.

IG is a post-hoc attribution method, not a literal transcript of model reasoning. Its results depend on the baseline, tokenisation and model. Token deletion produces unnatural text, so the deletion probe is diagnostic evidence, not a causal test.

### Secondary sentiment analysis

VADER (Hutto & Gilbert, 2014) scores each clause, and the compound score is mapped to Positive (≥ 0.05), Negative (≤ −0.05) or Neutral. VADER has no Mixed output, so with default settings it can never predict the Mixed class. The analysis then:

1. compares VADER with gold sentiment over all four classes;
2. builds a sentiment profile for each destination (the share of each sentiment, and net sentiment = positive share − negative share) under gold, rule and BERT attribution, using both gold and VADER sentiment labels;
3. measures each profile's error against the reference profile, built from gold destinations and gold sentiment;
4. reports joint accuracy: the share of clauses where a method's destination **and** VADER's sentiment both match gold;
5. computes review-level bootstrap intervals for these statistics.

The reference comparison separates two sources of error:

| Profile | Error it measures |
|---|---|
| Rule or BERT destinations + gold sentiment | Attribution error alone |
| Gold destinations + VADER sentiment | VADER's error alone |
| Rule or BERT destinations + VADER sentiment | End-to-end error of a fully automatic pipeline |

Rule clauses labelled `Unclear` are not attributed to any destination. Mean profile errors are averaged only over destinations with at least 5 attributed clauses under gold **and** every method, so all methods are compared on the same destinations. Excluded destinations are listed in `sentiment_summary.json`.

## Setup and usage

### Requirements

- **Python 3.11** (the version used in CI). Python 3.10+ is likely to work.
- `requirements.txt`: core packages for data handling, rules, metrics and tests.
- `requirements-bert.txt`: PyTorch, Transformers (pinned to `4.46.3`, the tested API version) and Captum, for BERT training and the explainability audits.
- `requirements-sentiment.txt`: `vaderSentiment`, for the sentiment analysis only.
- `requirements-optional.txt`: only for the legacy clause-building utilities in `src/cruise_xai/source_utils.py`. These also need spaCy's English model: `python -m spacy download en_core_web_sm`.
- **Internet access on first BERT run.** The pretrained `distilbert-base-uncased` weights (about 250 MB) are downloaded from Hugging Face and cached. No model weights are committed to this repository.
- **GPU optional.** A GPU speeds up training, but the default run (3 epochs, batch size 8) works on CPU. If the default PyTorch wheel is unsuitable for your platform, install a CPU-only or CUDA build from the official PyTorch installation selector.

### 1. Get the code and create a virtual environment

Clone the repository and move into it. These two commands are the same in PowerShell, Command Prompt, Git Bash and macOS/Linux terminals, and they need [Git](https://git-scm.com/downloads):

```bash
git clone https://github.com/Mysauce1/XAI_Port_Attribution-Project.git
cd XAI_Port_Attribution-Project
```

Without Git, open [the repository page](https://github.com/Mysauce1/XAI_Port_Attribution-Project), choose **Code → Download ZIP**, unzip it, and open a terminal in the unzipped folder. GitHub usually names that folder `XAI_Port_Attribution-Project-main`.

Run every command after this from the repository root. Pick the row for your operating system and terminal:

| OS | Terminal | Create the environment | Activate it |
|---|---|---|---|
| Windows | PowerShell | `py -3.11 -m venv .venv` | `.venv\Scripts\Activate.ps1` |
| Windows | Command Prompt (cmd) | `py -3.11 -m venv .venv` | `.venv\Scripts\activate.bat` |
| Windows | Git Bash (bash) | `py -3.11 -m venv .venv` | `source .venv/Scripts/activate` |
| macOS | Terminal (zsh or bash) | `python3.11 -m venv .venv` | `source .venv/bin/activate` |
| Linux | bash | `python3.11 -m venv .venv` | `source .venv/bin/activate` |

PowerShell may block the activation script. If it does, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

**Bash, full setup.** This works in Git Bash on Windows and in bash or zsh on macOS and Linux:

```bash
# Skip these two lines if you already cloned the repository above.
git clone https://github.com/Mysauce1/XAI_Port_Attribution-Project.git
cd XAI_Port_Attribution-Project

# Create the environment. If `python` is not found, use `py -3.11` (Windows)
# or `python3.11` / `python3` (macOS/Linux).
python -m venv .venv

# Activate it: Windows puts the script in .venv/Scripts, macOS/Linux in .venv/bin.
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

# Confirm the environment's Python is the one in use.
python --version

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-bert.txt        # needed for BERT and the explainability audit
pip install -r requirements-sentiment.txt   # needed for the sentiment analysis
```

In every terminal, the `pip install` and `python scripts/...` commands below are the same once the environment is active. They are written on single lines so they work in PowerShell, cmd and bash alike.

### 2. Prepare and inspect the data

```bash
python scripts/prepare_data.py
python scripts/describe_data.py
```

- `prepare_data.py` validates the annotations and creates `data/processed/review_splits.csv` the first time. On later runs it reports that it reused the existing file, and `--seed` is ignored. It prints destination and sentiment labels by split and warns about split/class cells with fewer than 10 clauses.
- `describe_data.py` writes label counts by split, low-support cells (`outputs/tables/data_quality_summary.json`) and a label-distribution figure. Inspect them before continuing.

Every later script stops with an error if the split file is missing, so no step can silently create a different split. Don't regenerate the split after seeing test results unless you're deliberately starting a new experiment.

### 3. Run the rule baseline

```bash
python scripts/run_rules.py --window-size 2
```

This writes predictions with full traces for every clause, held-out metrics, per-class metrics, and raw and row-normalised confusion matrices. `src/forward_window.py` is a legacy entry point for the same baseline; use `scripts/run_rules.py`, which refuses to run without the frozen split.

### 4. Train BERT

A single training run looks like this:

```bash
python scripts/train_bert.py --epochs 3 --batch-size 8 --seed 42
```

This trains on the train split, selects the best epoch on validation macro-F1, and saves the result to `outputs/bert/`. Training needs the Hugging Face download the first time. A run saves the model plus checkpoints, which can take more than 1 GB of disk.

To retrain into the same folder, add `--overwrite`. It deletes the previous run in `--output-dir` *before* training starts, so if training fails the old model is gone too. It refuses to delete a folder that doesn't look like a BERT training run. Copy any model you want to keep first.

#### Choosing unweighted vs class-weighted BERT

The destination classes are imbalanced. For example, only 30 of 1,233 clauses are Multiple Destinations. `--class-weighting balanced` weights the training loss by inverse class frequency, using the same formula as scikit-learn's `class_weight="balanced"`. The default, `none`, is the standard unweighted loss. Whether weighting helps is an empirical question, so make the choice on **validation** data, before the test set is touched:

1. **Train both settings into separate folders**, with everything else identical:

   ```bash
   python scripts/train_bert.py --class-weighting none --output-dir outputs/bert_unweighted --epochs 3 --batch-size 8 --seed 42
   python scripts/train_bert.py --class-weighting balanced --output-dir outputs/bert_balanced --epochs 3 --batch-size 8 --seed 42
   ```

2. **Compare validation macro-F1.** Each run records it as `best_validation_metric` in its `training_metadata.json`:

   ```bash
   python -c "import json; [print(d, json.load(open(f'outputs/{d}/training_metadata.json'))['best_validation_metric']) for d in ('bert_unweighted', 'bert_balanced')]"
   ```

3. **Pick the setting with the higher validation macro-F1.** Decide your tie-break rule before you look; keeping the simpler unweighted model is a sensible default. In the reported run, the balanced model won (0.646 vs 0.612), so it was used for every result above.
4. **Evaluate only the chosen model on test.** Pass its `model/` folder to every later step:

   ```bash
   python scripts/evaluate_bert.py --model-dir outputs/bert_balanced/model
   python scripts/run_xai_audit.py --model-dir outputs/bert_balanced/model --n-examples 1000 --ig-steps 32
   ```

   Don't run `evaluate_bert.py` on both models and keep the better test score. That turns the test set into a selection set and overstates performance.
5. **Use the same setting for any repeated-seed run.** Pass `--class-weighting balanced` (or `none`) to `run_bert_seeds.py`.
6. **Record the decision**: both validation scores, the chosen setting and the date. The class weights used are already saved in `training_metadata.json`.

If you'd rather keep the default paths (`outputs/bert/model`), retrain the chosen setting into `outputs/bert` after deciding. Then `evaluate_bert.py` and `run_xai_audit.py` work without `--model-dir`. Keep the same seed and settings so the model matches the one you validated.

#### Keep the model and its training metadata together

```text
outputs/bert_balanced/          (or outputs/bert/)
├── model/
│   ├── config.json
│   ├── tokenizer files and model weights
│   └── label_map.json
└── training_metadata.json
```

Always pass the **`model/` folder** as `--model-dir`; `training_metadata.json` must sit in its parent folder. Prediction and the explainability functions read the training `max_length` from that file. If you copy only `model/`, they raise `FileNotFoundError`. In that case, restore the metadata file or pass the same `max_length` explicitly through the Python API. A different truncation length can change predictions and explanations.

`training_metadata.json` records:

- the training length, class counts, class weighting and class weights;
- the selected checkpoint, validation metric and evaluation history;
- the seed and library versions;
- the run's start and finish times and the exact command line.

### 5. Evaluate BERT and compare the methods

```bash
python scripts/evaluate_bert.py --model-dir outputs/bert_balanced/model
```

This writes BERT's predictions and held-out metrics, then aligns them clause by clause with the rule predictions. It produces:

- the comparison groups;
- the paired review-level bootstrap;
- the side-by-side metrics table and figure.

Run it with the same `--output-dir` (default `outputs`) as `run_rules.py`, because it reads the rule predictions from there.

### 6. Run the explainability audit

```bash
python scripts/run_xai_audit.py --model-dir outputs/bert_balanced/model --n-examples 1000 --ig-steps 32
```

`--n-examples` larger than the test set audits every test clause, which is what the poster reports (207). Auditing all of them avoids questions about which clauses were sampled. A smaller number audits a random sample, fixed by `--seed`. The model is loaded once and reused for every clause. Before interpreting the results, review the tokenisation and the edited text.

### 7. Run the sentiment analysis (secondary)

```bash
python scripts/run_sentiment.py
```

Run this after `run_rules.py` and, optionally, `evaluate_bert.py`. It only reads saved predictions and gold annotations, and it skips any method whose prediction file is missing. It analyses the test split by default (`--split`). `--mixed-min` optionally lets VADER output Mixed when its positive and negative proportions both reach that value; choose it on train or validation data only.

### 8. Optional: repeated-seed BERT runs

> ⚠️ **Computationally expensive.** `run_bert_seeds.py` trains and saves a **complete model for every seed**. Each run writes model weights and training checkpoints to `outputs/bert_multi_seed/seed_<N>/`. Five seeds can use **several GB of disk space**, plus substantial RAM or GPU memory while training, and on CPU the full run can take hours.
>
> Before running it:
> - check your free disk space;
> - try two seeds first (`--seeds 42 43`);
> - once the metrics are written, delete each seed's checkpoint folders if you don't need them.

```bash
python scripts/run_bert_seeds.py --seeds 42 43 44 45 46 --class-weighting balanced
```

This reuses the frozen split and writes per-seed test metrics, plus a mean and standard deviation summary, to `outputs/bert_multi_seed/`. It shows how much BERT's score varies with the seed alone. **Don't use these test results to pick a "best" seed.** The poster reports a single seed; this script is for checking that limitation.

### 9. Run the tests

```bash
pytest -q
```

The tests cover:

- label normalisation, split integrity and the split-file guard;
- the rule baseline: context carry-forward, window expiry, ship-topic transitions, port comparisons and abstention;
- metrics and low-support flags, and the destination bootstrap;
- class-weight computation and the BERT data helpers;
- the explainability helpers;
- the sentiment layer.

The sentiment tests use a stand-in scorer, so they don't need `vaderSentiment`. GitHub Actions runs the tests on Python 3.11 with `requirements.txt` only. The BERT and explainability helper tests are skipped there because they need PyTorch; they run locally once `requirements-bert.txt` is installed.

### Reproducing the poster, end to end

```bash
python scripts/prepare_data.py
python scripts/describe_data.py
python scripts/run_rules.py --window-size 2
python scripts/train_bert.py --class-weighting none --output-dir outputs/bert_unweighted --seed 42
python scripts/train_bert.py --class-weighting balanced --output-dir outputs/bert_balanced --seed 42
# compare validation macro-F1 (step 4), then evaluate only the chosen model:
python scripts/evaluate_bert.py --model-dir outputs/bert_balanced/model
python scripts/run_xai_audit.py --model-dir outputs/bert_balanced/model --n-examples 1000 --ig-steps 32
python scripts/run_sentiment.py
```

**Suggested reading order for the outputs:**

1. Label support by split.
2. Rule traces for a sample of correct and incorrect clauses.
3. The comparison groups.
4. IG examples with high and low attribution.
5. The sentiment summary.
6. Every result with its interval and low-support flags.

## Outputs

Metrics and figures are generated locally and are not committed to the repository. The poster's figures come from these files.

| Output | What it is for |
|---|---|
| `outputs/rule_predictions_all.csv` | Every rule decision with its full trace |
| `outputs/rule_test_cases.csv` | Test clauses with the rule's trace, for error analysis |
| `outputs/rule_metrics.json`, `rule_metrics_per_class.csv` | Held-out rule metrics, including `low_support_classes` |
| `outputs/bert_predictions_all.csv` | BERT predictions and (uncalibrated) class scores |
| `outputs/bert_metrics.json`, `bert_metrics_per_class.csv` | Held-out BERT metrics |
| `outputs/figures/*_confusion_matrix.png`, `*_confusion_matrix_normalized.png` | Confusion matrices for each method, raw and row-normalised |
| `outputs/method_comparison_test.csv` | The same test clauses, both predictions and a correctness group |
| `outputs/tables/comparison_group_counts.csv` | Counts for the six agreement/abstention groups |
| `outputs/tables/method_metrics_comparison.csv` | Side-by-side headline metrics with low-support classes |
| `outputs/tables/paired_review_bootstrap.csv` / `.json` | Per-method and BERT-minus-rule intervals for seven metrics |
| `outputs/tables/data_quality_summary.json` | Label counts by split and low-support cells |
| `outputs/explanations/integrated_gradients_tokens.csv` | Token attributions and convergence diagnostics |
| `outputs/explanations/integrated_gradients_reliability_reruns.csv` | Higher-step IG reruns for flagged clauses (only if any) |
| `outputs/explanations/faithfulness_perturbation.csv` | Top, random and lowest-attribution deletion results |
| `outputs/explanations/stability_punctuation_probe.csv` | Punctuation probe per clause, with eligibility reasons |
| `outputs/explanations/xai_audit_summary.json` | Explainability audit summary, including the top − random bootstrap interval |
| `outputs/sentiment/sentiment_summary.json` | Sentiment settings, VADER-vs-gold metrics, joint accuracy, profile errors and notes |
| `outputs/sentiment/destination_sentiment_profiles.csv`, `..._profile_errors.csv` | Sentiment shares per destination for each attribution source, and their errors |
| `outputs/sentiment/sentiment_bootstrap_intervals.csv` | Review-level intervals for the sentiment statistics |
| `outputs/sentiment/figures/destination_net_sentiment.png` | Net sentiment per destination for each attribution source |
| `outputs/bert_multi_seed/` | Optional per-seed metrics and summary |

## Class imbalance safeguards

There are only 30 Multiple Destinations clauses in total, and the test set has 6 Multiple Destinations clauses and 3 Mixed-sentiment clauses. The project handles this in three places:

- **Measurement.** Macro-F1 (every class weighted equally) is the primary metric and the BERT selection metric. Balanced accuracy, weighted F1 and per-class scores with support are always reported. Classes with fewer than 10 gold clauses are listed as low-support, and the scripts print a warning.
- **Visibility.** Row-normalised confusion matrices make rare classes as readable as common ones. `prepare_data.py` and `describe_data.py` flag split/class cells below 10 clauses.
- **Training.** Class weighting is opt-in and is chosen on validation data (see above).

The split is grouped by review but not stratified by class. It is deliberately **not** regenerated to fix thin classes, because re-splitting after seeing results would break the experimental protocol. Low-support classes are reported as a limitation instead. A future experiment could use scikit-learn's `StratifiedGroupKFold`, which balances classes while keeping reviews together.

## Responsible AI considerations and limitations

**Limitations of the evidence**

- **Small, synthetic data.** The corpus is 130 GPT-3-generated reviews from one three-port itinerary. The results do not show that the methods generalise to real guest reviews, other itineraries or other travel domains. The test set has only 6 Multiple Destinations clauses (rule 1/6 correct, BERT 0/6), so per-class results for rare classes are highly uncertain.
- **Unequal context.** The rules carry a port up to 2 sentences forward but never look ahead. BERT sees ±1 sentence: one back and one ahead. Each method therefore has evidence the other lacks. The comparison is between the two systems *as configured*, not between rules and transformers in general.
- **One BERT seed.** The reported BERT results come from a single training seed. The bootstrap intervals reflect test-sampling uncertainty, not training variability; `run_bert_seeds.py` exists to measure that.
- **Explanations are post hoc.** IG attributions are not causal, and the deletion probe creates unnatural text. A faithful attribution shows what the model *uses*, not whether that use is *correct*: BERT faithfully relying on the nearest port name is exactly what causes its topic-change errors.
- **Narrow stability check.** The punctuation probe changed only 46 inputs. There is no human-validated paraphrase suite.
- **General-purpose sentiment scorer.** VADER is not tuned to cruise reviews and cannot output Mixed, so its errors (up to 0.27 in net sentiment) are larger than the attribution effects being measured. Its errors belong to VADER, not to either attribution method.
- **Imperfect clause boundaries.** A few segments still join two thoughts but carry one label, and alias-only port mentions are invisible to the rules.
- **Annotation.** The labels follow a draft guideline (v0.1). No inter-annotator agreement is reported here, so treat the gold labels as one careful annotator's reading, not ground truth.
- **No interactive explorer.** This version has no interactive explorer; results are files and figures.

**Failure modes to watch for**

- **Rules: cascade.** A shipboard sentence clears the active port, so every later clause about that port abstains.
- **Rules: keyword misfires.** For example, "buffet" in a beach-excursion clause fires the ship rule.
- **Rules: no look-ahead.** A lead-in clause that comes before the port is named cannot be attributed.
- **BERT: topic changes.** It follows the nearest port name, before or after the clause, even across a change of topic.
- **BERT: class default.** With no port in view, it defaults to the majority class (Costa Maya).
- **BERT: confident errors.** It is often confidently wrong, and its scores are not calibrated.

**Misuse risks and deployment considerations**

- **Reputational harm through misattribution.** Port-level sentiment summaries can affect local businesses, such as taxi drivers and tour operators, who are never asked. Misattributed complaints hurt the wrong community. Negative comments rely on context more often than positive ones, which makes complaints especially exposed.
- **Over-trusting explanations.** An IG highlight on "Roatan" can look like a justification, but it only shows what the model used. Pair BERT outputs with human review, especially for rare categories like port comparisons.
- **Treating abstention as failure.** The rule system's 12% abstention is a feature for auditability, since a human can review exactly those clauses. Hiding coverage behind a single accuracy number misrepresents both methods.
- **Privacy, for any real-data extension.** Real reviews contain personal details; the synthetic corpus here had author names removed. Any extension to real reviews needs consent and terms-of-service review and de-identification.

More detail is in `docs/methodology.md` and `docs/responsible_ai.md`.

## Future work

- Test on a larger set of **real guest reviews** with more multi-port clauses.
- Give both methods the **same 2-sentence lookback**, so differences reflect the method rather than the context it sees.
- Train BERT over **several seeds** and report the spread.
- Check IG against a **second attribution method**, and add a human-validated paraphrase suite for stability.
- **Refine clause splitting**, fixing the "and" rule and adverbial "yet", and add port aliases to the rules before re-labelling.
- Measure **inter-annotator agreement** and freeze a versioned gold standard.

## Repository layout

```text
XAI_Port_Attribution-Project/
├── configs/project.yaml          # documents the settings used by the scripts
├── data/
│   ├── raw/cruise_reviews.jsonl  # sanitised synthetic corpus
│   └── processed/
│       ├── annotated_reviews.csv # 1,233 gold-labelled clauses
│       └── review_splits.csv     # frozen review-level split
├── docs/
│   ├── methodology.md            # method details and design decisions
│   ├── responsible_ai.md         # stakeholders, risks and limitations
│   ├── references.md             # full reference list
│   └── GITHUB_SETUP.md           # publishing and CI notes
├── outputs/                      # generated reports and figures (not committed)
├── scripts/
│   ├── prepare_data.py           # validates annotations; creates or reuses the split
│   ├── describe_data.py          # label distributions and data-quality checks
│   ├── run_rules.py              # rule baseline and held-out evaluation
│   ├── train_bert.py             # fine-tunes DistilBERT (optionally class-weighted)
│   ├── evaluate_bert.py          # BERT test evaluation and rule-vs-BERT comparison
│   ├── run_xai_audit.py          # Integrated Gradients, deletion and stability probes
│   ├── run_sentiment.py          # secondary VADER sentiment analysis
│   └── run_bert_seeds.py         # optional repeated-seed runs (expensive)
├── src/
│   ├── forward_window.py         # legacy CLI entry point for the rule baseline
│   └── cruise_xai/
│       ├── data.py               # loading, label normalisation, splits, BERT input text
│       ├── rules.py              # forward-window rule system with decision traces
│       ├── bert_model.py         # training, prediction, class weights
│       ├── explainability.py     # Integrated Gradients, deletion and stability probes
│       ├── metrics.py            # metrics, low-support flags, confusion matrices
│       ├── bootstrap.py          # paired review-level bootstrap
│       ├── pipeline.py           # runs and compares both methods
│       ├── sentiment.py          # secondary VADER sentiment layer
│       └── source_utils.py       # legacy clause-building utilities (from the data repo)
├── tests/                        # unit tests (pytest)
├── requirements.txt              # core: data, rules, metrics, tests
├── requirements-bert.txt         # BERT training and explainability
├── requirements-sentiment.txt    # VADER sentiment analysis
└── requirements-optional.txt     # legacy clause-building utilities
```

Dataset construction (sampling, clause splitting and the annotation template) and the annotation guidelines live in **[processed_cruise_data_setup](https://github.com/Mysauce1/processed_cruise_data_setup)**.

## References

- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*. (Integrated Gradients)
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv:1910.01108*.
- Kokhlikyan, N., et al. (2020). Captum: A unified and generic model interpretability library for PyTorch. *arXiv:2009.07896*.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206–215.
- Jacovi, A., & Goldberg, Y. (2020). Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness? *ACL*.
- DeYoung, J., et al. (2020). ERASER: A benchmark to evaluate rationalized NLP models. *ACL*. (deletion-based faithfulness)
- Adebayo, J., et al. (2018). Sanity checks for saliency maps. *NeurIPS*.

See `docs/references.md` for the full list.
