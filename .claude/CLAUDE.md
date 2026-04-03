# CLAUDE.md — Real-Time In-Game Chat Sentiment Analysis Pipeline

## Project Overview

This is an academic research project building an end-to-end pipeline for real-time in-game chat sentiment analysis:

**Live game chat (screen capture) → OCR text extraction → sentiment/intent classification (fine-tuned DistilBERT) → live overlay + sentiment timeline visualization**

This is a semester-long team project (1–3 people) worth 30% of the final grade, split across a proposal presentation, final presentation, and written report.

---

## Dataset: CONDA (Weld et al., 2021, ACL-IJCNLP)

- **Source**: Dota 2 in-game chat, English only
- **Size**: 44,869 utterances across 1,921 matches
- **Annotation**: Dual-level (we use utterance-level only — single-task, not dual-task)

### Intent Labels (Utterance Level) — Our Target

| Label | Meaning              | Example                    |
|-------|----------------------|----------------------------|
| E     | Explicit toxicity    | "ur trash noob report"     |
| I     | Implicit toxicity    | "not a good pudg"          |
| A     | Action               | "pausing", "gg reporting"  |
| O     | Other / neutral      | "push top", "nice"         |

### Slot Labels (Token Level) — NOT used in this project

Six classes: T (toxicity word), S (game slang), C (character name), D (Dota term), P (pronoun), O (other). We do NOT use these — our task is single-task intent classification only.

### Class Imbalance

The dataset is heavily skewed toward class O (other/neutral). E and especially I are minority classes. This is a known challenge — address with class-weighted loss (`CrossEntropyLoss(weight=...)`), NOT synthetic oversampling or paraphrase augmentation.

### Known Dataset Limitations

- The 60-second conversation boundary is methodologically arbitrary — does not reflect true conversational threading
- Non-English messages are excluded entirely, introducing bias
- The [SEPA] token merging of consecutive same-user messages can dilute toxicity signals
- ~45K utterances is relatively small by modern NLP standards — risk of overfitting on minority classes

---

## Model: DistilBERT-CONDA

Always refer to the model as **"fine-tuned DistilBERT"** or **"DistilBERT-CONDA"** to distinguish from the arXiv 2025 paper's cross-domain approach (which trained on Wikipedia comments, not in-domain game chat).

### Architecture

- 6-layer transformer encoder, distilled from BERT
- 768-dimensional hidden space, 12 attention heads
- 40% less memory, ~2x inference speed vs full BERT
- Retains 97% of BERT's language understanding
- [CLS] token → classification head → one of four intent labels (E/I/A/O)

### Game Time Token (Novel Contribution)

We prepend a normalized game time token to each input: `[TIME=0.84]`

- Provides temporal context as a substitute for conversational history
- Supported by CONDA's own temporal findings and Märtens et al.'s kill-event toxicity patterns
- Implementation decisions still needed: normalization strategy, binning vs continuous, tokenizer handling

---

## Pipeline Architecture

```
Stage 1: Screen Capture    → OpenCV, fixed ROI, frame differencing for new messages
Stage 2: OCR Extraction    → Tesseract v4 / EasyOCR, OpenCV preprocessing
Stage 3: Text Preprocessing → Light cleaning (see rules below)
Stage 4: DistilBERT-CONDA  → Intent classification with confidence scores
Stage 5: Visualization     → Bounding boxes, sentiment labels, timeline chart
```

---

## Preprocessing Rules

### What to do:
- Expand contractions ("I'm" → "I am")
- Retain emoticons — they carry sentiment signal
- Keep game slang as-is ("gg", "ez", "afk" are semantically meaningful)
- Remove non-UTF8 characters
- Filter messages under 2 alphanumeric tokens

### What NOT to do:
- Do NOT aggressively lowercase — "EASY GAME" vs "easy game" carries different emotional weight (though DistilBERT's tokenizer lowercases by default with uncased models, so decide on cased vs uncased)
- Do NOT strip all punctuation — "???" and "!!!" carry signal
- Do NOT over-clean slang — it IS the data, not noise

### Data Augmentation Policy:
- OCR noise simulation (character swaps, deletions, substitutions) → YES, defensible
- Class balance augmentation or vocabulary expansion → NO, not well-justified
- Paraphrase augmentation for minority classes → NO, risk of artifacts

---

## Five Novel Contributions (Differentiators)

1. **Live screen capture + OCR layer** — no prior work does real-time capture-to-classification
2. **Game time as temporal context** — `[TIME=0.84]` token prepended to inputs
3. **Single-task intent classification** — not dual-task like CONDA's original setup
4. **End-to-end latency measurement** — benchmarking the full pipeline speed
5. **Live visualization output** — sentiment timeline + bounding box overlays

---

## Related Work (Key Papers)

| Paper | Key Contribution | Relevance |
|---|---|---|
| Weld et al. (2021) — CONDA | Dual-level annotated Dota 2 dataset (E/I/A/O + T/S/C/D/P/O) | Our primary dataset |
| Märtens et al. (2015) | Lexicon-based toxicity, kill-event triggers, winner/loser patterns | Supports temporal analysis argument |
| Ghosh (2021) | Word2Vec / Bag-of-Words on game chat | Shows evolution from classical ML |
| arXiv 2025 benchmarking paper | DistilBERT + LLMs on gaming chat, 94.3% F1 | Trained on Wikipedia — our in-domain approach is the contribution |
| Sanh et al. (2019) — DistilBERT | Knowledge distillation from BERT | Model architecture reference |

---

## Evaluation Strategy

- **Primary metric**: F1-score (per-class, not just macro — accuracy will lie due to class imbalance)
- **Additional**: Precision, Recall, Confusion matrix
- **Testing approaches**: Held-out test set + real pipeline testing (OCR → model, measuring error propagation)
- **Latency**: Measure end-to-end pipeline speed (capture → classification)

---

## Code Conventions

- Python 3.10+
- PyTorch for model training
- Hugging Face Transformers for DistilBERT
- OpenCV for screen capture and image preprocessing
- Tesseract / EasyOCR for OCR
- Keep code modular — one script/module per pipeline stage
- Use descriptive variable names, comment non-obvious decisions

---

## Project Structure (Target)

```
project/
├── CLAUDE.md
├── .claudeignore
├── data/
│   ├── raw/              # Original CONDA files
│   ├── processed/        # Cleaned, split data
│   └── augmented/        # OCR noise augmented data (if used)
├── src/
│   ├── preprocessing/    # Data loading, cleaning, splitting
│   ├── model/            # DistilBERT training, evaluation
│   ├── pipeline/         # Screen capture, OCR, inference
│   └── visualization/    # Overlay rendering, timeline charts
├── notebooks/            # Exploration, EDA, experiments
├── results/              # Metrics, confusion matrices, plots
└── presentations/        # Slide content and diagrams
```