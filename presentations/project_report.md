# Real-Time In-Game Chat Sentiment Analysis: An End-to-End Capture, OCR, and Classification Pipeline

**Course Project Report — Numerical Methods, Fall 2025**

---

## 1. Project Background

Online multiplayer games are one of the most active venues for real-time text communication, and also one of the most consistent sources of toxic and abusive language. In-game chat has unusual properties that make it a hard target for off-the-shelf sentiment models: utterances are short, full of game-specific slang and abbreviations ("gg", "ez", "afk", "mid diff"), heavily punctuated, and often deliberately misspelled to evade automated filters. Existing platform moderation is mostly based on keyword blocklists, which are reactive and brittle.

Our project builds an **end-to-end real-time pipeline** that takes a live game window as input and produces sentiment-labeled overlays as output:

> **Live game chat (screen capture) → OCR text extraction → intent classification (fine-tuned DistilBERT) → live overlay + sentiment labels**

The classifier is trained on the CONDA dataset (Weld et al., ACL-IJCNLP 2021), which is the only large publicly available dataset of in-game chat with utterance-level intent annotations. The pipeline is deployed and tested against live VALORANT gameplay.

The project has three concrete goals:

1. **Train a single-task intent classifier** (fine-tuned DistilBERT on CONDA) that achieves competitive macro-F1 despite severe class imbalance.
2. **Test whether normalized game time helps classification** by injecting a `[TIME=x.xx]` token at the front of each utterance — a practical substitute for the conversational history we cannot recover from screen capture alone.
3. **Build the full real-time pipeline**: screen capture, ROI-based OCR, model inference, and a transparent click-through overlay, with the engineering needed to run continuously without re-classifying static chat or re-reading the scoreboard every tick.

We treat the time-token experiment as an honest ablation: we report the result whether it helped or not.

---

## 2. Related Work

| Paper | Contribution | Relevance to our work |
|---|---|---|
| Weld et al. (2021) — *CONDA: A CONtextual Dual-Annotated Dataset for In-Game Toxicity Understanding and Detection* | 44,869 dual-annotated Dota 2 utterances (4 intent classes E/I/A/O × 6 slot classes) | Our primary dataset. We use the utterance-level intent labels only. |
| Märtens et al. (2015) — *Toxicity Detection in Multiplayer Online Games* | Lexicon-based classifier; established that toxicity is event-triggered (after deaths, late-game) and asymmetric between winners and losers | Direct motivation for trying a temporal feature: if toxicity has time structure, a time-aware model should benefit. |
| Ghosh (2021) — *Sentiment Analysis of In-Game Chat* | Word2Vec / Bag-of-Words baselines on game chat | Establishes the classical-ML floor that transformer models clear by a wide margin. |
| arXiv 2501 (2025) benchmarking paper | DistilBERT and LLMs on gaming-style toxicity, ~94% F1, but trained on Wikipedia comments | Motivates our **in-domain** training. Our contribution over this work is fine-tuning directly on CONDA rather than transferring from out-of-domain text. |
| Sanh et al. (2019) — *DistilBERT* | Knowledge distillation from BERT: 40% smaller, ~2× faster, retains ~97% of language understanding | Architectural choice. Real-time capture demands a small model; DistilBERT is the standard pick at this size. |

The closest prior work to ours is the arXiv 2025 benchmarking paper. The key methodological difference is that they train on out-of-domain text (Wikipedia toxicity) and evaluate on game chat, while we train on CONDA directly. To our knowledge, no prior work performs **screen capture → OCR → in-domain classification → live overlay** as an integrated real-time system.

---

## 3. Dataset

### 3.1 Source

CONDA contains 44,869 English utterances drawn from 1,921 Dota 2 matches. Every utterance carries a single intent label from {E, I, A, O}:

| Label | Meaning | Example |
|---|---|---|
| E | Explicit toxicity | "ur trash noob report" |
| I | Implicit toxicity | "not a good pudg" |
| A | Action | "pausing", "gg reporting" |
| O | Other / neutral | "push top", "nice" |

CONDA also provides token-level slot labels (T/S/C/D/P/O for toxicity word, slang, character, Dota term, pronoun, other). We do **not** use the slot labels — our task is single-task intent classification.

### 3.2 Splits

| Split | Utterances |
|---|---:|
| Train | 26,921 |
| Validation | 8,973 (after dropping one NaN row) |
| Test | 8,974 (provided unlabeled — unused for evaluation) |

Because the public test set has no labels released, we report all results on the validation split. This is the same convention used in subsequent CONDA work.

### 3.3 Class distribution

The training split is heavily imbalanced toward the neutral class:

| Class | Count | % of train | Loss weight |
|---|---:|---:|---:|
| A | 1,719 | 6.4 % | 3.915 |
| E | 3,528 | 13.1 % | 1.907 |
| I | 1,692 | 6.3 % | 3.977 |
| O | 19,982 | 74.2 % | 0.337 |

Weights are computed as `total / (num_classes × class_count)`, so a model that predicts O on every input would achieve ~74% accuracy but a macro-F1 of roughly 0.21. This makes macro-F1 the only honest top-line metric.

### 3.4 Preprocessing

The preprocessing pipeline (`model_tuning/preprocessing.ipynb`) performs:

1. **Column reduction.** We keep only `utterance`, `chatTime`, `intentClass`. We drop `Id`, `conversationId`, `playerId`, `matchId`, `playerSlot`, `slotClasses`, `slotTokens`.
2. **Time normalization.** `chatTime` is given in seconds since match start; pre-game lobby chat has negative values. We clip to `[-90, 3600]` seconds and rescale to `[0, 1]`:
   ```
   chatTime_norm = (clip(t, -90, 3600) - (-90)) / (3600 - (-90))
   ```
   This is the value injected into the `[TIME=x.xx]` token.
3. **`[SEPA]` removal.** CONDA concatenates consecutive same-user messages with a `[SEPA]` separator. We strip it because the model has no need to know the message boundary; concatenated utterances are still single classification units.
4. **Duplicate inspection.** We flag duplicate `(utterance, chatTime)` pairs (~425 in train) but do not remove them — short repeated phrases like "GG WP" are legitimate in-game communication.
5. **NaN handling.** One NaN utterance was dropped from validation.
6. **Label validation.** All labels confirmed in {A, E, I, O}.

In line with our preprocessing rules, we deliberately **do not** lowercase, do not strip punctuation, and do not remove slang or emoticons. Casing carries emotional weight ("EZ" vs. "ez"), repeated punctuation carries intensity ("???"), and slang *is* the data.

---

## 4. Method

### 4.1 Model

We fine-tune `distilbert-base-cased` with a four-way classification head on top of the `[CLS]` token embedding. DistilBERT is a 6-layer transformer, 768-dimensional hidden state, 12 attention heads, ~66M parameters. It retains roughly 97% of BERT's language understanding at ~40% the size and ~2× the inference speed (Sanh et al., 2019). This combination matters in our setting because the pipeline must keep up with live chat at sub-second latency on commodity hardware.

We use the **cased** variant deliberately. Uncased models map "EZ" and "ez" to the same token; in toxic-chat detection, capitalization is signal, not noise.

The classification head is randomly initialized — the load report at the start of training confirms `pre_classifier` and `classifier` weights are missing from the pre-trained checkpoint and freshly initialized.

### 4.2 Game-time token (novel contribution)

Our principal methodological contribution at the model level is prepending a normalized game-time token to every utterance:

```
[TIME=0.84] ur trash noob report
```

The motivation: the screen-capture pipeline sees one chat line at a time and cannot reconstruct the conversational thread. Prior work (Märtens et al., 2015) showed that toxicity is concentrated late in matches and after specific in-game events. A normalized scalar in `[0, 1]` is a cheap proxy for "where in the match are we?" and was hypothesized to substitute for the missing context.

A practical implementation note worth stating clearly: `[TIME=0.84]` is **not** a registered special token. DistilBERT's WordPiece tokenizer splits it into subword pieces (`[`, `time`, `=`, `0`, `.`, `8`, `4`, `]`), so the model has to learn the temporal signal across roughly seven subword positions rather than from a single learned embedding. We chose this over `tokenizer.add_tokens()` + `model.resize_token_embeddings()` for simplicity and to keep the pre-trained vocabulary intact; we discuss the consequences in §6.

### 4.3 Class-weighted loss

We address class imbalance with weighted cross-entropy:

```python
CLASS_WEIGHTS = torch.tensor([3.915, 1.907, 3.977, 0.337])  # [A, E, I, O]
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
```

We deliberately **do not** use synthetic oversampling, paraphrase augmentation, or vocabulary-expansion augmentation. Paraphrase augmentation in particular tends to introduce stylistic artifacts that the model can latch onto, and the CONDA dataset is small enough (~45K utterances) that such artifacts would be visible in validation.

### 4.4 Training configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Schedule | Linear warmup (10%) → linear decay |
| Batch size | 32 |
| Max sequence length | 64 tokens |
| Epochs | 5 |
| Gradient clipping | max-norm 1.0 |
| Loss | Class-weighted cross-entropy |
| Random seed | 42 |
| Hardware | CUDA GPU |

Selection criterion is **best macro-F1 on the validation set**, evaluated after every epoch. A 64-token sequence length covers >98% of CONDA utterances; the small fraction of longer messages are typically copy-pasted spam.

### 4.5 Pipeline architecture

The deployment pipeline (`src/pipeline.py`) runs the following loop at a 0.4-second interval:

```
Stage 1: Screen capture     mss → OpenCV (BGRA → BGR), 4 ROIs
Stage 2: Frame differencing chat ROI absdiff < 0.5 → skip OCR entirely
Stage 3: OCR                EasyOCR (chat) + EasyOCR-digits (timer/score)
Stage 4: Time reconstruction (rounds × 100 + (100 − round_seconds)) → normalize
Stage 5: Classification     DistilBERT-CONDA, batched, with (text, time_bin) cache
Stage 6: Overlay            transparent click-through Tkinter window, colored bboxes
```

Several engineering decisions directly shape the system's real-time behavior:

- **Frame differencing** is the single largest cost saver. Chat is static for the vast majority of ticks; running OCR and the model on every frame would be wasteful. We skip the entire downstream stack when the chat ROI's mean per-pixel absdiff against the previous frame is below 0.5.
- **Decoupled metadata refresh.** The round timer and scoreboard are re-read at most every 8 seconds on the success path, with a 2-second cooldown after a parse failure. Because the `[TIME]` token is normalized over a ~30-minute match, a few seconds of timer drift is below the granularity that the model can resolve from the rounded `:.2f` representation.
- **`(text, time_bin)` prediction cache.** When the same utterance appears with the same rounded time, we re-use the cached label rather than re-running the model. A bounded LRU policy caps the cache at 512 entries.
- **ROI-based OCR.** The four ROIs (chat, time, won, lost) are pixel-defined in `src/game_info.json` for VALORANT at a fixed resolution. EasyOCR (GPU-aware) handles the chat region after a 4× upscale + grayscale + invert preprocessing pass; the digit ROIs use an allowlist (`0123456789` or `0123456789:`) and a binary threshold to suppress the round-timer's anti-aliasing.
- **Time reconstruction.** VALORANT rounds are 100 seconds. We compute total elapsed time as `(rounds_won + rounds_lost - 1) × 100 + (100 − round_seconds)` and feed that through the same `[-90, 3600] → [0, 1]` normalization used at training time.
- **Overlay.** A full-screen transparent Tkinter window with `WS_EX_LAYERED | WS_EX_TRANSPARENT` extended styles makes the overlay click-through, so it doesn't capture mouse input from the game. Labels are color-coded: E=red, I=orange, A=yellow, O=green.

### 4.6 Ablation design

To test the time-token contribution, we run two identical training runs that differ only in the `use_time_token` flag in the `CONDADataset` constructor:

- **with-time:** input = `"[TIME={chatTime:.2f}] {utterance}"`
- **no-time:** input = `"{utterance}"`

Everything else — seed, hyperparameters, data, hardware, selection criterion — is held fixed. Both checkpoints are evaluated on the same validation split with the same `evaluate()` script.

---

## 5. Experimental Results

### 5.1 Headline results (validation, single seed)

| Metric | no time | with [TIME] | Δ (with − no) |
|---|---:|---:|---:|
| UCA (accuracy) | 0.9145 | 0.9100 | **−0.0046** |
| Macro F1 | **0.8377** | 0.8309 | **−0.0067** |
| F1[E] Explicit | 0.8579 | 0.8521 | −0.0057 |
| F1[I] Implicit | 0.7421 | 0.7336 | −0.0085 |
| F1[A] Action | 0.8003 | 0.7906 | −0.0097 |
| F1[O] Other | 0.9504 | 0.9473 | −0.0030 |

**The time token did not help.** It produced a small but consistent regression on every metric and every class. The two largest drops are on class A (Action) and class I (Implicit toxicity) — the two least frequent classes, which are most sensitive to any input change.

This is a **negative result**, and we report it as such. In an unseeded earlier run the same direction held (with-time 0.8309 vs. no-time 0.8331), so the regression is reproducible across two paired runs at this seed.

### 5.2 Confusion structure (with-time, evaluation order E/I/A/O)

```
              pred:     E      I      A      O
true E  (1183):       1043     23     26     91
true I   (582):         23    420     17    122   ← 21% I → O
true A   (580):         18     10    487     65   ← 11% A → O
true O  (6628):        181    110    122   6215
```

Three patterns dominate:

1. **Implicit toxicity is the hardest class.** F1 ≈ 0.74. The dominant failure mode is I → O (21% of true-I utterances are predicted O). By definition, implicit toxicity uses neutral surface lexicon ("not a good pudg"), so it shares vocabulary with neutral chat. This is exactly the ambiguity CONDA's authors flagged.
2. **Action confuses with Other.** Calls like "pausing" and "gg reporting" overlap with neutral coordination, producing an 11% A → O leak.
3. **Explicit toxicity is the easiest class.** F1 ≈ 0.85. Explicit toxicity carries high-signal lexicon (slurs, "trash", "noob", "report"), and the cased model picks it up reliably.

The no-time confusion matrix shows the same three patterns slightly stronger in the model's favor — for example, I → O is 122 with-time vs. 118 no-time.

### 5.3 Training dynamics

Both runs follow the same trajectory across 5 epochs:

| Epoch | Train loss (no-time / with-time) | Val loss | Macro F1 (no-time / with-time) |
|---|---|---|---|
| 1 | 0.749 / 0.756 | 0.51 / 0.49 | 0.804 / 0.791 |
| 2 | 0.401 / 0.415 | 0.48 / 0.47 | 0.802 / 0.822 |
| 3 | 0.306 / 0.320 | 0.58 / 0.52 | 0.824 / 0.814 |
| 4 | 0.235 / 0.248 | 0.60 / 0.58 | 0.827 / 0.822 |
| 5 | 0.181 / 0.191 | 0.68 / 0.65 | **0.838** / **0.831** |

Two observations:

- **Validation loss rises after epoch 2** while macro-F1 keeps climbing through epoch 5. This is mild overfitting that nevertheless improves the F1-relevant decision surface — the loss penalty on confidently-correct O predictions grows even as the per-class boundaries sharpen. Because we select on macro-F1, not loss, this works in our favor.
- **Five epochs is at the edge.** Earlier stopping (epoch 3 or 4) would lose ~0.01 F1. We did not run beyond 5 epochs; both runs were peaking, but we cannot rule out further gains.

### 5.4 Why the time token did not help

The negative result has a few plausible explanations, none of which we can fully separate without further experiments:

1. **Token-budget cost.** `[TIME=0.84]` consumes roughly seven of the 64 available tokens for what amounts to one scalar. For short utterances (the majority), this is fine; for long ones, it crowds out actual content.
2. **Subword fragmentation.** Because we did not register `[TIME=...]` as a special token, the model has to learn the temporal signal across multiple subword positions and combine it into the `[CLS]` representation. A registered special token with a fresh learned embedding would have been a fairer test.
3. **Saturation by lexical signal.** DistilBERT-cased is already pulling 0.838 macro-F1 from text alone. The lexical cues that correlate with intent ("gg" near match-end, "report" anywhere) are largely already captured by the model — adding time as a redundant feature does not improve the decision and the small per-input perturbation is, on net, slightly harmful.
4. **Weak utterance-level temporal structure.** Märtens et al. found *event-triggered* toxicity (post-kill, post-objective), not a smooth function of match elapsed time. A normalized scalar in `[0, 1]` may not encode the right granularity — the moments that matter are seconds after a kill, not the minute of the match.

This is an honest ablation, and we believe the result is more informative than a naive positive: it suggests that for in-domain, lexically-strong tasks, scalar temporal features provided as inline text tokens contribute little or nothing.

### 5.5 End-to-end pipeline behavior

The deployed pipeline runs against live VALORANT gameplay with the following observed properties (qualitative, from `PIPELINE_PROFILE=1` runs):

- **Frame differencing** suppresses OCR on the large majority of ticks where chat is static.
- **Metadata reads** dominate the first tick after a chat change but are amortized over the 8-second refresh window.
- **Classification cache hit-rate** is high in practice because in-game chat repeats heavily ("gg", "ez", "nice").
- **Overlay updates** are inexpensive — Tkinter canvas redraws at our rate are not a bottleneck.

We did not finalize a quantitative end-to-end latency table; this is the most concrete piece of follow-up work.

---

## 6. Limitations

We list these openly because the project is best understood with them on the table:

- **Single seed.** The ablation is reported from one seed (42). The earlier non-seeded run reproduces the direction of the regression but the magnitude is not statistically established. Three-seed runs would let us report mean ± std and put the negative result on firmer ground.
- **Cross-domain deployment.** The model is trained on Dota 2 chat and deployed against VALORANT. Generic toxicity terms transfer cleanly; game-specific terms (hero names, agent names, ability names) do not. We did not quantify the cross-domain degradation.
- **OCR error propagation is unmeasured.** OCR character errors are a real input perturbation in deployment; the planned OCR-noise augmentation experiment was not run.
- **Time math is VALORANT-specific.** The 100-second-round assumption in `_parse_time` does not generalize to Dota 2 (continuous timer) or other games.
- **No special-token registration for `[TIME]`.** As discussed, fragmentation may have hurt the with-time condition more than necessary.
- **No cased-vs-uncased ablation.** We selected cased for sentiment-signal reasons but did not run the comparison.
- **Hardcoded ROIs.** Pixel coordinates are fixed for one resolution; deployment to a different resolution requires re-measuring the four ROIs.
- **No latency table.** The instrumentation exists (`PIPELINE_PROFILE=1`); the numbers do not.

---

## 7. Conclusion

We built and trained a fine-tuned DistilBERT classifier on the CONDA in-game chat dataset, achieving **macro-F1 = 0.838 / accuracy = 0.915** on the validation split. We then deployed it inside an end-to-end pipeline that captures live VALORANT gameplay, runs OCR on the chat and scoreboard ROIs, classifies each new utterance, and renders sentiment-coded overlays on the game screen in real time.

Our principal experimental contribution is an honest ablation of a normalized game-time token. The hypothesis — that match elapsed time, prepended as `[TIME=x.xx]`, would substitute for the conversational history that screen capture cannot recover — was not supported. The time-augmented model **regressed by 0.7 macro-F1** against the text-only baseline, with the largest losses on the rarest classes. We attribute the regression to a combination of token-budget cost, subword fragmentation of an unregistered token, redundancy with already-strong lexical signal, and the weak utterance-level temporal structure of CONDA itself.

The pipeline-level contributions — frame-differenced capture, decoupled metadata refresh, prediction caching, and a click-through overlay — are, to our knowledge, the first integrated real-time capture-to-overlay system for in-game chat sentiment.

The most useful next steps are: a multi-seed re-run with a registered `[TIME]` special token, an OCR-noise robustness study, and a quantitative latency profile of the deployed pipeline.

---

## Appendix A — Reproducibility

| Item | Path |
|---|---|
| Preprocessing | [model_tuning/preprocessing.ipynb](../model_tuning/preprocessing.ipynb) |
| Dataset class | [src/model/CONDA_dataset.py](../src/model/CONDA_dataset.py) |
| Training (with time) | [model_tuning/model.py](../model_tuning/model.py) |
| Training (no time) | [model_tuning/train_no_time.py](../model_tuning/train_no_time.py) |
| Evaluation (with time) | [src/model/evaluate.py](../src/model/evaluate.py) |
| Evaluation (no time) | [src/model/evaluate_no_time.py](../src/model/evaluate_no_time.py) |
| Live pipeline | [src/pipeline.py](../src/pipeline.py) |
| OCR | [src/ocr/capture_screen.py](../src/ocr/capture_screen.py) |
| Overlay | [src/visualization/overlay.py](../src/visualization/overlay.py) |
| Classifier inference | [src/model/toxicity_classifier.py](../src/model/toxicity_classifier.py) |
| Plot generation | [src/model/plot_results.py](../src/model/plot_results.py) |
| Results table | [src/model/results/results_table.md](../src/model/results/results_table.md) |
| Confusion plots | [src/model/results/plots/](../src/model/results/plots/) |
| Training logs | [logs/](../logs/) |

## Appendix B — Hyperparameter reference

```
model_name      distilbert-base-cased
num_labels      4
max_length      64
batch_size      32
learning_rate   2e-5
epochs          5
warmup_ratio    0.1
seed            42
class_weights   [3.915, 1.907, 3.977, 0.337]   # A, E, I, O
optimizer       AdamW
schedule        linear warmup + linear decay
grad_clip       max-norm 1.0
selection       max macro-F1 on validation
```
