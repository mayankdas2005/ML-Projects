# SMS Spam / Smishing Classifier

Three models compared on the **same** train/val/test split: a TF-IDF + LogisticRegression
baseline (`SMS_Spam_Classifier_Baseline.ipynb`), a from-scratch PyTorch mean-pooling
embedding model (`SMS_Spam_Classifier_NN.ipynb`), and a 1D CNN with a cosine LR schedule
(`SMS_Spam_Classifier_CNN.ipynb`). Three-way classification: `ham` / `spam` / `smishing`.

Dataset: Mishra & Soni SMS Phishing dataset (Mendeley),
`Data/SMS-Spam-Data/Dataset_5971.csv`, 5971 raw rows -> 5831 after de-duplication.

**Split (identical across all three notebooks):** 80/20 stratified into train_temp/test,
then train_temp split 90/10 (stratified) into train/val. So train ~72% (4197 rows),
val ~8% (467 rows), test = 20% (1167 rows), `random_state=42` throughout. The test set is
touched exactly once per model, at the end.

## Data fixes applied (all three notebooks)

**1. Stripped a leading `\t` artifact.** 153 rows (2.6%) carried a literal leading tab
character. All 153 were `spam`/`smishing` - **zero** were `ham`. This is a leakage-shaped
artifact from how the source data was assembled, not a real linguistic signal. It has no
measurable effect on any of these models (none of the tokenizers used - TF-IDF's or the
custom whitespace one - treat a leading tab as a token), but it's stripped so it can't
silently leak into anything whitespace-sensitive later.

**2. Added `<URL>` / `<PHONE>` placeholders.** Regex-based masking before tokenizing:
`<URL>` for `http(s)://...`, `www...`, and bare domains (`.com`, `.co.uk`, `.org`, etc.);
`<PHONE>` for digit runs of 7+ characters (optional `+`, internal spaces/dashes allowed).
220 rows got a `<URL>` substitution, 625 got `<PHONE>`. Deliberately not matched: 4-6
digit SMS short-codes (e.g. "text CHAT to 86688") - in this corpus they look identical to
prices/quantities, and masking them indiscriminately looked more likely to hurt than help.

## Methodology note: the baseline split changed

An earlier version of `SMS_Spam_Classifier_Baseline.ipynb` used a plain 80/20 split and
swept 4 variants (plain/balanced x original/masked), picking masked+balanced (macro-F1
0.896) after comparing all 4 *on the test set* - a mild form of test-set leakage. That
80/20 split also gave the baseline more training data (80%) than the NN/CNN notebooks
ever get (72%, since they carve out a validation split for checkpointing).

This has been corrected: the baseline notebook now uses the identical 72/8/20 split as
the NN and CNN notebooks, commits to one config in advance (masked text,
`class_weight="balanced"`), and is evaluated once. The corrected number is **0.907** -
*higher* than the old 0.896, not lower. That's not "more data would have helped more";
it's more likely that the 8% carved out into validation happened to remove some of the
dataset's noisier/mislabeled rows from training (see Known data issues below). Don't read
too much into the exact direction of that one delta - the point of the fix is that 0.907
is now a number earned on the same footing as the NN and CNN numbers, not that 0.907 is
intrinsically more "correct" than 0.896 would have been on a true re-split.

## Results (identical split, test set n=1167)

| Model | Accuracy | Macro-F1 | Recall (ham) | Recall (smishing) | Recall (spam) | Ham msgs flagged as spam/smishing |
|---|---:|---:|---:|---:|---:|---:|
| **TF-IDF + LogisticRegression (balanced)** | **0.972** | **0.907** | 0.994 | 0.899 | 0.824 | 6 |
| Embedding + mean pooling (PyTorch NN) | 0.955 | 0.866 | 0.984 | 0.862 | 0.758 | 15 |
| 1D CNN + cosine LR schedule | 0.964 | 0.884 | 0.995 | 0.835 | 0.791 | 5 |

## Findings

1. **The TF-IDF+LogisticRegression baseline wins outright - neither neural model beats
   it, once compared on the same split.** This reverses the earlier tentative read (from
   before this split fix) that the CNN might be closing in on or passing the baseline.
   At this dataset size (~4.2k training messages), sparse exact-token features beat both
   a from-scratch mean-pooled embedding and a from-scratch CNN. It also overturns the
   older "neural models trade spam recall for fewer ham false positives" narrative: the
   corrected baseline has both the best recall on every class *and* a competitive ham
   false-positive count (6, between the CNN's 5 and the NN's 15) - it isn't trading
   anything away anymore.
2. `class_weight="balanced"` was, historically, the single biggest lever for the sklearn
   baseline (see the markdown note inside `SMS_Spam_Classifier_Baseline.ipynb` for the
   old ablation numbers) - without it, spam recall on this data drops below 0.5.
3. **Spam/smishing confusion is still the dominant error for every model**, and part of
   it is genuine label noise rather than a model failing to learn: the test set contains
   a "Bloomberg -Message center... Why wait?" message that appears twice with two
   different labels (once `smishing`, once `spam`) - both copies are in the baseline's
   current error list, which is as much a dataset problem as a model one.
4. A known blind spot carried over from earlier analysis: real-world Indian promotional
   text (real estate, health checkups, finance) shares none of the lexical cues this
   dataset's UK-style spam uses, and has been missed by every model tried so far.
5. Found a latent bug (not yet triggered, not yet fixed): `CNNModel.forward` in
   `SMS_Spam_Classifier_CNN.ipynb` references `self.min` in its short-sequence padding
   branch, but the attribute is named `self.min_len`. It's never hit on this dataset
   (no batch's longest sequence is shorter than the largest kernel, 4), so training ran
   fine, but it would raise `AttributeError` on different data.

## Next steps

- **Latency benchmark (ms/message, CPU)** for all three models - not measured yet for any
  of them, and it's the comparison that matters most given the target role's emphasis on
  CPU-scale, low-latency inference. Worth calling out: the baseline isn't just the most
  accurate model above, it's also almost certainly the cheapest to run per message - that
  combination is worth confirming with real numbers, not assumed.
- **Multi-seed runs for the CNN** before drawing any conclusion about its variance - less
  urgent now that it's clearly behind the baseline rather than close to it, but still
  useful to know how noisy 0.884 actually is.
- **Frozen embeddings + LogisticRegression** (cheap, diagnostic, do first): MiniLM
  `all-MiniLM-L6-v2` or DistilBERT `distilbert-base-uncased` as a feature extractor only
  (no fine-tuning), pooled embeddings into sklearn `LogisticRegression`. Use raw text,
  not the `<PHONE>`/`<URL>` masked version - pretrained tokenizers handle real numbers/
  URLs fine, masking may not help here. Purpose: isolate how much of any later gain is
  pretrained knowledge vs. fine-tuning.
- **Full fine-tuning**: DistilBERT and MiniLM via `AutoModelForSequenceClassification`,
  the model's own `AutoTokenizer` (not the custom whitespace vocab), lr 2e-5 to 5e-5,
  AdamW, 3-4 epochs, short warmup + decay. Run >= 3 seeds given the small training set.
  CPU fine-tuning will be slow; Colab/Kaggle free GPU is the practical option.
- Fix the `self.min` typo in `CNNModel.forward` before it's relied on with different data.
