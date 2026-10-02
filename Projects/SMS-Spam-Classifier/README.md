# SMS Spam / Smishing Classifier

Six models compared on the **same** train/val/test split: a TF-IDF + LogisticRegression
baseline (`SMS_Spam_Classifier_Baseline.ipynb`), a from-scratch PyTorch mean-pooling
embedding model (`SMS_Spam_Classifier_NN.ipynb`), a 1D CNN with a cosine LR schedule
(`SMS_Spam_Classifier_CNN.ipynb`), frozen `all-MiniLM-L6-v2` sentence embeddings fed into
LogisticRegression (`SMS_Spam_Classifier_FrozenEmbeddings.ipynb`), and full fine-tunes of
MiniLM (`SMS_Spam_Classifier_MiniLMFineTuned.ipynb`) and DistilBERT
(`SMS_Spam_Classifier_DistilBERTFineTuned.ipynb`). Three-way classification: `ham` /
`spam` / `smishing`.

Dataset: Mishra & Soni SMS Phishing dataset (Mendeley),
`Data/SMS-Spam-Data/Dataset_5971.csv`, 5971 raw rows -> 5831 after de-duplication.

**Split (identical across all six notebooks):** 80/20 stratified into train_temp/test,
then train_temp split 90/10 (stratified) into train/val. So train ~72% (4197 rows),
val ~8% (467 rows), test = 20% (1167 rows), `random_state=42` throughout. The test set is
touched exactly once per model, at the end.

## Data fixes applied (all six notebooks)

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
The frozen-embeddings and fine-tuned-transformer notebooks (MiniLM and DistilBERT) skip
this masking deliberately - see their own sections below.

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

## Methodology note: CNN and NN re-measured across seeds

Both `SMS_Spam_Classifier_NN.ipynb` and `SMS_Spam_Classifier_CNN.ipynb` originally ran
once, seed 42. Once the fine-tuned transformers showed how much a single seed's number
can mislead on this small a dataset, the same multi-seed treatment (5 seeds, 0-4) was
applied here too.

NN's original number (0.866) held up fine - it lands near the top of the new 5-seed
range (0.846-0.869, mean **0.856 ± 0.008**), favorable but not a fluke.

**CNN's original number (0.884) did not hold up.** It's higher than the *maximum* of all
5 new seeds (0.848-0.861, mean **0.854 ± 0.004**), not just outside one standard
deviation - a genuine outlier, not noise. The practical effect: CNN is no longer the
closest from-scratch challenger to the TF-IDF baseline. Once measured fairly, **CNN and
NN are statistically tied with each other** (0.854 vs. 0.856, within each other's seed
noise), and both are clearly behind TF-IDF (0.907) and both fine-tuned transformers
(~0.914). The gap between from-scratch models and TF-IDF is larger than this project
previously reported, not smaller.

## Results (identical split, test set n=1167)

| Model | Accuracy | Macro-F1 | Recall (ham) | Recall (smishing) | Recall (spam) | Ham msgs flagged as spam/smishing |
|---|---:|---:|---:|---:|---:|---:|
| **Fine-tuned DistilBERT (best of 5 seeds)** | **0.979** | **0.925** | 0.996 | 0.927 | 0.857 | **4** |
| Fine-tuned DistilBERT (mean of 5 seeds) | - | 0.914 ± 0.008 | - | - | - | - |
| Fine-tuned MiniLM (best of 5 seeds) | 0.974 | 0.920 | 0.989 | 0.945 | 0.857 | 11 |
| Fine-tuned MiniLM (mean of 5 seeds) | - | 0.914 ± 0.005 | - | - | - | - |
| TF-IDF + LogisticRegression (balanced) | 0.972 | 0.907 | 0.994 | 0.899 | 0.824 | 6 |
| NN mean pooling (best of 5 seeds) | 0.955 | 0.869 | 0.981 | 0.862 | 0.791 | 18 |
| NN mean pooling (mean of 5 seeds) | - | 0.856 ± 0.008 | - | - | - | - |
| 1D CNN + cosine LR schedule (best of 5 seeds) | 0.959 | 0.861 | 0.995 | 0.807 | 0.758 | 5 |
| 1D CNN + cosine LR schedule (mean of 5 seeds) | - | 0.854 ± 0.004 | - | - | - | - |
| Frozen MiniLM embeddings + LogisticRegression | 0.948 | 0.860 | 0.967 | 0.927 | 0.769 | 32 |

## Frozen embeddings (diagnostic, no fine-tuning)

`all-MiniLM-L6-v2` as a feature extractor only - no fine-tuning, no gradient ever touches
its weights - pooled sentence embeddings fed into `LogisticRegression(class_weight=
"balanced")`. Uses **raw text, not the `<PHONE>`/`<URL>` masked version**: a pretrained
encoder has already seen phone numbers and URLs during its own pretraining and tokenizes
them fine on its own, so masking would only throw away information for no benefit here.
Purpose: isolate how much of any future transformer gain is pretrained knowledge alone
vs. something fine-tuning adds - this result says "not much, on its own."

It lands *below* the sparse TF-IDF baseline (0.860 vs. 0.907 macro-F1) and by far the
worst on ham false positives (32, more than double the next-worst model). It does have
the best smishing recall of any model so far (0.927), but at the cost of flagging a lot
of ordinary messages as smishing to get there. Takeaway: general-purpose sentence
embeddings, used as-is, don't carry this dataset's specific spam signals (exact phone
numbers, specific scam phrasing) as well as sparse exact-token features do at this data
size. If a transformer is going to beat the baseline, it'll need to come from
fine-tuning, not from pretrained knowledge alone.

Reproducibility note: first run downloads `all-MiniLM-L6-v2` from the Hugging Face Hub
(~90MB), so it needs internet access once; after that it's cached locally and the
notebook runs offline.

## Fine-tuned MiniLM

Same `all-MiniLM-L6-v2`, same raw text, same split - but this time a full fine-tune:
every parameter (encoder + a freshly-initialized classification head) gets gradients,
via `AutoModelForSequenceClassification`. `AdamW`, lr `2e-5`, linear warmup + decay,
class-weighted `CrossEntropyLoss`, 6 epochs, checkpointed on best validation macro-F1.
Run on Google Colab's free GPU tier (`SMS_Spam_Classifier_MiniLMFineTuned.ipynb` isn't
executed in this repo - full fine-tuning on this machine's CPU is estimated at 10-30
minutes per run, versus well under a minute on a free T4 - so the notebook is checked in
unexecuted; these are the real numbers from actually running it there).

Run across **5 seeds** given how small the training set is (~4.2k messages) - fine-tuning
small data is high variance, and a single run's number isn't trustworthy on its own.
Test macro-F1 per seed: 0.9201, 0.9046, 0.9173, 0.9164, 0.9119 - **mean 0.914, std 0.005**.
Every single seed beat the TF-IDF baseline (0.907), not just the best one - this is a real,
reproducible win, not a lucky draw.

The checkpoint saved and reported in the table above is the **best of those 5 seeds**
(0.920), not the mean. Worth being explicit about that distinction, for the same reason
0.896 wasn't a fair baseline number earlier in this project: 0.920 describes this one
saved checkpoint, 0.914 ± 0.005 is the honest "what should I expect if I fine-tune this
again" number. Best-seed detail:

```
              precision    recall  f1-score   support

         ham      0.997     0.989     0.993       967
    smishing      0.896     0.945     0.920       109
        spam      0.839     0.857     0.848        91

    accuracy                          0.974      1167
   macro avg      0.910     0.930     0.920      1167

[[956   2   9]
 [  0 103   6]
 [  3  10  78]]
```

6 epochs vs. the first attempt's 4 made no real difference (0.913±0.003 at 4 epochs/3
seeds vs. 0.914±0.005 at 6 epochs/5 seeds) - per-seed val-F1 curves show the classic
small-dataset pattern of peaking then dipping somewhere in epochs 3-6, which the
best-checkpoint logic already handles regardless of how many epochs you let it run.

The fine-tuned checkpoint (~90MB) isn't committed to this repo - it's a one-minute Colab
run away, not worth versioning. The notebook's save/zip/download cells are commented out
for that reason; uncomment them if you want the actual weights.

## Fine-tuned DistilBERT

Same recipe again, `distilbert-base-uncased` instead of MiniLM - 66M params, 768 hidden
dim, vs. MiniLM's 22M params, 384 hidden dim. Raw text, same split, full fine-tune,
`AdamW` lr `2e-5`, linear warmup + decay, 6 epochs, 5 seeds, checkpointed on best
validation macro-F1. Also run on Colab, also checked in unexecuted
(`SMS_Spam_Classifier_DistilBERTFineTuned.ipynb`) for the same reason.

Test macro-F1 per seed: 0.9167, 0.9130, 0.9255, 0.9157, 0.9014 - **mean 0.914, std 0.008**.
All 5 seeds beat the TF-IDF baseline again. But compare that mean to MiniLM's **0.914 ±
0.005** - these two numbers are the same within either model's own seed-to-seed noise.
**A model 3x the size bought no reliable quality improvement here.** The best individual
seed (0.925, 4 ham false positives) is the best single result in the whole project - but
"best of 5 seeds" is the same kind of number 0.896 and 0.920 were: a ceiling, not an
expectation. The honest comparison between MiniLM and DistilBERT is mean vs. mean, and
by that measure they're tied.

```
              precision    recall  f1-score   support

         ham      0.997     0.996     0.996       967
    smishing      0.910     0.927     0.918       109
        spam      0.867     0.857     0.862        91

    accuracy                          0.979      1167
   macro avg      0.924     0.927     0.925      1167

[[963   0   4]
 [  0 101   8]
 [  3  10  78]]
```

Given the tie on quality, the deciding factor between MiniLM and DistilBERT for this
project is whichever is cheaper to run - exactly what the latency benchmark below is for.
Absent that number, MiniLM is the more defensible default: same expected accuracy, a
third of the parameters.

## Findings

1. **Fine-tuning is what finally beats the TF-IDF baseline - nothing else did.** Every
   from-scratch model (NN: 0.856 ± 0.008, CNN: 0.854 ± 0.004) and the frozen-embedding
   diagnostic (0.860) all lost to TF-IDF+LR (0.907) once fairly measured; full fine-tunes
   of MiniLM (0.914 ± 0.005) and DistilBERT (0.914 ± 0.008) both got past it, consistently
   across every seed, not as a fluke of one run.
2. **The CNN's original headline number (0.884) was a lucky seed, not a real result.**
   Across 5 fresh seeds it never got above 0.861 (mean 0.854 ± 0.004) - the single-seed
   number was higher than the max of the new range, not just a high draw within it. Once
   measured fairly, CNN and NN are statistically tied with each other (0.854 vs. 0.856),
   not "CNN nearly caught the baseline" as the one-seed number suggested. See the
   methodology note above for the full comparison.
3. **Fine-tuned MiniLM and DistilBERT are statistically tied - a 3x bigger model bought
   nothing reliable here.** Their means (0.914 ± 0.005 vs. 0.914 ± 0.008) overlap well
   within either model's own seed noise. The best individual checkpoint of either model
   (DistilBERT's best seed: 0.925, 4 ham false positives) looks better than the other, but
   "best of 5 seeds" is a ceiling, not an expectation - see the dedicated sections above
   for why that distinction matters. Pending the latency benchmark below, MiniLM is the
   more defensible choice: same expected quality, a third of the parameters.
4. `class_weight="balanced"` was, historically, the single biggest lever for the sklearn
   baseline (see the markdown note inside `SMS_Spam_Classifier_Baseline.ipynb` for the
   old ablation numbers) - without it, spam recall on this data drops below 0.5.
5. **Spam/smishing confusion is still the dominant error for every model**, and part of
   it is genuine label noise rather than a model failing to learn: the test set contains
   a "Bloomberg -Message center... Why wait?" message that appears twice with two
   different labels (once `smishing`, once `spam`) - both copies are in the baseline's
   current error list, which is as much a dataset problem as a model one.
6. A known blind spot carried over from earlier analysis: real-world Indian promotional
   text (real estate, health checkups, finance) shares none of the lexical cues this
   dataset's UK-style spam uses, and has been missed by every model tried so far.
7. Found a latent bug (not yet triggered, not yet fixed): `CNNModel.forward` in
   `SMS_Spam_Classifier_CNN.ipynb` references `self.min` in its short-sequence padding
   branch, but the attribute is named `self.min_len`. It's never hit on this dataset
   (no batch's longest sequence is shorter than the largest kernel, 4), so training ran
   fine, but it would raise `AttributeError` on different data.
8. **Frozen pretrained embeddings alone don't beat the sparse baseline, but fine-tuning
   the same model does** - frozen MiniLM scored 0.860, fine-tuned MiniLM scored 0.914 ±
   0.005. That ~5.4pp jump, on the identical base model, is the cleanest evidence yet that
   the gain was sitting in task adaptation, not in pretrained knowledge alone.

## Next steps

- **Latency benchmark (ms/message)** for all six models - not measured yet for any of
  them, and now the single most important number left to collect. MiniLM and DistilBERT
  are tied on quality, both beat TF-IDF+LR on quality, but TF-IDF+LR is almost certainly
  far cheaper per message, and MiniLM is almost certainly cheaper than DistilBERT. "How
  much accuracy per millisecond" is the real comparison, and nothing else left in this
  project can substitute for actually measuring it.
- Fix the `self.min` typo in `CNNModel.forward` before it's relied on with different data.
