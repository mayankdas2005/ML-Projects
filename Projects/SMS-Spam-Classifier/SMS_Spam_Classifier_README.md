# SMS Spam / Smishing Classifier — Baseline Notes (Draft)

Status: TF-IDF + Logistic Regression baseline, plus a from-scratch PyTorch embedding model,
three-way classification (`ham` / `spam` / `smishing`).
Dataset: `Data/SMS-Spam-Data/Dataset_5971.csv`, 5971 raw rows -> 5831 after de-duplication.
Split: 80/20 stratified, `random_state=42`, test set n=1167.

## Data fixes applied

**1. Stripped a leading `\t` artifact.**
153 rows (2.6%) carried a literal leading tab character. All 153 were `spam`/`smishing` —
**zero** were `ham`. This is a leakage-shaped artifact from how the source data was
assembled (looks like a remnant of tab-separated concatenation), not a real linguistic
signal. It had **no measurable effect** on this baseline, because `TfidfVectorizer`'s
default tokenizer already discards whitespace — but it's stripped now so it can't silently
leak into any future whitespace-sensitive feature (char n-grams, raw text length, prompting
an LLM with the raw string, etc.). Side effect: exact-duplicate-row count (`df.duplicated()`
on the untouched frame) rose from 17 to 84 once the tab was gone, because some rows were
otherwise byte-identical. The dataset's actual dedup step already normalizes case/whitespace
before dropping duplicates, so the final row count (5831) is unchanged either way.

**2. Added `<URL>` / `<PHONE>` placeholders.**
Regex-based masking before vectorizing: `<URL>` for `http(s)://...`, `www...`, and bare
domains (`.com`, `.co.uk`, `.org`, etc.); `<PHONE>` for digit runs of 7+ characters
(optional `+`, internal spaces/dashes allowed). This matched in 220 rows for URL and 625
rows for phone. Deliberately **not** matched: 4-6 digit SMS short-codes (e.g. "text CHAT to
86688") — in this corpus they're visually identical to prices/quantities, and masking them
indiscriminately looked more likely to hurt than help.

## Baseline results (same train/test split across all four rows)

| run                  | accuracy | macro-F1 | recall (ham) | recall (smishing) | recall (spam) |
|----------------------|---------:|---------:|-------------:|-------------------:|--------------:|
| original / plain     |   0.9417 |   0.8183 |        1.0000 |              0.8165 |         0.4725 |
| original / balanced  |   0.9649 |   0.8872 |        0.9938 |              0.8991 |         0.7363 |
| masked / plain       |   0.9409 |   0.8079 |        0.9990 |              0.8257 |         0.4615 |
| **masked / balanced**|**0.9683**|**0.8959**|       0.9928 |              0.8991 |     **0.7912** |

## PyTorch model (mean-pooled embeddings + linear head)

Same masked text, tokenized on whitespace, vocab built from train-only tokens with count
>= 2 (`SMS_Spam_Classifier_Pytorch.ipynb`). Architecture: `Embedding(vocab_size, 64,
padding_idx=0)` -> mean-pool over non-pad tokens -> `Dropout(0.3)` -> `Linear(64, 3)`.
Class-weighted `CrossEntropyLoss`, Adam (`lr=3e-3`), 50 epochs, checkpoint selected by
best validation macro-F1 (epoch 15, val macro-F1 0.8561 — training loss kept falling to
under 0.02 while val F1 plateaued/oscillated in the 0.82-0.86 range after that, the usual
overfitting signature on a training set this size, ~4.2k rows).

Test set (same 1167-row split as the TF-IDF runs):

| class    | precision | recall | f1-score |
|----------|----------:|-------:|---------:|
| ham      |     0.987 |  0.984 |    0.986 |
| smishing |     0.879 |  0.862 |    0.870 |
| spam     |     0.726 |  0.758 |    0.742 |
| **macro**|   **0.864**|**0.868**|**0.866**|

accuracy 0.955. Confusion matrix (rows = true ham/smishing/spam):

```
[[952   1  14]
 [  3  94  12]
 [ 10  12  69]]
```

The same notebook also fits a bag-of-words + `LogisticRegression(class_weight="balanced")`
baseline over the same vocab (`model2`), but that cell currently scores on the
**validation** split (467 rows), not test — macro-F1 0.85 there. That number isn't on the
same split as everything else in this doc, so don't compare it directly yet; it needs
re-scoring on `X_test_bow`/`y_test_encoded` first (see Open questions).

## Findings

1. **`class_weight="balanced"` is the dominant lever, by far.** It moves macro-F1 by
   +0.069 and spam recall by +26pp, for a ~0.6-1pp cost in ham recall. Any config without
   it leaves spam recall stuck below 0.5 — worse than a coin flip on catching spam.
2. **Placeholders move the numbers, but the direction depends on the class weighting** —
   this is the finding the task asked me to check for:
   - With `balanced` (the config we're actually shipping): masking helps —
     macro-F1 +0.0087, spam recall +5.5pp, smishing recall unchanged, ham recall down
     ~0.1pp (1 more false positive out of 967).
   - With plain weighting: masking very slightly *hurts* — macro-F1 -0.0104, spam recall
     -1.1pp.
   - **Decision: keep the placeholders.** The production config is `balanced` either way,
     the effect there is a real (if modest) improvement, and collapsing numbers/URLs to a
     shared token should generalize better to phone numbers and domains never seen in
     training — a claim this single split can't fully test, but the direction of the
     evidence supports it. Given n=91 for the spam test class, treat these deltas as
     suggestive, not conclusive — worth re-checking under cross-validation before leaning
     on the placeholder result too hard.
3. **Spam is the weakest class in every configuration** — always the lowest per-class
   recall, and the main source of confusion is with `smishing`, not `ham`.
4. **The from-scratch PyTorch embedding model loses to the sparse TF-IDF baseline** —
   macro-F1 0.866 vs. 0.896 for masked/balanced TF-IDF+LR, despite being a neural model.
   With ~4.2k training messages and a 64-dim embedding learned from scratch (no
   pretrained weights), there isn't enough data for it to beat exact-token sparse
   features + logistic regression. Not a dead end, just means the win has to come from
   somewhere other than "swap in a neural net" — pretrained embeddings, more data, or a
   different architecture, not this one as-is.

## Error notes (from the original/balanced run, 40 misclassified test examples)

- **Dominant confusion: spam <-> smishing**, not spam/smishing <-> ham. Promotional spam
  ("Call 0800... to claim your prize") and phishing-style smishing ("Your account has been
  suspended, call ... to verify") share almost identical surface form — urgency language,
  a call-to-action, a phone number or URL. The label boundary between the two classes looks
  genuinely fuzzy in several of these examples (e.g. `#631`/`#998`, the "Bloomberg -Message
  center" text, labeled `smishing` but reads like a generic promotional/phishing blend).
- **A handful of `ham` false positives are short, numeric-adjacent messages** that pattern-match
  spam superficially: "what is your account number?", "Why didn't u call on your lunch?",
  "70 mins but i had to stop somewhere first." These are casual conversational messages
  that happen to contain digits or phrases ("call", "account number") that are strongly
  associated with spam/smishing in training.
- **A few `spam`/`smishing` false negatives are short and low-signal** once digits/URLs are
  the only spam-like content: "ringtoneking 84484", "08714712388 between 10am-7pm Cost 10p" —
  almost no surrounding text for TF-IDF to key off, which is exactly the case placeholder
  masking is meant to help with (turning that number into a shared `<PHONE>` token) and
  the balanced-model spam recall gain suggests it's doing some of that work.

## Open questions for next steps

- Re-score `model2` (BOW + LogisticRegression) on `X_test_bow`/`y_test_encoded` instead
  of validation, so it's on the same split as everything else in this doc.
- Spam recall (0.79) is still the weakest number in the table — worth deciding whether
  that's acceptable or needs a stronger model / more features before moving on.
- The placeholder effect was checked on one split; worth a quick cross-validation pass if
  it's going to inform a real decision rather than just this baseline note.
- Spam vs. smishing may be a harder label boundary than ham vs. everything-else; worth
  checking a few of the ambiguous examples above against the original labeling guidelines.
