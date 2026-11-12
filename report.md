# NimbusPay Ticket Triage — Report

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` · **Final adapter:** QLoRA (base loaded in 4-bit nf4) — _update if E3 (16-bit) wins_
**Best dev result:** _TBD_ (run `E5_aug`) — exact match _TBD_, mean field accuracy _TBD_

---

## 1. Data audit (Part A)

**Method.** (1) Structural checks on every label: valid JSON, exactly the 8 keys, valid types and values.
(2) Leakage and duplicate checks: ticket text normalised (lowercased, punctuation and whitespace collapsed) and compared with dev, test and the rest of train.
(3) A rule-based reference extractor written from SCHEMA.md (`rules.py`), which agrees with **all 200 dev labels on every non-category field**. Wherever it disagreed with a train label, I read the row and grouped the disagreement into an issue type. No issue type was accepted without reading examples.
(4) Category was audited separately with out-of-fold TF-IDF + logistic regression, a confident-learning approach (5-fold, 97% accuracy). I read every confident disagreement.

| Issue | Rows | How found | Action |
|---|---:|---|---|
| Duplicate ticket (near-identical text, identical label) | 100 | normalised-text match within train | drop (keep first) |
| **Ticket also in dev** (30 exact + 15 differing only in spacing or punctuation) | 45 | normalised-text match against dev | **drop**: the rules forbid training on dev, and these rows would inflate dev scores |
| Truncated / unparseable label JSON | 25 | `json.loads` fails | drop: fields unknowable |
| Junk ticket (`test`, `asdf asdf`, empty) with a full label | 10 | body < 20 chars | drop |
| Old schema (v1): key `prio` with `low/medium/high`, dates as `DD/MM/YYYY` | 70 rows (100 issues) | key-set check | **fix**: in all 70 rows `low/medium/high` maps exactly to the P3/P2/P1 the spec rules give, so rename + convert |
| Priority contradicts the spec rules | 60 | rule disagreement | fix: re-derive |
| Amount stored in rupees, not paise ("INR 167" → 167; "2.2 lakh" → 220000) | 25 | label × 100 = extracted amount | fix: × 100 |
| `txn_id` not normalised (`np 81633 47892`, `NP-…`) | 20 | regex | fix: `NP` + 10 digits |
| **Result** | 1,445 → **1,265** | | 180 dropped, 175 fixed; every row in `cleaning_log.csv` |

**Checked and found clean:** `category` (the confident disagreements were all classifier errors, e.g. "a payment I don't recognise" is correctly `fraud`, not `refund`), `txn_date`, `channel`, `language` and `needs_human`. Every residual disagreement on those fields was a junk ticket.
After cleaning, every kept label agrees with the reference extractor.

**A correction to my own first pass.** My first audit "fixed" 69 `needs_human` labels and dropped all 70 v1 rows. Both decisions were wrong:
- Its repeat-contact phrase list missed "This is my second complaint on the same issue" and "I have already written 4 times about this". I extracted the real phrase set from the data instead: 8 phrase families, none ever labelled `false`. Those 69 "fixes" were turning correct labels into wrong ones.
- The v1 rows turned out to be recoverable.

Two distractors are worth noting: "my earlier ticket number is #…" is **not** a repeat contact (dev labels it `false`), and "tried my PIN too many times" is a retry, not a contact.

**Gaps that remain after cleaning, which motivate E5.** `Received` dates cover only Jun–Sep 2026, so there is no example of a year-less date that must resolve to the previous year. "parso" appears 5 times and "kal" 13 times. Fee distractors appear 9 times.

## 2. Baselines and experiments (Parts B–D)

All runs are on dev (200 rows) with `score.py`. Each fine-tuning run changes **one** thing relative to E2.
Shared settings: LoRA on all attention + MLP projections, α = 2r, dropout 0.05, LR 2e-4 cosine with 5% warm-up, batch 4 × gradient accumulation 4 = 16, fp16 (the T4 has no bf16), seed 42, and greedy decoding with batch size 1 (the graders' setting).

| Run | What changed | Dev exact | Dev mean field acc | Peak GPU (GB) | Train min |
|---|---|---:|---:|---:|---:|
| B1 | base model, fixed prompt | | | | – |
| B2 | base model, SCHEMA rules in prompt | | | | – |
| E1 | raw `train.jsonl` instead of cleaned | | | | |
| E2 | **reference**: cleaned data, 4-bit QLoRA, r=16, 3 epochs | | | | |
| E3 | 16-bit LoRA instead of 4-bit QLoRA | | | | |
| E4 | 1 epoch instead of 3 | | | | |
| E5 | + 800 rule-targeted augmented rows | | | | |

_Fill from the table cell in the notebook (Part D), which rebuilds it from the saved `result.json` files._

**Main settings and why.**
- *Model:* the 1.5B size keeps each run short enough for 5+ runs in free-tier quota.
- *r = 16 on all linear layers:* 18.5M trainable parameters, well under the 50M cap. r = 64 would exceed it.
- *`MAX_LEN`:* the maximum measured length rounded up to 64, because truncation would cut the label, which sits at the end of the sequence. Some tickets carry long quoted email threads; see the length histogram in the notebook.
- *Loss:* only on the label JSON plus `<|im_end|>`. Learning the end-of-turn token is what makes the model stop after the JSON. The vocabulary projection runs only at label positions, through a custom `compute_loss` that matches HF's built-in loss to within 2e-7. Without it, a batch of 4 long quoted-thread tickets (p99 = 1,858 tokens, max = 2,214) needs about 18 GB of fp32 logits and runs out of memory, which I measured. With it, the worst-case batch peaks at 6.2 GB (4-bit) / 7.6 GB (16-bit).
- *Reproducibility:* a local test showed that predictions from the in-memory QLoRA model differ from predictions after reloading base + adapter, because `prepare_model_for_kbit_training` upcasts norms and embeddings to fp32. Batched generation also differs from bs=1. So every scored prediction comes from a fresh reload at bs=1, the graders' exact path. The base-model baselines use bs=8 to save time.
- *E1 caveat:* raw data contains the 45 dev tickets, so E1's dev score is optimistically biased. That bias works against the cleaning, so a win for E2 is conservative.

## 3. Error analysis (Part E)

_TBD from the notebook's Part E cells: the weakest fields and categories, then 10+ failures, each with an explanation._

## 4. With one more week

_TBD after the results. Candidates: a 3B model for date arithmetic; augmentation aimed at the confusions found in Part E; checking whether the "harder" test cases match the augmentation targets._

## 5. Use of AI tools

I used Claude Code (Anthropic) as a coding assistant. It helped write and debug `rules.py`, `audit.py`, `augment.py` and the notebook, and it suggested audit checks; the duplicate/dev-leak check and the out-of-fold category check came from it. It also drafted this report. _[Edit to match what you actually did, e.g. which flagged rows you checked yourself and which decisions you changed.]_ All training ran on Colab, and every number in this report comes from those runs.
