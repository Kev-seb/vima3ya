# NimbusPay Ticket Triage — Report

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` · **Final adapter:** E5_aug, QLoRA (base loaded in 4-bit nf4), r=16, 18.5M trainable params
**Best dev result:** exact match **91.5%**, mean field accuracy **98.2%**, valid JSON 100%

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
| B1 | base model, fixed prompt | 0.0% | 0.0% | 2.8 | – |
| B2 | base model, SCHEMA rules in prompt | 0.0% | 0.4% | 3.8 | – |
| E2 | **reference**: cleaned data, 4-bit QLoRA, r=16, 3 epochs | 80.0% | 96.4% | 7.35 | 39.1 |
| E5 | + 800 rule-targeted augmented rows | **91.5%** | **98.2%** | 6.20 | 50.9 |
| E1 | raw `train.jsonl` instead of cleaned | not run | | | |
| E3 | 16-bit LoRA instead of 4-bit QLoRA | not run | | | |
| E4 | 1 epoch instead of 3 | not run | | | |
| E6 | E5 with `priority`/`needs_human` emitted last (code in notebook) | not run | | | |

**Honest status:** I ran out of Colab GPU time before the deadline. Each run took about 1 hour on the free T4 (training plus bs=1 dev generation), and the free tier allows one session at a time. Only B1/B2, E2 and E5 finished. The code for E1, E3, E4 and E6 is in the notebook (`CONFIGS`) and runs by setting `RUN`. The assignment's required comparisons (raw vs clean, 16- vs 4-bit, epochs) are therefore **not** evidenced by numbers here.

**Baselines score 0** because the base model wraps its JSON in markdown fences (```json … ```), and stripping them is not allowed. As a diagnostic only (not a submitted score): with the fences removed, B2 reaches 48.6% field accuracy and still 0% exact. The format alone justifies fine-tuning.

**E2 → E5 (+11.5 points exact).** E2's errors were dominated by `priority` (84%) and `needs_human` (92.5%). E5 lifts these to 93.5% and 97%, plus small gains on `txn_date`/`channel`. Peak memory is lower in E5 only because of batch composition (it is the same code).

**Main settings and why.**
- *Model:* the 1.5B size keeps each run short enough for 5+ runs in free-tier quota.
- *r = 16 on all linear layers:* 18.5M trainable parameters, well under the 50M cap. r = 64 would exceed it.
- *`MAX_LEN`:* the maximum measured length rounded up to 64, because truncation would cut the label, which sits at the end of the sequence. Some tickets carry long quoted email threads; see the length histogram in the notebook.
- *Loss:* only on the label JSON plus `<|im_end|>`. Learning the end-of-turn token is what makes the model stop after the JSON. The vocabulary projection runs only at label positions, through a custom `compute_loss` that matches HF's built-in loss to within 2e-7. Without it, a batch of 4 long quoted-thread tickets (p99 = 1,858 tokens, max = 2,214) needs about 18 GB of fp32 logits and runs out of memory, which I measured. With it, the worst-case batch peaks at 6.2 GB (4-bit) / 7.6 GB (16-bit).
- *Reproducibility:* a local test showed that predictions from the in-memory QLoRA model differ from predictions after reloading base + adapter, because `prepare_model_for_kbit_training` upcasts norms and embeddings to fp32. Batched generation also differs from bs=1. So every scored prediction comes from a fresh reload at bs=1, the graders' exact path. The base-model baselines use bs=8 to save time.
- *E1 caveat:* raw data contains the 45 dev tickets, so E1's dev score is optimistically biased. That bias works against the cleaning, so a win for E2 is conservative.

## 3. Error analysis (Part E)

E5 on dev: 17 of 200 rows have at least one wrong field. By field: priority 13, needs_human 6, category 4, txn_date 3, amount 2, txn_id 1.
By category (exact): kyc, offers, account_access 100%; other 94%; refund, payment_failed, fraud 87%.
**The weakest field is `priority`**, and the cause is clear: the amount is extracted correctly (99%), but the model misplaces it relative to the ₹5,000 / ₹50,000 thresholds. `needs_human` errors are almost all knock-on errors, since P1 forces `true`.

| id | Wrong field: gold → pred | Why |
|---|---|---|
| dv-00005 | priority P2 → P1, needs_human | ₹49,680 is just under the ₹50,000 threshold; the model rounds it up |
| dv-00034 | priority P2 → P1 | ₹36,737 refund/failed: threshold comparison wrong |
| dv-00056 | priority P2 → P1 | ₹44,206.75: same as above |
| dv-00081 | priority P2 → P1 | ₹24,444: same as above |
| dv-00037 | priority P1 → P2 | ₹2,84,669: written without commas ("284669 rupees"), so it reads as smaller |
| dv-00063 | priority P3 → P2 | ₹3,308 refund, below ₹5,000, predicted P2 |
| dv-00127 / 00138 | P3 → P2 / P2 → P3 | ₹4,836 and ₹14,842.25: both on the wrong side of ₹5,000 |
| dv-00044 | priority P2 → P3 | ₹8,377 refund; the repeat-contact phrase seems to distract it (needs_human is right) |
| dv-00093 | category fraud → refund | "a random charge got added to my card": an unrecognised charge is fraud per SCHEMA, but it sounds like a billing error |
| dv-00157 | fraud → account_access, all txn fields null | "someone asked for my OTP and took my money": the OTP mention pulls it to account_access, which nulls every txn field |
| dv-00177 | payment_failed → fraud | "red flag on my top up" read as fraud; the customer made the payment |
| dv-00194 | other → account_access | "lost card, can I re-activate it": card questions are `other` |
| dv-00018 / 00090 | txn_date off by 1 day | "kal" and "parso" resolved one day wrong, even after augmentation |
| dv-00017 | amount 18082725 → 18082525 | Copying a long digit string: one digit changed |

**Why priority is hard for this setup:** the label JSON lists `priority` *before* `amount_paise`, so the model must commit to the priority before it has written the amount. A 1.5B model also compares magnitudes poorly when numbers come in Indian grouping (1,80,827), plain digits or k/lakh.

## 4. With one more week

1. **Run E6** (already coded): emit `priority` and `needs_human` last, so they are generated after the amount. Scoring ignores key order. This is the most direct fix for the top error.
2. **Finish the required comparisons**: E1 (raw vs clean), E3 (16-bit vs 4-bit), E4 (epochs), with 2 seeds each, since a 200-row dev set has ±2–3 points of noise.
3. **Augment around the thresholds**: amounts within ±10% of ₹5,000 and ₹50,000 in every format (commas, Indian grouping, k, lakh), plus fraud phrasings that do not use "fraud" ("random charge", "OTP and took money").
4. Try **Qwen2.5-3B** in 4-bit for better number comparison and relative dates. It still fits the T4.

## 5. Use of AI tools

I used Claude Code (Anthropic) as a coding assistant. It helped write and debug `rules.py`, `audit.py`, `augment.py` and the notebook, and it suggested audit checks; the duplicate/dev-leak check and the out-of-fold category check came from it. It also drafted this report. _[Edit to match what you actually did, e.g. which flagged rows you checked yourself and which decisions you changed.]_ All training ran on Colab, and every dev number in this report comes from those runs. Because the GPU time ran out, `predictions_test.jsonl` was generated locally (RTX 4060) with the same code path: base model in 4-bit nf4, the E5 adapter, chat template, greedy decoding, `max_new_tokens=200`, bs=1. The notebook's final mode regenerates it on a T4.
