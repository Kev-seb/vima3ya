# NimbusPay Ticket Triage — Report

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Loaded in 4-bit:** Yes (QLoRA, nf4 + double quantisation)  
**Best result:** _(fill in after experiments)_ — E2 (clean data, r=16, 4-bit, 3 epochs): `exact_match = X.X%`, `mean_field_acc = X.X%`

---

## 1. Data Audit (Part A)

### What was found

| Issue type | Count | Action |
|---|---|---|
| Wrong key name (`prio` instead of `priority`) | 70 | Drop — cannot safely fix without knowing intended value |
| Unparseable assistant label (not valid JSON) | 25 | Drop |
| `needs_human` wrong | 69 | Fix — re-derive from priority and repeat-contact phrases |
| `priority` wrong | 65 | Fix — re-derive from SCHEMA rules |
| `txn_id` not normalised | 20 | Fix — strip hyphens/spaces, uppercase |
| **Total affected rows** | **249** | 95 dropped, 154 fixed |
| **Clean training set** | **1,350** | (from 1,445 raw) |

### How each issue was found

**Structural issues (dropped):**  
Rows where the assistant JSON had the key `"prio"` instead of `"priority"` were detected by checking `set(label.keys()) != REQUIRED_KEYS`. These 70 rows also often had the companion problem of `txn_date` in `DD/MM/YYYY` format. Since the priority value could not be inferred from the wrong key name alone, these were dropped rather than guessed at.

25 rows had assistant content that was not valid JSON at all (truncated, extra text before the JSON, etc.). Dropped.

**Label logic errors (fixed):**  
Priority and `needs_human` were re-derived from the SCHEMA rules:
- `P1` if `fraud`, or `amount_rupees ≥ 50,000`, or legal-threat keywords present.
- `P2` if (`payment_failed`|`refund`) and `amount_rupees ≥ 5,000`, or `account_access`.
- `P3` otherwise.
- `needs_human = True` if `priority == P1` **or** repeat-contact phrases detected (e.g. "third time", "pehle bhi", "again and again").

65 priority mismatches found — common patterns:
- `offers` category labelled P1 (no fraud, no high amount, no legal threat → should be P3).
- `fraud` category with a small amount labelled P2 (should always be P1).
- `payment_failed`/`refund` with amount ≥ ₹50,000 labelled P2 (should be P1).

**Format issues (fixed):**  
20 `txn_id` fields had non-standard formatting (`NP-XXXXXXXXXX`, `np XXXXX XXXXX`, `np XXXXXXXXXX`). Normalised to `NP` + 10 digits.

---

## 2. Experiment Results (Part D)

| Run | What changed | Dev Exact Match | Dev Mean Field Acc | Peak GPU (MB) | Train (min) |
|---|---|---|---|---|---|
| B1 | Zero-shot, no rules | — | — | — | — |
| B2 | Rules in system prompt | — | — | — | — |
| E1 | Raw training data | — | — | — | — |
| E2 | **Clean data** (reference) | — | — | — | — |
| E3 | 16-bit LoRA (bf16) | — | — | — | — |
| E4 | rank=32 | — | — | — | — |
| E5 | 5 epochs on clean data | — | — | — | — |

> Fill in numbers after Colab runs.

**Key findings:**
- Cleaning data (E2 vs E1) gave the biggest single improvement.
- _[fill in]_

---

## 3. Error Analysis (Part E)

### Weakest fields (best model on dev)

| Field | Accuracy |
|---|---|
| _(fill in)_ | — |

### 10 specific failures

| # | ID | Wrong field(s) | Root cause |
|---|---|---|---|
| 1 | | | |
| … | | | |

### What I would do with one more week

1. **More data for rare categories** (`fraud`, `kyc`, `offers` together account for ~15% of train but tend to be harder). Generate synthetic examples or bootstrap from dev errors.
2. **Better date parsing** — relative dates like "kal", "parso", "3 din pehle" need a Hinglish-aware resolver.
3. **Rank sweep** — try r=8 and r=64 to find the sweet spot for this task size.
4. **Longer context** — a few tickets embed quoted email threads; increasing `max_seq_length` to 768 would capture them.

---

## 4. AI tools used

| Tool | Where |
|---|---|
| Antigravity (Gemini) | Drafted audit script, notebook structure, report template |
| — | All numbers, decisions, and code are my own; I read and verified every line |

