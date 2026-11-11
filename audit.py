"""
Part A: NimbusPay train.jsonl data audit & cleaning script.

Usage:
    python audit.py --input candidate_pack/data/train.jsonl \
                    --output candidate_pack/data/train_clean.jsonl \
                    --log    candidate_pack/cleaning_log.csv

Checks performed (in priority order):
  1. JSON / structure validity
  2. Required keys present (exactly 8)
  3. No extra keys
  4. Field types (category str, priority str, amount_paise int|null, etc.)
  5. Allowed-values validation
  6. Business-rule re-derivation:
       a. priority derivation
       b. needs_human derivation
       c. txn fields must be null for account_access / kyc / other
  7. Field-format checks:
       a. txn_date ISO format YYYY-MM-DD
       b. txn_id NP + 10 digits
  8. Duplicate IDs
"""

import argparse, json, re, csv
from collections import defaultdict
from datetime import date, timedelta

# ─── Schema constants ────────────────────────────────────────────────────────
REQUIRED_KEYS = {"category","priority","amount_paise","txn_date","txn_id","channel","language","needs_human"}
CATEGORIES = {"payment_failed","refund","fraud","account_access","kyc","offers","other"}
PRIORITIES  = {"P1","P2","P3"}
CHANNELS    = {"upi","card","netbanking","wallet"}
LANGUAGES   = {"en","hi-en"}
NULL_TXN_CATEGORIES = {"account_access","kyc","other"}  # txn fields always null

# Repeat-contact phrases → needs_human = true
REPEAT_PATTERNS = [
    r'\bthird time\b', r'\b3rd time\b', r'\bagain and again\b',
    r'\baar baar\b', r'\bbaar-baar\b',
    r'\bcontacted (support|you|again)\b',
    r'\bprevious(ly)? (contact|ticket|complaint)\b',
    r'\beach time\b', r'\bevery time\b',
    r'\bpehle bhi\b', r'\bpehle (bhi )?(complaint|contact)\b',
    r'\bfollowing up\b',
]

# Legal threat phrases → P1
LEGAL_PATTERNS = [
    r'\bRBI\b', r'\bombudsman\b', r'\bconsumer court\b',
    r'\blawyer\b', r'\blegal notice\b', r'\blegal action\b',
    r'\bforum\b.*\bcourt\b', r'\bcourt\b',
]

# ─── Helpers ─────────────────────────────────────────────────────────────────
def _has_repeat_contact(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in REPEAT_PATTERNS)

def _has_legal_threat(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in LEGAL_PATTERNS)

def _derive_priority(cat: str, amount_paise, ticket_text: str) -> str:
    amount_rupees = (amount_paise / 100) if isinstance(amount_paise, int) else 0
    if cat == "fraud":
        return "P1"
    if amount_rupees >= 50_000:
        return "P1"
    if _has_legal_threat(ticket_text):
        return "P1"
    if cat in ("payment_failed","refund") and amount_rupees >= 5_000:
        return "P2"
    if cat == "account_access":
        return "P2"
    return "P3"

def _derive_needs_human(priority: str, ticket_text: str) -> bool:
    return priority == "P1" or _has_repeat_contact(ticket_text)

def _valid_txn_id(s) -> bool:
    return bool(s and re.fullmatch(r'NP\d{10}', str(s)))

def _valid_date(s) -> bool:
    if not isinstance(s, str):
        return False
    return bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', s))

def _parse_received(user_content: str):
    """Return date object from 'Received: YYYY-MM-DD' header."""
    m = re.search(r'Received:\s*(\d{4}-\d{2}-\d{2})', user_content)
    if m:
        y, mo, d_ = map(int, m.group(1).split('-'))
        return date(y, mo, d_)
    return None

# ─── Audit one label dict ────────────────────────────────────────────────────
def audit_label(row_id: str, label: dict, ticket_text: str):
    """
    Returns (fixed_label, issues) where issues is a list of str descriptions.
    fixed_label is the repaired dict (may equal the input if no issues found).
    """
    issues = []
    L = dict(label)  # working copy

    # 1. Required keys
    missing = REQUIRED_KEYS - set(L.keys())
    extra   = set(L.keys()) - REQUIRED_KEYS
    if missing:
        issues.append(f"missing_keys:{missing}")
    if extra:
        issues.append(f"extra_keys:{extra}")
        for k in extra:
            del L[k]

    # Patch missing required keys to null so we can continue
    for k in missing:
        L[k] = None

    # 2. category
    cat = L.get("category")
    if cat not in CATEGORIES:
        issues.append(f"bad_category:{cat!r}")
        # Cannot safely fix → will be dropped by caller

    # 3. priority
    pri = L.get("priority")
    if pri not in PRIORITIES:
        issues.append(f"bad_priority_value:{pri!r}")
        pri = None  # will re-derive below

    # 4. amount_paise
    amt = L.get("amount_paise")
    if amt is not None:
        if isinstance(amt, bool) or not isinstance(amt, int):
            issues.append(f"amount_paise_wrong_type:{type(amt).__name__}:{amt!r}")
            # Try coercing float → int
            try:
                L["amount_paise"] = int(float(amt))
                amt = L["amount_paise"]
                issues.append("amount_paise_coerced_to_int")
            except Exception:
                L["amount_paise"] = None
                amt = None

    # 5. txn_date
    txn_date = L.get("txn_date")
    if txn_date is not None and not _valid_date(txn_date):
        issues.append(f"txn_date_bad_format:{txn_date!r}")
        # Try common re-formats
        fixed = None
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%m/%d/%Y"):
            try:
                from datetime import datetime
                fixed = datetime.strptime(str(txn_date), fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                pass
        L["txn_date"] = fixed
        if fixed:
            issues.append(f"txn_date_reformatted_to:{fixed}")
        else:
            issues.append("txn_date_unrecoverable_set_null")

    # 6. txn_id
    txn_id = L.get("txn_id")
    if txn_id is not None:
        raw = str(txn_id).upper().replace(" ", "").replace("-", "")
        if re.fullmatch(r'NP\d{10}', raw):
            if txn_id != raw:
                issues.append(f"txn_id_normalized:{txn_id!r}->{raw!r}")
                L["txn_id"] = raw
        else:
            issues.append(f"txn_id_bad_format:{txn_id!r}")
            L["txn_id"] = None

    # 7. channel
    ch = L.get("channel")
    if ch is not None and ch not in CHANNELS:
        issues.append(f"bad_channel:{ch!r}")
        L["channel"] = None

    # 8. language
    lang = L.get("language")
    if lang not in LANGUAGES:
        issues.append(f"bad_language:{lang!r}")
        L["language"] = "en"  # conservative default

    # 9. needs_human type
    nh = L.get("needs_human")
    if not isinstance(nh, bool):
        issues.append(f"needs_human_wrong_type:{type(nh).__name__}:{nh!r}")
        # Coerce
        L["needs_human"] = bool(nh)

    # ── Business-rule re-derivation ──────────────────────────────────────────
    if cat in CATEGORIES:
        # a. txn fields must be null for null-txn categories
        if cat in NULL_TXN_CATEGORIES:
            for field in ("amount_paise", "txn_date", "txn_id", "channel"):
                if L.get(field) is not None:
                    issues.append(f"txn_field_should_be_null_for_{cat}:{field}={L[field]!r}")
                    L[field] = None

        # b. priority derivation
        derived_pri = _derive_priority(cat, L.get("amount_paise"), ticket_text)
        if L.get("priority") != derived_pri:
            issues.append(f"priority_mismatch:label={L.get('priority')!r}->derived={derived_pri!r}")
            L["priority"] = derived_pri

        # c. needs_human derivation
        derived_nh = _derive_needs_human(derived_pri, ticket_text)
        if L.get("needs_human") != derived_nh:
            issues.append(f"needs_human_mismatch:label={L.get('needs_human')!r}->derived={derived_nh!r}")
            L["needs_human"] = derived_nh

    return L, issues

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="candidate_pack/data/train.jsonl")
    ap.add_argument("--output", default="candidate_pack/data/train_clean.jsonl")
    ap.add_argument("--log",    default="candidate_pack/cleaning_log.csv")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rows = []
    parse_errors = []
    with open(args.input, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                parse_errors.append((lineno, str(e)))

    print(f"Loaded {len(rows)} rows ({len(parse_errors)} JSON parse errors in outer envelope)")

    # Duplicate ID check
    id_counts = defaultdict(list)
    for i, r in enumerate(rows):
        id_counts[r.get("id")].append(i)
    dup_ids = {k for k,v in id_counts.items() if len(v) > 1}
    if dup_ids:
        print(f"Duplicate IDs: {dup_ids}")

    log_rows = []   # (id, action, reason)
    clean    = []
    drop_ids = set()

    issue_summary = defaultdict(int)

    for r in rows:
        row_id = r.get("id", "UNKNOWN")

        # Parse assistant label
        try:
            msgs = r["messages"]
            assistant_content = msgs[-1]["content"]
            label = json.loads(assistant_content)
        except Exception as e:
            log_rows.append((row_id, "drop", f"unparseable_assistant_label:{e}"))
            drop_ids.add(row_id)
            issue_summary["unparseable_assistant_label"] += 1
            continue

        if not isinstance(label, dict):
            log_rows.append((row_id, "drop", "assistant_label_not_dict"))
            drop_ids.add(row_id)
            issue_summary["assistant_label_not_dict"] += 1
            continue

        # Get ticket text for rule re-derivation
        try:
            ticket_text = msgs[1]["content"]  # user message
        except Exception:
            ticket_text = ""

        fixed, issues = audit_label(row_id, label, ticket_text)

        # Unfixable: bad category
        if any(i.startswith("bad_category") for i in issues) or any(i.startswith("missing_keys") for i in issues):
            log_rows.append((row_id, "drop", "; ".join(issues)))
            drop_ids.add(row_id)
            issue_summary["dropped_bad_structure"] += 1
            if args.verbose:
                print(f"DROP {row_id}: {issues}")
            continue

        if issues:
            reason = "; ".join(issues)
            log_rows.append((row_id, "fix", reason))
            for iss in issues:
                issue_summary[iss.split(":")[0]] += 1
            if args.verbose:
                print(f"FIX  {row_id}: {issues}")

            # Write fixed label back
            new_msgs = list(msgs)
            new_msgs[-1] = {"role": "assistant", "content": json.dumps(fixed, ensure_ascii=False)}
            clean.append({**r, "messages": new_msgs})
        else:
            clean.append(r)

    # Write clean JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write cleaning log
    with open(args.log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "action", "reason"])
        w.writerows(log_rows)

    # Summary
    total      = len(rows)
    n_dropped  = len(drop_ids)
    n_fixed    = len([x for x in log_rows if x[1] == "fix"])
    n_clean    = len(clean)
    print(f"\n{'='*60}")
    print(f"Total rows:    {total}")
    print(f"Dropped:       {n_dropped}")
    print(f"Fixed:         {n_fixed}")
    print(f"Clean output:  {n_clean}")
    print(f"\nIssue breakdown:")
    for k, v in sorted(issue_summary.items(), key=lambda x: -x[1]):
        print(f"  {k:<45} {v:>5}")
    print(f"\nWrote clean data → {args.output}")
    print(f"Wrote cleaning log → {args.log}")


if __name__ == "__main__":
    main()

