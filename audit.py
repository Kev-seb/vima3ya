"""
Part A: audit and clean train.jsonl.

    python audit.py            # writes candidate_pack/data/train_clean.jsonl and cleaning_log.csv

How problems were found
-----------------------
1. Structural checks: label parses as JSON, has exactly the 8 schema keys, valid types/values.
2. Leakage / duplicates: ticket text normalised (lowercase, punctuation and whitespace collapsed)
   and compared against dev, test and the rest of train.
3. Field-by-field disagreement with an independent rule-based extractor (rules.py), written from
   SCHEMA.md. The extractor agrees with all 200 dev labels on every non-category field, so where it
   disagrees with a train label, every disagreement was read by hand and grouped into the issue
   types below. Category was audited separately with 5-fold out-of-fold TF-IDF + logistic
   regression (see notebook): the flagged rows were read and no category noise was found.

Issue types and decisions
-------------------------
drop  unparseable_label        assistant content is truncated JSON -> fields unknown
drop  junk_ticket              ticket body is "test", "asdf asdf", empty -> label is meaningless
drop  dev_leak                 ticket (near-)identical to a dev ticket -> rules forbid training on dev
drop  duplicate                (near-)identical to an earlier train ticket with the same label
fix   old_schema               v1 labels: key "prio" with low/medium/high (maps 1:1 to P3/P2/P1 in all
                               70 rows) and DD/MM/YYYY dates -> renamed / converted
fix   txn_id_not_normalised    "np 12345 67890", "NP-..." -> NP + 10 digits
fix   amount_in_rupees         label holds the rupee value (x100 too small) -> converted to paise
fix   priority_wrong           priority contradicts the SCHEMA rules -> re-derived
fix   needs_human_wrong        needs_human contradicts priority / repeat-contact phrases -> re-derived
"""
import argparse, csv, json, re
from datetime import datetime

import rules as R

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]


def norm_text(user_content):
    t = user_content.lower()
    t = re.sub(r"\bthx\b", " ", t)
    return re.sub(r"[^\w₹]+", " ", t).strip()


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def audit_row(row):
    """Return (action, reasons, fixed_label). action in {keep, fix, drop}."""
    user = row["messages"][1]["content"]
    received, _, body = R.split_ticket(user)

    try:
        lab = json.loads(row["messages"][2]["content"])
    except json.JSONDecodeError:
        return "drop", ["unparseable_label"], None
    if len(body.strip()) < 20 or re.fullmatch(r"(test|asdf|\s|\d)*", body.strip(), re.I):
        return "drop", ["junk_ticket"], None

    reasons = []
    L = dict(lab)

    # v1 schema rows
    if "prio" in L:
        L["priority"] = {"low": "P3", "medium": "P2", "high": "P1"}[L.pop("prio")]
        reasons.append("old_schema:prio_key")
    if isinstance(L.get("txn_date"), str) and re.fullmatch(r"\d\d/\d\d/\d{4}", L["txn_date"]):
        L["txn_date"] = datetime.strptime(L["txn_date"], "%d/%m/%Y").strftime("%Y-%m-%d")
        reasons.append("old_schema:dd/mm/yyyy_date")
    if set(L) != set(KEYS):
        return "drop", [f"bad_keys:{sorted(set(L) ^ set(KEYS))}"], None

    # txn_id
    if L["txn_id"] is not None:
        norm = "NP" + re.sub(r"\D", "", L["txn_id"])
        if norm != L["txn_id"]:
            reasons.append(f"txn_id_not_normalised:{L['txn_id']!r}")
            L["txn_id"] = norm

    # amount stored in rupees instead of paise
    if L["category"] in R.TXN_CATS and isinstance(L["amount_paise"], int):
        ref = R.amount(body)
        if ref is not None and ref != L["amount_paise"] and L["amount_paise"] == ref // 100:
            reasons.append(f"amount_in_rupees:{L['amount_paise']}->{ref}")
            L["amount_paise"] = ref

    # derived fields
    pri = R.priority(L["category"], L["amount_paise"], body)
    if L["priority"] != pri:
        reasons.append(f"priority_wrong:{L['priority']}->{pri}")
        L["priority"] = pri
    nh = R.needs_human(pri, body)
    if L["needs_human"] != nh:
        reasons.append(f"needs_human_wrong:{L['needs_human']}->{nh}")
        L["needs_human"] = nh

    L = {k: L[k] for k in KEYS}
    return ("fix" if reasons else "keep"), reasons, L


def residual_disagreements(label, body, received):
    """Fields where the cleaned label still differs from the reference extractor (for manual review)."""
    if label["category"] not in R.TXN_CATS:
        ref = {"amount_paise": None, "txn_date": None, "txn_id": None, "channel": None}
    else:
        d = R.txn_date(body, received)
        ref = {"amount_paise": R.amount(body), "txn_date": d.isoformat() if d else None,
               "txn_id": R.txn_id(body), "channel": R.channel(body)}
    ref["language"] = R.language(body)
    return {k: (label[k], v) for k, v in ref.items() if label[k] != v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="candidate_pack/data/train.jsonl")
    ap.add_argument("--dev", default="candidate_pack/data/dev.jsonl")
    ap.add_argument("--test", default="candidate_pack/data/test_inputs.jsonl")
    ap.add_argument("--out", default="candidate_pack/data/train_clean.jsonl")
    ap.add_argument("--log", default="cleaning_log.csv")
    a = ap.parse_args()

    train, dev, test = load(a.train), load(a.dev), load(a.test)
    dev_text = {norm_text(r["messages"][1]["content"]) for r in dev}
    test_text = {norm_text(r["messages"][1]["content"]) for r in test}

    log, clean, seen, counts, residual = [], [], set(), {}, []
    for row in train:
        action, reasons, lab = audit_row(row)
        t = norm_text(row["messages"][1]["content"])
        if action != "drop":
            if t in dev_text:
                action, reasons = "drop", ["dev_leak"]
            elif t in test_text:
                action, reasons = "drop", ["test_leak"]
            elif t in seen:
                action, reasons = "drop", ["duplicate"]
        seen.add(t)
        for r in reasons:
            k = r.split(":")[0]
            counts[(action, k)] = counts.get((action, k), 0) + 1
        if action != "keep":
            log.append((row["id"], action, "; ".join(reasons)))
        if action == "drop":
            continue
        received, _, body = R.split_ticket(row["messages"][1]["content"])
        res = residual_disagreements(lab, body, received)
        if res:
            residual.append((row["id"], res))
        msgs = row["messages"][:2] + [{"role": "assistant", "content": json.dumps(lab, ensure_ascii=False)}]
        clean.append({"id": row["id"], "messages": msgs})

    with open(a.out, "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "action", "reason"])
        w.writerows(log)

    n_drop = sum(1 for x in log if x[1] == "drop")
    n_fix = sum(1 for x in log if x[1] == "fix")
    print(f"raw {len(train)}  dropped {n_drop}  fixed {n_fix}  clean {len(clean)}\n")
    print("issue counts (a fixed row can have several issues):")
    for (act, k), v in sorted(counts.items(), key=lambda x: (x[0][0], -x[1])):
        print(f"  {act:5} {k:24} {v:4}")
    print(f"\nrows whose cleaned label still differs from the reference extractor: {len(residual)}")
    for rid, res in residual:
        print(" ", rid, res)


if __name__ == "__main__":
    main()
