"""
Experiment E5 (own idea): rule-targeted data augmentation, using only train_clean.jsonl + SCHEMA.md.

Why: the cleaned train set has blind spots that the spec explicitly covers:
  * Received dates are only Jun-Sep 2026 -> zero examples where a year-less date ("28 Dec")
    must resolve to the previous year;
  * "parso" (5 rows), "kal" (13), fee distractors (9) are rare;
  * transaction details inside account_access / kyc / other tickets (must still be null) are rare.

How: every synthetic ticket reuses a real *intent* sentence harvested from a cleaned train row
(so category + language come from a human-style label), then adds slot sentences written in the
same phrasings train uses (amount, date, txn id, channel, distractors, repeat-contact, legal threat).
The label is built from the values we inserted, and the ticket is kept only if the independent
rule extractor (rules.py, 100% agreement with dev on non-category fields) produces the same label.

    python augment.py --n 800 --out candidate_pack/data/train_aug.jsonl
"""
import argparse, json, random, re
from datetime import date, timedelta

import rules as R

SYSTEM = "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."
KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
NAMES = ["Imran Khan", "Rahul Verma", "Meera Joshi", "Vikram Singh", "Pooja Nair", "Kavya Reddy", "Fatima Sheikh",
         "Arjun Das", "Rohan Gupta", "Ananya Bose", "Aarav Mehta", "Sneha Iyer"]
SUBJECTS = ["Help needed", "Issue with my account", "Problem with NimbusPay", "Complaint", "Need support"]
GREET = {"en": ["Hi,", "Hello,", "Hey,", "Hi team,", "Good morning,", "Dear NimbusPay support,", "URGENT!!", ""],
         "hi-en": ["Hello ji,", "Namaste,", "Bhai,", "Sir,", "Hi,", ""]}
CLOSE = {"en": ["Please help.", "Thanks.", "Waiting for your reply.", "Kindly resolve this asap.", "Please look into it.", ""],
         "hi-en": ["Please help karo.", "Dhanyavaad.", "Reply ka wait kar raha hoon.", "Jaldi solve karo please.", ""]}
DEVICE = {"en": ["Using the Android app.", "I am on the iOS app.", "I am using app version {v}.", "App version is {v}.",
                 "My phone is a Redmi Note 12.", "My phone is a Pixel 7a."],
          "hi-en": ["Android app use kar raha hoon.", "App version {v} hai.", "Mera phone iPhone 13 hai.",
                    "Mera phone Samsung M34 hai."]}
REPEAT = {"en": ["This is the third time I am contacting you.", "This is my second complaint on the same issue.",
                 "I have contacted support about this before.", "I have already written {n} times about this.",
                 "I keep raising this again and again."],
          "hi-en": ["Teesri baar bol raha hoon.", "Baar baar complaint kar raha hoon.", "Pehle bhi complaint kar chuka hoon."]}
LEGAL = {"en": ["I will go to consumer court.", "My lawyer will send a legal notice.", "I will complain to the RBI ombudsman.",
                "I will escalate this to the RBI.", "Otherwise I will approach the consumer court.",
                "I will take legal action."],
         "hi-en": ["Consumer court jaunga.", "RBI mein complaint karunga.", "Lawyer se legal notice bhejunga."]}
# distractors that must NOT change any field
EARLIER_TICKET = {"en": "My earlier ticket number is #{t}.", "hi-en": "Pichla ticket number #{t} tha."}
BANK_REF = {"en": "Bank ref no {b}.", "hi-en": "Bank ref {b} hai."}
RETRY = {"en": ["I tried several times.", "I tried multiple times but it did not work."], "hi-en": ["Maine kai baar try kiya."]}
QUOTE = ("> On {d}, NimbusPay Support wrote:\n> Thank you for contacting NimbusPay. Your request has been registered "
         "with ticket #{t}. Our team will get back to you within 48 working hours. Please do not share your OTP, PIN "
         "or card details with anyone, including people who claim to be from NimbusPay.")

CHANNEL_PHRASES = {
    "en": {"upi": ["Mode: UPI.", "It was a UPI transaction."],
           "netbanking": ["Mode: internet banking.", "It was a netbanking transaction."],
           "wallet": ["Mode: wallet.", "It was from the NimbusPay wallet."],
           "card": ["Mode: credit card.", "Mode: debit card.", "It was a card transaction."]},
    "hi-en": {"upi": ["UPI se hua tha."], "netbanking": ["Netbanking se hua tha."], "wallet": ["Wallet se hua tha."],
              "card": ["Card se hua tha.", "Credit card wala transaction tha."]},
}
ID_PHRASES = {"en": ["The txn id is {i}.", "You can check {i}.", "Transaction ID {i}.", "Transaction reference {i}.",
                     "Ref: {i}", "ID: {i}"],
              "hi-en": ["Transaction ID {i} hai.", "{i} check kar lo.", "Txn id {i}.", "Reference {i} hai."]}
AMOUNT_PHRASES = {"en": ["It was for {a}.", "{a} is the amount.", "The amount was {a}.", "Amount: {a}", "It is {a} in total.",
                         "{a} was the value.", "Total {a}."],
                  "hi-en": ["{a} kat gaye.", "Amount {a} tha.", "{a} ka transaction tha.", "Rakam {a} hai.", "Total {a} ka tha."]}
DISTRACT_AMT = {"en": ["My account balance is {a} right now.", "FYI my daily limit is {a}.", "The fee charged was {a}."],
                "hi-en": ["Account mein abhi {a} pade hain."]}


def fmt_inr(n):
    """Indian digit grouping: 123456 -> 1,23,456"""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:]); head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


def make_amount(rng, hard):
    """Return (surface text, paise). Formats seen in train + spec (k, lakh)."""
    kind = rng.choice(["k", "lakh", "plain", "plain", "decimal"] if hard else ["plain", "plain", "plain", "decimal", "k"])
    if kind == "k":
        v = rng.choice([rng.randint(1, 99) / 10, rng.randint(10, 99) / 10 * 10, rng.randint(1, 49)])
        rupees = v * 1000
        txt = (f"{v:g}k")
    elif kind == "lakh":
        v = rng.choice([1, 1.5, 2, 2.5, 1.2, 3, 0.5, 1.75])
        rupees = v * 100000
        txt = f"{v:g} " + rng.choice(["lakh", "lakhs"])
    else:
        r = rng.choice([rng.randint(50, 999), rng.randint(1000, 9999), rng.randint(10000, 99999), rng.randint(100000, 300000)])
        paise = rng.choice([0, 0, 0, 25, 50, 75, 99]) if kind == "decimal" else 0
        num = fmt_inr(r) if rng.random() < 0.6 else str(r)
        if paise:
            num += f".{paise:02d}"
        rupees = r + paise / 100
        txt = rng.choice([f"Rs. {num}", f"Rs {num}", f"₹{num}", f"INR {num}", f"{num} rupees", f"{num} rupaye"])
    return txt, int(round(rupees * 100))


def make_id(rng):
    d = "".join(str(rng.randint(0, 9)) for _ in range(10))
    txt = rng.choice([f"NP{d}", f"np{d}", f"NP-{d}", f"NP {d}", f"np {d[:5]} {d[5:]}"])
    return txt, "NP" + d


def make_date(rng, received, lang):
    """Return (sentence, date). Emphasises relative dates and year-less dates crossing a year boundary."""
    kind = rng.choice(["rel", "rel", "noyear", "noyear", "full", "numeric"])
    if kind == "rel":
        n = rng.choice([0, 1, 2, 3, 4, 5, 6])
        d = received - timedelta(days=n)
        if lang == "hi-en":
            word = {0: "aaj hi hua yeh.", 1: "kal ki baat hai.", 2: "parso ki baat hai."}.get(n, f"{n} din pehle ki baat hai.")
            return word[0].upper() + word[1:], d
        word = {0: "today", 1: "yesterday", 2: "day before yesterday"}.get(n, f"{n} days ago")
        return rng.choice([f"It was {word}.", f"This happened {word}.", f"It took place {word}."]), d
    if kind == "noyear":
        # 50%: a month after the received month -> must resolve to previous year
        back = rng.randint(1, 300) if rng.random() < 0.5 else rng.randint(1, 60)
        d = received - timedelta(days=back)
        txt = rng.choice([f"{d.day} {MON[d.month-1]}", f"{MON[d.month-1]} {d.day}"])
    elif kind == "full":
        d = received - timedelta(days=rng.randint(0, 120))
        txt = f"{d.day}{'th' if 10 <= d.day % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(d.day % 10,'th')} {MONTH_FULL[d.month-1]} {d.year}"
    else:
        d = received - timedelta(days=rng.randint(0, 120))
        txt = rng.choice([d.strftime("%d/%m/%Y"), d.strftime("%d-%m-%y")])
    if lang == "hi-en":
        return rng.choice([f"{txt} ki transaction hai.", f"Yeh {txt} ko hua."]), d
    return rng.choice([f"Transaction date {txt}.", f"This happened on {txt}.", f"Dated {txt}.", f"It took place on {txt}."]), d


# ── harvest intents from cleaned train ─────────────────────────────────────
SLOTISH = re.compile(
    r"\d|np\b|rupees|rupaye|\brs\b|inr|₹|lakh|mode:|se hua tha|transaction tha|it was a \w+ transaction|nimbuspay wallet|"
    r"ticket|bank ref|app version|my phone|mera phone|android app|ios app|today|yesterday|ago|pehle|kal\b|parso|aaj|"
    r"please help|thanks|thank you|waiting for|kindly resolve|look into|dhanyavaad|reply ka|jaldi solve|"
    r"regards|subject:|urgent!!|^\s*(hi|hello|hey|namaste|bhai|sir|good morning|dear)\b[^.?!]*,\s*$|"
    + R.REPEAT_RE.pattern + "|" + R.LEGAL_RE.pattern, re.I)


def harvest(rows):
    intents = []
    for r in rows:
        lab = json.loads(r["messages"][2]["content"])
        _, src, body = R.split_ticket(r["messages"][1]["content"])
        body = R.unquoted(body)
        body = re.sub(r"^Subject:.*\n", "", body).replace("[IVR transcript]", "").replace("@NimbusPayCare", "")
        body = re.sub(r"\n\s*Regards,[\s\S]*$", "", body)
        body = re.sub(r"^\s*(Hi team|Hi|Hello ji|Hello|Hey|Namaste|Bhai|Sir|Good morning|Dear NimbusPay support),", "", body.strip())
        body = body.replace("URGENT!!", "")
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", body) if s.strip()]
        keep = [s for s in sents if not SLOTISH.search(s)]
        if not keep:
            continue
        text = " ".join(keep)
        # the intent alone must not already carry slot information or a different language signal
        if R.legal(text) or R.repeat(text) or R.language(text) != lab["language"] and lab["language"] == "en":
            continue
        intents.append({"text": text, "category": lab["category"], "language": lab["language"],
                        "card_in_intent": R.channel(text)})
    return intents


def build(rng, intent):
    lang, cat = intent["language"], intent["category"]
    received = date(2026, 1, 1) + timedelta(days=rng.randint(0, 364))
    src = rng.choice(["app-chat", "email", "twitter", "ivr-transcript"])
    hard = rng.random() < 0.7
    is_txn = cat in R.TXN_CATS
    parts, lab = [intent["text"]], {k: None for k in KEYS}
    lab["category"], lab["language"] = cat, lang

    slots = []
    if is_txn or rng.random() < 0.2:  # non-txn tickets sometimes mention txn details that must stay null
        if rng.random() < 0.75:
            t, paise = make_amount(rng, hard)
            slots.append(rng.choice(AMOUNT_PHRASES[lang]).format(a=t))
            if is_txn:
                lab["amount_paise"] = paise
        if rng.random() < 0.7:
            t, d = make_date(rng, received, lang)
            slots.append(t)
            if is_txn:
                lab["txn_date"] = d.isoformat()
        if rng.random() < 0.6:
            t, i = make_id(rng)
            slots.append(rng.choice(ID_PHRASES[lang]).format(i=t))
            if is_txn:
                lab["txn_id"] = i
        if intent["card_in_intent"]:
            if is_txn:
                lab["channel"] = intent["card_in_intent"]
        elif rng.random() < 0.55:
            ch = rng.choice(list(CHANNEL_PHRASES[lang]))
            slots.append(rng.choice(CHANNEL_PHRASES[lang][ch]))
            if is_txn:
                lab["channel"] = ch
    # distractors
    if rng.random() < (0.45 if hard else 0.15) and is_txn and lab["amount_paise"] is not None:
        t, _ = make_amount(rng, False)
        slots.append(rng.choice(DISTRACT_AMT[lang]).format(a=t))
    if rng.random() < 0.25:
        slots.append(EARLIER_TICKET[lang].format(t=rng.randint(10000, 99999)))
    if rng.random() < 0.2:
        slots.append(BANK_REF[lang].format(b=rng.randint(10**11, 10**12 - 1)))
    if rng.random() < 0.1 and cat in ("payment_failed", "account_access"):
        slots.append(rng.choice(RETRY[lang]))
    if rng.random() < 0.3:
        slots.append(rng.choice(DEVICE[lang]).format(v=f"{rng.randint(4,7)}.{rng.randint(0,9)}.{rng.randint(0,20)}"))
    if rng.random() < 0.2:
        slots.append(rng.choice(REPEAT[lang]).format(n=rng.randint(3, 5)))
    if rng.random() < 0.15:
        slots.append(rng.choice(LEGAL[lang]))
    rng.shuffle(slots)
    body_core = " ".join(parts + slots + [rng.choice(CLOSE[lang])]).strip()
    greet = rng.choice(GREET[lang])

    if src == "email":
        text = f"Subject: {rng.choice(SUBJECTS)}\n\n" + (greet + "\n" if greet else "") + body_core
        text += f"\n\nRegards,\n{rng.choice(NAMES)}"
        if rng.random() < 0.35:  # quoted thread: its dates are not the txn date
            for _ in range(rng.randint(1, 3)):
                qd = received - timedelta(days=rng.randint(1, 40))
                text += "\n\n" + QUOTE.format(d=f"{qd.day:02d} {MON[qd.month-1]} {qd.year}", t=rng.randint(10000, 99999))
    else:
        text = (greet + " " if greet else "") + body_core
        if src == "twitter":
            text = "@NimbusPayCare " + text
        elif src == "ivr-transcript":
            text = "[IVR transcript] " + text

    body = text
    lab["priority"] = R.priority(cat, lab["amount_paise"], body)
    lab["needs_human"] = R.needs_human(lab["priority"], body)
    user = f"Received: {received.isoformat()} ({received.strftime('%a')})\nSource: {src}\n\n{text}"
    return user, {k: lab[k] for k in KEYS}


def check(user, lab):
    """Independent re-extraction; the synthetic label must agree on every rule-checkable field."""
    received, _, body = R.split_ticket(user)
    if lab["category"] in R.TXN_CATS:
        d = R.txn_date(body, received)
        ref = {"amount_paise": R.amount(body), "txn_date": d.isoformat() if d else None,
               "txn_id": R.txn_id(body), "channel": R.channel(body)}
    else:
        ref = {"amount_paise": None, "txn_date": None, "txn_id": None, "channel": None}
    ref["language"] = R.language(body)
    return all(lab[k] == v for k, v in ref.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="candidate_pack/data/train_clean.jsonl")
    ap.add_argument("--out", default="candidate_pack/data/train_aug.jsonl")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(l) for l in open(a.train, encoding="utf-8")]
    intents = harvest(rows)
    out, rejected = [], 0
    while len(out) < a.n:
        user, lab = build(rng, rng.choice(intents))
        if not check(user, lab):
            rejected += 1
            continue
        out.append({"id": f"aug-{len(out)+1:05d}", "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(lab, ensure_ascii=False)}]})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"intents harvested: {len(intents)}  generated: {len(out)}  rejected by checker: {rejected}")


if __name__ == "__main__":
    main()
