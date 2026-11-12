"""
Rule-based reference extractor, written from SCHEMA.md and patterns observed in train.jsonl.

Used ONLY for auditing / cleaning the training data (Part A) and for generating
augmentation labels. It is never used at inference time: the submitted model
gets only the fixed system prompt + ticket, and its raw output is submitted.
"""
import re
from datetime import date, timedelta

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

LEGAL_RE = re.compile(r"\b(rbi|ombudsman|consumer court|lawyer|legal notice|legal action)\b", re.I)
REPEAT_RE = re.compile(
    # The 8 phrase families that mark repeat contact in train (each has 0 occurrences with needs_human=false
    # outside P1). NOT included: "tried too many / several times" (a retry, not a prior contact) and
    # "my earlier ticket number is #..." (a distractor: dev labels it false).
    r"(third time i am contacting|second complaint|contacted support about this before|already written \w+ times|"
    r"again and again|teesri baar|baar baar|pehle bhi complaint)", re.I)
TXN_CATS = {"payment_failed", "refund", "fraud", "offers"}


def split_ticket(user_content):
    head, _, body = user_content.partition("\n\n")
    m = re.search(r"Received:\s*(\d{4})-(\d{2})-(\d{2})", head)
    received = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    src = re.search(r"Source:\s*(\S+)", head).group(1)
    return received, src, body


def unquoted(body):
    return "\n".join(l for l in body.split("\n") if not l.lstrip().startswith(">"))


# ── txn_id ──────────────────────────────────────────────────────────────────
def txn_id(body):
    for m in re.finditer(r"\bnp[\s\-_#:.]*((?:\d[\s\-]*){10})(?!\d)", body, re.I):
        digits = re.sub(r"\D", "", m.group(1))
        if len(digits) == 10:
            return "NP" + digits
    return None


# ── channel ────────────────────────────────────────────────────────────────
def channel(body):
    t = unquoted(body).lower()
    t = re.sub(r"\bs\w{4,6} card\b", "", t)  # "scratch card" (incl. typos) is an offer, not a channel
    if re.search(r"\bupi\b", t):
        return "upi"
    if re.search(r"net ?banking|internet banking", t):
        return "netbanking"
    if re.search(r"\bwallet\b", t):
        return "wallet"
    if re.search(r"\bcard\b|\bvisa\b|mastercard", t):
        return "card"
    return None


# ── amount ─────────────────────────────────────────────────────────────────
NUM = r"(\d[\d,]*(?:\.\d+)?)"
MULT = r"(?:\s*(k|lakh|lakhs|lac)\b)?"
AMT_PATTERNS = [
    re.compile(r"(?:rs\.?|inr|₹)\s*" + NUM + MULT, re.I),
    re.compile(r"(?<![\w\-.,])" + NUM + MULT + r"\s*(?:rupees|rupaye|rupay|rs\b|inr\b)", re.I),
    re.compile(r"(?<![\w.])" + NUM + r"\s*(k|lakh|lakhs|lac)\b", re.I),
]
IGNORE_CTX = re.compile(r"balance|limit|\bfees?\b|charges?\b|account mein abhi|pade hain", re.I)


def _to_paise(num, mult):
    v = float(num.replace(",", ""))
    if mult:
        mult = mult.lower()
        v *= 1000 if mult == "k" else 100000
    return int(round(v * 100))


def amount_candidates(body):
    out = []
    body = re.sub(r"\b(rs)\.", r"\1", body, flags=re.I)  # "Rs." must not end a sentence
    body = re.sub(r"\bnp[\s\-_#:.]*(?:\d[\s\-]?){9}\d", " ", body, flags=re.I)  # ids are not amounts
    body = re.sub(r"\b\d{11,12}\b", " ", body)  # bank refs
    for sent in re.split(r"(?<=[.!?])\s+|\n", body):
        seen = set()
        for pat in AMT_PATTERNS:
            for m in pat.finditer(sent):
                if m.start() in seen:
                    continue
                seen.add(m.start())
                # balance/limit/fee words just before ("my daily limit is 2.5k")
                # or just after ("Account mein abhi ₹790 pade hain") mark a distractor
                ctx = sent[max(0, m.start() - 30): m.end() + 12]
                out.append((_to_paise(m.group(1), m.group(2)), bool(IGNORE_CTX.search(ctx)), sent))
    return out


def amount(body):
    for paise, ignore, _ in amount_candidates(body):
        if not ignore:
            return paise
    return None


# ── date ───────────────────────────────────────────────────────────────────
HINDI_NUM = {"ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5}


def _recent(received, month, day):
    for y in (received.year, received.year - 1):
        try:
            d = date(y, month, day)
        except ValueError:
            continue
        if d <= received:
            return d
    return None


def txn_date(body, received):
    t = unquoted(body)
    t = re.sub(r"\d+\.\d+\.\d+", " ", t)  # app versions
    tl = t.lower()
    # explicit numeric d/m/y
    m = re.search(r"\b(\d{1,2})([/\-])(\d{1,2})\2(\d{2}|\d{4})\b", t)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
        y = y + 2000 if y < 100 else y
        try:
            return date(y, mo, d)
        except ValueError:
            pass
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        return date(*map(int, m.groups()))
    mon = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + mon + r"\.?(?:,?\s+(20\d\d)\b)?", tl)
    if m:
        d, mo = int(m.group(1)), MONTHS[m.group(2)]
        return date(int(m.group(3)), mo, d) if m.group(3) else _recent(received, mo, d)
    m = re.search(r"\b" + mon + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(20\d\d)\b)?", tl)
    if m:
        mo, d = MONTHS[m.group(1)], int(m.group(2))
        return date(int(m.group(3)), mo, d) if m.group(3) else _recent(received, mo, d)
    if re.search(r"\bday before yesterday\b|\bparso\b", tl):
        return received - timedelta(days=2)
    if re.search(r"\btoday\b|\baaj\b", tl):
        return received
    if re.search(r"\byesterday\b|\bkal\b", tl):
        return received - timedelta(days=1)
    m = re.search(r"\b(\d+|ek|do|teen|char|chaar|paanch|panch|a|one|two|three|four|five|six|seven)\s+(days?|din)\s+(ago|pehle|back)", tl)
    if m:
        w = m.group(1)
        words = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, **HINDI_NUM}
        n = int(w) if w.isdigit() else words[w]
        return received - timedelta(days=n)
    m = re.search(r"\b(a|one|\d+)\s+weeks?\s+ago|\bpichle hafte\b|\blast week\b", tl)
    if m:
        n = 1 if not m.group(1) or not m.group(1).isdigit() else int(m.group(1))
        return received - timedelta(days=7 * n)
    return None


def has_day_before_yesterday(tl):
    return "day before yesterday" in tl


# ── language ───────────────────────────────────────────────────────────────
HINGLISH = re.compile(r"\b(hai|hain|nahi|nahin|kar|karo|raha|rahi|hoon|mera|meri|mere|paise|ki|ko|se|tha|thi|"
                      r"gaya|gaye|kya|kyun|kab|abhi|jaldi|bhai|yeh|woh|aur|ho|mein|karna|kaise|karun|hua|atka|"
                      r"kinta|kitna|ka|aa|mila|kat|pehle|baar|kiya|wapas|lekin|dhanyavaad)\b", re.I)


def language(body):
    return "hi-en" if len(HINGLISH.findall(unquoted(body))) >= 2 else "en"


# ── priority / needs_human ─────────────────────────────────────────────────
def legal(body):
    return bool(LEGAL_RE.search(body))


def repeat(body):
    return bool(REPEAT_RE.search(body))


def priority(cat, amt, body):
    rupees = (amt or 0) / 100
    if cat == "fraud" or rupees >= 50000 or legal(body):
        return "P1"
    if (cat in ("payment_failed", "refund") and rupees >= 5000) or cat == "account_access":
        return "P2"
    return "P3"


def needs_human(pri, body):
    return pri == "P1" or repeat(body)
