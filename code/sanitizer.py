import re
import unicodedata
from typing import Dict, Tuple

_PATTERNS = [
    ("credit_card", re.compile(r'\b(?:\d[ -]*?){13,19}\b')),
    ("partial_card", re.compile(r'\b(?:card\s*)?(?:ending|last\s*4)\s*(?:in\s*)?\d{4}\b', re.I)),
    ("ssn", re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b')),
    ("passport", re.compile(r'\b[A-Z][0-9]{7,8}\b')),
    ("transaction_id", re.compile(r'\b(?:cs|pi|txn|tx|order|ch)_[A-Za-z0-9_/-]{6,}\b', re.I)),
    ("email", re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')),
    ("phone", re.compile(r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b')),
    ("ip_address", re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')),
]

_INJECTION_PATTERNS = [
    re.compile(r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions', re.I),
    re.compile(r'ignore\s+everything\s+(?:above|before)', re.I),
    re.compile(r'you\s+are\s+now\s+(?:a|an)\s+', re.I),
    re.compile(r'system\s*:\s*', re.I),
    re.compile(r'admin\s*:\s*', re.I),
    re.compile(r'<\|im_start\|>', re.I),
    re.compile(r'<\|im_end\|>', re.I),
    re.compile(r'\[INST\]', re.I),
    re.compile(r'\[/INST\]', re.I),
    re.compile(r'###\s*(?:system|instruction)', re.I),
    re.compile(r'override\s+(?:your|the)\s+(?:instructions|rules|guidelines)', re.I),
    re.compile(r'disregard\s+(?:your|the)\s+(?:instructions|rules)', re.I),
    re.compile(r'new\s+instructions?\s*:', re.I),
    re.compile(r'(?:show|reveal|print|display).{0,40}(?:system prompt|hidden rules|internal logic|retrieved documents)', re.I),
    re.compile(r'affiche.{0,60}(?:regles internes|documents recuperes|logique exacte)', re.I),
]


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return normalized.encode("ascii", "ignore").decode("ascii")


def detect_injection(text: str) -> bool:
    normalized = _normalize(text)
    for pat in _INJECTION_PATTERNS:
        if pat.search(text) or pat.search(normalized):
            return True
    return False


def mask_pii(text: str) -> str:
    masked = text
    for name, pattern in _PATTERNS:
        masked = pattern.sub(f"[{name.upper()}]", masked)
    return masked


def sanitize_with_report(text: str) -> Tuple[str, bool, Dict[str, int]]:
    text = text or ""
    report = {}
    cleaned = text
    for name, pattern in _PATTERNS:
        cleaned, count = pattern.subn(f"[{name.upper()}]", cleaned)
        if count:
            report[name] = count
    return cleaned, detect_injection(text), report


def sanitize_input(text: str) -> Tuple[str, bool]:
    cleaned, injection_detected, _ = sanitize_with_report(text)
    return cleaned, injection_detected


def safe_concat(parts: list) -> str:
    cleaned = []
    for p in parts:
        if p:
            c = mask_pii(str(p))
            cleaned.append(c)
    return "\n\n".join(cleaned)
