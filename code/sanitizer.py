import re
from typing import Tuple

_PATTERNS = [
    ("credit_card", re.compile(r'\b(?:\d[ -]*?){13,19}\b')),
    ("ssn", re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b')),
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
]


def detect_injection(text: str) -> bool:
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            return True
    return False


def mask_pii(text: str) -> str:
    masked = text
    masked = _PATTERNS[0][1].sub("[CREDIT_CARD]", masked)
    masked = _PATTERNS[1][1].sub("[SSN]", masked)
    masked = _PATTERNS[2][1].sub("[EMAIL]", masked)
    masked = _PATTERNS[3][1].sub("[PHONE]", masked)
    masked = _PATTERNS[4][1].sub("[IP_ADDRESS]", masked)
    return masked


def sanitize_input(text: str) -> Tuple[str, bool]:
    injection_detected = detect_injection(text)
    cleaned = mask_pii(text)
    return cleaned, injection_detected


def safe_concat(parts: list) -> str:
    cleaned = []
    for p in parts:
        if p:
            c = mask_pii(str(p))
            cleaned.append(c)
    return "\n\n".join(cleaned)
