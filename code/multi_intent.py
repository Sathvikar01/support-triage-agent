import re
from typing import List


_EXPLICIT_SPLIT_PATTERNS = [
    re.compile(r'\b(?:also|additionally|furthermore|moreover|in addition|another thing|one more thing)\b', re.I),
    re.compile(r'\b(?:secondly|thirdly|finally)\b', re.I),
]

_LIST_ITEM_PATTERN = re.compile(r'(?m)^\s*(?:[-*]|\d+[.)])\s+')
_ACTION_TERMS = [
    "can", "could", "help", "need", "want", "lost", "stolen", "refund",
    "dispute", "remove", "delete", "change", "update", "reset", "restore",
    "access", "error", "failed", "failing", "blocked", "unable", "not working",
    "charge", "login", "report", "cancel", "pause", "setup", "set up",
]


def detect_compound(text: str) -> bool:
    if text.count("?") >= 2:
        return True
    if _LIST_ITEM_PATTERN.search(text):
        return True
    return any(pat.search(text) for pat in _EXPLICIT_SPLIT_PATTERNS)


def split_intents(text: str) -> List[str]:
    text = (text or "").strip()
    if not text or not detect_compound(text):
        return [text] if text else []

    if _LIST_ITEM_PATTERN.search(text):
        parts = [_clean_part(part) for part in _LIST_ITEM_PATTERN.split(text)]
        parts = [part for part in parts if _looks_actionable(part)]
        if len(parts) > 1:
            return parts

    if text.count("?") >= 2:
        parts = [_clean_part(part) for part in re.split(r'(?<=\?)\s+', text)]
        parts = [part for part in parts if _looks_actionable(part)]
        if len(parts) > 1:
            return parts

    for pattern in _EXPLICIT_SPLIT_PATTERNS:
        parts = [_clean_part(part) for part in pattern.split(text)]
        parts = [part for part in parts if _looks_actionable(part)]
        if len(parts) > 1:
            return parts

    return [text]


def _clean_part(text: str) -> str:
    return re.sub(r'\b(?:and|but)\s*$', '', (text or "").strip(), flags=re.I).strip()


def _looks_actionable(text: str) -> bool:
    lowered = (text or "").lower()
    if len(lowered.split()) < 3:
        return False
    return any(term in lowered for term in _ACTION_TERMS)
