import re
from typing import List, Tuple


_INTERNAL_DISCLOSURE_PATTERNS = [
    re.compile(r'\b(?:show|reveal|print|display|expose|share|give)\b.{0,30}\bsystem\s+prompt\b', re.I),
    re.compile(r'\bmy\s+(?:system\s+prompt|hidden\s+instructions?|internal\s+rules)\b', re.I),
    re.compile(r'\bthe\s+(?:system\s+prompt|hidden\s+rules)\s+(?:is|are|says|contains)\b', re.I),
    re.compile(r'\bhidden\s+instruction', re.I),
    re.compile(r'\bretrieved\s+documents?\s+(?:show|contain|include|say)\b', re.I),
    re.compile(r'\braw\s+context\b', re.I),
    re.compile(r'\bdeveloper\s+message\b', re.I),
]


def validate_response(response: str, sources: list, status: str = "replied") -> Tuple[bool, List[str]]:
    flags = []
    text = response or ""

    if status == "replied" and not text.strip():
        flags.append("empty_response")

    if status == "replied" and not sources:
        flags.append("missing_sources")

    for pat in _INTERNAL_DISCLOSURE_PATTERNS:
        if pat.search(text):
            flags.append("internal_disclosure")
            break

    markdown_noise = len(re.findall(r"(?m)^#{1,6}\s", text)) + text.count("---") + text.count("![")
    if markdown_noise >= 2:
        flags.append("raw_markdown_leakage")

    if len(text) > 2500:
        flags.append("overlong_response")

    return not flags, flags

