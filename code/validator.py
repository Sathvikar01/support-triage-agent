import re
from typing import List, Tuple


_INTERNAL_DISCLOSURE = [
    "system prompt",
    "hidden instruction",
    "internal logic",
    "retrieved document",
    "raw context",
    "developer message",
]


def validate_response(response: str, sources: list, status: str = "replied") -> Tuple[bool, List[str]]:
    flags = []
    text = response or ""
    lowered = text.lower()

    if status == "replied" and not text.strip():
        flags.append("empty_response")

    if status == "replied" and not sources:
        flags.append("missing_sources")

    if any(marker in lowered for marker in _INTERNAL_DISCLOSURE):
        flags.append("internal_disclosure")

    markdown_noise = len(re.findall(r"(?m)^#{1,6}\s", text)) + text.count("---") + text.count("![")
    if markdown_noise >= 2:
        flags.append("raw_markdown_leakage")

    if len(text) > 2500:
        flags.append("overlong_response")

    return not flags, flags
