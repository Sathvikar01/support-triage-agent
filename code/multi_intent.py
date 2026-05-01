import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SubIntent:
    text: str
    index: int
    escalation: Optional[Dict[str, Any]] = None
    response: str = ""
    status: str = "replied"
    product_area: str = ""
    request_type: str = "product_issue"
    justification: str = ""


_SPLIT_PATTERNS = [
    re.compile(r'\b(?:also|additionally|furthermore|moreover|in addition|another thing|one more thing)\b', re.I),
    re.compile(r'(?<=[.?])\s+(?=[A-Z])', re.M),
    re.compile(r'\b(?:and|but)\s+(?=I\s|I\'m|I\'ve|I\'d|my\s|can\s|could\s|please\s|how\s|what\s|where\s|when\s|why\s|is\s|are\s|was\s|were\s)', re.I),
]

_MULTI_INDICATORS = [
    re.compile(r'\b\d+[.)]\s', re.M),
    re.compile(r'^\s*[-*]\s', re.M),
    re.compile(r'\?\s+.*\?', re.M),
]


def detect_compound(text: str) -> bool:
    question_marks = text.count('?')
    if question_marks >= 2:
        return True
    for pat in _MULTI_INDICATORS:
        if pat.search(text):
            return True
    return False


def split_intents(text: str) -> List[str]:
    if not detect_compound(text):
        return [text.strip()]

    parts = []
    for pat in _SPLIT_PATTERNS:
        splits = pat.split(text)
        if len(splits) > 1:
            parts = [s.strip() for s in splits if s.strip() and len(s.strip()) > 20]
            if len(parts) > 1:
                return parts

    return [text.strip()]


def merge_results(intents: List[SubIntent]) -> Dict[str, Any]:
    if len(intents) == 1:
        return {
            "status": intents[0].status,
            "response": intents[0].response,
            "product_area": intents[0].product_area,
            "request_type": intents[0].request_type,
            "justification": intents[0].justification,
        }

    escalated = [i for i in intents if i.status == "escalated"]
    if escalated:
        combined_response = "\n\n".join(
            f"**Issue {i.index + 1}:** {i.response}" for i in intents
        )
        combined_justification = "Multiple intents detected; escalation triggered for at least one sub-issue."
        return {
            "status": "escalated",
            "response": combined_response,
            "product_area": escalated[0].product_area,
            "request_type": escalated[0].request_type,
            "justification": combined_justification,
        }

    combined_response = "\n\n".join(
        f"**Issue {i.index + 1}:** {i.response}" for i in intents
    )
    combined_justification = " | ".join(i.justification for i in intents if i.justification)
    return {
        "status": "replied",
        "response": combined_response,
        "product_area": intents[0].product_area,
        "request_type": intents[0].request_type,
        "justification": combined_justification,
    }
