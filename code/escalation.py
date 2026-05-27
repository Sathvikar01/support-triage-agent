import re
import unicodedata
from typing import Optional, Dict, Any, List
from config import ESCALATION_KEYWORDS, ESCALATION_RESPONSE_TEMPLATES, MIN_CONFIDENCE


_COMPILED_KEYWORDS: Dict[str, List[re.Pattern]] = {}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()


def _get_compiled_keywords() -> Dict[str, List[re.Pattern]]:
    if not _COMPILED_KEYWORDS:
        for rule_name, keywords in ESCALATION_KEYWORDS.items():
            _COMPILED_KEYWORDS[rule_name] = [
                re.compile(re.escape(_normalize(kw)))
                for kw in keywords
            ]
    return _COMPILED_KEYWORDS


_DEFENSIVE_MARKERS = [
    "how do i prevent",
    "how to prevent",
    "protect against",
    "best practices",
    "mitigate",
    "avoid sql injection",
    "prevent sql injection",
]


def _is_defensive_security_question(text: str) -> bool:
    return any(marker in text for marker in _DEFENSIVE_MARKERS)


def check_hard_rules(text: str, subject: str = "") -> Optional[str]:
    combined = _normalize(text + " " + subject)
    compiled = _get_compiled_keywords()
    refund_score = 0
    for rule_name, patterns in compiled.items():
        if rule_name == "unauthorized_action" and _is_defensive_security_question(combined):
            continue
        if rule_name == "refund_demand":
            for pat in patterns:
                if pat.search(combined):
                    refund_score += 1
            continue
        for pat in patterns:
            if pat.search(combined):
                return rule_name
    if refund_score >= 2:
        return "refund_demand"
    return None


def assess_escalation(
    text: str,
    subject: str = "",
    retrieval_score: float = 0.0,
    rerank_score: float = 0.0,
    rerank_threshold: float = 0.05,
    confidence: float = 0.0,
    context_count: int = 0,
    source_companies: Optional[list] = None,
    expected_company: str = "Unknown",
    enforce_confidence: bool = True,
) -> Dict[str, Any]:
    hard_rule = check_hard_rules(text, subject)
    if hard_rule:
        template = ESCALATION_RESPONSE_TEMPLATES.get(hard_rule, ESCALATION_RESPONSE_TEMPLATES["out_of_scope"])
        return {
            "escalated": True,
            "reason": hard_rule,
            "response_template": template,
        }

    if enforce_confidence:
        if context_count <= 0:
            return {
                "escalated": True,
                "reason": "insufficient_context",
                "response_template": ESCALATION_RESPONSE_TEMPLATES["insufficient_context"],
            }

        if expected_company not in ("Unknown", "None", "", None):
            companies = {c for c in (source_companies or []) if c}
            if companies and expected_company not in companies and confidence < 0.7:
                return {
                    "escalated": True,
                    "reason": "corpus_mismatch",
                    "response_template": ESCALATION_RESPONSE_TEMPLATES["corpus_mismatch"],
                }

        if confidence < MIN_CONFIDENCE and rerank_score < rerank_threshold:
            return {
                "escalated": True,
                "reason": "insufficient_context",
                "response_template": ESCALATION_RESPONSE_TEMPLATES["insufficient_context"],
            }

    return {
        "escalated": False,
        "reason": None,
        "response_template": None,
    }

