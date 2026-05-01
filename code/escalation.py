import re
from typing import Optional, Dict, Any, List
from config import ESCALATION_KEYWORDS, ESCALATION_RESPONSE_TEMPLATES


def check_hard_rules(text: str, subject: str = "") -> Optional[str]:
    combined = (text + " " + subject).lower()
    for rule_name, keywords in ESCALATION_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in combined:
                return rule_name
    return None


def assess_escalation(
    text: str,
    subject: str = "",
    retrieval_score: float = 0.0,
    rerank_score: float = 0.0,
    rerank_threshold: float = 0.05,
) -> Dict[str, Any]:
    hard_rule = check_hard_rules(text, subject)
    if hard_rule:
        template = ESCALATION_RESPONSE_TEMPLATES.get(hard_rule, ESCALATION_RESPONSE_TEMPLATES["out_of_scope"])
        return {
            "escalated": True,
            "reason": hard_rule,
            "response_template": template,
        }

    return {
        "escalated": False,
        "reason": None,
        "response_template": None,
    }
