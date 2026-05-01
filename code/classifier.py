import re
from typing import Optional, Tuple
from config import COMPANY_KEYWORDS, PRODUCT_AREAS, REQUEST_TYPE_KEYWORDS


def infer_company(text: str, subject: str = "") -> str:
    combined = (text + " " + subject).lower()
    scores = {}
    for company, keywords in COMPANY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in combined)
        scores[company] = score
    if max(scores.values()) == 0:
        return "Unknown"
    return max(scores, key=scores.get)


def classify_request_type(text: str, subject: str = "") -> str:
    combined = (text + " " + subject).lower().strip()

    if _is_clearly_invalid(combined):
        return "invalid"

    bug_keywords = REQUEST_TYPE_KEYWORDS.get("bug", [])
    bug_score = sum(1 for kw in bug_keywords if kw.lower() in combined)

    feature_keywords = REQUEST_TYPE_KEYWORDS.get("feature_request", [])
    feature_score = sum(1 for kw in feature_keywords if kw.lower() in combined)

    if bug_score > 0:
        return "bug"
    if feature_score > 0:
        return "feature_request"
    return "product_issue"


def _is_clearly_invalid(text: str) -> bool:
    text = text.strip()
    word_count = len(text.split())

    off_topic_patterns = [
        r'\biron man\b', r'\bactor\b', r'\bmovie\b', r'\bfilm\b',
        r'\bwhat is the name of\b', r'\bwho played\b',
        r'\bjoke\b', r'\briddle\b', r'\bfunny\b',
    ]
    for pat in off_topic_patterns:
        if re.search(pat, text, re.I):
            return True

    if word_count < 5:
        pure_greetings = [
            r'^(hi|hello|hey|thanks|thank you|bye|ok|yes|no)\s*[!?.]*$',
        ]
        for pat in pure_greetings:
            if re.match(pat, text, re.I):
                return True

    if word_count < 8 and ('happy to help' in text or 'out of scope' in text):
        return True

    return False


def classify_product_area(text: str, subject: str = "", company: str = "Unknown") -> str:
    combined = (text + " " + subject).lower()
    if company in PRODUCT_AREAS:
        areas = PRODUCT_AREAS[company]
        scores = {}
        for area, keywords in areas.items():
            score = sum(1 for kw in keywords if kw.lower() in combined)
            scores[area] = score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

    for comp, areas in PRODUCT_AREAS.items():
        scores = {}
        for area, keywords in areas.items():
            score = sum(1 for kw in keywords if kw.lower() in combined)
            scores[area] = score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

    return "general"


def classify_ticket(issue: str, subject: str = "", company: str = "None") -> dict:
    if company in (None, "None", "", "null"):
        inferred_company = infer_company(issue, subject)
    else:
        inferred_company = company

    request_type = classify_request_type(issue, subject)
    product_area = classify_product_area(issue, subject, inferred_company)

    return {
        "company": inferred_company,
        "request_type": request_type,
        "product_area": product_area,
    }
