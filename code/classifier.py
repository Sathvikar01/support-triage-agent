import re
from config import COMPANY_KEYWORDS, PRODUCT_AREAS, REQUEST_TYPE_KEYWORDS


def _keyword_score(text: str, keywords: list) -> int:
    score = 0
    for kw in keywords:
        pattern = r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)"
        if re.search(pattern, text):
            score += 1
    return score


def infer_company(text: str, subject: str = "") -> str:
    combined = (text + " " + subject).lower()
    scores = {}
    for company, keywords in COMPANY_KEYWORDS.items():
        score = _keyword_score(combined, keywords)
        scores[company] = score
    if max(scores.values()) == 0:
        return "Unknown"
    return max(scores, key=scores.get)


def classify_request_type(text: str, subject: str = "") -> str:
    combined = (text + " " + subject).lower().strip()

    if _is_clearly_invalid(combined):
        return "invalid"

    bug_keywords = REQUEST_TYPE_KEYWORDS.get("bug", [])
    bug_score = _keyword_score(combined, bug_keywords)

    feature_keywords = REQUEST_TYPE_KEYWORDS.get("feature_request", [])
    feature_score = _keyword_score(combined, feature_keywords)

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
            score = _keyword_score(combined, keywords)
            scores[area] = score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return "general"

    best_area = "general"
    best_score = 0
    for comp, areas in PRODUCT_AREAS.items():
        for area, keywords in areas.items():
            score = _keyword_score(combined, keywords)
            if score > best_score:
                best_score = score
                best_area = area

    return best_area


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
