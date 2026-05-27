import re
from typing import Dict, List

from config import COMPANY_KEYWORDS, PRODUCT_AREAS, REQUEST_TYPE_KEYWORDS


_COMPILED_COMPANY_PATTERNS: Dict[str, List[re.Pattern]] = {}
_COMPILED_PRODUCT_PATTERNS: Dict[str, Dict[str, List[re.Pattern]]] = {}
_COMPILED_REQUEST_PATTERNS: Dict[str, List[re.Pattern]] = {}


def _compile_keyword_patterns(keywords: List[str]) -> List[re.Pattern]:
    return [
        re.compile(r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)")
        for kw in keywords
    ]


def _get_company_patterns() -> Dict[str, List[re.Pattern]]:
    if not _COMPILED_COMPANY_PATTERNS:
        for company, keywords in COMPANY_KEYWORDS.items():
            _COMPILED_COMPANY_PATTERNS[company] = _compile_keyword_patterns(keywords)
    return _COMPILED_COMPANY_PATTERNS


def _get_product_patterns() -> Dict[str, Dict[str, List[re.Pattern]]]:
    if not _COMPILED_PRODUCT_PATTERNS:
        for company, areas in PRODUCT_AREAS.items():
            _COMPILED_PRODUCT_PATTERNS[company] = {}
            for area, keywords in areas.items():
                _COMPILED_PRODUCT_PATTERNS[company][area] = _compile_keyword_patterns(keywords)
    return _COMPILED_PRODUCT_PATTERNS


def _get_request_patterns() -> Dict[str, List[re.Pattern]]:
    if not _COMPILED_REQUEST_PATTERNS:
        for rtype, keywords in REQUEST_TYPE_KEYWORDS.items():
            _COMPILED_REQUEST_PATTERNS[rtype] = _compile_keyword_patterns(keywords)
    return _COMPILED_REQUEST_PATTERNS


def _keyword_score(text: str, patterns: List[re.Pattern]) -> int:
    return sum(1 for pat in patterns if pat.search(text))


def infer_company(text: str, subject: str = "") -> str:
    combined = (text + " " + subject).lower()
    patterns = _get_company_patterns()
    scores = {company: _keyword_score(combined, pats) for company, pats in patterns.items()}
    if max(scores.values()) == 0:
        return "Unknown"
    return max(scores, key=scores.get)


def classify_request_type(text: str, subject: str = "") -> str:
    combined = (text + " " + subject).lower().strip()

    if _is_clearly_invalid(combined):
        return "invalid"

    bug_config = REQUEST_TYPE_KEYWORDS.get("bug", {})
    feature_patterns = _get_request_patterns().get("feature_request", [])

    bug_score = 0
    if isinstance(bug_config, dict):
        for kw in bug_config.get("strong", []):
            if re.search(r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)", combined):
                bug_score += 3
        for kw in bug_config.get("moderate", []):
            if re.search(r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)", combined):
                bug_score += 2
        for kw in bug_config.get("weak", []):
            if re.search(r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)", combined):
                bug_score += 1
    else:
        bug_score = _keyword_score(combined, _get_request_patterns().get("bug", []))

    feature_score = _keyword_score(combined, feature_patterns)

    if bug_score >= 2:
        return "bug"
    if feature_score > 0:
        return "feature_request"
    return "product_issue"


_INVALID_PATTERNS = [
    re.compile(r'\biron man\b', re.I),
    re.compile(r'\bactor\b', re.I),
    re.compile(r'\bmovie\b', re.I),
    re.compile(r'\bfilm\b', re.I),
    re.compile(r'\bwhat is the name of\b', re.I),
    re.compile(r'\bwho played\b', re.I),
    re.compile(r'\bjoke\b', re.I),
    re.compile(r'\briddle\b', re.I),
]
_GREETING_PATTERN = re.compile(r'^(hi|hello|hey|thanks|thank you|bye|ok|okay|yes|no|hi there|hey there|hello there)\s*[!?.]*$', re.I)


def _is_clearly_invalid(text: str) -> bool:
    text = text.strip()
    word_count = len(text.split())

    for pat in _INVALID_PATTERNS:
        if pat.search(text):
            return True

    if word_count < 5 and _GREETING_PATTERN.match(text):
        return True

    if word_count < 8 and ('happy to help' in text or 'out of scope' in text):
        return True

    return False


def classify_product_area(text: str, subject: str = "", company: str = "Unknown") -> str:
    combined = (text + " " + subject).lower()
    product_patterns = _get_product_patterns()

    if company in product_patterns:
        areas = product_patterns[company]
        scores = {area: _keyword_score(combined, pats) for area, pats in areas.items()}
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return "general"

    best_area = "general"
    best_score = 0
    for comp, areas in product_patterns.items():
        for area, pats in areas.items():
            score = _keyword_score(combined, pats)
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

