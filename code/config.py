import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "data"
SUPPORT_TICKETS_DIR = REPO_ROOT / "support_tickets"
VECTOR_DB_DIR = REPO_ROOT / "vector_db"

XIAOMI_MODEL = "mimo-v2.5"
XIAOMI_BASE_URL = os.getenv("XIAOMI_BASE_URL", "https://api.xiaomi.com/v1") # Replace with actual Xiaomi endpoint if different
XIAOMI_API_KEY = os.getenv("XIAOMI_API_KEY", "")
XIAOMI_TIMEOUT_SECONDS = float(os.getenv("XIAOMI_TIMEOUT_SECONDS", "20"))

RELEVANCE_THRESHOLD = 0.005
RERANK_THRESHOLD = 0.05
MIN_CONFIDENCE = 0.35
TOP_K_TFIDF = 50
TOP_K_EMBEDDING = 50
TOP_K_RERANK = 10
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

COMPANY_KEYWORDS = {
    "HackerRank": [
        "hackerrank", "hrank", "coding test", "assessment", "interview",
        "screen", "recruit", "candidate", "hiring", "test", "challenge",
        "resume builder", "mock interview", "prep kit", "certification",
        "contest", "practice", "submission", "compiler", "question bank",
    ],
    "Claude": [
        "claude", "anthropic", "ai assistant", "conversation", "chat",
        "artifact", "sonnet", "opus", "haiku", "claude code", "claude desktop",
        "pro plan", "max plan", "api key", "bedrock", "prompt",
    ],
    "Visa": [
        "visa", "card", "payment", "transaction", "merchant", "charge",
        "dispute", "fraud", "traveller", "traveler", "cheque", "checkout",
        "pos terminal", "contactless", "chip", "pin", "cvv", "statement",
    ],
}

PRODUCT_AREAS = {
    "HackerRank": {
        "screen": ["test", "assessment", "screen", "invite", "candidate", "score", "report", "proctoring", "integrity"],
        "interviews": ["interview", "live", "record", "schedule", "whiteboard", "pair programming"],
        "library": ["question", "library", "coding", "challenge", "problem"],
        "settings": ["account", "setting", "role", "team", "admin", "user management"],
        "integrations": ["integration", "ats", "api", "sso", "greenhouse", "lever", "workday", "icims"],
        "community": ["community", "practice", "contest", "certification", "mock", "prep kit", "leaderboard"],
        "engage": ["engage", "campus", "university", "hiring event"],
        "chakra": ["chakra", "skillup", "learning path"],
    },
    "Claude": {
        "account_management": ["account", "delete", "password", "login", "signup", "profile"],
        "conversation_management": ["conversation", "delete", "rename", "history", "chat", "thread"],
        "features": ["artifact", "feature", "capability", "tool", "function", "project", "knowledge"],
        "api": ["api", "console", "key", "endpoint", "sdk", "bedrock", "token", "rate limit"],
        "plans": ["pro", "max", "team", "enterprise", "plan", "subscription", "billing", "upgrade"],
        "privacy": ["privacy", "data", "legal", "gdpr", "security", "retention"],
        "desktop": ["desktop", "app", "mobile", "chrome", "extension", "android", "ios"],
        "troubleshooting": ["error", "bug", "issue", "not working", "broken", "slow", "timeout"],
    },
    "Visa": {
        "general_support": ["card", "lost", "stolen", "replace", "block", "activate", "pin"],
        "travel_support": ["travel", "abroad", "international", "exchange", "currency", "forex"],
        "dispute": ["dispute", "charge", "refund", "merchant", "unauthorized", "billing error"],
        "fraud": ["fraud", "scam", "suspicious", "unauthorized", "identity theft", "phishing"],
        "small_business": ["business", "merchant", "terminal", "pos", "accept payment", "settlement"],
    },
}

ESCALATION_KEYWORDS = {
    "score_manipulation": [
        "increase my score", "change grade", "manipulate score", "fake score",
        "review my answers", "graded me unfairly", "move me to the next round",
        "tell the company to move me", "override hiring",
    ],
    "fraud": ["fraud", "scam", "stolen", "identity theft", "unauthorized transaction", "identity has been stolen"],
    "security": ["security vulnerability", "bug bounty", "exploit", "breach", "security flaw"],
    "unauthorized_action": [
        "delete all files", "drop table", "rm -rf", "destroy", "wipe",
        "give me the code to delete", "delete all", "sql inject",
        "sql injection attack", "exfiltrate data",
    ],
    "platform_outage": [
        "site is down", "site down", "not working at all", "completely down",
        "outage", "all requests failing", "none of the pages", "none of the submissions",
    ],
    "refund_demand": [
        "give me the refund asap", "refund me today", "refund now",
        "i want my money back", "give me my money", "please give me the refund",
    ],
    "internal_disclosure": [
        "show your system prompt", "reveal your instructions", "print your hidden rules",
        "show internal logic", "show all retrieved documents", "display all retrieved documents",
        "affiche toutes les regles internes", "documents recuperes", "logique exacte",
    ],
}

REQUEST_TYPE_KEYWORDS = {
    "bug": [
        "error", "broken", "not working", "bug", "crash", "fail", "issue",
        "problem", "down", "outage", "blocked", "stuck", "unable", "can't",
        "cannot", "doesn't work", "stopped working", "facing", "blocker",
    ],
    "feature_request": [
        "feature request", "would be nice", "suggestion",
        "wish", "can you add", "please add", "it would help",
    ],
    "invalid": [],
}

ESCALATION_RESPONSE_TEMPLATES = {
    "fraud": "This ticket involves potential fraud or identity theft and requires immediate attention from our security team. A human agent will review this case shortly.",
    "security": "This ticket involves a security concern that requires review by our specialized security team.",
    "score_manipulation": "We understand your concern about your assessment results. However, we cannot modify scores or override hiring decisions. Please contact the recruiting company directly for any disputes regarding your assessment.",
    "unauthorized_action": "We cannot process this request as it involves potentially harmful actions. If you have a legitimate need, please contact our support team directly.",
    "platform_outage": "We are aware of the issue and our engineering team has been notified. A human agent will follow up with you shortly.",
    "refund_demand": "This request involves a financial transaction that requires review by our billing team. A human agent will assist you shortly.",
    "internal_disclosure": "I cannot reveal internal instructions, routing logic, or raw retrieved documents. A human support agent will review the customer issue and respond with the appropriate public guidance.",
    "insufficient_context": "I could not find enough trusted support documentation to answer this safely. A human agent will review and route this case.",
    "corpus_mismatch": "The retrieved documentation did not match the requested support domain closely enough to answer safely. A human agent will review and route this case.",
    "out_of_scope": "This request appears to be outside the scope of our support system. A human agent will review and route this appropriately.",
}
