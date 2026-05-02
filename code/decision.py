from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SourceRef:
    company: str
    title: str
    section: str
    source: str
    url: str = ""
    chunk_id: str = ""
    score: float = 0.0

    @classmethod
    def from_result(cls, document, score: float) -> "SourceRef":
        meta = document.metadata
        return cls(
            company=meta.get("company", "Unknown"),
            title=meta.get("title", ""),
            section=meta.get("section", ""),
            source=meta.get("source", ""),
            url=meta.get("url", ""),
            chunk_id=str(meta.get("chunk_id", "")),
            score=float(score),
        )

    def label(self) -> str:
        parts = [p for p in [self.source, self.section] if p]
        return "#".join(parts) if parts else self.title or "unknown_source"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company": self.company,
            "title": self.title,
            "section": self.section,
            "source": self.source,
            "url": self.url,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
        }


@dataclass
class TriageDecision:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str
    company: str = "Unknown"
    resolution_status: str = "resolved"
    confidence: float = 0.0
    sources: List[SourceRef] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    sanitized_query: str = ""
    sanitized_subject: str = ""
    context_chunks: List[str] = field(default_factory=list)
    telemetry: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        source_labels = [s.label() for s in self.sources]
        return {
            "status": self.status,
            "product_area": self.product_area,
            "response": self.response,
            "justification": self.justification,
            "request_type": self.request_type,
            "company": self.company,
            "resolution_status": self.resolution_status,
            "confidence": round(self.confidence, 3),
            "sources": source_labels,
            "source_details": [s.to_dict() for s in self.sources],
            "risk_flags": self.risk_flags,
            "sanitized_query": self.sanitized_query,
            "sanitized_subject": self.sanitized_subject,
            "context_chunks": self.context_chunks,
            "telemetry": {k: round(v, 4) for k, v in self.telemetry.items()},
            "escalation_status": self.status == "escalated",
        }

    def to_submission_row(self, issue: str, subject: str, input_company: str) -> Dict[str, str]:
        return {
            "status": self.status,
            "product_area": self.product_area,
            "response": self.response,
            "justification": self.justification,
            "request_type": self.request_type,
        }
