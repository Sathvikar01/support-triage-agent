import re
from time import perf_counter
from typing import Any, Dict, List, Optional

from classifier import classify_ticket
from config import RERANK_THRESHOLD
from corpus_loader import load_corpus
from decision import SourceRef, TriageDecision
from escalation import assess_escalation
from multi_intent import split_intents
from responder import generate_response, template_response
from retriever import HybridRetriever
from sanitizer import sanitize_with_report
from validator import validate_response


_DEFAULT_RETRIEVER: Optional[HybridRetriever] = None


def build_default_retriever() -> HybridRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        documents = load_corpus()
        _DEFAULT_RETRIEVER = HybridRetriever(documents)
        _DEFAULT_RETRIEVER.build()
    return _DEFAULT_RETRIEVER


def process_ticket(query: str, subject: str = "", company: str = "None") -> Dict[str, Any]:
    retriever = build_default_retriever()
    return triage_decision(query, subject, company, retriever).to_dict()


def triage_ticket(
    issue: str,
    subject: str = "",
    company: str = "None",
    retriever: HybridRetriever = None,
) -> Dict[str, Any]:
    return triage_decision(issue, subject, company, retriever).to_dict()


def triage_decision(
    issue: str,
    subject: str = "",
    company: str = "None",
    retriever: HybridRetriever = None,
) -> TriageDecision:
    started = perf_counter()
    cleaned_issue, issue_injection, issue_pii = sanitize_with_report(issue)
    cleaned_subject, subject_injection, subject_pii = sanitize_with_report(subject)
    risk_flags = [f"pii:{key}" for key in sorted({*issue_pii.keys(), *subject_pii.keys()})]

    if not cleaned_issue.strip():
        return TriageDecision(
            status="replied",
            response="I am sorry, this ticket does not include a support issue to route.",
            product_area="general",
            request_type="invalid",
            justification="Empty ticket body.",
            company=company if company not in ("", "None", None) else "Unknown",
            resolution_status="out_of_scope",
            confidence=1.0,
            risk_flags=risk_flags,
            sanitized_query=cleaned_issue,
            sanitized_subject=cleaned_subject,
            telemetry={"total_seconds": perf_counter() - started},
        )

    if issue_injection or subject_injection:
        return TriageDecision(
            status="escalated",
            response="This ticket contains instruction-like or internal-disclosure content. A human support agent will review it before any response is sent.",
            product_area="security",
            request_type="invalid",
            justification="Escalated due to prompt-injection or internal-disclosure indicators.",
            company=company if company not in ("", "None", None) else "Unknown",
            resolution_status="high_risk",
            confidence=1.0,
            risk_flags=[*risk_flags, "prompt_injection"],
            sanitized_query=cleaned_issue,
            sanitized_subject=cleaned_subject,
            telemetry={"total_seconds": perf_counter() - started},
        )

    intents = split_intents(cleaned_issue)
    if len(intents) <= 1:
        decision = _process_single(cleaned_issue, cleaned_subject, company, retriever, risk_flags)
        decision.telemetry["total_seconds"] = perf_counter() - started
        return decision

    decisions = [
        _process_single(intent_text, cleaned_subject, company, retriever, risk_flags)
        for intent_text in intents
    ]
    merged = _merge_decisions(decisions)
    merged.sanitized_query = cleaned_issue
    merged.sanitized_subject = cleaned_subject
    merged.telemetry["total_seconds"] = perf_counter() - started
    return merged


def _process_single(
    issue: str,
    subject: str,
    company: str,
    retriever: HybridRetriever,
    inherited_risk_flags: Optional[List[str]] = None,
) -> TriageDecision:
    started = perf_counter()
    inherited_risk_flags = inherited_risk_flags or []
    classification = classify_ticket(issue, subject, company)
    inferred_company = classification["company"]
    request_type = classification["request_type"]
    product_area = classification["product_area"]

    hard_escalation = assess_escalation(issue, subject, enforce_confidence=False)
    if hard_escalation["escalated"]:
        reason = hard_escalation["reason"]
        return TriageDecision(
            status="escalated",
            response=hard_escalation["response_template"],
            product_area=product_area,
            request_type=request_type,
            justification=_build_escalation_justification(reason, inferred_company, product_area, "hard_rule"),
            company=inferred_company,
            resolution_status=_resolution_from_reason(reason),
            confidence=1.0,
            risk_flags=[*inherited_risk_flags, reason],
            sanitized_query=issue,
            sanitized_subject=subject,
            telemetry={"decision_seconds": perf_counter() - started},
        )

    if request_type == "invalid":
        return TriageDecision(
            status="replied",
            response="I am sorry, this is outside the supported HackerRank, Claude, and Visa support scope.",
            product_area=product_area,
            request_type="invalid",
            justification=_build_invalid_justification(inferred_company, product_area),
            company=inferred_company,
            resolution_status="out_of_scope",
            confidence=0.9,
            risk_flags=inherited_risk_flags,
            sanitized_query=issue,
            sanitized_subject=subject,
            telemetry={"decision_seconds": perf_counter() - started},
        )

    retrieval_started = perf_counter()
    results = []
    context_chunks: List[str] = []
    sources: List[SourceRef] = []
    confidence = 0.0
    rerank_score = 0.0

    if retriever:
        query = f"{subject} {issue}".strip()
        results = retriever.retrieve(query, top_k=10, company=inferred_company)
        context_chunks = [doc.content for doc, _ in results]
        sources = [SourceRef.from_result(doc, score) for doc, score in results]
        confidence = retriever.estimate_confidence(results)
        rerank_score = results[0][1] if results else 0.0

    source_companies = [source.company for source in sources]
    escalation = assess_escalation(
        issue,
        subject,
        retrieval_score=rerank_score,
        rerank_score=rerank_score,
        rerank_threshold=RERANK_THRESHOLD,
        confidence=confidence,
        context_count=len(context_chunks),
        source_companies=source_companies,
        expected_company=inferred_company,
    )
    if escalation["escalated"]:
        reason = escalation["reason"]
        return TriageDecision(
            status="escalated",
            response=escalation["response_template"],
            product_area=product_area,
            request_type=request_type,
            justification=_build_escalation_justification(reason, inferred_company, product_area, "confidence_gate", confidence),
            company=inferred_company,
            resolution_status=_resolution_from_reason(reason),
            confidence=confidence,
            sources=sources,
            risk_flags=[*inherited_risk_flags, reason],
            sanitized_query=issue,
            sanitized_subject=subject,
            context_chunks=context_chunks,
            telemetry={
                "retrieval_seconds": perf_counter() - retrieval_started,
                "decision_seconds": perf_counter() - started,
            },
        )

    generation_started = perf_counter()
    response = generate_response(
        query=issue,
        context_chunks=context_chunks,
        company=inferred_company,
        product_area=product_area,
        request_type=request_type,
        subject=subject,
        sources=sources,
    )
    if not response:
        response = template_response(issue, context_chunks, inferred_company, product_area, subject, sources)

    response = _clean_response(response)

    valid, validation_flags = validate_response(response, sources, status="replied")
    if not valid:
        fallback = template_response(issue, context_chunks, inferred_company, product_area, subject, sources)
        fallback = _clean_response(fallback)
        fallback_valid, fallback_flags = validate_response(fallback, sources, status="replied")
        if fallback_valid:
            response = fallback
            validation_flags = validation_flags + ["llm_response_replaced"]
        else:
            return TriageDecision(
                status="escalated",
                response="I could not produce a safe, source-grounded response for this ticket. A human support agent will review it.",
                product_area=product_area,
                request_type=request_type,
                justification=f"Escalated after response validation failed: {', '.join(fallback_flags)}.",
                company=inferred_company,
                resolution_status="insufficient_context",
                confidence=confidence,
                sources=sources,
                risk_flags=[*inherited_risk_flags, *validation_flags, *fallback_flags],
                sanitized_query=issue,
                sanitized_subject=subject,
                context_chunks=context_chunks,
                telemetry={
                    "retrieval_seconds": perf_counter() - retrieval_started,
                    "generation_seconds": perf_counter() - generation_started,
                    "decision_seconds": perf_counter() - started,
                },
            )

    return TriageDecision(
        status="replied",
        response=response,
        product_area=product_area,
        request_type=request_type,
        justification=_grounded_justification(inferred_company, product_area, request_type, confidence, sources, validation_flags),
        company=inferred_company,
        resolution_status="resolved",
        confidence=confidence,
        sources=sources,
        risk_flags=[*inherited_risk_flags, *validation_flags],
        sanitized_query=issue,
        sanitized_subject=subject,
        context_chunks=context_chunks,
        telemetry={
            "retrieval_seconds": perf_counter() - retrieval_started,
            "generation_seconds": perf_counter() - generation_started,
            "decision_seconds": perf_counter() - started,
        },
    )


def _clean_response(response: str) -> str:
    cleaned = re.sub(r"(?m)^#{1,6}\s+.*$", "", response or "")
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
    cleaned = re.sub(r"\\([>!#])", r"\1", cleaned)
    cleaned = re.sub(r"_Last updated:.*?_\s*", "", cleaned)
    cleaned = re.sub(r"\(Last updated.*?\)\s*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _merge_decisions(decisions: List[TriageDecision]) -> TriageDecision:
    from collections import Counter
    escalated = [decision for decision in decisions if decision.status == "escalated"]
    selected = escalated[0] if escalated else decisions[0]
    status = "escalated" if escalated else "replied"
    response = "\n\n".join(
        f"Issue {index + 1}: {decision.response}" for index, decision in enumerate(decisions)
    )
    sources = _unique_sources([source for decision in decisions for source in decision.sources])
    confidence = min(decision.confidence for decision in decisions) if decisions else 0.0
    risk_flags = sorted({flag for decision in decisions for flag in decision.risk_flags})
    companies = sorted({decision.company for decision in decisions if decision.company})

    product_areas = [d.product_area for d in decisions]
    request_types = [d.request_type for d in decisions]
    area_counts = Counter(product_areas)
    merged_area = area_counts.most_common(1)[0][0] if area_counts else selected.product_area
    type_counts = Counter(request_types)
    merged_type = type_counts.most_common(1)[0][0] if type_counts else selected.request_type

    return TriageDecision(
        status=status,
        response=response,
        product_area=merged_area,
        request_type=merged_type,
        justification="Multiple intents detected; responses were composed per sub-issue.",
        company=companies[0] if len(companies) == 1 else "Multiple",
        resolution_status="escalated" if escalated else "resolved",
        confidence=confidence,
        sources=sources,
        risk_flags=risk_flags,
        context_chunks=[chunk for decision in decisions for chunk in decision.context_chunks],
        telemetry={
            "sub_intent_count": len(decisions),
            "sub_intents_escalated": len(escalated),
        },
    )


def _unique_sources(sources: List[SourceRef]) -> List[SourceRef]:
    seen = set()
    unique = []
    for source in sources:
        key = (source.source, source.section, source.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def _resolution_from_reason(reason: Optional[str]) -> str:
    mapping = {
        "insufficient_context": "insufficient_context",
        "corpus_mismatch": "corpus_mismatch",
        "score_manipulation": "high_risk",
        "fraud": "high_risk",
        "security": "high_risk",
        "unauthorized_action": "high_risk",
        "internal_disclosure": "high_risk",
        "platform_outage": "escalated",
        "refund_demand": "escalated",
    }
    return mapping.get(reason, "escalated")


def _build_escalation_justification(
    reason: str, company: str, product_area: str, gate: str, confidence: float = 1.0
) -> str:
    reason_labels = {
        "fraud": "fraud or identity theft indicators detected",
        "security": "security concern requiring specialized review",
        "score_manipulation": "score modification request that cannot be fulfilled",
        "unauthorized_action": "potentially harmful action request",
        "platform_outage": "reported platform-wide service disruption",
        "refund_demand": "financial transaction requiring billing team review",
        "internal_disclosure": "attempt to extract internal system information",
        "insufficient_context": "insufficient documentation to provide a grounded answer",
        "corpus_mismatch": "retrieved documentation does not match the expected support domain",
    }
    label = reason_labels.get(reason, reason)
    return f"Escalated: {label}. Company={company}, area={product_area}, gate={gate}, confidence={confidence:.2f}."


def _build_invalid_justification(company: str, product_area: str) -> str:
    return (
        f"Ticket classified as outside support scope. "
        f"Company={company}, area={product_area}. "
        f"No matching support documentation found for the given content."
    )


def _grounded_justification(
    company: str,
    product_area: str,
    request_type: str,
    confidence: float,
    sources: List[SourceRef],
    validation_flags: List[str],
) -> str:
    source_text = "; ".join(source.label() for source in sources[:3]) or "no source"
    validation_text = ""
    if validation_flags:
        validation_text = f" Validation flags: {', '.join(validation_flags)}."
    return (
        f"Grounded response for {company}/{product_area} ({request_type}); "
        f"confidence={confidence:.2f}; sources={source_text}.{validation_text}"
    )

