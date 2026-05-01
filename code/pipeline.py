from typing import Dict, Any, List
from sanitizer import sanitize_input, safe_concat
from multi_intent import split_intents, detect_compound, merge_results, SubIntent
from classifier import classify_ticket
from escalation import assess_escalation
from responder import generate_response, template_response
from retriever import HybridRetriever
from config import RERANK_THRESHOLD


def triage_ticket(
    issue: str,
    subject: str = "",
    company: str = "None",
    retriever: HybridRetriever = None,
) -> Dict[str, Any]:
    cleaned_issue, injection_detected = sanitize_input(issue)
    cleaned_subject, _ = sanitize_input(subject) if subject else (subject, False)

    if injection_detected:
        return {
            "status": "escalated",
            "response": "This ticket has been flagged for review due to unusual content patterns. A human agent will review it shortly.",
            "product_area": "security",
            "request_type": "invalid",
            "justification": "Potential prompt injection detected in ticket content.",
        }

    intents = split_intents(cleaned_issue)

    if len(intents) <= 1:
        return _process_single(cleaned_issue, cleaned_subject, company, retriever)

    sub_intents = []
    for i, intent_text in enumerate(intents):
        result = _process_single(intent_text, cleaned_subject, company, retriever)
        sub_intents.append(SubIntent(
            text=intent_text,
            index=i,
            response=result["response"],
            status=result["status"],
            product_area=result["product_area"],
            request_type=result["request_type"],
            justification=result["justification"],
        ))
    return merge_results(sub_intents)


def _process_single(
    issue: str,
    subject: str,
    company: str,
    retriever: HybridRetriever,
) -> Dict[str, Any]:
    classification = classify_ticket(issue, subject, company)
    inferred_company = classification["company"]
    request_type = classification["request_type"]
    product_area = classification["product_area"]

    if request_type == "invalid":
        return {
            "status": "replied",
            "response": "I am sorry, this is out of scope from my capabilities.",
            "product_area": product_area,
            "request_type": "invalid",
            "justification": "Ticket content is not a valid support request.",
        }

    context_chunks = []
    retrieval_score = 0.0
    rerank_score = 0.0

    if retriever:
        query = f"{subject} {issue}" if subject else issue
        results = retriever.retrieve(query, top_k=10)
        if results:
            context_chunks = [r[0].content for r in results]
            retrieval_score = results[0][1] if results else 0.0
            rerank_score = results[0][1] if results else 0.0

    escalation = assess_escalation(
        issue, subject,
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
        rerank_threshold=RERANK_THRESHOLD,
    )

    if escalation["escalated"]:
        return {
            "status": "escalated",
            "response": escalation["response_template"],
            "product_area": product_area,
            "request_type": request_type,
            "justification": f"Escalated due to: {escalation['reason']}",
        }

    llm_response = generate_response(
        query=issue,
        context_chunks=context_chunks,
        company=inferred_company,
        product_area=product_area,
        request_type=request_type,
        subject=subject,
    )

    if llm_response:
        response = llm_response
    else:
        response = template_response(issue, context_chunks, inferred_company, product_area)

    return {
        "status": "replied",
        "response": response,
        "product_area": product_area,
        "request_type": request_type,
        "justification": f"Response generated based on {inferred_company} support documentation for {product_area}.",
    }
