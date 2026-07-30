"""LangGraph pipeline for the Citizen AI conversational agent.

Nodes
-----
router_node      Classify agency (NELFUND / CAC / unclear) and intent.
retrieval_node   Query ChromaDB for relevant knowledge chunks.
answer_node      Generate a streaming answer from retrieved context.
citation_node    Attach source citations and optional checklist.

Retry behaviour
---------------
All outbound calls to the NVIDIA API are wrapped with retry logic for
transient failures (timeouts, connection errors).  The answer streaming
node only retries when NO tokens have been received yet — if streaming
started successfully and then fails mid-stream, the error propagates
immediately (retrying would duplicate partial output).
"""

import asyncio
import json
from typing import Optional

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import HumanMessage, SystemMessage

from memory import get_checkpointer
from schemas import Checklist, ChecklistItem
from llm import get_router_llm, get_answer_llm, get_embeddings
from vectorstore import get_collection


# ── Retry helpers ──

_MAX_RETRIES = 3
_RETRYABLE_KEYWORDS = ["timeout", "read timed out", "connection", "connect",
                       "503", "502", "429", "too many requests", "rate limit",
                       "service unavailable", "internal server error",
                       "bad gateway"]


def _is_retryable(exc: Exception) -> bool:
    """Return True when *exc* looks like a transient upstream failure."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _RETRYABLE_KEYWORDS)


async def _async_retry(coro_factory, label: str = "operation"):
    """Run an async callable up to ``_MAX_RETRIES`` times on retryable errors.

    The callable is a zero-argument factory (typically a lambda) so a fresh
    call/stream is created on each attempt — essential for streaming where
    the iterator cannot be rewound.
    """
    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            if not _is_retryable(e):
                raise
            if attempt < _MAX_RETRIES:
                wait = 2 ** attempt  # 2, 4, 8 seconds
                print(f"  ⏳ {label} timed out, retry {attempt}/{_MAX_RETRIES - 1} "
                      f"in {wait}s...")
                await asyncio.sleep(wait)
    # All retries exhausted
    raise last_error  # type: ignore[union-attr]


def _sync_retry(fn, label: str = "operation"):
    """Run a synchronous callable up to ``_MAX_RETRIES`` times on retryable errors."""
    import time
    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if not _is_retryable(e):
                raise
            if attempt < _MAX_RETRIES:
                wait = 2 ** attempt
                print(f"  ⏳ {label} timed out, retry {attempt}/{_MAX_RETRIES - 1} "
                      f"in {wait}s...")
                time.sleep(wait)
    raise last_error  # type: ignore[union-attr]


# ── Graph State ──


class CitizenAIState(dict):
    """Mutable state dict for the Citizen AI conversation graph.

    Using ``dict`` subclass (instead of ``TypedDict``) so generator nodes
    can yield partial state updates naturally.
    """
    thread_id: str
    user_query: str
    chat_history: list        # list[dict] — previous {role, content} messages
    detected_agency: str      # "NELFUND" | "CAC" | "unclear" | specific out-of-scope topic
    detected_intent: str      # "eligibility" | "process_steps" | "fees" |
                              # "documents_required" | "unclear"
    retrieved_chunks: list    # list[dict] — top-K chunks from vector store
    draft_answer: str
    final_answer: str
    citations: list           # list[dict] — {source_name, page, agency}
    checklist: Optional[Checklist]
    needs_clarification: bool
    clarification_question: str
    is_out_of_scope: bool     # True when query is clearly about unsupported domain
    out_of_scope_topic: str   # e.g. "tax", "passport", "driver_license", or empty
    no_grounding_found: bool  # True when retrieval returned nothing useful


# ── Constants ──

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant for the Nigerian Education Loan Fund (NELFUND) "
    "and the Corporate Affairs Commission (CAC). "
    "Answer the user's question using ONLY the provided context below. "
    "If the context does not contain enough information to answer, say so "
    "explicitly and recommend consulting the relevant official portal "
    "(nelfund.gov.ng or cac.gov.ng). "
    "Never invent facts, fees, amounts, or procedural steps. "
    "Keep your response to a maximum of two paragraphs."
)

# Mapping from detected_intent to a clarifying question for the user.
_INTENT_CLARIFICATION = {
    "eligibility": "Are you asking about who qualifies, or the requirements?",
    "process_steps": "Are you asking about the application steps or procedure?",
    "fees": "Are you asking about the costs or fees involved?",
    "documents_required": "Are you asking about the required documents or paperwork?",
    "unclear": "Could you provide more detail about what you'd like to know?",
}


# ── Helpers ──


def _format_chat_history(history: list[dict], max_turns: int = 4) -> str:
    """Format chat history into a text block for the router prompt."""
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for msg in history[-max_turns:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"  {role}: {content}")
    return "\n".join(lines) + "\n"


def _format_chunks_for_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context block for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        source = meta.get("source_name", "Unknown")
        page = meta.get("page", "")
        page_info = f" (Page {page})" if page else ""
        parts.append(f"[Source: {source}{page_info}]\n{chunk['text']}")
    return "\n\n".join(parts)


def _generate_checklist(state: CitizenAIState) -> Optional[Checklist]:
    """Ask the answer LLM to produce a structured checklist from context."""
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return None

    context_text = "\n\n".join(c["text"] for c in chunks[:4])

    messages = [
        SystemMessage(content=(
            "Extract a step-by-step checklist from the context below. "
            "Respond with valid JSON only, using this schema:\n"
            '{\n'
            '  "title": "Application Steps",\n'
            '  "items": [{"text": "Step description", "id": "0"}],\n'
            '  "documents": ["Required doc 1", "Required doc 2"],\n'
            '  "official_cost": "Cost or empty string",\n'
            '  "processing_time": "Time or empty string",\n'
            '  "portal_url": "URL or empty string"\n'
            '}\n'
            'Use empty strings for any field not found in the context.'
        )),
        HumanMessage(content=f"Context:\n{context_text}"),
    ]

    llm = get_answer_llm()
    # Use non-streaming invoke — this is a short structured extraction
    # (with retry since it calls the NVIDIA API)
    response = _sync_retry(
        lambda: llm.invoke(messages),
        label="generate_checklist",
    )
    text = response.content.strip() if response else ""

    # Strip markdown fences if the model wraps JSON in them
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    try:
        items = [
            ChecklistItem(id=str(i), text=item["text"], status="todo")
            for i, item in enumerate(data.get("items", []))
        ]
        return Checklist(
            title=data.get("title", "Application Steps"),
            items=items,
            documents=data.get("documents", []),
            official_cost=data.get("official_cost", ""),
            processing_time=data.get("processing_time", ""),
            portal_url=data.get("portal_url", ""),
        )
    except (KeyError, TypeError):
        return None


# ── Node implementations ──


def router_node(state: CitizenAIState) -> dict:
    """Classify agency and intent using the router LLM.

    Uses conversation history so a follow-up question (e.g. "what documents
    do I need?") can inherit the agency from a previous turn without the
    user restating it.
    
    Also detects out-of-scope queries (e.g. tax, passport, visa, driver's license).
    """
    llm = get_router_llm()

    history_text = _format_chat_history(state.get("chat_history", []))
    user_query = state['user_query']

    prompt = (
        f"{history_text}"
        "You are a classifier for Nigerian government services. "
        "Given the user query, determine the agency and intent.\n\n"
        "Agency options:\n"
        "- NELFUND: Nigerian Education Loan Fund (student loans, education)\n"
        "- CAC: Corporate Affairs Commission (business / company registration)\n"
        "- unclear: if it doesn't relate to NELFUND or CAC\n\n"
        "Intent options:\n"
        "- eligibility: who qualifies, requirements, conditions\n"
        "- process_steps: how to apply, steps, procedure\n"
        "- fees: costs, charges, how much\n"
        "- documents_required: required documents, paperwork\n"
        "- unclear: if the intent is not clear\n\n"
        f"User query: {user_query}\n\n"
        "Respond with JSON only: {\"agency\": \"...\", \"intent\": \"...\"}"
    )

    response = _sync_retry(
        lambda: llm.invoke([HumanMessage(content=prompt)]),
        label="router_node",
    )
    text = (response.content or "").strip() if response else ""

    # Extract JSON from possible markdown code fences
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {}

    agency = result.get("agency", "unclear")
    intent = result.get("intent", "unclear")

    # Validate against allowed values
    if agency not in ("NELFUND", "CAC", "unclear"):
        agency = "unclear"
    if intent not in ("eligibility", "process_steps", "fees", "documents_required", "unclear"):
        intent = "unclear"

    print(f"[router_node] Classified -> Agency: {agency} | Intent: {intent}")

    # ── Out-of-scope detection ──
    # If agency is "unclear", check if it's actually out-of-scope (tax, passport, visa, etc.)
    # rather than genuinely ambiguous about NELFUND/CAC
    if agency == "unclear":
        # First, check for common greeting patterns which should NOT be treated as out-of-scope
        greeting_keywords = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "how are you", "what's up"]
        query_lower = user_query.lower().strip()
        is_greeting = any(greeting in query_lower for greeting in greeting_keywords) and len(user_query) < 50
        
        if not is_greeting:
            # Check if the query is about a different domain
            out_of_scope_prompt = (
                f"Is the following query clearly about a topic OTHER than Nigerian student loans (NELFUND) "
                f"or business registration (CAC)? Common out-of-scope topics include taxes, passport issues, "
                f"visa policies, driver's license renewal, immigration, healthcare, etc.\n"
                f"Respond 'true' only if the query is CLEARLY about one of these other domains. "
                f"If you're uncertain or it could possibly be about NELFUND/CAC, respond 'false'.\n\n"
                f"Query: {user_query}\n\n"
                f"Respond with JSON only: {{\"is_out_of_scope\": true|false, \"topic\": \"tax|passport|visa|driver_license|immigration|healthcare|etc|unclear\"}}"
            )
            
            out_scope_response = _sync_retry(
                lambda: llm.invoke([HumanMessage(content=out_of_scope_prompt)]),
                label="router_node.out_of_scope_check",
            )
            out_scope_text = (out_scope_response.content or "").strip() if out_scope_response else ""
            
            # Extract JSON from possible markdown code fences
            if "```" in out_scope_text:
                out_scope_text = out_scope_text.split("```")[1]
                if out_scope_text.startswith("json"):
                    out_scope_text = out_scope_text[4:]
                out_scope_text = out_scope_text.strip()
            
            try:
                out_scope_result = json.loads(out_scope_text)
            except json.JSONDecodeError:
                out_scope_result = {"is_out_of_scope": False, "topic": "unclear"}
            
            is_out_of_scope = out_scope_result.get("is_out_of_scope", False)
            out_of_scope_topic = out_scope_result.get("topic", "unclear")
        else:
            is_out_of_scope = False
            out_of_scope_topic = "unclear"
        
        if is_out_of_scope:
            print(f"[router_node] Out-of-scope detected: topic={out_of_scope_topic}")
            out_of_scope_message = (
                "Right now I can only help with questions about NELFUND (Nigerian Education Loan Fund) "
                "and CAC (Corporate Affairs Commission). I'm not able to assist with that yet — but we're "
                "actively expanding, and in the coming days I'll be able to help with things like tax issues, "
                "driver's license matters, international passport services, visa policies, and more. "
                "Thanks for your patience, and feel free to ask me anything about NELFUND or CAC in the meantime!"
            )
            return {
                "detected_agency": out_of_scope_topic,  # e.g., "tax", "passport"
                "detected_intent": intent,
                "is_out_of_scope": True,
                "out_of_scope_topic": out_of_scope_topic,
                "needs_clarification": False,
                "clarification_question": "",
                "draft_answer": out_of_scope_message,
                "final_answer": out_of_scope_message,
                "citations": [],
                "checklist": None,
            }

    if agency == "unclear" or intent == "unclear":
        # Build a helpful clarification question
        if agency == "unclear" and intent == "unclear":
            question = (
                "I'd be happy to help! Are you asking about "
                "**NELFUND** (student loans) or **CAC** (business registration)? "
                "And what would you like to know — eligibility, application steps, "
                "fees, or required documents?"
            )
        elif agency == "unclear":
            question = (
                "Could you please specify which agency you're asking about? "
                "**NELFUND** (student loans) or **CAC** (business registration)?"
            )
        else:
            question = _INTENT_CLARIFICATION.get(intent, _INTENT_CLARIFICATION["unclear"])

        return {
            "detected_agency": agency,
            "detected_intent": intent,
            "is_out_of_scope": False,
            "out_of_scope_topic": "",
            "needs_clarification": True,
            "clarification_question": question,
            "draft_answer": question,
            "final_answer": question,
            "citations": [],
            "checklist": None,
        }

    return {
        "detected_agency": agency,
        "detected_intent": intent,
        "is_out_of_scope": False,
        "out_of_scope_topic": "",
        "needs_clarification": False,
        "clarification_question": "",
    }


def retrieval_node(state: CitizenAIState) -> dict:
    """Query ChromaDB for relevant chunks, filtered by agency."""
    agency = state.get("detected_agency", "")
    query = state.get("user_query", "")
    collection = get_collection()
    embedder = get_embeddings()

    print(f"[retrieval_node] Query: '{query}' | Agency: {agency}")

    # 1. Embed the query (with retry)
    try:
        query_embedding = _sync_retry(
            lambda: embedder.embed_query(query),
            label="retrieval_node.embed",
        )
    except Exception as exc:
        print(f"[retrieval_node] embedding failed: {exc}")
        return {"retrieved_chunks": [], "no_grounding_found": True}

    # 2. Query ChromaDB — filter by agency; return 6 candidates
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=6,
            where={"agency": agency},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        print(f"[retrieval_node] query failed: {exc}")
        return {"retrieved_chunks": [], "no_grounding_found": True}

    # 3. Parse results (ChromaDB nests everything in list-of-lists per query)
    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    print(f"[retrieval_node] Retrieved {len(documents)} candidates from ChromaDB")

    if not documents:
        print(f"[retrieval_node] No documents found for agency '{agency}'")
        return {"retrieved_chunks": [], "no_grounding_found": True}

    # 4. Filter by similarity threshold
    # The collection uses cosine distance; distance = 1 - cosine_similarity.
    # Lower distance = more similar.  Threshold of 0.7 ≈ cosine sim > 0.3.
    SIMILARITY_THRESHOLD = 0.7

    chunks = []
    for i, doc in enumerate(documents):
        dist = distances[i] if i < len(distances) else None
        if dist is not None and dist >= SIMILARITY_THRESHOLD:
            print(f"[retrieval_node] Filtering out chunk (dist={dist:.3f} >= threshold)")
            continue  # below cosine-similarity cutoff
        chunks.append({
            "text": doc,
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "distance": dist,
        })

    if not chunks:
        print(f"[retrieval_node] All chunks filtered out by similarity threshold")
        return {"retrieved_chunks": [], "no_grounding_found": True}

    print(f"[retrieval_node] Keeping {len(chunks)} chunks after filtering")
    return {
        "retrieved_chunks": chunks,
        "no_grounding_found": False,
    }


async def answer_node(state: CitizenAIState) -> dict:
    """Generate a streaming answer using the answer LLM.

    Uses async streaming internally so ``astream_events`` on the compiled graph
    can capture ``on_chat_model_stream`` events for token-by-token delivery to
    the client.
    """
    llm = get_answer_llm()
    chunks = state.get("retrieved_chunks", [])

    print(f"[answer_node] Received {len(chunks)} chunks to contextualize")

    if not chunks:
        fallback = (
            "I'm sorry, I couldn't find information in my knowledge base to "
            "answer your question. For accurate information please visit the "
            "official NELFUND (nelfund.gov.ng) or CAC (cac.gov.ng) portal."
        )
        print(f"[answer_node] No chunks; returning fallback message")
        return {
            "draft_answer": fallback,
        }

    context_text = _format_chunks_for_context(chunks)

    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Context:\n{context_text}\n\nQuestion: {state['user_query']}"
        )),
    ]

    full = ""
    # Wrap the streaming call in a retry that only fires if NO tokens
    # have been received yet (retrying mid-stream would duplicate output).
    async def _stream():
        nonlocal full
        full = ""
        async for chunk in llm.astream(messages):
            content = getattr(chunk, "content", "") if chunk else ""
            if content:
                full += content
        return full

    try:
        await _async_retry(lambda: _stream(), label="answer_node")
        print(f"[answer_node] Generated {len(full)} characters")
    except Exception as exc:
        if not full:
            print(f"[answer_node] Streaming failed: {exc}")
            raise  # no partial output — safe to surface the error
        # Partial tokens were already yielded to the frontend via
        # the LangGraph astream_events machinery.  Return what we have.
        print("  ⚠ answer_node: streaming failed mid-response, "
              f"using {len(full)} chars received so far.")

    return {"draft_answer": full}


def citation_node(state: CitizenAIState) -> dict:
    """Attach source citations and optional checklist.

    Builds a deduplicated list of ``{source_name, page, agency}`` from every
    chunk returned by the retriever.  If the detected intent is
    ``process_steps``, it also generates a structured ``Checklist`` from the
    retrieved context.
    """
    chunks = state.get("retrieved_chunks", [])
    draft = state.get("draft_answer", "")

    print(f"[citation_node] Processing {len(chunks)} chunks, draft length: {len(draft)}")

    # ── Deduplicated citations ──
    seen = set()
    citations = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source_name", "")
        page = meta.get("page", 0)
        key = (source, page)
        if key not in seen and source:
            seen.add(key)
            citations.append({
                "source_name": source,
                "page": page,
                "agency": meta.get("agency", ""),
            })

    print(f"[citation_node] Generated {len(citations)} citations")

    result: dict = {
        "citations": citations,
    }

    # ── Checklist (only for process_steps) ──
    checklist = None
    if state.get("detected_intent") == "process_steps" and chunks:
        try:
            checklist = _generate_checklist(state)
            print(f"[citation_node] Generated checklist with {len(checklist.items) if checklist else 0} items")
        except Exception as exc:
            print(f"[citation_node] checklist generation failed: {exc}")
            checklist = None
    result["checklist"] = checklist

    # ── Final answer (append source footer to draft) ──
    if citations:
        footer_parts = ["\n\n**Sources:**"]
        for c in citations:
            footer_parts.append(f"- {c['source_name']} (Page {c['page']})")
        result["final_answer"] = draft + "\n".join(footer_parts)
    else:
        result["final_answer"] = draft

    return result


# ── Conditional edge ──


def route_after_router(state: CitizenAIState) -> str:
    """Route after router node.
    
    If clarification or out-of-scope, skip retrieval and go straight to END.
    Otherwise, proceed to retrieval.
    """
    if state.get("needs_clarification") or state.get("is_out_of_scope"):
        return "end"
    return "retrieval"


# ── Compile ──


def build_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Build and compile the Citizen AI state graph."""
    builder = StateGraph(CitizenAIState)

    builder.add_node("router", router_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("answer", answer_node)
    builder.add_node("citation", citation_node)

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        route_after_router,
        {"end": END, "retrieval": "retrieval"},
    )
    builder.add_edge("retrieval", "answer")
    builder.add_edge("answer", "citation")
    builder.add_edge("citation", END)

    checkpointer = checkpointer or get_checkpointer()
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
