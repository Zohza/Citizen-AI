"""Agentic RAG loop for Citizen AI using tool calling.

The agent behaves like a conversational assistant that decides when to use
retrieval tools vs respond directly. It maintains conversation history and
uses Groq's function calling to control the flow.
"""

import json
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
from vectorstore import get_collection
from llm import get_embeddings
from schemas import Citation

# Tool definitions for function calling

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_nelfund",
            "description": (
                "Search the NELFUND (Nigerian Education Loan Fund) knowledge base "
                "for information about student loans, eligibility, application process, "
                "fees, required documents, and other loan-related topics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query about NELFUND",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_cac",
            "description": (
                "Search the CAC (Corporate Affairs Commission) knowledge base "
                "for information about business registration, company formation, "
                "corporate matters, and related topics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query about CAC",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a warm, knowledgeable assistant for Nigerian government services. "
    "You help users with questions about NELFUND (Nigerian Education Loan Fund - student loans) "
    "and CAC (Corporate Affairs Commission - business registration).\n\n"
    "Your behavior:\n"
    "1. When a user asks about NELFUND or CAC topics that need specific facts (eligibility, "
    "application steps, fees, documents, etc.), call the appropriate retrieval tool "
    "(retrieve_nelfund or retrieve_cac) to search the knowledge base.\n"
    "2. For greetings, small talk, or general conversation, respond naturally without calling tools.\n"
    "3. When a question is vague or ambiguous (e.g., just 'loan' or 'help'), ask a friendly "
    "clarifying question instead of calling a tool. Ask in your own words based on context.\n"
    "4. If a user asks about topics outside NELFUND and CAC (e.g., tax, visas, driver's license, "
    "passport, immigration, healthcare), politely explain that you currently only help with NELFUND "
    "and CAC, but mention you're expanding soon. Use a warm, apologetic tone.\n"
    "5. Keep responses to a maximum of two paragraphs when providing factual information.\n"
    "6. Never invent facts, fees, amounts, or procedural steps. If the knowledge base doesn't have "
    "the information, say so and suggest the official portal (nelfund.gov.ng or cac.gov.ng).\n"
    "7. Use the conversation history to maintain context across multiple turns."
)


async def retrieve_documents(
    query: str,
    agency: str,
) -> tuple[list[dict], list[Citation]]:
    """Retrieve documents from the vector store."""
    collection = get_collection()
    embeddings = get_embeddings()

    # Embed the query
    query_embedding = embeddings.embed_query(query)

    # Query the collection with agency filter
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=6,
        where={"agency": agency},
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    citations = []

    if results and results["documents"] and len(results["documents"]) > 0:
        for i, doc_text in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}

            # ChromaDB uses distance; filter by cosine distance < 0.3 (similarity > 0.7)
            if distance < 0.3:
                chunks.append({
                    "text": doc_text,
                    "metadata": metadata,
                })
                
                source_name = metadata.get("source_name", "Unknown")
                page = metadata.get("page", 0)
                if page and page != "":
                    citations.append(Citation(
                        source_name=source_name,
                        page=int(page) if isinstance(page, (int, str)) else 0,
                        agency=agency,
                    ))

    return chunks, citations


def _format_retrieved_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into context for the LLM."""
    if not chunks:
        return ""
    
    parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source_name", "Unknown")
        page = meta.get("page", "")
        page_info = f" (Page {page})" if page else ""
        parts.append(f"[Source: {source}{page_info}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


async def run_agent(
    user_message: str,
    thread_id: str,
    chat_history: list[dict],
) -> AsyncGenerator[dict, None]:
    """Run the agentic loop for a single user turn.
    
    Yields dictionaries with:
    - {"type": "token", "data": "token text"} for streaming tokens
    - {"type": "final", "data": {...}} for final metadata (citations, checklist)
    """
    
    print(f"[agent] Starting for thread {thread_id}: {user_message[:60]}...")
    
    # Initialize the LLM without tools first (Groq tool calling has strict format requirements)
    try:
        llm = ChatOpenAI(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            temperature=0.2,
            max_tokens=2048,
            timeout=60,
        )
        print(f"[agent] LLM initialized successfully")
    except Exception as e:
        print(f"[agent] Error initializing LLM: {e}")
        yield {"type": "token", "data": f"Error: {str(e)[:100]}"}
        yield {"type": "final", "data": {"citations": []}}
        return

    # Build messages
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    for msg in chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    
    messages.append(HumanMessage(content=user_message))

    print(f"[agent] Built message history with {len(messages)} messages")

    # First call: Let the model decide what to do (with tools bound)
    print(f"[agent] Calling LLM with tools...")
    
    try:
        llm_with_tools = llm.bind_tools(TOOLS, tool_choice="auto")
        response = await llm_with_tools.ainvoke(messages)
        print(f"[agent] Got response from LLM")
    except Exception as e:
        # If tool calling fails, fall back to direct response without tools
        print(f"[agent] Tool calling failed, falling back to direct response: {e}")
        try:
            response = await llm.ainvoke(messages)
        except Exception as e2:
            print(f"[agent] Direct call also failed: {e2}")
            yield {"type": "token", "data": f"Error: {str(e2)[:100]}"}
            yield {"type": "final", "data": {"citations": []}}
            return
    
    # Check if the model wants to call tools
    tool_calls = getattr(response, "tool_calls", [])
    citations = []
    retrieved_context = ""

    print(f"[agent] Tool calls: {len(tool_calls)}")

    if tool_calls:
        print(f"[agent] Model called {len(tool_calls)} tool(s)")
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_query = tool_args.get("query", "")

            print(f"[agent]   Tool: {tool_name}, Query: {tool_query[:50]}...")

            try:
                if tool_name == "retrieve_nelfund":
                    chunks, tool_citations = await retrieve_documents(tool_query, "NELFUND")
                elif tool_name == "retrieve_cac":
                    chunks, tool_citations = await retrieve_documents(tool_query, "CAC")
                else:
                    chunks = []
                    tool_citations = []

                citations.extend(tool_citations)
                retrieved_context = _format_retrieved_context(chunks)

                # Add assistant response and tool result to messages
                messages.append(response)
                messages.append(ToolMessage(
                    tool_call_id=tool_call.get("id", ""),
                    content=retrieved_context if retrieved_context else "No relevant documents found.",
                ))
            except Exception as e:
                print(f"[agent] Error executing tool: {e}")
                messages.append(response)
                messages.append(ToolMessage(
                    tool_call_id=tool_call.get("id", ""),
                    content=f"Error retrieving documents: {str(e)[:100]}",
                ))

        # Second call: Generate final answer with context
        print(f"[agent] Calling LLM again with tool results...")
        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            print(f"[agent] Error on second call: {e}")
            yield {"type": "token", "data": f"Error generating response: {str(e)[:100]}"}
            yield {"type": "final", "data": {"citations": citations}}
            return

    # Extract and stream the final answer
    final_answer = response.content or ""
    
    print(f"[agent] Final answer ({len(final_answer)} chars), citations: {len(citations)}")
    
    if final_answer:
        # Stream tokens with natural chunking
        chunk_size = 100
        for i in range(0, len(final_answer), chunk_size):
            chunk = final_answer[i:i + chunk_size]
            yield {"type": "token", "data": chunk}

    # Emit final metadata
    yield {
        "type": "final",
        "data": {
            "citations": [c.model_dump() for c in citations],
            "checklist": None,
        }
    }
