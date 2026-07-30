# Citizen AI — Build Instructions

Scope: NELFUND + CAC only. Stack: existing React/Vite frontend (moved into `/frontend`) + new FastAPI backend (in `/backend`) with LangGraph, ChromaDB, and NVIDIA 
NIM.

Repo layout to produce:

```
citizen-ai/
├── frontend/     # existing React/Vite app moves here, unchanged except for the API call
└── backend/      # new FastAPI + LangGraph + ChromaDB service
```

Execute the following steps in order. Do not skip ahead or reorder.

---

## Step 1 — Restructure into `/frontend` and `/backend`

1. Create a root folder `citizen-ai/`.
2. Move all existing project files (`src/`, `public/`, `package.json`, `vite.config.ts`, `tsconfig*.json`, etc.) into `citizen-ai/frontend/`.
3. Confirm the frontend still runs unchanged: `cd frontend && npm run dev` should serve the app exactly as before, with no functional changes yet.
4. Create `citizen-ai/backend/` as an empty folder, ready for Step 2.

---

## Step 2 — Scaffold the backend

Inside `backend/`, create:

```
backend/
├── main.py
├── graph.py
├── ingestion.py
├── vectorstore.py
├── llm.py
├── memory.py
├── schemas.py
├── config.py
├── requirements.txt
└── data/
    ├── chroma/
    └── pdfs/
```

`requirements.txt`:
```
fastapi
uvicorn
langgraph
langchain-nvidia-ai-endpoints
langchain-community
chromadb
python-multipart
python-dotenv
```

`config.py`: load from `.env` — `NVIDIA_API_KEY`, `NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1`, `CHROMA_PERSIST_DIR=./data/chroma`, `MEMORY_DB_PATH=./data/memory.sqlite`.

---

## Step 3 — Vector store (`vectorstore.py`)

1. Set up a persistent ChromaDB client using `CHROMA_PERSIST_DIR`.
2. Create/load a single collection named `citizen_ai_kb`.
3. Expose a function `get_collection()` that other modules import.

---

## Step 4 — PDF ingestion (`ingestion.py`)

1. Accept a PDF file path, an `agency` value (`"NELFUND"` or `"CAC"`), and a `doc_type` value.
2. Extract text from the PDF, preserving page numbers.
3. Chunk text into ~500–800 token pieces with ~50–100 token overlap.
4. For each chunk, store metadata: `agency`, `doc_type`, `source_name` (the filename), `page`.
5. Embed each chunk (via NVIDIA NIM embeddings, or a local embedding model if NIM embeddings are unavailable) and upsert into the `citizen_ai_kb` collection.
6. Add a CLI entry point: running `python ingestion.py` processes every PDF found in `backend/data/pdfs/`, using the filename to help determine `agency` (e.g. prompt for agency/doc_type per file, or read from a small `manifest.json` placed alongside the PDFs mapping filename → agency/doc_type).

---

## Step 5 — NVIDIA NIM client (`llm.py`)

1. Instantiate two `ChatNVIDIA` clients:
   - `router_llm` — a fast/small model, used for classification only.
   - `answer_llm` — a stronger model, with `streaming=True`, used for final answer generation.
2. Instantiate an embeddings client for use by `ingestion.py` and the retrieval step.
3. Read model names from `config.py` so they can be changed without touching this file.

---

## Step 6 — LangGraph pipeline (`graph.py`)

1. Define state:
```python
class CitizenAIState(TypedDict):
    thread_id: str
    user_query: str
    chat_history: list
    detected_agency: str
    detected_intent: str
    retrieved_chunks: list
    draft_answer: str
    final_answer: str
    citations: list
    checklist: dict | None
    needs_clarification: bool
    clarification_question: str
```

2. Build node **`router_node`**:
   - Input: `user_query`, `chat_history`.
   - Use `router_llm` to classify `detected_agency` (`NELFUND` / `CAC` / `unclear`) and `detected_intent` (`eligibility` / `process_steps` / `fees` / `documents_required` / `unclear`).
   - Use `chat_history` so a follow-up question can inherit the agency from a previous turn.
   - If either value is `unclear`, set `needs_clarification = True` and generate one clarifying question. Do not proceed to retrieval in this case.

3. Build node **`retrieval_node`**:
   - Input: `detected_agency`, `detected_intent`, `user_query`.
   - Query the `citizen_ai_kb` collection, filtered by `agency` metadata (and `doc_type` if the intent maps cleanly to one).
   - Return top 4–6 chunks by similarity.
   - If no chunk clears a minimum similarity threshold, set a flag on state indicating no grounding was found.

4. Build node **`answer_node`**:
   - Input: `user_query`, `retrieved_chunks`.
   - Use `answer_llm` to generate `draft_answer`, using ONLY the retrieved chunks as context.
   - System instruction: answer only from the provided context; if the context does not cover the question, say so explicitly and recommend the official NELFUND/CAC portal — never invent facts, fees, or steps.
   - This node must yield tokens as a generator/stream, not a single blocking return.

5. Build node **`citation_node`**:
   - Input: `draft_answer`, `retrieved_chunks`.
   - Attach `citations`: a list of `{source_name, page, agency}` for each chunk actually used.
   - If `detected_intent == "process_steps"`, generate a `checklist` object with this shape (matching the frontend's existing `Checklist` type): title, list of steps, required documents, official cost, processing time, portal URL.
   - Set `final_answer`.

6. Wire edges: `router_node → (needs_clarification: END) | (else: retrieval_node) → answer_node → citation_node → END`.

7. Compile the graph with a checkpointer for persistent memory:
```python
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string(MEMORY_DB_PATH)
graph = builder.compile(checkpointer=memory)
```
If `langgraph.checkpoint.sqlite` is unavailable in the installed version, use `MemorySaver` instead and note in a code comment that memory will not survive a process restart.

8. Every invocation of the graph must pass `config={"configurable": {"thread_id": thread_id}}` so LangGraph loads/saves history per thread automatically.

---

## Step 7 — FastAPI app (`main.py`)

1. Create the FastAPI app with CORS enabled for the frontend's origin.
2. On startup, load the vectorstore collection and the compiled LangGraph graph once (not per-request).
3. Endpoint `POST /chat`:
   - Body: `{thread_id: str, query: str}`.
   - Stream the graph's `answer_node` output token-by-token to the client via Server-Sent Events.
   - After streaming completes, send one final structured SSE event containing `{citations, checklist, detected_agency}`.
4. Endpoint `POST /admin/upload`:
   - Accepts a PDF file, `agency`, and `doc_type` as form data.
   - Saves the PDF into `backend/data/pdfs/`, then calls the same chunk/embed/upsert logic as `ingestion.py`.
5. Endpoint `GET /health`: returns `{status: "ok"}`.

---

## Step 8 — Frontend integration (`frontend/src/App.tsx`)

1. Remove `classifyIntent()`, `retrieveKnowledge()`, `generateNELFUNDResponse()`, `generateCACResponse()`.
2. Add a `sendToBackend(threadId: string, query: string)` function that:
   - Opens a stream to `POST http://localhost:8000/chat` with `{thread_id: threadId, query}`.
   - Appends incoming tokens to the current assistant message's text as they arrive.
   - On the final structured SSE event, sets `citations` and `checklist` on that message using the existing `Message` type fields — no changes needed to `MainContent.tsx` rendering logic.
3. Pass the frontend's existing thread ID (from `generateId()` / the active thread) as `thread_id` on every call — this must be the same ID across the whole conversation in that thread.
4. Keep `detectScam()` and `detectLanguage()` exactly as they are, running client-side before any backend call.
5. Update `frontend/src/types.ts` only if new fields are needed on `Message` to hold streaming state (e.g. an `isStreaming: boolean` flag) — do not change existing type shapes for `citations` or `checklist`.

---

## Step 9 — Verify end-to-end, in this order

1. `POST /health` returns OK.
2. Run `ingestion.py` against the NELFUND and CAC PDFs placed in `backend/data/pdfs/`; confirm chunks appear in the ChromaDB collection with correct metadata.
3. Call the compiled graph directly (a small test script, not via the API) with one NELFUND query and one CAC query; confirm each returns a correct, cited answer.
4. Send a two-turn conversation to the graph directly (e.g. "how do I apply for NELFUND" then "what documents do I need") using the same `thread_id`; confirm the second answer correctly infers NELFUND without it being restated.
5. Start the FastAPI server; call `POST /chat` via curl or Postman; confirm SSE streaming works and the final citations/checklist event arrives.
6. Wire the frontend's `sendToBackend` call in; confirm the chat UI streams responses and renders citations/checklists with no visual regressions.
7. Test one out-of-scope query (e.g. asking about a passport/NIMC process) and confirm it triggers the clarification path rather than a hallucinated answer.