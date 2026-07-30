"""FastAPI application for the Citizen AI backend.

Endpoints
---------
GET  /health         Health check.
POST /chat           Stream an answer to a user query (SSE) using agentic RAG.
POST /admin/upload   Upload a PDF for ingestion.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent import run_agent
from ingestion import ingest_pdf, ingest_all_pdfs
from schemas import ChatRequest, HealthResponse, FinalEventData, Citation, ErrorEventData, classify_error
from vectorstore import get_collection

# ── In-memory chat history storage ──
# Maps thread_id -> list[dict] of {"role": "user"|"assistant", "content": "..."}
_chat_histories = {}


# ── SSE helper ──


def _sse(event: str, data: object) -> str:
    """Format a Server-Sent Event line."""
    return f"data: {json.dumps({'event': event, 'data': data}, default=str)}\n\n"


# ── Lifespan ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the vectorstore collection on startup and ingest PDFs."""
    print("[lifespan] Initializing vector store...")
    _ = get_collection()
    
    print("[lifespan] Ingesting PDFs from ./data/pdfs...")
    try:
        results = ingest_all_pdfs()
        print(f"[lifespan] Ingestion complete: {len(results)} documents processed")
    except Exception as exc:
        print(f"[lifespan] WARNING: PDF ingestion failed: {exc}")
    
    yield


# ── App ──

app = FastAPI(title="Citizen AI Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(body: ChatRequest):
    """Stream a conversational answer token-by-token via Server-Sent Events using agentic RAG.

    The final SSE event carries ``citations`` and ``checklist``.
    """
    thread_id = body.thread_id
    user_query = body.query

    # ── Load previous chat history ──
    chat_history = _chat_histories.get(thread_id, [])

    # ── SSE event stream ──

    async def event_stream():
        """Stream tokens and metadata from the agent."""
        final_answer = ""
        final_citations = []

        try:
            # Run the agent
            async for item in run_agent(user_query, thread_id, chat_history):
                if item.get("type") == "token":
                    token = item.get("data", "")
                    final_answer += token
                    yield _sse("token", token)
                
                elif item.get("type") == "final":
                    final_data_dict = item.get("data", {})
                    final_citations = final_data_dict.get("citations", [])

        except Exception as exc:
            print(f"[chat] Agent error: {exc}")
            err_data = ErrorEventData(
                error_type=classify_error(str(exc)),
                detail=str(exc),
            )
            yield _sse("error", err_data.model_dump(mode="json"))
            return

        # ── Emit final metadata ──
        final_data = FinalEventData(
            citations=final_citations,
            checklist=None,
            detected_agency="",
        )
        yield _sse("final", final_data.model_dump(mode="json"))

        # ── Update chat history ──
        if final_answer:
            _chat_histories[thread_id] = chat_history + [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": final_answer},
            ]

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/admin/upload")
async def admin_upload(
    file: UploadFile = File(...),
    agency: str = Form(...),
    doc_type: str = Form(...),
):
    """Upload a PDF file and ingest it into the knowledge base."""
    pdfs_dir = Path("./data/pdfs")
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    # Save file to disk
    file_path = pdfs_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    # Ingest into vector store
    try:
        chunks = ingest_pdf(str(file_path), agency=agency, doc_type=doc_type)
        return {
            "filename": file.filename,
            "agency": agency,
            "doc_type": doc_type,
            "chunks_ingested": len(chunks),
        }
    except Exception as exc:
        return {
            "filename": file.filename,
            "agency": agency,
            "doc_type": doc_type,
            "error": str(exc),
        }
