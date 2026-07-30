# Citizen AI

A multilingual conversational AI assistant that helps Nigerian citizens navigate government services — currently supporting **NELFUND** (Nigerian Education Loan Fund) and **CAC** (Corporate Affairs Commission).

Users can ask questions about student loan applications, business registration, eligibility, fees, required documents, and application procedures. The assistant responds with citations, step-by-step checklists, and links to official portals — all in English, Pidgin, Hausa, Yoruba, or Igbo.

---

## Architecture

```
frontend/          React + Vite + TypeScript + shadcn/ui
backend/           FastAPI + LangGraph + ChromaDB + Groq LLM
data/pdfs/         Source PDFs ingested into the knowledge base
```

The backend uses an **agentic RAG** pipeline: a router classifies the user's intent and agency, a retriever fetches relevant chunks from a ChromaDB vector store, and an answer LLM generates a grounded response with source citations. Responses stream token-by-token via Server-Sent Events.

---

## Features

- **Agentic RAG** — Intent classification, vector retrieval, and grounded answer generation via LangGraph
- **Two government agencies** — NELFUND (student loans) and CAC (business registration)
- **Multilingual** — English, Pidgin, Hausa, Yoruba, and Igbo
- **Streaming responses** — Real-time token streaming from backend to frontend via SSE
- **Source citations** — Every answer cites its source document and page number
- **Step-by-step checklists** — Structured application checklists with progress tracking
- **Scam detection** — Client-side keyword detection flags potential fraud attempts
- **PDF ingestion** — Upload PDFs via API or CLI to grow the knowledge base
- **Conversation history** — Thread-based chat with persistent history via SQLite
- **Dark mode** — System-preference-aware theme toggle

---

## Tech Stack

### Backend
- **FastAPI** — REST + SSE API server
- **LangGraph** — Agentic state machine for routing, retrieval, and answering
- **Groq** — LLM inference (llama-3.3-70b-versatile) via OpenAI-compatible API
- **NVIDIA NIM** — Embeddings (nv-embedqa-e5-v5)
- **ChromaDB** — Vector store with cosine similarity search
- **LangChain** — PDF loading, text splitting, and LLM tool-calling wrappers

### Frontend
- **React 19** — UI framework
- **Vite** — Build tool and development server
- **TypeScript** — Type safety
- **Tailwind CSS v4** — Utility-first styling
- **shadcn/ui** — Accessible components built on Radix UI
- **Framer Motion** — Animations

---

## Getting Started

### Prerequisites

- Node.js 18+ or Bun
- Python 3.10+
- A Groq API key (https://console.groq.com/keys)
- An NVIDIA API key for embeddings (https://build.nvidia.com/)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Create .env file with your API keys
cp .env.example .env
# Edit .env: set GROQ_API_KEY and NVIDIA_API_KEY

# Start the server
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install    # or bun install
npm run dev    # or bun run dev
```

The frontend runs at `http://localhost:3000` (or the next available port) and connects to the backend at `http://localhost:8000`.

---

## API Endpoints

| Method | Path              | Description                                  |
|--------|-------------------|----------------------------------------------|
| GET    | `/health`         | Health check                                 |
| POST   | `/chat`           | Stream an answer to a user query (SSE)       |
| POST   | `/admin/upload`   | Upload a PDF for ingestion into the KB       |

### Chat Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "my-thread", "query": "How do I apply for a NELFUND loan?"}'
```

The server responds with a Server-Sent Events stream:
```
data: {"event":"token","data":"To apply..."}
data: {"event":"final","data":{"citations":[...],"checklist":{...}}}
data: [DONE]
```

---

## PDF Ingestion

### Automatic (startup)

Place PDFs in `backend/data/pdfs/` before starting the server. On startup, the application automatically ingests all PDFs into the vector store.

Optionally include a `manifest.json` alongside the PDFs to specify agency and document type:

```json
{
  "NELFUND_guidelines.pdf": {"agency": "NELFUND", "doc_type": "guidelines"},
  "CAC_fees.pdf": {"agency": "CAC", "doc_type": "fee_schedule"}
}
```

Without a manifest, filenames starting with `NELFUND_` or `CAC_` are auto-detected.

### Manual via API

```bash
curl -X POST http://localhost:8000/admin/upload \
  -F "file=@document.pdf" \
  -F "agency=NELFUND" \
  -F "doc_type=guidelines"
```

### Manual via CLI

```bash
cd backend
python ingestion.py --file data/pdfs/NELFUND_guidelines.pdf --agency NELFUND --doc-type guidelines
python ingestion.py --dir data/pdfs
```

---

## Project Structure

```
Citizen-AI/
├── backend/
│   ├── main.py           # FastAPI server and endpoints
│   ├── graph.py          # LangGraph state machine (router → retrieval → answer → citation)
│   ├── agent.py          # Tool-calling agent implementation (alternative to graph.py)
│   ├── ingestion.py      # PDF loading, chunking, embedding, and upserting
│   ├── vectorstore.py    # ChromaDB client and collection management
│   ├── llm.py            # LLM and embeddings client factories (Groq + NVIDIA)
│   ├── config.py         # Environment variable configuration
│   ├── schemas.py        # Pydantic request/response models
│   ├── memory.py         # LangGraph checkpoint persistence (SQLite / in-memory)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx       # Main application component with SSE streaming logic
│   │   ├── constants.ts  # Knowledge base, multilingual strings, prompt cards
│   │   ├── types.ts      # TypeScript type definitions
│   │   ├── main.tsx      # Application entry point
│   │   └── components/   # UI components (shadcn/ui and custom)
│   ├── package.json
│   └── vite.config.ts
└── data/
    └── pdfs/             # Source PDFs for ingestion
```

---

## Multilingual Support

Citizen AI detects the user's language from their message and responds in kind. Supported languages:

- English
- Nigerian Pidgin
- Hausa
- Yoruba
- Igbo

Language detection uses keyword matching against curated word lists. Quick-prompt cards and the welcome screen adapt to the detected language.

---

## Scam Detection

The frontend includes a real-time scam keyword detection system. When a user message contains known scam indicators (requests for agent fees, processing charges, middleman payments, etc.), a prominent warning banner appears in the assistant's response.

---

## Contributing

Contributions are welcome. Feel free to submit issues and pull requests.

---

## License

Built as part of the Citizen AI project.
