"""PDF ingestion pipeline.

Extracts text from PDF files, chunks it, embeds via NVIDIA NIM,
and upserts into the ChromaDB ``citizen_ai_kb`` collection.
"""

import json
import os
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from vectorstore import get_collection
from llm import get_embeddings


def ingest_pdf(
    pdf_path: str,
    agency: str,
    doc_type: str,
) -> list[dict]:
    """Extract, chunk, embed, and upsert a single PDF into the vector store.

    Args:
        pdf_path: Path to the PDF file on disk.
        agency: ``"NELFUND"`` or ``"CAC"``.
        doc_type: The category of document (e.g. ``"guidelines"``, ``"faq"``).

    Returns:
        List of metadata dicts for every chunk that was upserted.
    """
    source_name = os.path.basename(pdf_path)

    # ── 1. Load text page-by-page ──
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # list of Document(page_content, metadata{"page": int, "source": str})

    if not pages:
        print(f"  WARNING: {pdf_path} contains no pages. Skipping.")
        return []

    # ── 2. Chunk each page independently ──
    # Per-page chunking ensures the ``page`` metadata stays correct.
    # An alternative would be to chunk the full text and track page ranges,
    # but per-page is simpler and page-spanning chunks are rare enough that
    # neighbouring chunks provide the context if needed.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: list[str] = []
    all_metadatas: list[dict] = []

    for page in pages:
        # PyPDFLoader uses 0-based page numbers; convert to 1-based for users
        page_number = page.metadata.get("page", 0) + 1
        page_text = page.page_content.strip()
        if not page_text:
            continue

        page_chunks = text_splitter.split_text(page_text)

        for chunk_text in page_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            all_chunks.append(chunk_text)
            all_metadatas.append(
                {
                    "agency": agency,
                    "doc_type": doc_type,
                    "source_name": source_name,
                    "page": page_number,
                }
            )

    if not all_chunks:
        print(f"  WARNING: No extractable text found in {source_name}. Skipping.")
        return []

    # ── 3. Embed all chunks via NVIDIA NIM ──
    embeddings_client = get_embeddings()
    try:
        embeddings = embeddings_client.embed_documents(all_chunks)
    except Exception as e:
        print(f"  ERROR: Failed to embed chunks from {source_name}: {e}")
        raise

    # ── 4. Upsert into ChromaDB ──
    ids = [str(uuid.uuid4()) for _ in all_chunks]
    collection = get_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
    )

    print(f"  ✓ {len(all_chunks)} chunks upserted from {source_name}")
    return all_metadatas


def _detect_agency_from_filename(filename: str) -> str | None:
    """Try to infer the agency from an uppercase filename prefix.

    Returns ``"NELFUND"``, ``"CAC"``, or ``None`` if the prefix is not recognised.
    """
    stem = filename.upper()
    if stem.startswith("NELFUND"):
        return "NELFUND"
    if stem.startswith("CAC"):
        return "CAC"
    return None


def ingest_all_pdfs(pdfs_dir: str = "./data/pdfs") -> list[dict]:
    """Ingest every PDF found in *pdfs_dir*, guided by an optional manifest.

    Behaviour (in priority order):

    1. If a ``manifest.json`` exists alongside the PDFs, it is used to map
       each filename → ``{"agency": "NELFUND"|"CAC", "doc_type": "…"}``.
       Files absent from the manifest are **skipped**.

    2. Without a manifest, the function tries to infer the agency from the
       filename prefix (``NELFUND_*`` or ``CAC_*``). The rest of the stem
       (after the first ``_``) is used as ``doc_type``.
       Files with unrecognised prefixes are **skipped**.

    Returns:
        A flat list of metadata dicts for every chunk upserted.
    """
    pdfs_path = Path(pdfs_dir)
    if not pdfs_path.is_dir():
        print(f"ERROR: {pdfs_dir} is not a valid directory.")
        return []

    # ── Load manifest ──
    manifest_path = pdfs_path / "manifest.json"
    manifest: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        print(f"Loaded manifest with {len(manifest)} file mapping(s).")
    else:
        print(
            "No manifest.json found. Will try to infer agency from filename "
            "(e.g. NELFUND_*.pdf, CAC_*.pdf)."
        )

    # ── Scan for PDFs ──
    pdf_files = sorted(pdfs_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdfs_dir}.")
        return []

    results: list[list[dict]] = []

    for pdf_file in pdf_files:
        filename = pdf_file.name
        agency: str | None = None
        doc_type: str | None = None

        # Priority 1: manifest entry
        if filename in manifest:
            entry = manifest[filename]
            agency = entry.get("agency")
            doc_type = entry.get("doc_type")

        # Priority 2: filename heuristic (only when there's NO manifest)
        elif not manifest:
            agency = _detect_agency_from_filename(filename)
            if agency:
                # Everything after "NELFUND_" or "CAC_" is the doc_type
                # e.g. NELFUND_guidelines.pdf → doc_type = "guidelines"
                parts = filename.split("_", 1)
                if len(parts) > 1:
                    # Strip the extension
                    doc_type = Path(parts[1]).stem
                else:
                    doc_type = "general"

        # Neither manifest nor heuristic matched
        if agency is None:
            print(
                f"  - Skipping {filename}: cannot determine agency "
                f"(no manifest entry and filename does not start with "
                f"NELFUND_ or CAC_)"
            )
            continue

        if agency not in ("NELFUND", "CAC"):
            print(
                f"  - Skipping {filename}: unknown agency '{agency}' "
                f"(must be NELFUND or CAC)."
            )
            continue

        if not doc_type:
            print(f"  - Skipping {filename}: no doc_type could be determined.")
            continue

        try:
            chunk_metas = ingest_pdf(str(pdf_file), agency=agency, doc_type=doc_type)
            results.append(chunk_metas)
        except Exception as e:
            print(f"  ✗ Failed to ingest {filename}: {e}")
            continue

    success_count = len(results)
    total_chunks = sum(len(r) for r in results)
    print(f"\nDone. {success_count}/{len(pdf_files)} PDFs ingested "
          f"({total_chunks} total chunks).")
    return [meta for batch in results for meta in batch]


# ── CLI entry point ──

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest PDFs into the Citizen AI knowledge base."
    )
    parser.add_argument(
        "--dir",
        default="./data/pdfs",
        help="Directory containing PDF files to ingest (default: ./data/pdfs)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Single PDF file to ingest (overrides --dir)",
    )
    parser.add_argument("--agency", default=None, help="NELFUND or CAC")
    parser.add_argument("--doc-type", default=None, help="Type of document")

    args = parser.parse_args()

    if args.file and args.agency and args.doc_type:
        result = ingest_pdf(args.file, agency=args.agency, doc_type=args.doc_type)
        print(f"Ingested {len(result)} chunks from {args.file}")
    elif args.file:
        print("ERROR: --file requires --agency and --doc-type too.")
    else:
        results = ingest_all_pdfs(args.dir)
        total = sum(1 for _ in results)
        print(f"Ingested {total} chunks across all PDFs.")
