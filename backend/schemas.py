"""Pydantic schemas for Citizen AI API request/response models.
"""

from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    thread_id: str
    query: str


class ChatResponseEvent(BaseModel):
    """An SSE event sent to the client during streaming."""

    event: str  # "token" | "final"
    data: str


class Citation(BaseModel):
    source_name: str
    page: int
    agency: str


class ChecklistItem(BaseModel):
    id: str
    text: str
    status: str = "todo"


class Checklist(BaseModel):
    title: str
    items: list[ChecklistItem]
    documents: list[str]
    official_cost: str
    processing_time: str
    portal_url: str


class FinalEventData(BaseModel):
    citations: list[Citation] = []
    checklist: Optional[Checklist] = None
    detected_agency: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"


_UPSTREAM_TIMEOUT_KEYWORDS = [
    "timeout", "read timed out", "connection",
]


def classify_error(error_message: str) -> str:
    """Classify an error string into a typed category for the frontend.

    Returns one of ``"upstream_timeout"``, ``"upstream_error"``,
    or ``"internal_error"``.
    """
    msg_lower = error_message.lower()
    if any(kw in msg_lower for kw in _UPSTREAM_TIMEOUT_KEYWORDS):
        return "upstream_timeout"
    if "nvidia" in msg_lower or "api" in msg_lower or "400" in msg_lower or "401" in msg_lower or "403" in msg_lower or "500" in msg_lower or "502" in msg_lower:
        return "upstream_error"
    return "internal_error"


class ErrorEventData(BaseModel):
    error_type: str  # "upstream_timeout" | "upstream_error" | "internal_error"
    detail: str = ""
