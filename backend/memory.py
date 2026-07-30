"""LangGraph checkpoint memory / persistence.

Uses SqliteSaver for persistent thread-history across restarts.
Falls back to MemorySaver (in-memory only) when SqliteSaver is unavailable.
"""

from config import MEMORY_DB_PATH


def get_checkpointer():
    """Return a LangGraph checkpointer for conversational memory.

    Tries ``SqliteSaver`` first (persists across restarts).
    Falls back to ``MemorySaver`` (ephemeral) if the sqlite checkpoint
    module is not available in the installed version of ``langgraph``.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver.from_conn_string(MEMORY_DB_PATH)
    except (ImportError, Exception):
        from langgraph.checkpoint.memory import MemorySaver

        # NOTE: MemorySaver is ephemeral — history is lost on process restart.
        return MemorySaver()
