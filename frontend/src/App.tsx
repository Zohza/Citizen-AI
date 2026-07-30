import { useState, useCallback, useEffect } from "react";
import { Toaster } from "sonner";
import type { Language, Thread, Message, Checklist, Citation } from "./types";
import {
  SCAM_KEYWORDS,
  LANG_KEYWORDS,
  STORAGE_KEY_THREADS,
  STORAGE_KEY_ACTIVE_THREAD,
  STORAGE_KEY_LANG,
  MAX_THREADS,
} from "./constants";
import MainContent from "./components/MainContent";

// ── Backend URL ──

const BACKEND_URL = "http://localhost:8000";

// ── Helpers ──

function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function detectLanguage(text: string): Language {
  const lower = text.toLowerCase();
  const scores: Record<string, number> = { en: 0, pidgin: 0, ha: 0, yo: 0, ig: 0 };
  for (const [lang, keywords] of Object.entries(LANG_KEYWORDS)) {
    for (const kw of keywords) {
      if (lower.includes(kw)) scores[lang] += 1;
    }
  }
  const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
  if (best && best[1] > 0 && best[0] !== "en") return best[0] as Language;
  return "en";
}

function detectScam(text: string): boolean {
  const lower = text.toLowerCase().replace(/\s+/g, " ");
  return SCAM_KEYWORDS.some((kw) => lower.includes(kw));
}

// ── Backend SSE streaming ──

/** Maps a backend citation to the frontend ``Citation`` shape. */
function citationsFromBackend(
  data: Array<{ source_name: string; page: number; agency: string }> | undefined,
): Citation[] | undefined {
  if (!data || data.length === 0) return undefined;
  return data.map((c, i) => ({
    sectionId: `${c.source_name}-p${c.page}-${i}`,
    source: c.agency === "CAC" ? "CAC" : "NELFUND",
    section: `${c.source_name} (Page ${c.page})`,
  }));
}

/** Maps a backend checklist payload to the frontend ``Checklist`` shape. */
function checklistFromBackend(
  data: Record<string, unknown> | null | undefined,
  agency: string,
  threadId: string,
): Checklist | undefined {
  if (!data) return undefined;
  const items = (data.items as Array<Record<string, unknown>>) ?? [];
  return {
    id: generateId(),
    threadId,
    type: agency === "CAC" ? "CAC" : "NELFUND",
    title: (data.title as string) || "Application Steps",
    items: items.map((item, i) => ({
      id: (item.id as string) ?? String(i),
      text: (item.text as string) ?? "",
      status: ((item.status as string) ?? "todo") as "todo" | "in_progress" | "completed",
    })),
    documents: (data.documents as string[]) ?? [],
    officialCost: (data.official_cost as string) ?? "",
    processingTime: (data.processing_time as string) ?? "",
    portalUrl: (data.portal_url as string) ?? "",
    createdAt: Date.now(),
  };
}

/**
 * Send a query to the Citizen AI backend and stream the response via SSE.
 *
 * @param callbacks.onToken - Called with each new token from the backend.
 * @param callbacks.onFinal - Called once with the final event (citations + checklist).
 * @param callbacks.onError - Called if the request fails or the backend returns an error.
 *     Receives ``{detail, errorType}`` where ``detail`` is the human-readable message
 *     and ``errorType`` is a machine-readable category (e.g. ``"upstream_timeout"``).
 */
async function sendToBackend(
  threadId: string,
  query: string,
  callbacks: {
    onToken: (token: string) => void;
    onFinal: (data: {
      citations: Array<{ source_name: string; page: number; agency: string }>;
      checklist?: Record<string, unknown> | null;
      detected_agency: string;
    }) => void;
    onError: (detail: string, errorType?: string) => void;
  },
): Promise<void> {
  try {
    const response = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: threadId, query }),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      callbacks.onError(
        `Server returned ${response.status}${text ? `: ${text}` : ""}`,
        response.status >= 500 ? "upstream_error" : "internal_error",
      );
      return;
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Split on newlines; keep the last (possibly incomplete) line in the buffer
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data: ")) continue;

        const payload = trimmed.slice(6); // strip "data: " prefix
        if (payload === "[DONE]") continue;

        try {
          const parsed = JSON.parse(payload) as {
            event: string;
            data: unknown;
          };
          const { event, data } = parsed;

          if (event === "token" && typeof data === "string") {
            callbacks.onToken(data);
          } else if (event === "final" && typeof data === "object" && data !== null) {
            callbacks.onFinal(data as Parameters<typeof callbacks.onFinal>[0]);
          } else if (event === "error") {
            // Support both string (legacy) and {error_type, detail} (v2) formats
            if (typeof data === "string") {
              callbacks.onError(data);
            } else if (typeof data === "object" && data !== null) {
              const err = data as Record<string, unknown>;
              callbacks.onError(
                (err.detail as string) ?? String(data),
                (err.error_type as string) ?? "internal_error",
              );
            }
          }
        } catch {
          // skip lines with malformed JSON
        }
      }
    }
  } catch (e) {
    callbacks.onError(e instanceof Error ? e.message : String(e), "internal_error");
  }
}

// ── App Component ──

export default function App() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [currentLang, setCurrentLang] = useState<Language>("en");
  const [darkMode, setDarkMode] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    try {
      const savedLang = localStorage.getItem(STORAGE_KEY_LANG);
      if (savedLang) setCurrentLang(savedLang as Language);

      const savedThreads = localStorage.getItem(STORAGE_KEY_THREADS);
      if (savedThreads) {
        const parsed: Thread[] = JSON.parse(savedThreads);
        setThreads(parsed);
      }

      const savedActive = localStorage.getItem(STORAGE_KEY_ACTIVE_THREAD);
      if (savedActive && savedThreads) {
        const parsed: Thread[] = JSON.parse(savedThreads);
        if (parsed.some((t) => t.id === savedActive)) {
          setActiveThreadId(savedActive);
        }
      }

      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      const savedDark = localStorage.getItem("citizen_ai_dark");
      if (savedDark !== null) {
        setDarkMode(savedDark === "true");
      } else {
        setDarkMode(prefersDark);
      }
    } catch (e) {
      // Ignore localStorage errors
    }
    setReady(true);
  }, []);

  useEffect(() => {
    if (!ready) return;
    try {
      localStorage.setItem(STORAGE_KEY_THREADS, JSON.stringify(threads));
    } catch (e) {
      if (threads.length > 10) {
        const trimmed = threads.slice(-10);
        setThreads(trimmed);
        try {
          localStorage.setItem(STORAGE_KEY_THREADS, JSON.stringify(trimmed));
        } catch { /* empty */ }
      }
    }
  }, [threads, ready]);

  useEffect(() => {
    if (!ready) return;
    try {
      if (activeThreadId) {
        localStorage.setItem(STORAGE_KEY_ACTIVE_THREAD, activeThreadId);
      } else {
        localStorage.removeItem(STORAGE_KEY_ACTIVE_THREAD);
      }
    } catch { /* empty */ }
  }, [activeThreadId, ready]);

  useEffect(() => {
    if (!ready) return;
    try {
      localStorage.setItem(STORAGE_KEY_LANG, currentLang);
    } catch { /* empty */ }
  }, [currentLang, ready]);

  useEffect(() => {
    if (!ready) return;
    try {
      localStorage.setItem("citizen_ai_dark", String(darkMode));
    } catch { /* empty */ }
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode, ready]);

  // ── Handlers ──

  const handleNewThread = useCallback(() => {
    const id = generateId();
    const thread: Thread = {
      id,
      title: `Chat ${threads.length + 1}`,
      messages: [],
      language: currentLang,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setThreads((prev) => [thread, ...prev].slice(0, MAX_THREADS));
    setActiveThreadId(id);
  }, [threads.length, currentLang]);

  const handleSelectThread = useCallback((id: string) => {
    setActiveThreadId(id);
  }, []);

  const handleDeleteThread = useCallback((id: string) => {
    setThreads((prev) => prev.filter((t) => t.id !== id));
    setActiveThreadId((prev) => (prev === id ? null : prev));
  }, []);

  const handleDeleteAllThreads = useCallback(() => {
    setThreads([]);
    setActiveThreadId(null);
  }, []);

  const handleToggleLang = useCallback((lang: Language) => {
    setCurrentLang(lang);
  }, []);

  const handleToggleDark = useCallback(() => {
    setDarkMode((prev) => !prev);
  }, []);

  const handleSendMessage = useCallback(
    (text: string) => {
      // ── 1. Client-side detection (unchanged) ──
      const detected = detectLanguage(text);
      if (detected !== "en" && currentLang === "en") {
        setCurrentLang(detected);
      }
      const isScam = detectScam(text);

      // ── 2. Ensure a thread exists ──
      let threadId = activeThreadId;
      if (!threadId) {
        threadId = generateId();
        const newThread: Thread = {
          id: threadId,
          title: text.length > 40 ? text.slice(0, 40) + "..." : text,
          messages: [],
          language: currentLang,
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        setThreads((prev) => [newThread, ...prev].slice(0, MAX_THREADS));
        setActiveThreadId(threadId);
      }

      // ── 3. Create user + placeholder assistant messages ──
      const userMsg: Message = {
        id: generateId(),
        role: "user",
        content: text,
        timestamp: Date.now(),
      };

      const assistantId = generateId();
      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
        isScamWarning: isScam,
        isStreaming: true,
      };

      // Stable references for the async closure
      const targetThreadId = threadId;

      // ── 4. Add both messages to state ──
      setThreads((prev) =>
        prev.map((t) => {
          if (t.id !== targetThreadId) return t;
          return {
            ...t,
            messages: [...t.messages, userMsg, assistantMsg],
            updatedAt: Date.now(),
          };
        }),
      );

      // ── 5. Stream from backend ──
      sendToBackend(targetThreadId, text, {
        onToken: (token: string) => {
          setThreads((prev) =>
            prev.map((t) => {
              if (t.id !== targetThreadId) return t;
              return {
                ...t,
                messages: t.messages.map((m) =>
                  m.id === assistantId ? { ...m, content: m.content + token } : m,
                ),
              };
            }),
          );
        },

        onFinal: (data) => {
          setThreads((prev) =>
            prev.map((t) => {
              if (t.id !== targetThreadId) return t;
              const updatedMessages = t.messages.map((m) => {
                if (m.id !== assistantId) return m;
                return {
                  ...m,
                  citations: citationsFromBackend(data.citations),
                  checklist: checklistFromBackend(data.checklist, data.detected_agency, targetThreadId),
                  isStreaming: false,
                };
              });
              const threadChecklist = data.checklist
                ? checklistFromBackend(data.checklist, data.detected_agency, targetThreadId)
                : undefined;
              return {
                ...t,
                messages: updatedMessages,
                updatedAt: Date.now(),
                checklist: threadChecklist ?? t.checklist,
              };
            }),
          );
        },

        onError: (error: string, errorType?: string) => {
          setThreads((prev) =>
            prev.map((t) => {
              if (t.id !== targetThreadId) return t;
              return {
                ...t,
                messages: t.messages.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        content: error,
                        isStreaming: false,
                        isError: true,
                        errorType,
                      }
                    : m,
                ),
              };
            }),
          );
        },
      });
    },
    [activeThreadId, currentLang],
  );

  if (!ready) {
    return (
      <div className="flex h-dvh items-center justify-center bg-zinc-50 dark:bg-zinc-900">
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-emerald-500 border-t-transparent" />
          <p className="text-sm text-zinc-400 dark:text-zinc-500">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <Toaster
        position="top-center"
        toastOptions={{
          style: {
            background: darkMode ? "#1e1e1e" : "#fff",
            color: darkMode ? "#e5e5e5" : "#27272a",
            border: darkMode ? "1px solid #333" : "1px solid #e4e4e7",
          },
        }}
      />
      <div className={darkMode ? "dark" : ""}>
        <MainContent
          threads={threads}
          activeThreadId={activeThreadId}
          currentLang={currentLang}
          darkMode={darkMode}
          onSendMessage={handleSendMessage}
          onNewThread={handleNewThread}
          onSelectThread={handleSelectThread}
          onDeleteThread={handleDeleteThread}
          onToggleLang={handleToggleLang}
          onToggleDark={handleToggleDark}
          onDeleteAllThreads={handleDeleteAllThreads}
        />
      </div>
    </>
  );
}
