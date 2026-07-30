import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  PaperPlaneTilt,
  ChatCircleDots,
  CheckCircle,
  WarningCircle,
  Translate,
  FileText,
  ListChecks,
  ArrowRight,
  Check,
  X,
  Clock,
  ArrowSquareOut,
  Copy,
  Sun,
  Moon,
  Plus,
  Trash,
  Sparkle,
  Circle,
  ClockAfternoon,
} from "@phosphor-icons/react";
import { toast } from "sonner";
import type {
  Message,
  Thread,
  ChecklistItem,
  Language,
  Citation,
} from "../types";
import { LANG_STRINGS, getPromptCards, APP_NAME } from "../constants";

// ── Props ──

interface MainContentProps {
  threads: Thread[];
  activeThreadId: string | null;
  currentLang: Language;
  darkMode: boolean;
  onSendMessage: (text: string) => void;
  onNewThread: () => void;
  onSelectThread: (id: string) => void;
  onDeleteThread: (id: string) => void;
  onToggleLang: (lang: Language) => void;
  onToggleDark: () => void;
  onDeleteAllThreads: () => void;
}

// ── Helpers ──

function formatTime(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDate(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleDateString([], { day: "numeric", month: "short" });
}

// ── Citation Badge ──

function CitationBadge({ citation }: { citation: Citation }) {
  const colors =
    citation.source === "NELFUND"
      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
      : "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-medium ${colors}`}
    >
      {citation.source === "NELFUND" ? (
        <CheckCircle size={12} weight="fill" />
      ) : (
        <FileText size={12} weight="fill" />
      )}
      {citation.source} KB, Section {citation.section}
    </span>
  );
}

// ── Scam Warning Banner ──

function ScamWarningBanner({ text }: { text: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -8, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -8, scale: 0.98 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="mx-2 mb-3 rounded-xl border-2 border-red-300 bg-red-50 p-4 dark:border-red-800 dark:bg-red-950/40"
    >
      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/50">
          <WarningCircle size={20} weight="fill" className="text-red-600 dark:text-red-400" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-bold uppercase tracking-wide text-red-700 dark:text-red-400">
            Scam Alert
          </p>
          <p className="mt-1 text-sm leading-relaxed text-red-800 dark:text-red-300">
            {text}
          </p>
        </div>
      </div>
    </motion.div>
  );
}

// ── Message Bubble ──

function MessageBubble({
  msg,
  lang,
  onCopy,
  onFollowUpClick,
}: {
  msg: Message;
  lang: Language;
  onCopy: (text: string) => void;
  onFollowUpClick: (text: string) => void;
}) {
  const s = LANG_STRINGS[lang];
  const isUser = msg.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`flex ${isUser ? "justify-end" : "justify-start"} px-2`}
    >
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 ${
          isUser
            ? "rounded-br-md bg-emerald-600 text-white dark:bg-emerald-700"
            : "rounded-bl-md bg-white text-zinc-800 shadow-sm ring-1 ring-zinc-100 dark:bg-zinc-800 dark:text-zinc-200 dark:ring-zinc-700"
        }`}
      >
        {/* Scam warning inside assistant message */}
        {msg.isScamWarning && (
          <div className="mb-3">
            <ScamWarningBanner text={s.scamBody} />
          </div>
        )}

        {/* Typing / thinking indicator */}
        {msg.isStreaming && msg.content.length === 0 ? (
          <div className="flex items-center gap-1.5 py-1" aria-label="Thinking">
            <span
              className="typing-dot h-2 w-2 rounded-full bg-emerald-400 dark:bg-emerald-500"
              style={{ animationDelay: "0ms" }}
            />
            <span
              className="typing-dot h-2 w-2 rounded-full bg-emerald-400 dark:bg-emerald-500"
              style={{ animationDelay: "200ms" }}
            />
            <span
              className="typing-dot h-2 w-2 rounded-full bg-emerald-400 dark:bg-emerald-500"
              style={{ animationDelay: "400ms" }}
            />
          </div>
        ) : msg.isError ? (
          <div className="flex items-start gap-2 rounded-lg bg-red-50 p-3 dark:bg-red-950/30">
            <WarningCircle
              size={20}
              weight="fill"
              className="mt-0.5 shrink-0 text-red-500"
            />
            <div className="min-w-0">
              <p className="text-sm font-semibold text-red-700 dark:text-red-400">
                {msg.errorType === "upstream_timeout"
                  ? "The AI service is taking too long to respond."
                  : msg.errorType === "upstream_error"
                    ? "The AI service returned an error."
                    : "Something went wrong."}
              </p>
              <p className="mt-1 text-xs leading-relaxed text-red-600 dark:text-red-300">
                {msg.content || "Please try again in a moment."}
              </p>
            </div>
          </div>
        ) : (
          <p className="text-[15px] leading-relaxed whitespace-pre-wrap">{msg.content}</p>
        )}

        {/* Citations */}
        {msg.citations && msg.citations.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-medium text-zinc-400 dark:text-zinc-500">
              {s.kbSource}:
            </span>
            {msg.citations.map((c, i) => (
              <CitationBadge key={i} citation={c} />
            ))}
          </div>
        )}

        {/* Follow-up suggestions */}
        {msg.followUpSuggestions && msg.followUpSuggestions.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {msg.followUpSuggestions.map((suggestion, i) => (
              <button
                key={i}
                onClick={() => onFollowUpClick(suggestion)}
                className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 transition-colors hover:bg-emerald-100 dark:bg-emerald-900/30 dark:text-emerald-300 dark:hover:bg-emerald-900/50"
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}

        {/* Checklist preview */}
        {msg.checklist && (
          <div className="mt-3 rounded-xl border border-emerald-200 bg-emerald-50/50 p-3 dark:border-emerald-800 dark:bg-emerald-900/20">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-emerald-700 dark:text-emerald-400">
              {msg.checklist.title}
            </p>
            <div className="space-y-1.5">
              {msg.checklist.items.slice(0, 4).map((item) => (
                <div key={item.id} className="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400">
                  {item.status === "completed" ? (
                    <CheckCircle size={14} weight="fill" className="text-emerald-500" />
                  ) : item.status === "in_progress" ? (
                    <ClockAfternoon size={14} weight="fill" className="text-amber-500" />
                  ) : (
                    <Circle size={14} className="text-zinc-300 dark:text-zinc-600" />
                  )}
                  <span className={item.status === "completed" ? "line-through opacity-60" : ""}>
                    {item.text}
                  </span>
                </div>
              ))}
              {msg.checklist.items.length > 4 && (
                <p className="text-[11px] text-zinc-400 dark:text-zinc-500">
                  +{msg.checklist.items.length - 4} more items
                </p>
              )}
            </div>
          </div>
        )}

        {/* Timestamp + copy */}
        <div className="mt-2 flex items-center justify-between gap-2">
          <span className="text-[11px] text-zinc-400 dark:text-zinc-500">
            {formatTime(msg.timestamp)}
          </span>
          {!isUser && (
            <button
              onClick={() => onCopy(msg.content)}
              className="text-zinc-400 transition-colors hover:text-zinc-600 dark:text-zinc-500 dark:hover:text-zinc-300"
              title={s.copy}
            >
              <Copy size={14} />
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
}

// ── Main Content ──

export default function MainContent({
  threads,
  activeThreadId,
  currentLang,
  darkMode,
  onSendMessage,
  onNewThread,
  onSelectThread,
  onDeleteThread,
  onToggleLang,
  onToggleDark,
  onDeleteAllThreads,
}: MainContentProps) {
  const s = LANG_STRINGS[currentLang];
  const [input, setInput] = useState("");
  const [showSidebar, setShowSidebar] = useState(false);
  const [showLangPicker, setShowLangPicker] = useState(false);
  const [tab, setTab] = useState<"chat" | "apps">("chat");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const langOptions: Language[] = ["en", "pidgin", "ha", "yo", "ig"];

  const activeThread = threads.find((t) => t.id === activeThreadId) || null;
  const messages = activeThread?.messages || [];

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  // Focus input
  useEffect(() => {
    inputRef.current?.focus();
  }, [activeThreadId]);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text) return;
    onSendMessage(text);
    setInput("");
  }, [input, onSendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleCopy = useCallback(
    (text: string) => {
      navigator.clipboard.writeText(text).then(() => {
        toast.success(s.copied);
      });
    },
    [s.copied]
  );

  const handlePromptClick = (text: string) => {
    onSendMessage(text);
  };

  const langLabels: Record<Language, string> = {
    en: "English",
    pidgin: "Pidgin",
    ha: "Hausa",
    yo: "Yoruba",
    ig: "Igbo",
  };

  const langNative: Record<Language, string> = {
    en: "English",
    pidgin: "Pidgin",
    ha: "Hausa",
    yo: "Yoruba",
    ig: "Igbo",
  };

  return (
    <div className="relative mx-auto flex h-dvh w-full max-w-2xl flex-col bg-zinc-50 dark:bg-zinc-900">
      {/* ── Top Bar ── */}
      <header className="flex shrink-0 items-center justify-between border-b border-zinc-200 bg-white px-4 py-3 dark:border-zinc-700 dark:bg-zinc-800">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className="flex h-9 w-9 items-center justify-center rounded-xl text-zinc-500 transition-colors hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-700"
          >
            {showSidebar ? <X size={20} /> : <ChatCircleDots size={20} weight="duotone" />}
          </button>
          <div>
            <h1 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
              {APP_NAME}
            </h1>
            <p className="text-[11px] text-zinc-400 dark:text-zinc-500">
              {langLabels[currentLang]}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          {/* Language picker */}
          <div className="relative">
            <button
              onClick={() => setShowLangPicker(!showLangPicker)}
              className="flex h-9 w-9 items-center justify-center rounded-xl text-zinc-500 transition-colors hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-700"
            >
              <Translate size={20} />
            </button>
            <AnimatePresence>
              {showLangPicker && (
                <motion.div
                  initial={{ opacity: 0, y: -4, scale: 0.96 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -4, scale: 0.96 }}
                  transition={{ duration: 0.15 }}
                  className="absolute right-0 top-10 z-50 w-40 rounded-2xl border border-zinc-200 bg-white p-1.5 shadow-lg dark:border-zinc-700 dark:bg-zinc-800"
                >
                  {langOptions.map((l) => (
                    <button
                      key={l}
                      onClick={() => {
                        onToggleLang(l);
                        setShowLangPicker(false);
                      }}
                      className={`flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm transition-colors ${
                        currentLang === l
                          ? "bg-emerald-50 font-semibold text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                          : "text-zinc-700 hover:bg-zinc-50 dark:text-zinc-300 dark:hover:bg-zinc-700"
                      }`}
                    >
                      <span className="text-xs opacity-50">{langNative[l]}</span>
                      <span>{langLabels[l]}</span>
                      {currentLang === l && <Check size={14} weight="bold" className="ml-auto" />}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Dark mode toggle */}
          <button
            onClick={onToggleDark}
            className="flex h-9 w-9 items-center justify-center rounded-xl text-zinc-500 transition-colors hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-700"
          >
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>
      </header>

      {/* ── Sidebar overlay ── */}
      <AnimatePresence>
        {showSidebar && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="absolute inset-0 z-40 bg-black/20 dark:bg-black/40"
            onClick={() => setShowSidebar(false)}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showSidebar && (
          <motion.aside
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            exit={{ x: -300 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="absolute left-0 top-0 z-50 flex h-full w-72 flex-col border-r border-zinc-200 bg-white dark:border-zinc-700 dark:bg-zinc-800"
          >
            <div className="flex items-center justify-between border-b border-zinc-100 p-4 dark:border-zinc-700">
              <h2 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                {s.threads}
              </h2>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => {
                    onNewThread();
                    setShowSidebar(false);
                  }}
                  className="flex h-8 w-8 items-center justify-center rounded-lg text-emerald-600 transition-colors hover:bg-emerald-50 dark:text-emerald-400 dark:hover:bg-emerald-900/30"
                >
                  <Plus size={18} weight="bold" />
                </button>
                {threads.length > 0 && (
                  <button
                    onClick={() => {
                      if (window.confirm("Delete all chats?")) {
                        onDeleteAllThreads();
                      }
                    }}
                    className="flex h-8 w-8 items-center justify-center rounded-lg text-zinc-400 transition-colors hover:bg-red-50 hover:text-red-500 dark:hover:bg-red-900/20"
                  >
                    <Trash size={16} />
                  </button>
                )}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-2">
              {threads.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <ChatCircleDots size={32} className="text-zinc-300 dark:text-zinc-600" />
                  <p className="mt-2 text-sm text-zinc-400 dark:text-zinc-500">{s.noApps}</p>
                </div>
              ) : (
                threads.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => {
                      onSelectThread(t.id);
                      setShowSidebar(false);
                    }}
                    className={`mb-1 flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left transition-colors ${
                      t.id === activeThreadId
                        ? "bg-emerald-50 dark:bg-emerald-900/30"
                        : "hover:bg-zinc-50 dark:hover:bg-zinc-700/50"
                    }`}
                  >
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-100 text-zinc-500 dark:bg-zinc-700 dark:text-zinc-400">
                      <ChatCircleDots size={18} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-zinc-900 dark:text-zinc-100">
                        {t.title}
                      </p>
                      <p className="text-[11px] text-zinc-400 dark:text-zinc-500">
                        {formatDate(t.updatedAt)}
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteThread(t.id);
                      }}
                      className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-zinc-300 opacity-0 transition-all hover:bg-red-50 hover:text-red-500 group-hover:opacity-100 dark:text-zinc-600 dark:hover:bg-red-900/20"
                    >
                      <X size={14} />
                    </button>
                  </button>
                ))
              )}
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* ── Main content area ── */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {tab === "chat" ? (
          <>
            {/* Messages */}
            <div className="flex-1 overflow-y-auto py-4">
              {messages.length === 0 ? (
                /* Welcome screen */
                <div className="flex flex-col items-center px-4 pt-8 text-center">
                  <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-100 dark:bg-emerald-900/40">
                    <Sparkle size={32} weight="duotone" className="text-emerald-600 dark:text-emerald-400" />
                  </div>
                  <h2 className="text-xl font-bold text-zinc-900 dark:text-zinc-100">
                    {s.welcome}
                  </h2>
                  <p className="mt-2 max-w-sm text-sm leading-relaxed text-zinc-500 dark:text-zinc-400">
                    {s.welcomeDesc}
                  </p>
                  <p className="mt-6 text-xs font-medium uppercase tracking-wider text-zinc-400 dark:text-zinc-500">
                    {s.selectOption}
                  </p>

                  {/* Quick prompt cards */}
                  <div className="mt-4 grid w-full max-w-sm grid-cols-1 gap-2 sm:grid-cols-2">
                    {getPromptCards(currentLang).map((card, i) => (
                      <motion.button
                        key={i}
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.08, duration: 0.3, ease: "easeOut" }}
                        onClick={() => handlePromptClick(card.text)}
                        className="flex items-center gap-2 rounded-2xl border border-zinc-200 bg-white px-4 py-3 text-left text-sm font-medium text-zinc-700 shadow-sm transition-all hover:border-emerald-300 hover:text-emerald-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:border-emerald-700 dark:hover:text-emerald-400"
                      >
                        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-emerald-50 dark:bg-emerald-900/30">
                          {card.intent === "NELFUND" ? (
                            <CheckCircle size={16} weight="fill" className="text-emerald-600 dark:text-emerald-400" />
                          ) : (
                            <FileText size={16} weight="fill" className="text-amber-600 dark:text-amber-400" />
                          )}
                        </span>
                        <span className="line-clamp-2 text-[13px]">{card.text}</span>
                      </motion.button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  {messages.map((msg) => (
                    <MessageBubble key={msg.id} msg={msg} lang={currentLang} onCopy={handleCopy} onFollowUpClick={onSendMessage} />
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Input area */}
            <div className="shrink-0 border-t border-zinc-200 bg-white px-3 pb-3 pt-2 dark:border-zinc-700 dark:bg-zinc-800">
              <div className="flex items-end gap-2 rounded-2xl border border-zinc-200 bg-zinc-50 px-4 py-2 focus-within:border-emerald-400 focus-within:ring-2 focus-within:ring-emerald-100 dark:border-zinc-700 dark:bg-zinc-900 dark:focus-within:border-emerald-600 dark:focus-within:ring-emerald-900/30">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={s.placeholder}
                  rows={1}
                  className="max-h-32 min-h-[24px] flex-1 resize-none bg-transparent text-[15px] text-zinc-900 placeholder-zinc-400 outline-none dark:text-zinc-100 dark:placeholder-zinc-500"
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim()}
                  className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-600 text-white transition-all hover:bg-emerald-700 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-emerald-700 dark:hover:bg-emerald-600"
                >
                  <PaperPlaneTilt size={20} weight="fill" />
                </button>
              </div>
            </div>
          </>
        ) : (
          /* ── My Applications Dashboard ── */
          <div className="flex-1 overflow-y-auto">
            <div className="p-4">
              <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                {s.myApps}
              </h2>
              <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                {s.noAppsDesc}
              </p>

              {threads.filter((t) => t.checklist).length === 0 ? (
                <div className="mt-8 flex flex-col items-center py-12 text-center">
                  <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-zinc-100 dark:bg-zinc-800">
                    <ListChecks size={32} className="text-zinc-300 dark:text-zinc-600" />
                  </div>
                  <p className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
                    {s.noApps}
                  </p>
                  <p className="mt-1 text-xs text-zinc-400 dark:text-zinc-500">
                    {s.noAppsDesc}
                  </p>
                </div>
              ) : (
                <div className="mt-4 space-y-4">
                  {threads
                    .filter((t) => t.checklist)
                    .map((t) => {
                      const cl = t.checklist!;
                      const completed = cl.items.filter((i) => i.status === "completed").length;
                      const total = cl.items.length;
                      const pct = total > 0 ? Math.round((completed / total) * 100) : 0;

                      return (
                        <motion.div
                          key={t.id}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800"
                        >
                          <div className="mb-3 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-100 dark:bg-emerald-900/40">
                                {cl.type === "NELFUND" ? (
                                  <CheckCircle size={18} weight="fill" className="text-emerald-600 dark:text-emerald-400" />
                                ) : (
                                  <FileText size={18} weight="fill" className="text-amber-600 dark:text-amber-400" />
                                )}
                              </span>
                              <div>
                                <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                                  {cl.title}
                                </p>
                                <p className="text-[11px] text-zinc-400 dark:text-zinc-500">
                                  {cl.type} {s.appProgress}
                                </p>
                              </div>
                            </div>
                            <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400">
                              {pct}%
                            </span>
                          </div>

                          <div className="mb-3 h-2 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-700">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${pct}%` }}
                              transition={{ duration: 0.8, ease: "easeOut" }}
                              className="h-full rounded-full bg-emerald-500"
                            />
                          </div>

                          <div className="space-y-1">
                            {cl.items.slice(0, 3).map((item) => {
                              const statusIcon =
                                item.status === "completed" ? (
                                  <CheckCircle size={14} weight="fill" className="text-emerald-500" />
                                ) : item.status === "in_progress" ? (
                                  <ClockAfternoon size={14} weight="fill" className="text-amber-500" />
                                ) : (
                                  <Circle size={14} className="text-zinc-300 dark:text-zinc-600" />
                                );
                              return (
                                <div key={item.id} className="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400">
                                  {statusIcon}
                                  <span className={item.status === "completed" ? "line-through opacity-60" : ""}>
                                    {item.text}
                                  </span>
                                </div>
                              );
                            })}
                            {cl.items.length > 3 && (
                              <p className="text-[11px] text-zinc-400 dark:text-zinc-500">
                                +{cl.items.length - 3} more
                              </p>
                            )}
                          </div>

                          <button
                            onClick={() => {
                              onSelectThread(t.id);
                              setTab("chat");
                            }}
                            className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-emerald-600 transition-colors hover:text-emerald-700 dark:text-emerald-400 dark:hover:text-emerald-300"
                          >
                            {s.appProgress}
                            <ArrowRight size={14} />
                          </button>
                        </motion.div>
                      );
                    })}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Bottom Navigation ── */}
      <nav className="flex shrink-0 items-center justify-around border-t border-zinc-200 bg-white px-2 py-1 dark:border-zinc-700 dark:bg-zinc-800">
        <button
          onClick={() => setTab("chat")}
          className={`flex flex-col items-center gap-0.5 rounded-xl px-5 py-2 text-[11px] font-medium transition-colors ${
            tab === "chat"
              ? "text-emerald-600 dark:text-emerald-400"
              : "text-zinc-400 dark:text-zinc-500"
          }`}
        >
          <ChatCircleDots size={22} weight={tab === "chat" ? "fill" : "regular"} />
          {s.chat}
        </button>
        <button
          onClick={() => onNewThread()}
          className="flex h-12 w-12 -translate-y-2 items-center justify-center rounded-full bg-emerald-600 text-white shadow-lg transition-all hover:bg-emerald-700 active:scale-95 dark:bg-emerald-700 dark:hover:bg-emerald-600"
        >
          <Plus size={24} weight="bold" />
        </button>
        <button
          onClick={() => setTab("apps")}
          className={`flex flex-col items-center gap-0.5 rounded-xl px-5 py-2 text-[11px] font-medium transition-colors ${
            tab === "apps"
              ? "text-emerald-600 dark:text-emerald-400"
              : "text-zinc-400 dark:text-zinc-500"
          }`}
        >
          <ListChecks size={22} weight={tab === "apps" ? "fill" : "regular"} />
          {s.myApps}
        </button>
      </nav>
    </div>
  );
}