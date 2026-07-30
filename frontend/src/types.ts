export type Language = "en" | "pidgin" | "ha" | "yo" | "ig";

export interface KnowledgeSection {
  id: string;
  source: "NELFUND" | "CAC";
  section: string;
  title: string;
  content: string;
  url?: string;
}

export interface ChecklistItem {
  id: string;
  text: string;
  status: "todo" | "in_progress" | "completed";
  notes?: string;
}

export interface Checklist {
  id: string;
  threadId: string;
  type: "NELFUND" | "CAC";
  title: string;
  items: ChecklistItem[];
  documents: string[];
  officialCost?: string;
  processingTime?: string;
  portalUrl?: string;
  createdAt: number;
}

export interface Citation {
  sectionId: string;
  source: "NELFUND" | "CAC";
  section: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  citations?: Citation[];
  isScamWarning?: boolean;
  followUpSuggestions?: string[];
  checklist?: Checklist;
  /** Set to ``true`` while the backend is streaming tokens for this message. */
  isStreaming?: boolean;
  /** Set to ``true`` when this message represents a backend error. */
  isError?: boolean;
  /** Machine-readable error category (``upstream_timeout``, etc.). */
  errorType?: string;
}

export interface Thread {
  id: string;
  title: string;
  messages: Message[];
  language: Language;
  createdAt: number;
  updatedAt: number;
  checklist?: Checklist;
}

export interface LangStrings {
  placeholder: string;
  send: string;
  newChat: string;
  myApps: string;
  chat: string;
  noApps: string;
  noAppsDesc: string;
  appProgress: string;
  documents: string;
  officialCost: string;
  processingTime: string;
  portalLink: string;
  printChecklist: string;
  downloadChecklist: string;
  scamTitle: string;
  scamBody: string;
  kbSource: string;
  section: string;
  todo: string;
  inProgress: string;
  completed: string;
  typeMessage: string;
  promptLoan: string;
  promptCAC: string;
  promptHow: string;
  promptCost: string;
  hello: string;
  welcome: string;
  welcomeDesc: string;
  selectOption: string;
  nelfund: string;
  cac: string;
  settings: string;
  about: string;
  darkMode: string;
  lightMode: string;
  thread: string;
  threads: string;
  delete: string;
  confirm: string;
  cancel: string;
  yes: string;
  no: string;
  loading: string;
  error: string;
  retry: string;
  close: string;
  back: string;
  next: string;
  finish: string;
  save: string;
  share: string;
  copy: string;
  copied: string;
  print: string;
  download: string;
  status: string;
  notCovered: string;
  official: string;
  free: string;
  warning: string;
  info: string;
  tip: string;
}

export interface LangDict {
  [key: string]: LangStrings;
}