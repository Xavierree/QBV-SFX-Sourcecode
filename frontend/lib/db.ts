// Local JSON-file store for submissions + events. The inference backend is stateless
// (embed/transcribe/caption only -- see /ingest's own docstring); this app is the one expected to
// track submission records and serve the admin views, per that docstring.
//
// Read-modify-write per call rather than a long-lived in-memory cache: Next dev (Turbopack) can
// re-instantiate route modules between requests, which would silently drop a module-level cache.
// Volume here is low (one person's SFX submissions), so the fs round-trip per call is a non-issue.
import fs from "node:fs";
import path from "node:path";

export type SubmissionStatus =
  | "pending"
  | "embedding"
  | "captioning"
  | "structuring"
  | "done"
  | "failed";

export interface SubmissionRecord {
  id: number;
  status: SubmissionStatus;
  gallery: string;
  original_filename: string;
  title: string | null;
  tags: string[] | null;
  description: string | null;
  caption_raw: string | null;
  transcript: string | null;
  // The relative path /ingest actually stored the clip under (e.g. "bo-eee-yoi-yoi-84691.wav") --
  // it renames on ingest rather than keeping original_filename, see lib/ingestJob.ts. Null until
  // the ingest response is parsed, and stays null if that response never included a recognizable
  // path/filename field.
  gallery_file: string | null;
  error: string | null;
  created_at: number;
  updated_at: number;
}

export type EventLevel = "info" | "warning" | "error";

export interface EventRecord {
  id: number;
  level: EventLevel;
  category: string;
  message: string;
  created_at: number;
  submission_id: number | null;
}

interface Store {
  nextSubmissionId: number;
  nextEventId: number;
  submissions: SubmissionRecord[];
  events: EventRecord[];
}

const DB_PATH = path.join(process.cwd(), "data", "db.json");
const MAX_EVENTS = 1000;

function emptyStore(): Store {
  return { nextSubmissionId: 1, nextEventId: 1, submissions: [], events: [] };
}

function loadStore(): Store {
  try {
    return JSON.parse(fs.readFileSync(DB_PATH, "utf8")) as Store;
  } catch {
    return emptyStore();
  }
}

// Atomic write (temp file + rename), avoids a reader ever seeing a half-written file.
function saveStore(store: Store) {
  fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
  const tmp = `${DB_PATH}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(store, null, 2), "utf8");
  fs.renameSync(tmp, DB_PATH);
}

function nowSec() {
  return Date.now() / 1000;
}

export function createSubmission(gallery: string, filename: string): SubmissionRecord {
  const store = loadStore();
  const record: SubmissionRecord = {
    id: store.nextSubmissionId++,
    status: "pending",
    gallery,
    original_filename: filename,
    title: null,
    tags: null,
    description: null,
    caption_raw: null,
    transcript: null,
    gallery_file: null,
    error: null,
    created_at: nowSec(),
    updated_at: nowSec(),
  };
  store.submissions.push(record);
  saveStore(store);
  return record;
}

export function updateSubmission(id: number, patch: Partial<SubmissionRecord>): SubmissionRecord | undefined {
  const store = loadStore();
  const record = store.submissions.find((s) => s.id === id);
  if (!record) return undefined;
  Object.assign(record, patch, { updated_at: nowSec() });
  saveStore(store);
  return record;
}

export function getSubmission(id: number): SubmissionRecord | undefined {
  return loadStore().submissions.find((s) => s.id === id);
}

export function listSubmissions(limit = 50): SubmissionRecord[] {
  return loadStore().submissions.slice(-limit).reverse();
}

export function addEvent(
  level: EventLevel,
  category: string,
  message: string,
  submissionId: number | null = null
): EventRecord {
  const store = loadStore();
  const record: EventRecord = {
    id: store.nextEventId++,
    level,
    category,
    message,
    created_at: nowSec(),
    submission_id: submissionId,
  };
  store.events.push(record);
  if (store.events.length > MAX_EVENTS) {
    store.events = store.events.slice(-MAX_EVENTS);
  }
  saveStore(store);
  return record;
}

export function listEvents(limit = 50): EventRecord[] {
  return loadStore().events.slice(-limit).reverse();
}

export function getStatsBase() {
  const store = loadStore();
  const submissions_by_status: Record<string, number> = {};
  for (const s of store.submissions) {
    submissions_by_status[s.status] = (submissions_by_status[s.status] || 0) + 1;
  }
  const events_by_level: Record<string, number> = {};
  for (const e of store.events) {
    events_by_level[e.level] = (events_by_level[e.level] || 0) + 1;
  }
  return {
    submissions_total: store.submissions.length,
    submissions_by_status,
    events_by_level,
  };
}
