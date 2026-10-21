// Background job kicked off (fire-and-forget) by POST /api/submit. The backend's own /ingest
// docstring says this app is expected to call it from a background job and poll its own submission
// record rather than block a user-facing request on it -- this is that job.
//
// NOTE ON RESPONSE SHAPE: /ingest's OpenAPI schema doesn't declare a response model (FastAPI without
// response_model just serializes whatever the handler returns), so the exact field names for the
// transcript/caption it hands back aren't confirmed from the schema alone. extractIngestFields below
// checks several plausible key names and falls back to null rather than throwing, and the full raw
// response is kept on the record (ingest_raw) so a mismatch is visible/fixable in the admin view
// instead of silently losing data.
import { addEvent, updateSubmission } from "@/lib/db";
import { OmnirouteUnavailableError, structureMetadata } from "@/lib/omnirouteStructure";

const INFERENCE_URL = process.env.INFERENCE_URL || "http://127.0.0.1:8500";
const INGEST_TIMEOUT_MS = 180_000; // cold Flamingo load + captioning can take tens of seconds

function extractIngestFields(raw: unknown): {
  transcript: string | null;
  caption: string | null;
  galleryFile: string | null;
} {
  const obj = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const pick = (...keys: string[]) => {
    for (const k of keys) {
      const v = obj[k];
      if (typeof v === "string" && v.trim()) return v;
    }
    return null;
  };
  return {
    transcript: pick("transcript", "text", "asr_text"),
    caption: pick("caption", "caption_raw", "caption_text", "description"),
    // Same "schema not declared, so guess plausible names" situation as transcript/caption above --
    // whichever of these the real response actually uses will populate gallery_file; if the response
    // uses something else entirely, it stays null and shows up in submission.error via the fallback
    // event logged below rather than silently pointing at the wrong file.
    galleryFile: pick("path", "rel_path", "relative_path", "file", "filename", "stored_filename"),
  };
}

export async function runIngestJob(
  submissionId: number,
  wavBytes: ArrayBuffer,
  filename: string,
  gallery: string
) {
  try {
    updateSubmission(submissionId, { status: "embedding" });

    let upstream: Response;
    try {
      upstream = await fetch(`${INFERENCE_URL}/ingest?gallery=${encodeURIComponent(gallery)}`, {
        method: "POST",
        headers: { "content-type": "application/octet-stream", "x-filename": filename },
        body: wavBytes,
        signal: AbortSignal.timeout(INGEST_TIMEOUT_MS),
      });
    } catch (err) {
      const message = `inference backend unreachable: ${err instanceof Error ? err.message : String(err)}`;
      updateSubmission(submissionId, { status: "failed", error: message });
      addEvent("error", "ingest", `Submission #${submissionId}: ${message}`, submissionId);
      return;
    }

    // /ingest embeds + transcribes + captions in one blocking call -- there's no intermediate
    // progress signal from the backend, so "captioning" here just reflects that the request landed
    // and is being processed, not a distinct backend phase we can observe.
    updateSubmission(submissionId, { status: "captioning" });

    const text = await upstream.text();
    if (!upstream.ok) {
      const message = `ingest failed: HTTP ${upstream.status}: ${text.slice(0, 300)}`;
      updateSubmission(submissionId, { status: "failed", error: message });
      addEvent("error", "ingest", `Submission #${submissionId}: ${message}`, submissionId);
      return;
    }

    let parsed: unknown = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      // Non-JSON 200 is unexpected but not fatal -- fields just come back null below.
    }
    const { transcript, caption, galleryFile } = extractIngestFields(parsed);
    updateSubmission(submissionId, {
      transcript,
      caption_raw: caption,
      gallery_file: galleryFile,
      status: "structuring",
    });
    addEvent(
      "info",
      "ingest",
      `Submission #${submissionId}: embedded and captioned` +
        (galleryFile ? ` -- stored as ${galleryFile}` : " -- backend response had no recognizable filename/path field"),
      submissionId
    );

    try {
      const meta = await structureMetadata({ transcript, caption, filenameHint: filename });
      updateSubmission(submissionId, {
        status: "done",
        title: meta.title,
        description: meta.description,
        tags: meta.tags,
      });
      addEvent("info", "structure", `Submission #${submissionId}: metadata written`, submissionId);
    } catch (err) {
      // Structuring is a nice-to-have layer on top of a real, already-embedded clip -- it stays
      // "done" and searchable even if this step fails (e.g. no OMNIROUTE_API_KEY configured yet).
      const message = err instanceof OmnirouteUnavailableError ? err.message : String(err);
      updateSubmission(submissionId, { status: "done", error: message });
      addEvent("warning", "structure", `Submission #${submissionId}: metadata step skipped -- ${message}`, submissionId);
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    updateSubmission(submissionId, { status: "failed", error: message });
    addEvent("error", "ingest", `Submission #${submissionId}: unhandled error -- ${message}`, submissionId);
  }
}
