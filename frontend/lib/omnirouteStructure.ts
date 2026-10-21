// Turns a raw transcript/caption from the inference backend into title/description/tags, via the
// OmniRoute gateway (OpenAI-compatible /chat/completions, Bearer key; see the manual, chapter 2.4).
//
// Optional layer: if OMNIROUTE_API_KEY isn't set, the caller (lib/ingestJob.ts) catches
// OmnirouteUnavailableError and still marks the submission "done" -- the clip is embedded and
// searchable either way, it just won't have generated metadata yet.

export class OmnirouteUnavailableError extends Error {}

const DEFAULT_MODEL = process.env.OMNIROUTE_MODEL || "";
const TIMEOUT_MS = 60_000;

function completionsURL(): string {
  const root = (process.env.OMNIROUTE_BASE_URL || "http://localhost:20128/v1").replace(/\/+$/, "");
  return root.endsWith("/chat/completions") ? root : `${root}/chat/completions`;
}

export interface StructuredMetadata {
  title: string;
  description: string;
  tags: string[];
}

function extractJson(text: string): unknown {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1] : text;
  return JSON.parse(candidate.trim());
}

export async function structureMetadata(opts: {
  transcript: string | null;
  caption: string | null;
  filenameHint: string;
}): Promise<StructuredMetadata> {
  const apiKey = process.env.OMNIROUTE_API_KEY;
  if (!apiKey) {
    throw new OmnirouteUnavailableError("OMNIROUTE_API_KEY is not set -- add it to .env.local");
  }
  if (!DEFAULT_MODEL) {
    throw new OmnirouteUnavailableError("OMNIROUTE_MODEL is not set -- add it to .env.local");
  }
  if (!opts.transcript && !opts.caption) {
    throw new OmnirouteUnavailableError("nothing to structure -- the backend returned no transcript or caption");
  }

  const prompt = [
    "You are labeling one sound-effect clip for a searchable library.",
    `Filename hint: ${opts.filenameHint}`,
    opts.caption ? `Caption from an audio-captioning model: ${opts.caption}` : null,
    opts.transcript ? `Transcript of any speech/vocalization in the clip: ${opts.transcript}` : null,
    "",
    "Reply with ONLY a JSON object, no markdown fence, shaped exactly as:",
    '{"title": "short human title", "description": "one sentence description", "tags": ["lowercase", "tag", "list"]}',
  ]
    .filter(Boolean)
    .join("\n");

  let upstream: Response;
  try {
    upstream = await fetch(completionsURL(), {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model: DEFAULT_MODEL,
        messages: [{ role: "user", content: prompt }],
        temperature: 0.3,
        max_tokens: 300,
        stream: false,
      }),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (err) {
    throw new OmnirouteUnavailableError(
      `could not reach Omniroute: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  const raw = await upstream.text();
  if (!upstream.ok) {
    throw new OmnirouteUnavailableError(`Omniroute returned HTTP ${upstream.status}: ${raw.slice(0, 300)}`);
  }

  let parsed: { choices?: Array<{ message?: { content?: string | null } }> };
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new OmnirouteUnavailableError(`Omniroute returned non-JSON: ${raw.slice(0, 300)}`);
  }

  const content = parsed.choices?.[0]?.message?.content?.trim();
  if (!content) {
    throw new OmnirouteUnavailableError("Omniroute returned an empty completion");
  }

  let meta: Partial<StructuredMetadata>;
  try {
    meta = extractJson(content) as Partial<StructuredMetadata>;
  } catch {
    throw new OmnirouteUnavailableError(`could not parse metadata JSON from model reply: ${content.slice(0, 300)}`);
  }

  return {
    title: typeof meta.title === "string" && meta.title.trim() ? meta.title.trim() : opts.filenameHint,
    description: typeof meta.description === "string" ? meta.description.trim() : "",
    tags: Array.isArray(meta.tags) ? meta.tags.filter((t): t is string => typeof t === "string") : [],
  };
}
