import { NextRequest, NextResponse } from "next/server";
import { addEvent, createSubmission } from "@/lib/db";
import { runIngestJob } from "@/lib/ingestJob";

export async function POST(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const gallery = searchParams.get("gallery") || "sfx";
  const filename = request.headers.get("x-filename") || "recording.wav";

  const body = await request.arrayBuffer();
  if (body.byteLength === 0) {
    return NextResponse.json(
      { error: "empty request body -- expected raw WAV bytes" },
      { status: 400 }
    );
  }

  const submission = createSubmission(gallery, filename);
  addEvent(
    "info",
    "submit",
    `Submission #${submission.id} received (${filename}, ${(body.byteLength / 1024).toFixed(1)} KB)`,
    submission.id
  );

  // Fire-and-forget: /ingest on the inference backend is slow (tens of seconds for a cold model
  // load), so this responds immediately with the submission id and the client polls
  // GET /api/submit/:id for progress instead of blocking this request on it.
  runIngestJob(submission.id, body, filename, gallery).catch((err) => {
    addEvent(
      "error",
      "ingest",
      `Submission #${submission.id}: unhandled job error -- ${err instanceof Error ? err.message : String(err)}`,
      submission.id
    );
  });

  return NextResponse.json({ id: submission.id });
}
