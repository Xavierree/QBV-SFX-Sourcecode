"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { audioBufferToWav, drawStaticWaveform } from "@/lib/wav";
import { apiPath } from "@/lib/apiPath";
import { CheckIcon, FileAudioIcon, PauseIcon, PlayIcon, UploadIcon, WaveIcon } from "../components/Icons";

type Submission = {
  id: number;
  status: "pending" | "embedding" | "captioning" | "structuring" | "done" | "failed";
  gallery: string;
  original_filename: string;
  title: string | null;
  tags: string[] | null;
  description: string | null;
  caption_raw: string | null;
  transcript: string | null;
  gallery_file: string | null;
  error: string | null;
};

const STAGE_LABEL: Record<Submission["status"], string> = {
  pending: "Queued…",
  embedding: "Embedding with the reference tower…",
  captioning: "Describing the sound (local ALM)…",
  structuring: "Writing metadata (Gemini via omniroute)…",
  done: "Done",
  failed: "Failed",
};

const STAGES: Submission["status"][] = ["pending", "embedding", "captioning", "structuring", "done"];

// One-word labels under the progress bar (the long stage text lives in the header).
const STAGE_SHORT: Record<(typeof STAGES)[number], string> = {
  pending: "Queued",
  embedding: "Embed",
  captioning: "Caption",
  structuring: "Metadata",
  done: "Live",
  failed: "Failed",
};

type PendingUpload = { blob: Blob; filename: string; duration: number };

export default function SubmitClient() {
  const [nameHint, setNameHint] = useState("");
  const [submission, setSubmission] = useState<Submission | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState("Choose an audio file, then Submit when you're ready.");
  const [pending, setPending] = useState<PendingUpload | null>(null);
  const [dragging, setDragging] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const playerRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);

  const stopPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = null;
  };

  const pollSubmission = (id: number) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      const resp = await fetch(apiPath(`/api/submit/${id}`), { cache: "no-store" });
      if (!resp.ok) return;
      const data: Submission = await resp.json();
      setSubmission(data);
      if (data.status === "done" || data.status === "failed") stopPolling();
    }, 1500);
  };

  useEffect(() => stopPolling, []);

  const submitRecording = async (wavBlob: Blob, filename: string) => {
    setSubmitting(true);
    setSubmission(null);
    setStatus(`Uploading ${filename}…`);
    try {
      const resp = await fetch(apiPath("/api/submit?gallery=sfx"), {
        method: "POST",
        headers: { "x-filename": nameHint || filename || "upload.wav" },
        body: wavBlob,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      setSubmission({ id: data.id, status: "pending" } as Submission);
      setStatus("Submitted — tracking progress below.");
      pollSubmission(data.id);
    } catch (err) {
      setStatus(`Submission failed: ${err instanceof Error ? err.message : "unknown error"}`);
    } finally {
      setSubmitting(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-picking the same file
    if (file) await processFile(file);
  };

  const processFile = async (file: File) => {
    setPending(null);
    setStatus(`Reading ${file.name}…`);
    try {
      const arrBuf = await file.arrayBuffer();
      const ctx = new AudioContext();
      const audioBuffer = await ctx.decodeAudioData(arrBuf);
      await ctx.close();
      if (canvasRef.current) drawStaticWaveform(canvasRef.current, audioBuffer);
      const wavBlob = audioBufferToWav(audioBuffer);
      setPending({ blob: wavBlob, filename: file.name, duration: audioBuffer.duration });
      setStatus(`Loaded ${file.name} — ${audioBuffer.duration.toFixed(1)}s. Add a name hint if you want, then Submit.`);
    } catch (err) {
      setStatus(`Could not read audio file: ${err instanceof Error ? err.message : "unknown error"}`);
    }
  };

  // Drag & drop: the whole upload card is the target. PreventDefault on
  // document-level drop stops the browser navigating to a dropped file when
  // the user misses the card; the card itself routes files into processFile.
  useEffect(() => {
    const prevent = (e: DragEvent) => e.preventDefault();
    document.addEventListener("dragover", prevent);
    document.addEventListener("drop", prevent);
    return () => {
      document.removeEventListener("dragover", prevent);
      document.removeEventListener("drop", prevent);
    };
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (submitting) return;
    const file = e.dataTransfer.files?.[0];
    if (file) processFile(file);
  };

  const handleSubmit = () => {
    if (!pending) return;
    submitRecording(pending.blob, pending.filename);
    setPending(null);
  };

  const stageIndex = submission ? STAGES.indexOf(submission.status === "failed" ? "done" : submission.status) : -1;

  // Drag-enter/leave fire per child element, so count them instead of toggling
  // on a single leave — the highlight must survive crossing into the buttons
  // and canvas inside the card.
  const dragDepth = useRef(0);

  const togglePlay = () => {
    const player = playerRef.current;
    if (!player || !submission?.gallery_file) return;
    if (playing && !player.paused) {
      player.pause();
      return;
    }
    const path = submission.gallery_file.split("/").map(encodeURIComponent).join("/");
    player.src = apiPath(`/api/audio/${submission.gallery}/${path}`);
    player.play();
    setPlaying(true);
    player.onended = () => setPlaying(false);
    player.onpause = () => setPlaying(false);
  };

  return (
    <div className="flex w-full flex-col gap-6">
      <header className="flex flex-col gap-3">
        <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl">Add to the gallery</h1>
        <p className="max-w-[55ch] text-sm text-muted-foreground sm:text-base">
          Upload a new sound effect. It gets embedded, transcribed, captioned, and tagged automatically — then
          you can find it again by mimicking it, just like every other clip in the SFX library.
        </p>
      </header>

      <section
        onDragEnter={(e) => {
          e.preventDefault();
          dragDepth.current += 1;
          setDragging(true);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={(e) => {
          e.preventDefault();
          dragDepth.current -= 1;
          if (dragDepth.current <= 0) {
            dragDepth.current = 0;
            setDragging(false);
          }
        }}
        onDrop={handleDrop}
        className={`glass-card flex flex-col gap-4 rounded-lg p-5 transition-colors sm:p-6 ${
          dragging ? "border-indigo bg-surface-strong" : ""
        }`}
      >
        <label className="flex flex-col gap-1.5 text-sm">
          <span className="font-medium">Name hint</span>
          <span className="text-xs text-muted-foreground">Optional — helps the metadata step, not required.</span>
          <input
            type="text"
            value={nameHint}
            onChange={(e) => setNameHint(e.target.value)}
            placeholder="e.g. car-door-slam.wav"
            className="rounded-md border border-border bg-card px-3 py-2.5 text-sm outline-none focus:border-indigo"
          />
        </label>

        <input ref={fileInputRef} type="file" accept="audio/*" onChange={handleFileChange} disabled={submitting} className="hidden" />

        {pending ? (
          <div className="flex flex-wrap items-center gap-3 rounded-md border border-border bg-card px-3.5 py-3">
            <FileAudioIcon width={16} height={16} className="shrink-0 text-indigo" />
            <span className="min-w-0 flex-1 truncate text-sm font-medium">{pending.filename}</span>
            <span className="text-xs text-muted-foreground">{pending.duration.toFixed(1)}s</span>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={submitting}
              className="rounded-full border border-border-strong px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-indigo hover:text-foreground disabled:opacity-50"
            >
              Change file
            </button>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={submitting}
            className={`group flex flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed px-6 py-9 text-center transition-colors disabled:opacity-50 ${
              dragging ? "border-indigo bg-surface-strong" : "border-border-strong hover:border-indigo"
            }`}
          >
            <UploadIcon
              width={20}
              height={20}
              className={`transition-colors ${dragging ? "text-indigo" : "text-muted-foreground group-hover:text-indigo"}`}
            />
            <span className={`text-sm font-medium ${dragging ? "text-indigo" : ""}`}>
              {dragging ? "Drop it here" : "Choose an audio file"}
            </span>
            <span className="text-xs text-muted-foreground">
              Drag &amp; drop, or click to browse. WAV, MP3 — anything the browser can decode.
            </span>
          </button>
        )}

        {pending && (
          <canvas ref={canvasRef} width={900} height={90} className="w-full rounded-md border border-border bg-card" />
        )}

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!pending || submitting}
            className="gradient-accent shrink-0 rounded-full px-5 py-2.5 text-sm font-semibold text-white shadow-lg transition-transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:hover:scale-100 disabled:active:scale-100"
          >
            {submitting ? "Submitting…" : "Submit"}
          </button>
          <p className="text-sm text-muted-foreground">{status}</p>
        </div>
      </section>

      {submission && (
        <section className="glass-card flex flex-col gap-4 rounded-lg p-5 sm:p-6">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-base font-semibold">Submission #{submission.id}</h2>
            <span
              className={`text-xs font-medium ${submission.status === "failed" ? "text-destructive" : "text-muted-foreground"}`}
            >
              {STAGE_LABEL[submission.status]}
            </span>
          </div>

          {submission.status !== "failed" && (
            <div className="flex flex-col gap-2">
              <div className="flex gap-1">
                {STAGES.map((s, i) => (
                  <div key={s} className={`h-1.5 flex-1 rounded-full ${i <= stageIndex ? "gradient-accent" : "bg-surface-strong"}`} />
                ))}
              </div>
              <div className="grid grid-cols-5 gap-1 text-center text-[10px]">
                {STAGES.map((s, i) => (
                  <span key={s} className={i <= stageIndex ? "font-semibold text-indigo" : "text-muted-foreground"}>
                    {STAGE_SHORT[s]}
                  </span>
                ))}
              </div>
            </div>
          )}

          {submission.status === "failed" && (
            <p className="rounded-md border border-border bg-card px-3 py-2 text-xs text-destructive">
              {submission.error || "Something went wrong."}
            </p>
          )}

          {submission.status === "done" && (
            <div className="flex flex-col gap-2 text-sm">
              {submission.title && (
                <p className="flex items-center gap-2 font-semibold">
                  <CheckIcon width={15} height={15} className="shrink-0 text-indigo" />
                  {submission.title}
                </p>
              )}
              {submission.description && (
                <p>
                  <span className="text-muted-foreground">Description: </span>
                  {submission.description}
                </p>
              )}
              {submission.tags && submission.tags.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {submission.tags.map((t) => (
                    <span key={t} className="rounded-full border border-border bg-surface px-2.5 py-0.5 text-xs text-muted-foreground">
                      {t}
                    </span>
                  ))}
                </div>
              )}
              {submission.error && (
                <p className="text-xs text-muted-foreground">
                  (metadata step: {submission.error} — the sound is still fully searchable)
                </p>
              )}
              {submission.gallery_file ? (
                <button
                  type="button"
                  onClick={togglePlay}
                  className="flex w-fit items-center gap-2 rounded-full border border-border-strong px-3 py-1.5 text-xs transition-colors hover:border-indigo"
                >
                  <span className="text-indigo">{playing ? <PauseIcon width={12} height={12} /> : <PlayIcon width={12} height={12} />}</span>
                  <span>Stored as {submission.gallery_file}</span>
                </button>
              ) : (
                <p className="text-xs text-muted-foreground">
                  (stored filename unknown — the backend response didn&apos;t include a field this app recognizes)
                </p>
              )}
              <p className="pt-2 text-xs text-muted-foreground">
                It&apos;s live —{" "}
                <Link href="/search" className="font-medium text-indigo underline underline-offset-[3px] hover:text-foreground">
                  go to Search
                </Link>{" "}
                and try mimicking it.
              </p>
            </div>
          )}
        </section>
      )}

      <audio ref={playerRef} preload="none" />
    </div>
  );
}
