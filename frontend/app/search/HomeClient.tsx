"use client";

import { useRef, useState } from "react";
import { useRecorder } from "@/lib/useRecorder";
import { apiPath } from "@/lib/apiPath";
import { EqBars, LayersIcon, MicIcon, PauseIcon, PlayIcon, StopIcon, WaveIcon } from "../components/Icons";

type Gallery = "sfx" | "qdev";
type ResultItem = { rank: number; file: string; score: number; tower_score?: number };
type SearchResponse = {
  gallery: Gallery;
  gallery_size: number;
  results: ResultItem[];
  transcript: string | null;
  query_ms: number;
};

const GALLERY_LABEL: Record<Gallery, string> = {
  sfx: "SFX Library",
  qdev: "qvim-dev",
};
const GALLERY_DESC: Record<Gallery, string> = {
  sfx: "9,708+ clip production library — an ensemble of two dual-tower encoders (MATPAC++ and BEATs) + soft-DTW head; queries with spoken words are also matched by their Whisper transcript.",
  qdev: "121-item benchmark set, 20 sound classes — the same ensemble + soft-DTW head, no transcript matching.",
};

export default function HomeClient() {
  const [gallery, setGallery] = useState<Gallery>("sfx");
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [playingFile, setPlayingFile] = useState<string | null>(null);
  const playerRef = useRef<HTMLAudioElement | null>(null);
  const lastWavBlobRef = useRef<Blob | null>(null);

  const runSearch = async (wavBlob: Blob, g: Gallery, updateStatus: (s: string) => void) => {
    setLoading(true);
    updateStatus("Searching…");
    try {
      const resp = await fetch(apiPath(`/api/search?gallery=${g}&k=10`), { method: "POST", body: wavBlob });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${resp.status}`);
      }
      const data: SearchResponse = await resp.json();
      setResults(data);
      updateStatus("Done. Record again to try another sound.");
    } catch (err) {
      setResults(null);
      updateStatus(`Search failed: ${err instanceof Error ? err.message : "unknown error"}`);
    } finally {
      setLoading(false);
    }
  };

  const { recording, status, setStatus, canvasRef, toggleRecording } = useRecorder((wavBlob, duration, updateStatus) => {
    lastWavBlobRef.current = wavBlob;
    updateStatus(`Captured ${duration.toFixed(1)}s — searching…`);
    runSearch(wavBlob, gallery, updateStatus);
  }, { autoStop: true });

  const handleGalleryChange = (g: Gallery) => {
    setGallery(g);
    setResults(null);
    setStatus("Click record, then make your sound (up to 10s).");
    if (lastWavBlobRef.current) {
      runSearch(lastWavBlobRef.current, g, setStatus);
    }
  };

  const togglePlay = (file: string) => {
    const player = playerRef.current;
    if (!player) return;
    const src = apiPath(`/api/audio/${gallery}/${file.split("/").map(encodeURIComponent).join("/")}`);
    if (playingFile === file && !player.paused) {
      player.pause();
      return;
    }
    player.src = src;
    player.play();
    setPlayingFile(file);
    player.onended = () => setPlayingFile(null);
    player.onpause = () => setPlayingFile(null);
  };

  const busy = recording || loading;

  return (
    <div className="flex w-full flex-col gap-6">
      <header className="flex flex-col gap-3">
        <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl">
          Hum <span className="bg-gradient-to-r from-indigo to-purple bg-clip-text text-transparent">to</span>{" "}
          Sound&nbsp;Effect
        </h1>
        <p className="max-w-[55ch] text-sm text-muted-foreground sm:text-base">
          Record a vocal imitation, get the closest sound effects, ranked. A browse tool, not an oracle.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1 text-xs text-muted-foreground">
            <span className="gradient-accent h-1.5 w-1.5 rounded-full" aria-hidden />
            All models run locally
          </span>
        </div>
      </header>

      <section className="glass-card flex flex-col gap-3 rounded-lg p-4 sm:flex-row sm:items-center sm:justify-between sm:p-5">
        <div className="inline-flex self-start rounded-full border border-border p-1" role="tablist" aria-label="Gallery">
          {(["sfx", "qdev"] as Gallery[]).map((g) => (
            <button
              key={g}
              type="button"
              role="tab"
              aria-selected={gallery === g}
              onClick={() => handleGalleryChange(g)}
              className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                gallery === g ? "gradient-accent text-white" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {GALLERY_LABEL[g]}
            </button>
          ))}
        </div>
        <span className="text-xs text-muted-foreground sm:max-w-[40ch] sm:text-right">{GALLERY_DESC[gallery]}</span>
      </section>

      <section className="glass-card flex flex-col gap-5 rounded-lg p-5 sm:p-6">
        <div className="flex flex-col gap-5 sm:flex-row sm:items-center">
          <button
            type="button"
            onClick={toggleRecording}
            aria-pressed={recording}
            aria-label={recording ? "Stop and search" : "Start recording"}
            className={`gradient-accent flex h-20 w-20 shrink-0 items-center justify-center rounded-full text-white shadow-lg transition-transform hover:scale-105 active:scale-95 ${
              recording ? "recording-pulse" : ""
            }`}
          >
            {recording ? <StopIcon width={26} height={26} /> : <MicIcon width={26} height={26} />}
          </button>
          <div className="flex min-w-0 flex-1 flex-col gap-1.5">
            <div className="flex items-center gap-2.5">
              <span className="text-sm font-semibold">
                {recording ? "Recording" : loading ? "Searching" : "Ready"}
              </span>
              {busy && <EqBars />}
            </div>
            <p className="truncate text-sm text-muted-foreground" title={status}>
              {status}
            </p>
          </div>
        </div>
        <div className="flex flex-col gap-2">
          <canvas ref={canvasRef} width={900} height={90} className="w-full rounded-md border border-border bg-card" />
          <p className="text-xs text-muted-foreground">
            Tap the mic, make your sound (up to 10s), tap again to search.
          </p>
        </div>
      </section>

      <section className="glass-card flex flex-col gap-4 rounded-lg p-5 sm:p-6">
        <div className="flex items-center justify-between gap-3">
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <LayersIcon width={16} height={16} className="text-indigo" />
            Results
          </h2>
          {results && (
            <span className="text-xs text-muted-foreground">
              {results.results.length} of {results.gallery_size.toLocaleString()} clips ·{" "}
              {results.query_ms.toFixed(0)}ms
            </span>
          )}
        </div>

        {results?.transcript && (
          <p className="border-l-2 border-indigo pl-3 text-xs text-muted-foreground">
            Recognised speech: <span className="font-medium text-foreground">&ldquo;{results.transcript}&rdquo;</span>
          </p>
        )}

        {loading ? (
          <div className="flex items-center justify-center gap-3 rounded-md border border-dashed border-border px-6 py-12 text-sm text-muted-foreground">
            <EqBars height={14} />
            Searching the library…
          </div>
        ) : !results ? (
          <div className="flex flex-col items-center gap-2 rounded-md border border-dashed border-border px-6 py-12 text-center">
            <WaveIcon width={22} height={22} className="text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              No query yet. Record a sound above and the closest matches land here, ranked.
            </p>
          </div>
        ) : results.results.length === 0 ? (
          <div className="flex flex-col items-center gap-2 rounded-md border border-dashed border-border px-6 py-12 text-center">
            <WaveIcon width={22} height={22} className="text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              No matches. Try a clearer, slightly longer recording.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            {results.results.map((r, i) => (
              <div
                key={r.file}
                className="rise-in flex items-center gap-3 rounded-md border border-transparent px-2.5 py-2.5 transition-colors hover:border-border hover:bg-surface-strong"
                style={{ animationDelay: `${Math.min(i, 8) * 45}ms` }}
              >
                <span
                  className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-xs font-bold ${
                    r.rank === 1 ? "gradient-accent text-white" : "bg-surface-strong text-muted-foreground"
                  }`}
                >
                  {r.rank}
                </span>
                <span className="min-w-0 flex-1 truncate text-sm" title={r.file}>
                  {r.file}
                </span>
                <span className="shrink-0 font-mono text-xs text-muted-foreground">{r.score.toFixed(3)}</span>
                <button
                  type="button"
                  onClick={() => togglePlay(r.file)}
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border-strong text-indigo transition-colors hover:border-indigo hover:bg-surface-strong active:scale-95"
                  aria-label={playingFile === r.file ? "Pause" : "Play"}
                >
                  {playingFile === r.file ? <PauseIcon /> : <PlayIcon />}
                </button>
              </div>
            ))}
          </div>
        )}
      </section>

      <audio ref={playerRef} preload="none" />
    </div>
  );
}
