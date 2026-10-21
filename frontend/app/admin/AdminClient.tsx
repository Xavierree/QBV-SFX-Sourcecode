"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiPath } from "@/lib/apiPath";

type Stats = {
  submissions_total: number;
  submissions_by_status: Record<string, number>;
  events_by_level: Record<string, number>;
  gallery_sizes: Record<string, number>;
};
type SubmissionRow = {
  id: number;
  status: string;
  gallery: string;
  original_filename: string;
  title: string | null;
  gallery_file: string | null;
  error: string | null;
  created_at: number;
};
type EventRow = {
  id: number;
  level: string;
  category: string;
  message: string;
  created_at: number;
  submission_id: number | null;
};

function fmtTime(unix: number) {
  return new Date(unix * 1000).toLocaleString();
}

type BackendStatus = "checking" | "up" | "down";

export default function AdminClient() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [submissions, setSubmissions] = useState<SubmissionRow[]>([]);
  const [events, setEvents] = useState<EventRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [starting, setStarting] = useState(false);
  const [startResult, setStartResult] = useState<string | null>(null);
  const [playingId, setPlayingId] = useState<number | null>(null);
  const playerRef = useRef<HTMLAudioElement | null>(null);

  // A start poll "owns" the status while it runs — the background interval
  // and manual Refresh must not paint over it with a mid-warm-up reading (the
  // backend is unreachable for its first seconds; that is "starting", not
  // "down"). Polls take a sequence number so a newer start cancels an
  // older one instead of interleaving.
  const pollIntentRef = useRef<"up" | "down" | null>(null);
  const pollSeqRef = useRef(0);
  const cancelledRef = useRef(false);

  const checkBackend = useCallback(async () => {
    if (pollIntentRef.current) return; // poll in flight — it owns the dot
    try {
      const r = await fetch(apiPath("/api/health"), { cache: "no-store" });
      const data = await r.json();
      setBackendStatus(data?.worker?.status === "ok" ? "up" : "down");
    } catch {
      setBackendStatus("down");
    }
  }, []);

  // Backend warm-up after start takes a few seconds.
  // Poll every 2s until the status reaches the expected state, up to ~80s.
  // While polling, the dot stays on "checking…" — never a false red/green —
  // and only the final honest state is reported if the poll exhausts.
  const pollBackendUntil = useCallback(async (want: "up" | "down") => {
    const seq = ++pollSeqRef.current;
    pollIntentRef.current = want;
    setBackendStatus("checking");
    for (let attempt = 0; attempt < 40; attempt++) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      if (cancelledRef.current || seq !== pollSeqRef.current) return;
      try {
        const r = await fetch(apiPath("/api/health"), { cache: "no-store" });
        const data = await r.json();
        const isUp = data?.worker?.status === "ok";
        if (isUp === (want === "up")) {
          setBackendStatus(isUp ? "up" : "down");
          pollIntentRef.current = null;
          return;
        }
        setBackendStatus("checking");
      } catch {
        if (want === "down") {
          setBackendStatus("down");
          pollIntentRef.current = null;
          return;
        }
        setBackendStatus("checking");
      }
    }
    try {
      const r = await fetch(apiPath("/api/health"), { cache: "no-store" });
      const data = await r.json();
      setBackendStatus(data?.worker?.status === "ok" ? "up" : "down");
    } catch {
      setBackendStatus("down");
    }
    pollIntentRef.current = null;
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const [statsResp, subsResp, eventsResp] = await Promise.all([
        fetch(apiPath("/api/admin/stats"), { cache: "no-store" }),
        fetch(apiPath("/api/admin/submissions?limit=50"), { cache: "no-store" }),
        fetch(apiPath("/api/admin/events?limit=50"), { cache: "no-store" }),
      ]);
      setStats(await statsResp.json());
      setSubmissions(await subsResp.json());
      setEvents(await eventsResp.json());
    } catch {
      setLoadError("Could not reach the admin API.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    const run = async () => {
      await Promise.resolve();
      if (!ignore) {
        checkBackend();
        load();
      }
    };
    run();
    // Keep the status honest while the page sits open: the backend can die on its own
    // (OOM, reboot), and the dot should show it without a manual refresh.
    const interval = setInterval(checkBackend, 15000);
    return () => {
      ignore = true;
      cancelledRef.current = true;
      clearInterval(interval);
    };
  }, [checkBackend, load]);

  const handleStartBackend = async () => {
    setStarting(true);
    setStartResult(null);
    try {
      const r = await fetch(apiPath("/api/admin/backend/start"), { method: "POST" });
      const data = await r.json();
      setStartResult(data.ok ? data.output || "triggered" : data.error || "failed");
    } catch (err) {
      setStartResult(err instanceof Error ? err.message : "failed");
    } finally {
      setStarting(false);
      pollBackendUntil("up");
    }
  };

  const togglePlay = (row: SubmissionRow) => {
    const player = playerRef.current;
    if (!player || !row.gallery_file) return;
    if (playingId === row.id && !player.paused) {
      player.pause();
      return;
    }
    const path = row.gallery_file.split("/").map(encodeURIComponent).join("/");
    player.src = apiPath(`/api/audio/${row.gallery}/${path}`);
    player.play();
    setPlayingId(row.id);
    player.onended = () => setPlayingId(null);
    player.onpause = () => setPlayingId(null);
  };

  return (
    <div className="flex w-full flex-col gap-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Admin</h1>
        <button
          type="button"
          onClick={() => {
            load();
            checkBackend();
          }}
          disabled={loading}
          className="rounded-full border border-border-strong px-3 py-1 text-xs text-muted-foreground hover:border-indigo disabled:opacity-50"
        >
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {loadError && <p className="text-sm text-destructive">{loadError}</p>}

      <section className="glass-card flex flex-col gap-3 rounded-lg p-6 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              backendStatus === "up" ? "bg-emerald-400" : backendStatus === "down" ? "bg-destructive" : "bg-muted-foreground"
            }`}
          />
          <span className="text-sm">
            Backend: <span className="font-medium">{backendStatus === "checking" ? "checking…" : backendStatus}</span>
          </span>
        </div>
        <div className="flex items-center gap-3">
          {startResult && <span className="text-xs text-muted-foreground">{startResult}</span>}
          <button
            type="button"
            onClick={handleStartBackend}
            disabled={starting}
            className="gradient-accent rounded-full px-4 py-1.5 text-xs font-medium text-white disabled:opacity-50"
          >
            {starting ? "Starting…" : "Start backend"}
          </button>
        </div>
      </section>

      {stats && (
        <section className="glass-card grid grid-cols-2 gap-4 rounded-lg p-6 sm:grid-cols-4">
          <Stat label="SFX gallery" value={stats.gallery_sizes.sfx ?? "—"} />
          <Stat label="qvim-dev gallery" value={stats.gallery_sizes.qdev ?? "—"} />
          <Stat label="Submissions" value={stats.submissions_total} />
          <Stat label="Errors logged" value={stats.events_by_level.error ?? 0} />
        </section>
      )}

      <section className="glass-card flex flex-col gap-3 rounded-lg p-6">
        <h2 className="text-lg font-medium">Submissions</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="text-muted-foreground">
                <th className="py-1.5 pr-3">ID</th>
                <th className="py-1.5 pr-3">Status</th>
                <th className="py-1.5 pr-3">Title / filename</th>
                <th className="py-1.5 pr-3">Gallery file</th>
                <th className="py-1.5 pr-3">Gallery</th>
                <th className="py-1.5 pr-3">Created</th>
                <th className="py-1.5">Error</th>
              </tr>
            </thead>
            <tbody>
              {submissions.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-3 text-muted-foreground">
                    No submissions yet.
                  </td>
                </tr>
              )}
              {submissions.map((s) => (
                <tr key={s.id} className="border-t border-border">
                  <td className="py-1.5 pr-3">{s.id}</td>
                  <td className="py-1.5 pr-3">
                    <span className={s.status === "failed" ? "text-destructive" : "text-foreground"}>{s.status}</span>
                  </td>
                  <td className="py-1.5 pr-3">{s.title || s.original_filename}</td>
                  <td className="py-1.5 pr-3">
                    {s.gallery_file ? (
                      <button
                        type="button"
                        onClick={() => togglePlay(s)}
                        className="inline-flex items-center gap-1.5 rounded-full border border-border-strong px-2 py-0.5 hover:border-indigo"
                        title={s.gallery_file}
                      >
                        <span>{playingId === s.id ? "❚❚" : "▶"}</span>
                        <span className="max-w-[14ch] truncate">{s.gallery_file}</span>
                      </button>
                    ) : (
                      <span className="text-muted-foreground">unknown</span>
                    )}
                  </td>
                  <td className="py-1.5 pr-3">{s.gallery}</td>
                  <td className="py-1.5 pr-3 whitespace-nowrap">{fmtTime(s.created_at)}</td>
                  <td className="py-1.5 text-destructive">{s.error || ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="glass-card flex flex-col gap-3 rounded-lg p-6">
        <h2 className="text-lg font-medium">Recent events</h2>
        <div className="flex flex-col gap-1.5 text-xs">
          {events.length === 0 && <p className="text-muted-foreground">No events logged.</p>}
          {events.map((e) => (
            <div key={e.id} className="flex items-start gap-2 border-t border-border pt-1.5 first:border-0 first:pt-0">
              <span
                className={`shrink-0 rounded-full px-2 py-0.5 font-medium ${
                  e.level === "error"
                    ? "bg-destructive/20 text-destructive"
                    : e.level === "warning"
                      ? "bg-surface-strong text-foreground"
                      : "text-muted-foreground"
                }`}
              >
                {e.level}
              </span>
              <span className="text-muted-foreground">[{e.category}]</span>
              <span className="min-w-0 flex-1 break-words">{e.message}</span>
              <span className="shrink-0 whitespace-nowrap text-muted-foreground">{fmtTime(e.created_at)}</span>
            </div>
          ))}
        </div>
      </section>

      <audio ref={playerRef} preload="none" />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-xl font-semibold">{value}</span>
    </div>
  );
}
