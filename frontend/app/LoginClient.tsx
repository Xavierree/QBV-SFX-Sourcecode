"use client";

import { useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { apiPath } from "@/lib/apiPath";
import { WaveIcon } from "./components/Icons";

// Same-origin paths only: "/x" yes, "//evil.com" or "/\\evil.com" (protocol-relative) no.
function safeNext(next: string | null): string {
  if (!next || !next.startsWith("/") || next.startsWith("//") || next.startsWith("/\\")) return "/search";
  return next;
}

export default function LoginClient() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [passcode, setPasscode] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!passcode.trim() || submitting) return;
    setSubmitting(true);
    setStatus(null);
    try {
      const resp = await fetch(apiPath("/api/qbv-login"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ passcode }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || !data.ok) {
        setStatus(data.error === "wrong passcode" ? "Wrong passcode." : data.error || "Login failed.");
        setPasscode("");
        inputRef.current?.focus();
        return;
      }
      router.replace(safeNext(searchParams.get("next")));
      router.refresh();
    } catch {
      setStatus("Could not reach the server. Try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex w-full flex-1 flex-col items-center justify-center gap-5 py-10">
      <div className="glass-card flex w-full max-w-sm flex-col gap-5 rounded-lg p-8">
        <div className="flex flex-col items-center gap-3 text-center">
          <span className="gradient-accent flex h-12 w-12 items-center justify-center rounded-xl text-white shadow-lg">
            <WaveIcon width={22} height={22} />
          </span>
          <div className="flex flex-col gap-1">
            <h1 className="text-2xl font-extrabold tracking-tight">QBV</h1>
            <p className="text-sm text-muted-foreground">Hum → sound effects — vocal-imitation sound-effect search.</p>
          </div>
        </div>
        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <label className="flex flex-col gap-1.5 text-sm font-medium">
            Passcode
            <input
              ref={inputRef}
              type="password"
              value={passcode}
              onChange={(e) => setPasscode(e.target.value)}
              autoFocus
              autoComplete="off"
              disabled={submitting}
              className="rounded-md border border-border bg-card px-3 py-2.5 text-sm outline-none focus:border-indigo"
            />
          </label>
          <button
            type="submit"
            disabled={submitting || !passcode.trim()}
            className="gradient-accent rounded-full px-4 py-2.5 text-sm font-semibold text-white shadow-lg transition-transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:hover:scale-100 disabled:active:scale-100"
          >
            {submitting ? "Checking…" : "Enter"}
          </button>
          {status && (
            <p role="alert" className="text-center text-xs text-destructive">
              {status}
            </p>
          )}
        </form>
      </div>
    </div>
  );
}
