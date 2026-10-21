import { NextRequest, NextResponse } from "next/server";
import {
  GATE_COOKIE_NAME,
  GATE_MAX_AGE_SECONDS,
  currentPasscode,
  passcodeMatches,
  signGateValue,
} from "@/lib/qbvGate";

// Per-IP attempt limit. In-memory is enough here: this app runs as one process
// (one `next start` process), and a restart only resets the window early.
const MAX_ATTEMPTS = 10;
const WINDOW_MS = 15 * 60 * 1000;
const attempts = new Map<string, { count: number; resetAt: number }>();

function clientIp(request: NextRequest): string {
  return (
    request.headers.get("cf-connecting-ip") ||
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    request.headers.get("x-real-ip") ||
    "unknown"
  );
}

function tooManyAttempts(ip: string): { limited: boolean; retryAfter: number } {
  const now = Date.now();
  if (attempts.size > 10_000) {
    for (const [key, entry] of attempts) if (entry.resetAt <= now) attempts.delete(key);
  }
  const entry = attempts.get(ip);
  if (!entry || entry.resetAt <= now) {
    attempts.set(ip, { count: 1, resetAt: now + WINDOW_MS });
    return { limited: false, retryAfter: 0 };
  }
  entry.count += 1;
  return { limited: entry.count > MAX_ATTEMPTS, retryAfter: Math.ceil((entry.resetAt - now) / 1000) };
}

export async function POST(request: NextRequest) {
  const ip = clientIp(request);
  const { limited, retryAfter } = tooManyAttempts(ip);
  if (limited) {
    return NextResponse.json(
      { ok: false, error: "Too many attempts. Try again later." },
      { status: 429, headers: { "retry-after": String(retryAfter) } }
    );
  }

  const body = await request.json().catch(() => ({}));
  const submitted = typeof body?.passcode === "string" ? body.passcode : "";

  const expected = currentPasscode();
  if (!expected) {
    return NextResponse.json(
      { ok: false, error: "no passcode is configured (QBV_PASSCODE is empty): the app is open" },
      { status: 503 }
    );
  }

  if (!passcodeMatches(submitted, expected)) {
    return NextResponse.json({ ok: false, error: "wrong passcode" }, { status: 401 });
  }

  attempts.delete(ip);
  const res = NextResponse.json({ ok: true });
  res.cookies.set({
    name: GATE_COOKIE_NAME,
    value: signGateValue(expected),
    httpOnly: true,
    // The app is served over plain http://localhost, so the cookie must not require https.
    secure: request.nextUrl.protocol === "https:",
    sameSite: "lax",
    path: "/",
    maxAge: GATE_MAX_AGE_SECONDS,
  });
  return res;
}
