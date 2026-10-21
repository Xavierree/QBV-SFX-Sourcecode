// Optional passcode gate for the whole app. One shared code, no account. It is configured entirely from
// environment variables (frontend/.env.local):
//   QBV_PASSCODE     the passcode; if it is empty or unset the gate is OFF and every page is open.
//   QBV_GATE_SECRET  secret used to sign the login cookie (required when QBV_PASSCODE is set);
//                    generate once with `openssl rand -hex 32`.
import crypto from "node:crypto";

export const GATE_COOKIE_NAME = "qbv-gate";
// Long-lived on purpose: the passcode only keeps casual visitors out, it is not a per-session login.
export const GATE_MAX_AGE_SECONDS = 365 * 24 * 3600;

/** The configured passcode, or null when the gate is disabled. */
export function currentPasscode(): string | null {
  return process.env.QBV_PASSCODE || null;
}

/** True when a passcode is configured, i.e. visitors must log in. */
export function gateEnabled(): boolean {
  return currentPasscode() !== null;
}

function secret(): string {
  const value = process.env.QBV_GATE_SECRET;
  if (!value) throw new Error("QBV_GATE_SECRET is not set -- add it to .env.local");
  return value;
}

/** Constant-time string compare (for the submitted passcode). */
export function passcodeMatches(submitted: string, expected: string): boolean {
  const a = crypto.createHash("sha256").update(submitted).digest();
  const b = crypto.createHash("sha256").update(expected).digest();
  return crypto.timingSafeEqual(a, b);
}

// The MAC key mixes in the passcode itself, so changing the passcode revokes
// every cookie issued under the old one -- no separate revocation list to keep.
function mac(payload: string, passcode: string): string {
  return crypto.createHmac("sha256", `${secret()}\0${passcode}`).update(payload).digest("hex");
}

/** HMAC-signed cookie value, a fixed marker + timestamp, signed so it can't be forged by just setting a cookie by hand. */
export function signGateValue(passcode: string): string {
  const payload = `ok.${Date.now()}`;
  return `${payload}.${mac(payload, passcode)}`;
}

export function verifyGateValue(value: string | undefined | null): boolean {
  if (!value) return false;
  const lastDot = value.lastIndexOf(".");
  if (lastDot <= 0) return false;
  const payload = value.slice(0, lastDot);
  const given = value.slice(lastDot + 1);

  const [marker, ts] = payload.split(".");
  const age = Date.now() - Number(ts);
  if (marker !== "ok" || !Number.isFinite(age) || age < 0 || age > GATE_MAX_AGE_SECONDS * 1000) return false;

  const passcode = currentPasscode();
  if (!passcode) return false;
  let expectedMac: string;
  try {
    expectedMac = mac(payload, passcode);
  } catch {
    return false;
  }
  const a = Buffer.from(given, "hex");
  const b = Buffer.from(expectedMac, "hex");
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}
