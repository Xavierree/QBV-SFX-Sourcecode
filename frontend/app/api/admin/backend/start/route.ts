import { NextResponse } from "next/server";
import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
// The backend lives next to the frontend on the same machine: <repo>/backend/scripts/run_server.sh.
// It is idempotent (no-ops if the backend is already up), so this is safe to call anytime.
// BACKEND_DIR overrides the location if the two folders are not siblings.
const BACKEND_DIR = process.env.BACKEND_DIR || path.join(process.cwd(), "..", "backend");
const SCRIPT_PATH = path.join(BACKEND_DIR, "scripts", "run_server.sh");

export async function POST() {
  try {
    const { stdout, stderr } = await execFileAsync("bash", [SCRIPT_PATH], { timeout: 20_000 });
    return NextResponse.json({ ok: true, output: (stdout + stderr).trim() });
  } catch (err) {
    const message = err instanceof Error ? err.message : "unknown error";
    return NextResponse.json({ ok: false, error: message }, { status: 502 });
  }
}
