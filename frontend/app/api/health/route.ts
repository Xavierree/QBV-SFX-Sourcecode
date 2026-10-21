import { NextResponse } from "next/server";

const INFERENCE_URL = process.env.INFERENCE_URL || "http://127.0.0.1:8500";

export async function GET() {
  try {
    const r = await fetch(`${INFERENCE_URL}/health`, { cache: "no-store" });
    const worker = await r.json();
    return NextResponse.json({ gateway: "ok", worker });
  } catch {
    return NextResponse.json(
      { gateway: "ok", worker: { status: "unreachable" } },
      { status: 503 }
    );
  }
}
