import { NextResponse } from "next/server";
import { getStatsBase } from "@/lib/db";

const INFERENCE_URL = process.env.INFERENCE_URL || "http://127.0.0.1:8500";

export async function GET() {
  const base = getStatsBase();

  let gallery_sizes: Record<string, number> = {};
  try {
    const r = await fetch(`${INFERENCE_URL}/health`, { cache: "no-store" });
    const data = await r.json();
    if (data?.galleries && typeof data.galleries === "object") {
      gallery_sizes = data.galleries;
    }
  } catch {
    // Backend unreachable -- stats still return with local submission/event counts, just no
    // live gallery sizes; the UI already renders "--" for a missing gallery key.
  }

  return NextResponse.json({ ...base, gallery_sizes });
}
