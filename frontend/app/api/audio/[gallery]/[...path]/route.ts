import { NextRequest, NextResponse } from "next/server";

// Proxies to the backend's own /audio endpoint (backend/server/main.py), which owns the gallery files.
const INFERENCE_URL = process.env.INFERENCE_URL || "http://127.0.0.1:8500";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ gallery: string; path: string[] }> }
) {
  const { gallery, path: parts } = await params;
  const upstreamPath = parts.map((p) => encodeURIComponent(p)).join("/");

  let upstream: Response;
  try {
    upstream = await fetch(`${INFERENCE_URL}/audio/${gallery}/${upstreamPath}`, { cache: "force-cache" });
  } catch {
    return NextResponse.json({ error: "inference service unreachable" }, { status: 502 });
  }
  if (!upstream.ok || !upstream.body) {
    return NextResponse.json({ error: "not found" }, { status: upstream.status || 404 });
  }

  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: {
      "content-type": upstream.headers.get("content-type") || "audio/wav",
      "cache-control": "public, max-age=31536000, immutable",
    },
  });
}
