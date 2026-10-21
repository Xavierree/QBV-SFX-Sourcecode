import { NextRequest, NextResponse } from "next/server";

// The Python inference service (backend/server/main.py) — internal, loopback-only, holds the
// models. This route is the only thing allowed to reach it, mirroring the old SFX/webapp Bun gateway's
// split: the browser never talks to the model-serving process directly.
const INFERENCE_URL = process.env.INFERENCE_URL || "http://127.0.0.1:8500";

export async function POST(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const gallery = searchParams.get("gallery");
  const k = searchParams.get("k") ?? "10";

  if (gallery !== "sfx" && gallery !== "qdev") {
    return NextResponse.json({ error: "gallery must be sfx or qdev" }, { status: 400 });
  }

  const body = await request.arrayBuffer();
  if (body.byteLength === 0) {
    return NextResponse.json(
      { error: "empty request body -- expected raw WAV bytes" },
      { status: 400 }
    );
  }

  let upstream: Response;
  try {
    upstream = await fetch(
      `${INFERENCE_URL}/search?gallery=${gallery}&k=${encodeURIComponent(k)}`,
      {
        method: "POST",
        headers: { "content-type": "application/octet-stream" },
        body,
      }
    );
  } catch {
    return NextResponse.json({ error: "inference service unreachable" }, { status: 502 });
  }

  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: { "content-type": "application/json" },
  });
}
