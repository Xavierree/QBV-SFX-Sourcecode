import { NextRequest, NextResponse } from "next/server";
import { listSubmissions } from "@/lib/db";

export async function GET(request: NextRequest) {
  const limit = Number(new URL(request.url).searchParams.get("limit")) || 50;
  return NextResponse.json(listSubmissions(limit));
}
