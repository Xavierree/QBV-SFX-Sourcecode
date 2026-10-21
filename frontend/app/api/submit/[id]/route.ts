import { NextRequest, NextResponse } from "next/server";
import { getSubmission } from "@/lib/db";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const numId = Number(id);
  const submission = Number.isFinite(numId) ? getSubmission(numId) : undefined;
  if (!submission) {
    return NextResponse.json({ error: "submission not found" }, { status: 404 });
  }
  return NextResponse.json(submission);
}
