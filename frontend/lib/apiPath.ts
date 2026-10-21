// Next's basePath (next.config.ts) only auto-prefixes next/link, next/image, and router
// navigation -- a plain fetch("/api/...") string or an <audio>/<img> src set from JS is never
// touched by it. Every client-side call to this app's own API routes needs to go through this
// instead of a hardcoded "/api/..." literal, or it silently breaks once NEXT_PUBLIC_BASE_PATH is
// set (requests land on whatever app the path-splitter's catch-all serves, not this one).
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";

export function apiPath(path: string): string {
  return `${BASE_PATH}${path}`;
}
