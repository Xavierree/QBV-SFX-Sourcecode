// app/not-found.tsx — custom 404 inside /qbv. Visitors without the passcode
// cookie are sent to the login page by proxy.ts before they get here.
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Page not found",
  robots: { index: false },
};

export default function NotFound() {
  return (
    <div className="flex w-full flex-1 flex-col items-center justify-center gap-6">
      <div className="glass-card flex w-full max-w-sm flex-col items-center gap-3 rounded-lg p-8 text-center">
        <span className="rounded-full border border-border px-3 py-0.5 text-xs text-muted-foreground">404</span>
        <h1 className="text-xl font-semibold tracking-tight">This page doesn&apos;t exist</h1>
        <p className="text-sm text-muted-foreground">The link may be old, or the address may have a typo.</p>
        <Link
          href="/search"
          className="mt-2 rounded-full gradient-accent px-5 py-2.5 text-sm font-medium text-white shadow-lg"
        >
          Back to search
        </Link>
      </div>
    </div>
  );
}
