"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import ThemeToggle from "./ThemeToggle";
import { WaveIcon } from "./Icons";

const LINKS = [
  { href: "/search", label: "Search" },
  { href: "/submit", label: "Submit" },
  { href: "/admin", label: "Admin" },
];

// Top bar shared by every unlocked page. One line on desktop (well under the
// 80px cap); on narrow screens the right cluster wraps below the wordmark.
// Sticky + translucent so content scrolling under it still reads as the app.
// `initialTheme` is the server-derived theme (from the qbv_theme cookie) so
// the toggle's first render matches the served HTML.
export default function NavBar({ initialTheme }: { initialTheme: "dark" | "light" }) {
  const pathname = usePathname();
  if (pathname === "/") return null; // the login page has no nav to a still-locked app

  return (
    <nav className="sticky top-0 z-20 border-b border-border bg-background/80 backdrop-blur-xl">
      <div className="mx-auto flex w-full max-w-3xl flex-wrap items-center justify-between gap-x-3 gap-y-2 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <Link href="/search" className="flex items-center gap-2.5" aria-label="QBV — hum to sound effects">
            <span className="gradient-accent flex h-8 w-8 items-center justify-center rounded-md text-white shadow-sm">
              <WaveIcon width={16} height={16} />
            </span>
            <span className="flex flex-col leading-none">
              <span className="text-sm font-bold tracking-tight">QBV</span>
              <span className="hidden text-[10px] text-muted-foreground sm:block">
                hum → sound effects
              </span>
            </span>
          </Link>
        </div>

        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-full border border-border p-1" role="navigation" aria-label="App sections">
            {LINKS.map((l) => {
              const active = pathname?.startsWith(l.href);
              return (
                <Link
                  key={l.href}
                  href={l.href}
                  className={`rounded-full px-3.5 py-1.5 text-sm font-medium transition-colors ${
                    active ? "gradient-accent text-white" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {l.label}
                </Link>
              );
            })}
          </div>
          <ThemeToggle theme={initialTheme} />
        </div>
      </div>
    </nav>
  );
}
