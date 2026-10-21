"use client";

// Theme switcher, now part of the top bar (see NavBar): a two-option segmented
// control with plain "Dark" / "Light" labels rather than a single icon button.
// QBV's main audience is older professors and lecturers, and a lone sun/moon icon is easy to
// misread — showing BOTH options with the active one highlighted makes the
// control impossible to miss.
//
// `theme` comes from the server layout (cookie-derived) as a prop so the first render matches the served
// HTML exactly (no hydration mismatch, no flash of the wrong segment active).
// The choice is persisted in the qbv_theme cookie and re-applied by the
// server on the next render.

import { useState } from "react";

type Theme = "dark" | "light";

// Theme cookie. Stored as
// a cookie rather than localStorage so the server can always render the right
// theme on first paint — no flash, no client-side bootstrapping.
export const THEME_COOKIE = "qbv_theme";
const THEME_MAX_AGE = 31536000; // 1 year

// Plain module function (not a component/hook) so the React Compiler's
// immutability rule doesn't flag the document mutations — a click handler is
// exactly where a theme side effect belongs.
function persistTheme(next: Theme) {
  document.documentElement.classList.toggle("light", next === "light");
  document.cookie = `${THEME_COOKIE}=${next}; path=/; max-age=${THEME_MAX_AGE}; SameSite=Lax`;
}

export default function ThemeToggle({ theme }: { theme: Theme }) {
  const [current, setCurrent] = useState<Theme>(theme);

  const apply = (next: Theme) => {
    if (next === current) return;
    setCurrent(next);
    // The class flips instantly (every color is a CSS variable), and the cookie
    // keeps it sticky for every other page/app and the next server render.
    persistTheme(next);
  };

  const segment = (value: Theme, label: string, icon: React.ReactNode) => {
    const active = current === value;
    return (
      <button
        type="button"
        aria-pressed={active}
        title={active ? `Now in ${label.toLowerCase()} mode` : `Switch to ${label.toLowerCase()} mode`}
        onClick={() => apply(value)}
        className={`flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-semibold transition-colors sm:text-sm ${
          active ? "gradient-accent text-white shadow" : "text-muted-foreground hover:text-foreground"
        }`}
      >
        {icon}
        {label}
      </button>
    );
  };

  return (
    <div
      role="group"
      aria-label="Page theme"
      className="flex items-center gap-0.5 rounded-full border border-border bg-card/70 p-1"
    >
      {segment("dark", "Dark", <MoonIcon />)}
      {segment("light", "Light", <SunIcon />)}
    </div>
  );
}

// Lucide's moon / sun glyphs, inlined (same icon style as SettingsMenu in this app).
const iconProps = {
  width: 15,
  height: 15,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

function MoonIcon() {
  return (
    <svg {...iconProps}>
      <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg {...iconProps}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="m4.93 4.93 1.41 1.41" />
      <path d="m17.66 17.66 1.41 1.41" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="m6.34 17.66-1.41 1.41" />
      <path d="m19.07 4.93-1.41 1.41" />
    </svg>
  );
}
