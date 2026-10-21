import type { Metadata } from "next";
import { Manrope, Geist_Mono } from "next/font/google";
import { cookies } from "next/headers";
import NavBar from "./components/NavBar";
import "./globals.css";

// Manrope replaced Geist site-wide (see Documentation/FONT-CHANGES.md): more stroke volume,
// much easier to read on the light theme this app defaults to. Geist_Mono
// (code/technical text) is unchanged.
const manropeSans = Manrope({
  variable: "--font-manrope",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const TITLE = "Hum to Sound Effect";
const DESCRIPTION =
  "Record a vocal imitation, get the closest sound effects, ranked. A vocal-imitation retrieval demo (an ensemble of two dual-tower encoders).";

// Icons and the social preview image come from the file conventions in this
// folder (favicon.ico, icon.png, apple-icon.png, opengraph-image.png) --
// proxy.ts lets those through without the passcode so link previews work.
export const metadata: Metadata = {
  metadataBase: new URL("http://localhost:3000"),
  title: TITLE,
  description: DESCRIPTION,
  openGraph: {
    type: "website",
    siteName: "Q-SFX",
    title: TITLE,
    description: DESCRIPTION,
  },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION },
};

export default async function RootLayout({ children }: LayoutProps<"/">) {
  // Theme comes from the shared qbv_theme cookie (set by ThemeToggle), read
  // server-side so the correct class ships with the HTML — no flash, no client
  // script. Default is LIGHT: this app's audience prefers it. The cookie is
  // site-wide (Path=/), shared with llm-portfolio, chat-bot, and the path
  // splitter's terms modal.
  const cookieStore = await cookies();
  const isLight = cookieStore.get("qbv_theme")?.value !== "dark";
  const initialTheme: "dark" | "light" = isLight ? "light" : "dark";

  return (
    <html
      lang="en"
      className={`${manropeSans.variable} ${geistMono.variable} h-full antialiased${isLight ? " light" : ""}`}
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        <NavBar initialTheme={initialTheme} />
        <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-4 py-10">{children}</main>
        <footer className="px-4 pb-8 text-center text-xs text-muted-foreground">
          <span className="rounded-full border border-border px-2 py-0.5">honest numbers</span> — a QBV
          (query-by-vocal-imitation) retrieval demo. All models run locally.
        </footer>
      </body>
    </html>
  );
}
