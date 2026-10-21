import type { NextConfig } from "next";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || ""; // empty: served at http://localhost:3000/

const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  // NOTE: this app's whole purpose is recording a mic clip in the
  // browser — microphone=() here would silently break every recording. Allow self, deny everything else.
  { key: "Permissions-Policy", value: "camera=(), microphone=(self), geolocation=(), payment=()" },
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "X-DNS-Prefetch-Control", value: "off" },
];

const nextConfig: NextConfig = {
  basePath,
  poweredByHeader: false,

  async headers() {
    return [
      { source: "/", headers: securityHeaders },
      { source: "/((?!api|_next).*)", headers: securityHeaders },
      {
        // Search/audio responses are per-visitor and must never be cached by a proxy/CDN in front.
        source: "/api/:path*",
        headers: [
          { key: "Cache-Control", value: "no-store, max-age=0" },
          { key: "X-Content-Type-Options", value: "nosniff" },
        ],
      },
    ];
  },
};

export default nextConfig;
