// Gates every page and API route behind the passcode cookie set by /api/qbv-login, but only when
// QBV_PASSCODE is set (see lib/qbvGate.ts). With no passcode the app is open.
import { NextRequest, NextResponse } from "next/server";
import { GATE_COOKIE_NAME, gateEnabled, verifyGateValue } from "@/lib/qbvGate";

const LOGIN_PATH = "/";
const SEARCH_PATH = "/search";

export default function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const basePath = request.nextUrl.basePath || process.env.NEXT_PUBLIC_BASE_PATH || "";
  if (!gateEnabled()) {
    // Gate off: the login page has nothing to unlock, send visitors straight to the search page.
    if (pathname === LOGIN_PATH || pathname === "") {
      return NextResponse.redirect(new URL(`${basePath}${SEARCH_PATH}`, request.url));
    }
    return NextResponse.next();
  }
  const authed = verifyGateValue(request.cookies.get(GATE_COOKIE_NAME)?.value);

  // A bare request to the basePath root (e.g. /qbv, no trailing slash) can come through here
  // with pathname === "" rather than "/" once Next strips the basePath -- match both.
  if (pathname === LOGIN_PATH || pathname === "") {
    // Already unlocked -- the login page itself is the one place a cookie means "skip me".
    if (authed) return NextResponse.redirect(new URL(`${basePath}${SEARCH_PATH}`, request.url));
    return NextResponse.next();
  }

  if (!authed) {
    // basePath + "/" would be e.g. "/qbv/" -- a needless extra trailing-slash redirect hop.
    const url = new URL(basePath || LOGIN_PATH, request.url);
    url.searchParams.set("next", pathname);
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

export const config = {
  // "/" as its own entry: the capturing-group pattern below never matches the bare basePath root
  // (e.g. /qbv with no trailing segment) -- confirmed by proxy.ts simply not running for that
  // request while every subpath (/qbv/search, /qbv/admin, ...) worked fine.
  // Icon and social-preview files are public so browsers and link unfurlers can fetch them.
  matcher: [
    "/",
    "/((?!api/qbv-login|_next/static|_next/image|favicon.ico|icon.png|apple-icon.png|opengraph-image.png).*)",
  ],
};
