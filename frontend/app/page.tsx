import { Suspense } from "react";
import LoginClient from "./LoginClient";

// LoginClient reads useSearchParams() (the "next" redirect target) -- App Router requires that
// behind a Suspense boundary or static prerendering of this page fails the build outright.
export default function Page() {
  return (
    <Suspense>
      <LoginClient />
    </Suspense>
  );
}
