// Logic tests for lib/endpointing.ts (run: node scripts/test-endpointing.mjs). Synthetic level traces at 60 frames/s.
import { Endpointer, DEFAULT_ENDPOINT } from "../lib/endpointing.ts";
import assert from "node:assert/strict";

const FPS = 60, DT = 1000 / FPS;
/** segs: [[durationMs, level], ...]; returns the stop time in ms, or null if never stopped within the trace. */
function run(segs, cfg = {}) {
  const ep = new Endpointer(cfg);
  let t = 0;
  for (const [dur, lvl] of segs) {
    const end = t + dur;
    for (; t < end; t += DT) if (ep.push(t, lvl) === "stop") return { stop: t, heard: ep.heardSound };
  }
  return { stop: null, heard: ep.heardSound };
}
const near = (a, b, tol = 150) => assert.ok(a !== null && Math.abs(a - b) <= tol, `expected ~${b} ms, got ${a}`);
const Q = 0.003, S = 0.1;

// 1. user never makes a sound: never stops by itself (the caller's hard maximum applies)
let r = run([[10000, Q]]); assert.equal(r.stop, null); assert.equal(r.heard, false);
// 2. quiet lead-in, 2 s of sound, then silence: stops 1.5 s after the sound ends
r = run([[500, Q], [2000, S], [5000, Q]]); near(r.stop, 500 + 2000 + 1500);
// 3. a 1.0 s pause inside the imitation does not stop it; the final silence does
r = run([[500, Q], [1000, S], [1000, Q], [1000, S], [5000, Q]]); near(r.stop, 500 + 1000 + 1000 + 1000 + 1500);
// 4. documented limitation: a 1.6 s pause inside the imitation stops it during the pause
r = run([[500, Q], [1000, S], [1600, Q], [1000, S], [3000, Q]]); near(r.stop, 500 + 1000 + 1500);
// 5. user starts at t = 0 (no quiet lead-in): still detected, stops after the sound ends
r = run([[2000, S], [5000, Q]]); near(r.stop, 2000 + 1500);
// 6. a 50 ms click is not a sound (onset needs 100 ms)
r = run([[500, Q], [50, 0.5], [6000, Q]]); assert.equal(r.stop, null); assert.equal(r.heard, false);
// 7. noisy room (floor 0.02): the sound is still found and the room noise afterwards counts as silence
r = run([[500, 0.02], [2000, 0.2], [5000, 0.02]]); near(r.stop, 500 + 2000 + 1500);
// 8. a very short sound still waits for the full silence window
r = run([[500, Q], [300, S], [5000, Q]]); near(r.stop, 500 + 300 + 1500);
// 9. never stops before minRecordMs (silence window and minimum both configurable)
r = run([[500, Q], [300, S], [5000, Q]], { silenceHoldMs: 100, minRecordMs: 2000 }); assert.ok(r.stop >= 2000, `stop ${r.stop}`);
console.log("endpointing: all 9 tests passed; defaults", JSON.stringify(DEFAULT_ENDPOINT));
