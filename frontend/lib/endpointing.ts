/**
 * Silence endpointing for the search recorder: stop a recording once the user has made a sound and then gone quiet.
 *
 * Pure logic (no DOM): feed it one (time in ms, RMS level 0..1) sample per animation frame; it answers "continue" or "stop".
 *  1. calibrate: the first `calibrationMs` give the room's noise floor (lowest smoothed level seen, so a user who starts
 *     immediately does not inflate it);
 *  2. onset: the (raw) level stays above max(onsetMinRms, floor x onsetFactor) for `onsetHoldMs`, so a click is not a sound;
 *  3. stop: after the onset, the level has stayed below max(silenceMinRms, floor x silenceFactor) for `silenceHoldMs`
 *     (and at least `minRecordMs` have been recorded).
 * Nothing here is learned or tuned on user data; the constants are heuristic defaults. The caller keeps its own hard
 * maximum recording time and the manual stop button.
 */
export interface EndpointConfig {
  calibrationMs: number;
  smoothFrames: number;
  maxFloor: number;
  onsetFactor: number;
  onsetMinRms: number;
  onsetHoldMs: number;
  silenceFactor: number;
  silenceMinRms: number;
  silenceHoldMs: number;
  minRecordMs: number;
}

export const DEFAULT_ENDPOINT: EndpointConfig = {
  calibrationMs: 400,
  smoothFrames: 5,
  maxFloor: 0.015,
  onsetFactor: 4,
  onsetMinRms: 0.02,
  onsetHoldMs: 100,
  silenceFactor: 2.5,
  silenceMinRms: 0.012,
  silenceHoldMs: 1500,
  minRecordMs: 800,
};

export type EndpointDecision = "continue" | "stop";

export class Endpointer {
  private readonly cfg: EndpointConfig;
  private recent: number[] = [];
  private floor = Number.POSITIVE_INFINITY;
  private onsetSince: number | null = null;
  private lastActiveMs = 0;
  private started = false;

  constructor(cfg: Partial<EndpointConfig> = {}) {
    this.cfg = { ...DEFAULT_ENDPOINT, ...cfg };
  }

  /** true once the user's sound has been detected */
  get heardSound(): boolean {
    return this.started;
  }

  push(tMs: number, rms: number): EndpointDecision {
    const c = this.cfg;
    this.recent.push(rms);
    if (this.recent.length > c.smoothFrames) this.recent.shift();
    const smooth = this.recent.reduce((a, b) => a + b, 0) / this.recent.length;

    if (tMs < c.calibrationMs) {
      if (this.recent.length >= c.smoothFrames) this.floor = Math.min(this.floor, smooth);
      return "continue";
    }
    if (!Number.isFinite(this.floor)) this.floor = smooth;
    const floor = Math.min(this.floor, c.maxFloor);
    const onsetThr = Math.max(c.onsetMinRms, floor * c.onsetFactor);
    const silenceThr = Math.max(c.silenceMinRms, floor * c.silenceFactor);

    if (!this.started) {
      if (rms >= onsetThr) {                                  // raw level: smoothing would stretch a click past the hold time
        if (this.onsetSince === null) this.onsetSince = tMs;
        if (tMs - this.onsetSince >= c.onsetHoldMs) {
          this.started = true;
          this.lastActiveMs = tMs;
        }
      } else {
        this.onsetSince = null;
      }
      return "continue";
    }
    if (smooth >= silenceThr) this.lastActiveMs = tMs;
    if (tMs - this.lastActiveMs >= c.silenceHoldMs && tMs >= c.minRecordMs) return "stop";
    return "continue";
  }
}

/** RMS (0..1) of a byte time-domain buffer from AnalyserNode.getByteTimeDomainData (silence = 128). */
export function rmsFromBytes(buf: Uint8Array): number {
  let s = 0;
  for (let i = 0; i < buf.length; i++) {
    const v = (buf[i] - 128) / 128;
    s += v * v;
  }
  return Math.sqrt(s / buf.length);
}
