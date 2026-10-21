// Shared inline SVG glyphs — same Lucide-style line set the rest of this app already
// inlines (see ThemeToggle / SettingsMenu): stroke currentColor, strokeWidth 2,
// no external icon dependency. Filled glyphs (play/pause/stop) use fill=currentColor.

type IconProps = {
  width?: number;
  height?: number;
  className?: string;
};

const strokeProps = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export function PlayIcon({ width = 14, height = 14, className }: IconProps) {
  return (
    <svg width={width} height={height} viewBox="0 0 24 24" aria-hidden className={className}>
      <path d="M8 5v14l11-7z" fill="currentColor" />
    </svg>
  );
}

export function PauseIcon({ width = 14, height = 14, className }: IconProps) {
  return (
    <svg width={width} height={height} viewBox="0 0 24 24" aria-hidden className={className}>
      <rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" />
      <rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" />
    </svg>
  );
}

export function MicIcon({ width = 24, height = 24, className }: IconProps) {
  return (
    <svg {...strokeProps} width={width} height={height} className={className}>
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <path d="M12 19v3" />
    </svg>
  );
}

export function StopIcon({ width = 24, height = 24, className }: IconProps) {
  return (
    <svg width={width} height={height} viewBox="0 0 24 24" aria-hidden className={className}>
      <rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" />
    </svg>
  );
}

export function UploadIcon({ width = 20, height = 20, className }: IconProps) {
  return (
    <svg {...strokeProps} width={width} height={height} className={className}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="m17 8-5-5-5 5" />
      <path d="M12 3v12" />
    </svg>
  );
}

export function WaveIcon({ width = 20, height = 20, className }: IconProps) {
  return (
    <svg {...strokeProps} width={width} height={height} className={className}>
      <path d="M2 13a2 2 0 0 0 2-2V7a2 2 0 0 1 4 0v13a2 2 0 0 0 4 0V4a2 2 0 0 1 4 0v13a2 2 0 0 0 4 0v-4a2 2 0 0 1 2-2" />
    </svg>
  );
}

export function LayersIcon({ width = 16, height = 16, className }: IconProps) {
  return (
    <svg {...strokeProps} width={width} height={height} className={className}>
      <path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z" />
      <path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65" />
      <path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65" />
    </svg>
  );
}

export function FileAudioIcon({ width = 16, height = 16, className }: IconProps) {
  return (
    <svg {...strokeProps} width={width} height={height} className={className}>
      <path d="M17.5 22h.5a2 2 0 0 0 2-2V7l-5-5H6a2 2 0 0 0-2 2v3" />
      <path d="M14 2v4a2 2 0 0 0 2 2h4" />
      <path d="M2 19a2 2 0 1 1 4 0v1a2 2 0 1 1-4 0v-4a6 6 0 0 1 12 0v4a2 2 0 1 1-4 0v-1a2 2 0 1 1 4 0" />
    </svg>
  );
}

export function CheckIcon({ width = 16, height = 16, className }: IconProps) {
  return (
    <svg {...strokeProps} width={width} height={height} className={className}>
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

/** Small live "now playing / busy" bar group; delays set per caller. */
export function EqBars({ count = 4, height = 16, className }: IconProps & { count?: number }) {
  return (
    <span aria-hidden className={`flex items-end gap-[3px] text-indigo ${className ?? ""}`}>
      {Array.from({ length: count }).map((_, i) => (
        <span
          key={i}
          className="eq-bar"
          style={{ height: `${height}px`, animationDelay: `${i * 0.12}s` }}
        />
      ))}
    </span>
  );
}
