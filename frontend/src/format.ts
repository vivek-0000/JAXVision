export const pct = (v: number, digits = 2) => `${(v * 100).toFixed(digits)}%`;
export const int = (v: number) => Math.round(v).toLocaleString("en-US");
export const compact = (v: number) =>
  new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v);

export function duration(seconds: number): string {
  if (seconds < 90) return `${seconds.toFixed(0)} s`;
  const m = seconds / 60;
  if (m < 90) return `${m.toFixed(1)} min`;
  return `${(m / 60).toFixed(1)} h`;
}
