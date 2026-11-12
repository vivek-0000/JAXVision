import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EpochRow } from "../types";

// Series colours match the --cobalt / --amber tokens (CSS vars work in SVG attributes).
export const TEST = "var(--cobalt)";
export const TRAIN = "var(--amber)";

const axis = { stroke: "var(--ink-3)", fontSize: 12 } as const;

export function Legend() {
  return (
    <div className="legend">
      <span><i style={{ background: TRAIN }} />Train</span>
      <span><i style={{ background: TEST }} />Test</span>
    </div>
  );
}

/** Train vs. test curve for one metric across epochs. */
export function EpochChart({
  rows,
  train,
  test,
  percent,
}: {
  rows: EpochRow[];
  train: keyof EpochRow;
  test: keyof EpochRow;
  percent?: boolean;
}) {
  const fmt = (v: number) => (percent ? `${(v * 100).toFixed(1)}%` : v.toFixed(3));
  const tick = (v: number) => (percent ? `${Math.round(v * 100)}%` : v.toFixed(2));
  return (
    <div style={{ height: 240 }}>
      <ResponsiveContainer>
        <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="epoch" {...axis} tickLine={false} axisLine={{ stroke: "var(--line)" }} />
          <YAxis {...axis} tickLine={false} axisLine={false} width={48} tickFormatter={tick} domain={percent ? [0, 1] : ["auto", "auto"]} />
          <Tooltip
            formatter={(v, name) => [fmt(Number(v)), name === train ? "Train" : "Test"]}
            labelFormatter={(l) => `Epoch ${l}`}
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 8, color: "var(--ink)" }}
          />
          <Line type="monotone" dataKey={train as string} stroke={TRAIN} strokeWidth={2} dot={rows.length < 12} isAnimationActive={false} />
          <Line type="monotone" dataKey={test as string} stroke={TEST} strokeWidth={2} dot={rows.length < 12} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
