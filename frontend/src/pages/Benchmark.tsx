import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, useResource } from "../api";
import { Gate, PageHead } from "../components/States";
import { int } from "../format";

const axis = { stroke: "var(--ink-3)", fontSize: 12 } as const;

export default function Benchmark() {
  const res = useResource(api.benchmark);
  return (
    <>
      <PageHead title="Benchmark">
        What compiling with jit and training across devices with pmap actually buys you on this hardware.
      </PageHead>
      <Gate res={res} command="python -m jaxvision.benchmark">
        {(b) => {
          const j = b.inference_jit_vs_eager;
          const last = b.training_scaling[b.training_scaling.length - 1];
          return (
            <>
              <dl className="stats" style={{ marginTop: 0 }}>
                <div className="stat"><dt>jit speed-up over eager</dt><dd>{j.speedup}x</dd></div>
                <div className="stat"><dt>Inference, jit</dt><dd>{j.jit_ms.toFixed(1)}<small>ms / {j.batch_size} images</small></dd></div>
                <div className="stat"><dt>Inference, eager</dt><dd>{j.eager_ms.toFixed(1)}<small>ms</small></dd></div>
                <div className="stat"><dt>One-time compile</dt><dd>{j.compile_time_s.toFixed(1)}<small>s</small></dd></div>
              </dl>

              {b.backend === "cpu" && b.num_devices > 1 && (
                <p className="panel" style={{ marginBottom: 20 }}>
                  These numbers come from {b.num_devices} <b>simulated</b> CPU devices that share the same physical cores,
                  so multi-device speed-ups here are small. On a real TPU slice each device has its own compute and the
                  scaling is close to linear.
                </p>
              )}

              <div className="grid-2">
                <section className="panel">
                  <h2>Training throughput by device count</h2>
                  <p className="sub">Images per second for 1, 2, 4... devices. Batch per device stays fixed, so total batch grows with devices.</p>
                  <div style={{ height: 240 }}>
                    <ResponsiveContainer>
                      <BarChart data={b.training_scaling} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                        <CartesianGrid stroke="var(--line)" vertical={false} />
                        <XAxis dataKey="devices" {...axis} tickLine={false} axisLine={{ stroke: "var(--line)" }} />
                        <YAxis {...axis} tickLine={false} axisLine={false} width={52} />
                        <Tooltip
                          formatter={(v) => [int(Number(v)), "Images/s"]}
                          labelFormatter={(l) => `${l} device${l === 1 ? "" : "s"}`}
                          contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 8, color: "var(--ink)" }}
                          cursor={{ fill: "var(--surface-2)" }}
                        />
                        <Bar dataKey="images_per_sec" fill="var(--cobalt)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>

                <section className="panel">
                  <h2>Scaling details</h2>
                  <p className="sub">Efficiency of 100% means doubling devices exactly doubles throughput.</p>
                  <div className="table-wrap">
                    <table>
                      <thead><tr><th>Devices</th><th>Batch</th><th>Step</th><th>Speed-up</th><th>Efficiency</th></tr></thead>
                      <tbody>
                        {b.training_scaling.map((r) => (
                          <tr key={r.devices}>
                            <td>{r.devices}</td><td>{r.global_batch}</td><td>{r.step_ms} ms</td>
                            <td>{r.speedup_vs_1_device}x</td><td>{(r.efficiency * 100).toFixed(0)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="sub" style={{ marginTop: 12 }}>
                    {last.devices} devices run {last.speedup_vs_1_device}x faster than 1.
                  </p>
                </section>
              </div>

              <section className="panel">
                <h2>Inference by batch size</h2>
                <p className="sub">Jit-compiled forward pass on a single device. Larger batches trade latency for throughput.</p>
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Batch size</th><th>Latency</th><th>Images/s</th></tr></thead>
                    <tbody>
                      {b.inference_batch_sweep.map((r) => (
                        <tr key={r.batch_size}>
                          <td>{r.batch_size}</td><td>{r.latency_ms} ms</td><td>{int(r.images_per_sec)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          );
        }}
      </Gate>
    </>
  );
}
