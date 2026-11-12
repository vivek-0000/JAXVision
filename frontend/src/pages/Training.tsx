import { api, useResource } from "../api";
import { Gate, PageHead } from "../components/States";
import { EpochChart, Legend } from "../components/Charts";
import { pct } from "../format";

export default function Training() {
  const res = useResource(api.metrics);
  return (
    <>
      <PageHead title="Training">Every epoch of the run, plus the settings that produced it.</PageHead>
      <Gate res={res} command="python -m jaxvision.train --dataset cifar10 --epochs 30">
        {(m) => {
          const best = Math.max(...m.history.map((h) => h.test_accuracy));
          const cfg = m.config as Record<string, unknown>;
          return (
            <div className="stack">
              <section className="panel">
                <h2>Learning curves</h2>
                <p className="sub">A growing gap between train and test accuracy means the model is starting to memorise.</p>
                <Legend />
                <EpochChart rows={m.history} train="train_accuracy" test="test_accuracy" percent />
              </section>

              <section className="panel">
                <h2>Epochs</h2>
                <p className="sub">Best test accuracy is in bold.</p>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Epoch</th><th>Train loss</th><th>Train acc</th><th>Test loss</th><th>Test acc</th>
                        <th>Learning rate</th><th>Time</th><th>Images/s</th>
                      </tr>
                    </thead>
                    <tbody>
                      {m.history.map((h) => (
                        <tr key={h.epoch} className={h.test_accuracy === best ? "best" : undefined}>
                          <td>{h.epoch}</td>
                          <td>{h.train_loss.toFixed(4)}</td>
                          <td>{pct(h.train_accuracy)}</td>
                          <td>{h.test_loss.toFixed(4)}</td>
                          <td>{pct(h.test_accuracy)}</td>
                          <td>{h.learning_rate.toExponential(2)}</td>
                          <td>{h.epoch_time_s.toFixed(1)} s</td>
                          <td>{Math.round(h.images_per_sec).toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="panel">
                <h2>Run settings</h2>
                <p className="sub">Saved with the results so the run can be repeated.</p>
                <div className="table-wrap">
                  <table>
                    <tbody>
                      {Object.entries(cfg).map(([k, v]) => (
                        <tr key={k}>
                          <td>{k}</td>
                          <td>{Array.isArray(v) ? v.join(", ") : String(v ?? "none")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </div>
          );
        }}
      </Gate>
    </>
  );
}
