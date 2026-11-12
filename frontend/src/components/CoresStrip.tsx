import type { Metrics } from "../types";
import { pct } from "../format";

/**
 * The hero. One lane per device that trained the model; a single sweep crosses
 * all lanes once on load to show the gradient all-reduce (jax.lax.pmean) that
 * keeps every core's weights identical after each step.
 */
export default function CoresStrip({ m }: { m: Metrics }) {
  const n = m.hardware.num_devices;
  const label = m.hardware.backend === "tpu" ? "TPU core" : m.hardware.backend === "cpu" ? "CPU device" : "Device";
  return (
    <section className="cores" aria-label="Training result">
      <div className="cores-top">
        <div>
          <div className="cores-acc">{pct(m.summary.final_test_accuracy)}</div>
          <p className="cores-acc-label">
            test accuracy on {m.dataset.name.toUpperCase()} after {m.summary.epochs} epochs
            {m.summary.best_test_accuracy > m.summary.final_test_accuracy &&
              ` (best epoch ${pct(m.summary.best_test_accuracy)})`}
          </p>
        </div>
        <div className="cores-meta">
          <b>{n} {label.toLowerCase()}{n === 1 ? "" : "s"}</b>, global batch {m.hardware.global_batch_size}
          {m.hardware.device_kind.toLowerCase() !== m.hardware.backend && (
            <>
              <br />
              {m.hardware.device_kind}
            </>
          )}
        </div>
      </div>

      <div className="lanes" role="img" aria-label={`${n} devices training in parallel`}>
        {Array.from({ length: Math.min(n, 16) }, (_, i) => (
          <div className="lane" key={i}>
            <span>{label.split(" ")[0]} {i}</span>
            <div className="lane-bar" />
          </div>
        ))}
        <div className="sync" />
      </div>

      <p className="cores-foot">
        Each lane trains on its own slice of every batch. After each step <code>pmean</code> averages the gradients
        across all {n} so the replicated weights never drift apart.
        {m.dataset.source.startsWith("synthetic") && " This run used generated data, not a real dataset."}
      </p>
    </section>
  );
}
