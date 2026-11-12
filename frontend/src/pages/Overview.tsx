import { api, useResource } from "../api";
import { Gate, PageHead } from "../components/States";
import CoresStrip from "../components/CoresStrip";
import { EpochChart, Legend } from "../components/Charts";
import { compact, duration, int } from "../format";

export default function Overview() {
  const res = useResource(api.metrics);
  return (
    <>
      <PageHead title="Overview">
        A CNN trained with JAX, Flax and Optax, with every training step spread across all available cores using pmap.
      </PageHead>
      <Gate res={res} command="python -m jaxvision.train --dataset cifar10 --epochs 30">
        {(m) => (
          <>
            <CoresStrip m={m} />
            <dl className="stats">
              <div className="stat"><dt>Parameters</dt><dd>{compact(m.model.parameters)}</dd></div>
              <div className="stat"><dt>Training time</dt><dd>{duration(m.summary.total_train_time_s)}</dd></div>
              <div className="stat"><dt>Throughput</dt><dd>{int(m.summary.mean_images_per_sec)}<small>img/s</small></dd></div>
              <div className="stat"><dt>Training images</dt><dd>{int(m.dataset.train_size)}</dd></div>
            </dl>
            <div className="grid-2">
              <section className="panel">
                <h2>Accuracy</h2>
                <p className="sub">Share of images classified correctly, per epoch</p>
                <Legend />
                <EpochChart rows={m.history} train="train_accuracy" test="test_accuracy" percent />
              </section>
              <section className="panel">
                <h2>Loss</h2>
                <p className="sub">Cross-entropy, lower is better</p>
                <Legend />
                <EpochChart rows={m.history} train="train_loss" test="test_loss" />
              </section>
            </div>
          </>
        )}
      </Gate>
    </>
  );
}
