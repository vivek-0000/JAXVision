import { useCallback, useRef, useState } from "react";
import { api, useResource } from "../api";
import { PageHead, Unavailable } from "../components/States";
import { pct } from "../format";
import type { Prediction } from "../types";

interface Shown {
  src: string;
  truth?: string;
  predictions: Prediction[];
}

export default function Playground() {
  const model = useResource(api.model);
  const samples = useResource(() => api.samples(10));
  const [shown, setShown] = useState<Shown | null>(null);
  const [picked, setPicked] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const classify = useCallback(async (blob: Blob, src: string, truth?: string) => {
    setBusy(true);
    setErr(null);
    try {
      const { predictions } = await api.predict(blob);
      setShown({ src, truth, predictions });
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Prediction failed.");
    } finally {
      setBusy(false);
    }
  }, []);

  const fromFile = (file?: File) => {
    if (!file) return;
    setPicked(null);
    classify(file, URL.createObjectURL(file));
  };

  const fromSample = async (i: number) => {
    const s = samples.data?.[i];
    if (!s) return;
    setPicked(i);
    classify(await (await fetch(s.image)).blob(), s.image, s.label);
  };

  if (model.error && !model.data) {
    return (
      <>
        <PageHead title="Playground" />
        <Unavailable res={model} command="python -m jaxvision.train --dataset cifar10 --epochs 30" />
      </>
    );
  }

  const classes = model.data?.classes ?? [];
  return (
    <>
      <PageHead title="Playground">
        Pick a test image or drop in your own. The trained model runs on the server and returns its top guesses.
      </PageHead>
      <div className="stack">
        {samples.data && samples.data.length > 0 && (
          <section className="panel">
            <h2>Test images</h2>
            <p className="sub">Held-out images the model never trained on.</p>
            <div className="samples">
              {samples.data.map((s, i) => (
                <button key={i} className="sample" aria-pressed={picked === i} onClick={() => fromSample(i)} aria-label={`Test image ${i + 1}`}>
                  <img src={s.image} alt="" />
                </button>
              ))}
            </div>
          </section>
        )}

        <section className="panel">
          <h2>Your own image</h2>
          <p className="sub">
            {classes.length > 0 ? `The model knows: ${classes.join(", ")}. ` : ""}
            Images are shrunk to {model.data ? `${model.data.image_shape[1]}x${model.data.image_shape[0]}` : "a small size"} pixels first.
          </p>
          <div
            className={`drop${over ? " over" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); fromFile(e.dataTransfer.files[0]); }}
          >
            <p style={{ marginBottom: 12 }}>Drop an image here</p>
            <input ref={input} type="file" accept="image/*" onChange={(e) => fromFile(e.target.files?.[0])} />
            <button className="btn primary" onClick={() => input.current?.click()}>Choose a file</button>
          </div>
        </section>

        {(busy || err || shown) && (
          <section className="panel" aria-live="polite">
            <h2>Prediction</h2>
            {busy && <p className="sub">Classifying...</p>}
            {err && <p className="sub" style={{ color: "var(--bad)" }}>{err}</p>}
            {shown && !busy && (
              <div className="preview" style={{ marginTop: 14 }}>
                <img src={shown.src} alt="The image being classified" />
                <div className="bars">
                  {shown.truth && (
                    <p className="sub" style={{ margin: 0 }}>
                      Actual label: <b>{shown.truth}</b>{" "}
                      <span className={`badge ${shown.predictions[0].label === shown.truth ? "good" : "amber"}`}>
                        {shown.predictions[0].label === shown.truth ? "Correct" : "Missed"}
                      </span>
                    </p>
                  )}
                  {shown.predictions.map((p, i) => (
                    <div key={p.label} className={`bar-row${i === 0 ? " top" : ""}`}>
                      <span>{p.label}</span>
                      <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(p.probability * 100, 1)}%` }} /></div>
                      <span>{pct(p.probability, 1)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </div>
    </>
  );
}
