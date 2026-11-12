# JAXVision: Distributed Image Classification on TPU

A CNN image classifier built with **JAX, Flax and Optax**, trained on CIFAR-10 or MNIST with every training step spread across all available accelerator cores (8 on a TPU v2/v3-8 or v5e-8) using `pmap`. It ships with a benchmark suite, a small API server and a React dashboard for exploring the results.

```
 jaxvision/  (Python)                server/  (Python)             frontend/  (React)
 ┌────────────────────────┐         ┌───────────────────┐         ┌───────────────────────┐
 │ train.py   pmap + pmean│  writes │ app.py  FastAPI   │  JSON   │ Overview  Training    │
 │ benchmark.py jit/pmap  │ ──────► │ /api/metrics      │ ──────► │ Benchmark Playground  │
 │ model.py   Flax CNN    │ artifacts/ /api/benchmark   │         │ (Vite + React + TS)   │
 │ data.py    CIFAR/MNIST │         │ /api/predict      │         └───────────────────────┘
 └────────────────────────┘         └───────────────────┘
```

## What it demonstrates

| Idea | Where |
| --- | --- |
| CNN in Flax with synchronised BatchNorm | `jaxvision/model.py` |
| `jax.jit` compiled execution (training and inference) | `jaxvision/benchmark.py`, `server/app.py` |
| `jax.pmap` data parallelism across cores | `jaxvision/train.py` |
| Gradient aggregation with `jax.lax.pmean` (all-reduce) | `jaxvision/train.py`, `train_step` |
| Optax optimiser: AdamW, warmup and cosine schedule, gradient clipping | `jaxvision/train.py`, `create_state` |
| Training metrics and model benchmarking | `artifacts/metrics.json`, `artifacts/benchmark.json` |

## Quick start

You need Python 3.10+ and Node 20+.

```bash
# 1. Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Train (downloads CIFAR-10 the first time, about 160 MB)
python -m jaxvision.train --dataset cifar10 --epochs 30

# 3. Start the API
uvicorn server.app:app --port 8000

# 4. Start the dashboard (second terminal)
cd frontend && npm install && npm run dev      # http://localhost:5173
```

Training writes everything the dashboard reads into `artifacts/`, so the Python side and the web side never talk to each other directly except through those files.

### No TPU? Test the multi-device code anyway

`pmap` runs on any set of JAX devices. This command pretends your CPU is 8 devices, so the full distributed code path (sharding, `pmean`, synchronised BatchNorm) runs on a laptop:

```bash
python -m jaxvision.train --dataset synthetic --epochs 3 --simulate-devices 8
```

`synthetic` is a generated dataset that needs no download. It is deliberately easy, so expect near 100% accuracy; it exists to test the plumbing, not to measure the model.

### Running on a TPU VM

```bash
# on a Cloud TPU VM (v2-8, v3-8, v5e-8 ...)
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt
python -m jaxvision.train --dataset cifar10 --epochs 30 --per-device-batch-size 128
```

Do not pass `--simulate-devices` on a TPU. JAX finds the 8 cores itself, and the startup log prints `backend=tpu devices=8`. The global batch is `per-device-batch-size x devices` (1024 with the defaults).

## How it is built, step by step

1. **Data (`data.py`).** Images stay as `uint8` arrays on the host. Each global batch is reshaped from `[1024, H, W, C]` to `[8, 128, H, W, C]`, one slice per device. Conversion to float and normalisation happen *inside* the compiled step, so only one byte per pixel crosses to the accelerator.
2. **Model (`model.py`).** Three blocks of `Conv-BN-ReLU` twice, max-pool and dropout (32, 64 and 128 channels), then global average pooling and two dense layers. About 323K parameters.
3. **Replicate.** `flax.jax_utils.replicate` copies the whole train state (weights, optimiser state, BatchNorm statistics) onto every device.
4. **Parallel step (`train.py`).** `jax.pmap` runs the same `train_step` on all devices at once. Each device computes loss and gradients on its own slice of the batch.
5. **Gradient aggregation.** `jax.lax.pmean(grads, "batch")` is an all-reduce: every device receives the *average* gradient. All devices then apply the identical update, so their weights stay bit-for-bit the same without ever being copied around. BatchNorm gets `axis_name="batch"` so its statistics are averaged across devices too.
6. **Augmentation.** Random flip and random crop run on-device inside the step (CIFAR-10 only; MNIST digits are not flipped).
7. **Evaluate.** A second `pmap` function sums correct predictions across devices with `psum`. The last partial batch is zero-padded and masked, so every test image counts exactly once.
8. **Save.** The final weights, model description, per-epoch metrics and benchmark numbers go to `artifacts/`. The best epoch's test accuracy is recorded in the metrics, but the saved weights are always from the last epoch, so the headline accuracy matches the model you serve.
9. **Serve and visualise.** FastAPI reads those files and runs predictions with a `jit`-compiled copy of the model. React shows it all.

## What each file does

### Python training package: `jaxvision/`

| File | Purpose |
| --- | --- |
| `__init__.py` | Reads `--simulate-devices` before JAX loads and sets the XLA flag that fakes N CPU devices. |
| `config.py` | Dataset descriptions (shape, classes, mean and std) and the `TrainConfig` dataclass holding every hyper-parameter. |
| `data.py` | Downloads and parses CIFAR-10 and MNIST, generates the synthetic dataset, normalises images, shards batches for `pmap`, and builds masked evaluation batches. |
| `model.py` | The Flax CNN, with optional synchronised BatchNorm across devices. |
| `train.py` | The main entry point. Creates the optimiser, replicates state, defines the `pmap`-ed train and eval steps (with `pmean`/`psum`), runs the epoch loop, and writes `metrics.json` and the checkpoint. |
| `benchmark.py` | Measures `jit` against eager execution, latency by batch size, and training throughput on 1, 2, 4 ... N devices. Writes `benchmark.json`. Also runnable on its own. |
| `checkpoint.py` | Saves and loads weights (`checkpoint.msgpack`) with the metadata (`model_meta.json`) needed to rebuild the model exactly. |

### API server: `server/`

| File | Purpose |
| --- | --- |
| `app.py` | FastAPI app. Serves `/api/metrics`, `/api/benchmark`, `/api/model`, `/api/samples` and `/api/predict`, and in production also serves the built React app. |

### React dashboard: `frontend/`

Vite, React 19 and TypeScript, with Recharts for charts. The styling is one plain CSS file built on design tokens (colours, radii and fonts as CSS variables), with automatic dark mode and a mobile layout.

| File | Purpose |
| --- | --- |
| `index.html` | Page shell, font loading and favicon. |
| `vite.config.ts` | Dev server on :5173 that proxies `/api` to the Python server on :8000. |
| `src/main.tsx`, `src/App.tsx` | App entry and the route table. |
| `src/api.ts` | Typed `fetch` wrapper plus the `useResource` hook (loading, error and reload state). |
| `src/types.ts` | TypeScript shapes of `metrics.json`, `benchmark.json` and API responses. |
| `src/format.ts` | Number and duration formatting helpers. |
| `src/styles.css` | Design tokens, layout, components and responsive rules. |
| `src/components/Layout.tsx` | Sidebar navigation and page shell. |
| `src/components/CoresStrip.tsx` | The overview hero: final accuracy plus one lane per training device. |
| `src/components/Charts.tsx` | Train-versus-test line chart shared by several pages. |
| `src/components/States.tsx` | Page header, loading skeleton, and the "no results yet" and "server offline" screens that tell you the command to run. |
| `src/pages/Overview.tsx` | Headline result, key numbers, accuracy and loss curves. |
| `src/pages/Training.tsx` | Full epoch table, learning curves and the saved run settings. |
| `src/pages/Benchmark.tsx` | `jit` speed-up, device-scaling chart and table, batch-size sweep. |
| `src/pages/Playground.tsx` | Classify held-out test images or your own upload. |

### Project root

| File | Purpose |
| --- | --- |
| `requirements.txt` | Python dependencies. |
| `tests/` | Pytest suite. `conftest.py` fakes 8 devices; `test_core.py` checks shapes, sharding, masking, and that replicas stay identical after a `pmap` step. |
| `.gitignore` | Keeps datasets, `artifacts/`, `node_modules/` and build output out of git. |

## Outputs in `artifacts/`

| File | Contents |
| --- | --- |
| `metrics.json` | Config, hardware, dataset info, per-epoch loss, accuracy, learning rate and throughput, and a summary. |
| `benchmark.json` | `jit` against eager timings, batch sweep, and multi-device scaling. |
| `samples.npz` | 32 held-out test images, so the Playground works without the dataset (used when deployed). |
| `checkpoint.msgpack` | Trained weights and BatchNorm statistics. |
| `model_meta.json` | Architecture settings and class names. |

## Useful options

```bash
python -m jaxvision.train --help
python -m jaxvision.train --dataset mnist --epochs 10
python -m jaxvision.train --dataset cifar10 --epochs 50 --widths 64 128 256   # bigger model
python -m jaxvision.train --train-subset 5000 --epochs 2                     # quick run
python -m jaxvision.benchmark                                                # re-run benchmarks only
```

## Production build

```bash
cd frontend && npm run build        # creates frontend/dist
uvicorn server.app:app --port 8000  # now serves the dashboard AND the API on :8000
```

## Deploying

See [DEPLOY.md](DEPLOY.md) for a step-by-step guide to putting the dashboard and API online on Render with no Docker. A `Dockerfile` is also included for hosts that prefer containers (Hugging Face Spaces, Fly.io, Railway). The files involved:

| File | Purpose |
| --- | --- |
| `render.yaml` | Render Blueprint: build and start commands, Python version, CPU-only JAX. |
| `requirements-deploy.txt` | The smaller dependency list the server needs (no tests). |
| `Dockerfile`, `.dockerignore` | Optional container build for other hosts. |
| `frontend/dist/` | The built dashboard, committed so the server needs no Node. |

## Tests

```bash
python -m pytest tests -q
```

## Your results

Fill this in after your TPU run, using `artifacts/metrics.json` (the dashboard shows the same numbers):

| Dataset | Devices | Epochs | Test accuracy | Training time | Images/s |
| --- | --- | --- | --- | --- | --- |
| CIFAR-10 | 8 | _N_ | _XX_% | _..._ | _..._ |

## Notes and limits

- `pmap` is the classic JAX approach to data parallelism and is what this project uses. Newer JAX code often uses `jax.jit` with sharding (`jax.sharding`) for the same job, and a future version could port to it.
- Training is data-parallel only: the whole model must fit on one core, which is easily true for this CNN.
- If a dataset download fails, add `--synthetic-fallback` to continue on generated data. The run is then marked as synthetic in the dashboard.
