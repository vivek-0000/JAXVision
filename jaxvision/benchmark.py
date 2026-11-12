"""Model benchmarking: what does `jit` and `pmap` actually buy us?

Three measurements, written to artifacts/benchmark.json:

  1. inference_jit_vs_eager   - single-device forward pass, compiled (jax.jit)
                                vs. op-by-op eager execution
  2. inference_batch_sweep    - jitted latency / throughput at several batch sizes
  3. training_scaling         - pmap training-step throughput using 1, 2, 4 ... N
                                devices (fixed per-device batch => weak scaling)

Run standalone against a saved checkpoint:
    python -m jaxvision.benchmark
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import flax.jax_utils as jax_utils
import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import load_checkpoint
from .config import DatasetSpec, TrainConfig
from .data import normalize
from .model import build_model


def _timeit(fn: Callable[[], object], warmup: int, iters: int) -> float:
    """Median seconds per call, with proper device synchronisation."""
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _random_images(spec: DatasetSpec, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(n,) + spec.image_shape).astype(np.uint8)


def bench_inference(variables, spec: DatasetSpec, widths, dropout, batch_size: int = 128) -> dict:
    model = build_model(spec.num_classes, widths, dropout)  # no axis_name: single device
    images = jnp.asarray(_random_images(spec, batch_size))

    def forward(vs, x):
        return model.apply(vs, normalize(x, spec), train=False)

    jitted = jax.jit(forward)
    eager = lambda: forward(variables, images)
    compiled = lambda: jitted(variables, images)

    t0 = time.perf_counter()
    jax.block_until_ready(compiled())  # first call = trace + XLA compile
    compile_s = time.perf_counter() - t0

    t_eager = _timeit(eager, warmup=1, iters=5)
    t_jit = _timeit(compiled, warmup=3, iters=20)
    return {
        "batch_size": batch_size,
        "compile_time_s": round(compile_s, 3),
        "eager_ms": round(t_eager * 1e3, 3),
        "jit_ms": round(t_jit * 1e3, 3),
        "speedup": round(t_eager / t_jit, 2),
        "jit_images_per_sec": round(batch_size / t_jit, 1),
    }


def bench_batch_sweep(variables, spec: DatasetSpec, widths, dropout, sizes=(1, 8, 32, 128, 512)) -> list:
    model = build_model(spec.num_classes, widths, dropout)
    jitted = jax.jit(lambda vs, x: model.apply(vs, normalize(x, spec), train=False))
    rows = []
    for bs in sizes:
        images = jnp.asarray(_random_images(spec, bs))
        t = _timeit(lambda: jitted(variables, images), warmup=2, iters=15)
        rows.append({
            "batch_size": bs,
            "latency_ms": round(t * 1e3, 3),
            "images_per_sec": round(bs / t, 1),
        })
    return rows


def bench_training_scaling(spec: DatasetSpec, cfg: TrainConfig, per_device_batch: int = 64, steps: int = 6) -> list:
    """Time the real pmapped train step on 1, 2, 4, ... N devices."""
    # Imported lazily to avoid a circular import (train imports benchmark).
    from .train import create_state, make_lr_schedule, make_train_step

    all_devices = jax.local_devices()
    counts = [1]
    while counts[-1] * 2 <= len(all_devices):
        counts.append(counts[-1] * 2)
    if counts[-1] != len(all_devices):
        counts.append(len(all_devices))

    rows = []
    base_ips = None
    for n in counts:
        devices = all_devices[:n]
        schedule = make_lr_schedule(cfg, 100)
        state = jax_utils.replicate(create_state(cfg, spec, schedule), devices)
        p_train = make_train_step(spec)
        rngs = jax_utils.replicate(jax.random.PRNGKey(0), devices)
        gb = per_device_batch * n
        x = _random_images(spec, gb).reshape((n, per_device_batch) + spec.image_shape)
        y = np.random.RandomState(1).randint(0, spec.num_classes, size=(n, per_device_batch)).astype(np.int32)

        state, m = p_train(state, x, y, rngs)  # compile
        jax.block_until_ready(m)
        t0 = time.perf_counter()
        for _ in range(steps):
            state, m = p_train(state, x, y, rngs)
        jax.block_until_ready(m)
        step_s = (time.perf_counter() - t0) / steps
        ips = gb / step_s
        base_ips = base_ips or ips
        rows.append({
            "devices": n,
            "global_batch": gb,
            "step_ms": round(step_s * 1e3, 2),
            "images_per_sec": round(ips, 1),
            "speedup_vs_1_device": round(ips / base_ips, 2),
            "efficiency": round(ips / base_ips / n, 3),
        })
    return rows


def run_benchmark(artifacts_dir: str, cfg: Optional[TrainConfig] = None, spec: Optional[DatasetSpec] = None,
                  state_for_inference=None) -> dict:
    variables, meta, ckpt_spec = load_checkpoint(artifacts_dir)
    spec = spec or ckpt_spec
    cfg = cfg or TrainConfig(dataset=spec.name, widths=tuple(meta["widths"]), dropout=meta["dropout"])
    widths, dropout = meta["widths"], meta["dropout"]
    print("[benchmark] jit vs eager inference ...")
    inf = bench_inference(variables, spec, widths, dropout)
    print("[benchmark] batch-size sweep ...")
    sweep = bench_batch_sweep(variables, spec, widths, dropout)
    print("[benchmark] pmap training scaling ...")
    scaling = bench_training_scaling(spec, cfg)

    dev = jax.local_devices()[0]
    result = {
        "backend": dev.platform,
        "device_kind": dev.device_kind,
        "num_devices": len(jax.local_devices()),
        "inference_jit_vs_eager": inf,
        "inference_batch_sweep": sweep,
        "training_scaling": scaling,
    }
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    (Path(artifacts_dir) / "benchmark.json").write_text(json.dumps(result, indent=2))
    print(f"[benchmark] jit speedup over eager: {inf['speedup']}x | "
          f"{scaling[-1]['devices']}-device training speedup: {scaling[-1]['speedup_vs_1_device']}x")
    print(f"[benchmark] saved {artifacts_dir}/benchmark.json")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--simulate-devices", type=int, default=None)
    run_benchmark(ap.parse_args().artifacts_dir)
