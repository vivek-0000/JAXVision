"""Distributed training with `jax.pmap`.

How the parallelism works (data parallelism, SPMD):

  1. The whole train state (params, optimizer state, BN statistics) is
     *replicated* onto every device with `flax.jax_utils.replicate`.
  2. Each global batch is reshaped to [n_devices, per_device_batch, H, W, C]
     and `pmap` hands one slice to each device.
  3. Every device computes the loss and gradients on its own slice.
  4. `jax.lax.pmean(grads, "batch")` is an all-reduce over the interconnect
     (ICI on a TPU pod slice) -> every device now holds the *same averaged
     gradient*, so the replicated weights stay bit-identical after the update.
  5. BatchNorm uses `axis_name="batch"` so its statistics are also averaged
     across devices (synchronised BN).

Run:
    python -m jaxvision.train --dataset cifar10 --epochs 30
    python -m jaxvision.train --dataset synthetic --epochs 3 --simulate-devices 8   # no TPU/internet needed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import flax.jax_utils as jax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from .checkpoint import save_checkpoint
from .config import DATASETS, TrainConfig
from .data import eval_batches, load_dataset, normalize, train_batches
from .model import build_model, count_params, init_variables

AXIS = "batch"  # name of the pmap axis; used by pmean/psum and by BatchNorm


class TrainState(train_state.TrainState):
    batch_stats: Any  # BatchNorm running mean / variance


# --------------------------------------------------------------------------- #
# On-device augmentation (random horizontal flip + random crop with 4px pad)
# --------------------------------------------------------------------------- #
def augment(rng, x):
    """x: float [B, H, W, C]. Done inside the compiled step - no host cost."""
    b, h, w, c = x.shape
    flip_key, crop_key = jax.random.split(rng)
    flip = jax.random.bernoulli(flip_key, 0.5, (b,))
    x = jnp.where(flip[:, None, None, None], x[:, :, ::-1, :], x)

    pad = 4
    xp = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    offs = jax.random.randint(crop_key, (b, 2), 0, 2 * pad + 1)

    def crop(img, off):
        return jax.lax.dynamic_slice(img, (off[0], off[1], 0), (h, w, c))

    return jax.vmap(crop)(xp, offs)


# --------------------------------------------------------------------------- #
# State creation
# --------------------------------------------------------------------------- #
def make_lr_schedule(cfg: TrainConfig, steps_per_epoch: int):
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = min(cfg.warmup_epochs * steps_per_epoch, total - 1) if total > 1 else 0
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=warmup,
        decay_steps=total,
        end_value=cfg.learning_rate * 0.01,
    )


def create_state(cfg: TrainConfig, spec, schedule) -> TrainState:
    model = build_model(spec.num_classes, cfg.widths, cfg.dropout, axis_name=AXIS)
    variables = init_variables(model, spec.image_shape, cfg.seed)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=cfg.weight_decay),
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
    )


# --------------------------------------------------------------------------- #
# pmapped steps
# --------------------------------------------------------------------------- #
def make_train_step(spec):
    def train_step(state: TrainState, images, labels, rng):
        # Distinct dropout / augmentation randomness on every device and step.
        rng = jax.random.fold_in(rng, jax.lax.axis_index(AXIS))
        rng = jax.random.fold_in(rng, state.step)
        aug_rng, drop_rng = jax.random.split(rng)

        x = normalize(images, spec)
        if spec.augment:
            x = augment(aug_rng, x)

        def loss_fn(params):
            logits, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                x,
                train=True,
                mutable=["batch_stats"],
                rngs={"dropout": drop_rng},
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            return loss, (logits, updates["batch_stats"])

        (loss, (logits, new_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        # ---- distributed gradient aggregation (all-reduce mean) ----------
        grads = jax.lax.pmean(grads, axis_name=AXIS)
        # ------------------------------------------------------------------

        state = state.apply_gradients(grads=grads, batch_stats=new_stats)
        metrics = {
            "loss": jax.lax.pmean(loss, AXIS),
            "accuracy": jax.lax.pmean((jnp.argmax(logits, -1) == labels).mean(), AXIS),
        }
        return state, metrics

    return jax.pmap(train_step, axis_name=AXIS, donate_argnums=(0,))


def make_eval_step(spec):
    def eval_step(state: TrainState, images, labels, mask):
        x = normalize(images, spec)
        logits = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats}, x, train=False
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        correct = (jnp.argmax(logits, -1) == labels).astype(jnp.float32)
        return {
            "loss": jax.lax.psum((loss * mask).sum(), AXIS),
            "correct": jax.lax.psum((correct * mask).sum(), AXIS),
            "count": jax.lax.psum(mask.sum(), AXIS),
        }

    return jax.pmap(eval_step, axis_name=AXIS)


def evaluate(p_eval, state, x, y, global_batch, n_devices):
    tot = {"loss": 0.0, "correct": 0.0, "count": 0.0}
    for xb, yb, mb in eval_batches(x, y, global_batch, n_devices):
        out = p_eval(state, xb, yb, mb)
        for k in tot:
            tot[k] += float(out[k][0])  # psum result is identical on every device
    return tot["loss"] / tot["count"], tot["correct"] / tot["count"]


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #
def train(cfg: TrainConfig) -> dict:
    devices = jax.local_devices()
    n_dev = cfg.num_devices or len(devices)
    if n_dev > len(devices):
        raise ValueError(f"Requested {n_dev} devices but only {len(devices)} available.")
    devices = devices[:n_dev]
    platform = devices[0].platform
    print(f"[jaxvision] backend={platform}  devices={n_dev}  ({devices[0].device_kind})")

    ds = load_dataset(
        cfg.dataset, cfg.data_dir, cfg.train_subset, cfg.test_subset, cfg.seed, cfg.synthetic_fallback
    )
    spec = ds.spec
    print(f"[jaxvision] dataset={spec.name} ({ds.source})  train={len(ds.x_train)}  test={len(ds.x_test)}")

    global_batch = cfg.per_device_batch_size * n_dev
    steps_per_epoch = len(ds.x_train) // global_batch
    if steps_per_epoch < 1:
        raise ValueError(
            f"Global batch {global_batch} > training set {len(ds.x_train)}. "
            "Lower --per-device-batch-size."
        )

    schedule = make_lr_schedule(cfg, steps_per_epoch)
    state = create_state(cfg, spec, schedule)
    n_params = count_params(state.params)
    print(f"[jaxvision] parameters={n_params:,}  global_batch={global_batch}  steps/epoch={steps_per_epoch}")

    state = jax_utils.replicate(state, devices)  # copy to every device
    p_train, p_eval = make_train_step(spec), make_eval_step(spec)
    # Same key on every device; each device folds in its axis_index inside the step.
    rngs = jax_utils.replicate(jax.random.PRNGKey(cfg.seed + 1), devices)
    np_rng = np.random.RandomState(cfg.seed)

    history = []
    best_acc = -1.0  # best test accuracy seen (reported only; the final weights are what get saved)
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        loss_acc = []
        for xb, yb in train_batches(ds.x_train, ds.y_train, global_batch, n_dev, np_rng):
            state, m = p_train(state, xb, yb, rngs)
            loss_acc.append(m)
        # block + average training metrics for the epoch
        tr_loss = float(np.mean([float(m["loss"][0]) for m in loss_acc]))
        tr_acc = float(np.mean([float(m["accuracy"][0]) for m in loss_acc]))
        epoch_time = time.time() - t0

        te_loss, te_acc = evaluate(p_eval, state, ds.x_test, ds.y_test, global_batch, n_dev)
        lr = float(schedule(int(state.step[0])))
        row = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 5),
            "train_accuracy": round(tr_acc, 5),
            "test_loss": round(te_loss, 5),
            "test_accuracy": round(te_acc, 5),
            "learning_rate": lr,
            "epoch_time_s": round(epoch_time, 3),
            "images_per_sec": round(steps_per_epoch * global_batch / epoch_time, 1),
        }
        history.append(row)
        print(
            f"[epoch {epoch:>3}/{cfg.epochs}] train loss {tr_loss:.4f} acc {tr_acc*100:5.2f}% | "
            f"test loss {te_loss:.4f} acc {te_acc*100:5.2f}% | {epoch_time:5.1f}s "
            f"({row['images_per_sec']:.0f} img/s)"
        )
        best_acc = max(best_acc, te_acc)

    total_time = time.time() - t_start

    # ---- persist ---------------------------------------------------------
    out = Path(cfg.artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_state = jax.device_get(jax_utils.unreplicate(state))
    save_checkpoint(
        cfg.artifacts_dir,
        {"params": final_state.params, "batch_stats": final_state.batch_stats},
        spec,
        cfg.widths,
        cfg.dropout,
    )
    # A few held-out test images, so the deployed server can show Playground
    # samples without needing the dataset (or internet access).
    pick = np.random.RandomState(0).choice(len(ds.x_test), size=min(32, len(ds.x_test)), replace=False)
    np.savez_compressed(out / "samples.npz", x=ds.x_test[pick], y=ds.y_test[pick])

    metrics = {
        "config": cfg.to_dict(),
        "dataset": {
            "name": spec.name,
            "source": ds.source,
            "train_size": int(len(ds.x_train)),
            "test_size": int(len(ds.x_test)),
            "class_names": list(spec.class_names),
        },
        "hardware": {
            "backend": platform,
            "device_kind": devices[0].device_kind,
            "num_devices": n_dev,
            "global_batch_size": global_batch,
        },
        "model": {"parameters": n_params, "widths": list(cfg.widths)},
        "history": history,
        "summary": {
            "epochs": cfg.epochs,
            "final_test_accuracy": history[-1]["test_accuracy"],
            "best_test_accuracy": round(best_acc, 5),
            "final_train_accuracy": history[-1]["train_accuracy"],
            "total_train_time_s": round(total_time, 2),
            "mean_images_per_sec": round(float(np.mean([h["images_per_sec"] for h in history[1:] or history])), 1),
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[jaxvision] final test accuracy: {history[-1]['test_accuracy']*100:.2f}%  (best {best_acc*100:.2f}%)")
    print(f"[jaxvision] saved checkpoint + metrics.json to {out}/")

    if not cfg.skip_benchmark:
        from .benchmark import run_benchmark

        run_benchmark(cfg.artifacts_dir, cfg, spec, state_for_inference=final_state)
    return metrics


def parse_args(argv=None) -> TrainConfig:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Train the JAXVision CNN with pmap.")
    p.add_argument("--dataset", choices=sorted(DATASETS), default=d.dataset)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--per-device-batch-size", type=int, default=d.per_device_batch_size)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup-epochs", type=int, default=d.warmup_epochs)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--widths", type=int, nargs="+", default=list(d.widths))
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--num-devices", type=int, default=None, help="default: all local devices")
    p.add_argument("--simulate-devices", type=int, default=None,
                   help="fake N CPU devices to test pmap without a TPU")
    p.add_argument("--data-dir", default=d.data_dir)
    p.add_argument("--artifacts-dir", default=d.artifacts_dir)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--synthetic-fallback", action="store_true",
                   help="use generated data if the dataset download fails")
    p.add_argument("--skip-benchmark", action="store_true")
    a = p.parse_args(argv)
    return TrainConfig(
        dataset=a.dataset, epochs=a.epochs, per_device_batch_size=a.per_device_batch_size,
        learning_rate=a.learning_rate, weight_decay=a.weight_decay, warmup_epochs=a.warmup_epochs,
        dropout=a.dropout, widths=tuple(a.widths), seed=a.seed, num_devices=a.num_devices,
        data_dir=a.data_dir, artifacts_dir=a.artifacts_dir, train_subset=a.train_subset,
        test_subset=a.test_subset, synthetic_fallback=a.synthetic_fallback,
        skip_benchmark=a.skip_benchmark,
    )


if __name__ == "__main__":
    train(parse_args())
