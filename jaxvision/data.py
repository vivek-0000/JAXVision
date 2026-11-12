"""Dataset loading, preprocessing and batching.

Images are kept on the host as uint8 NHWC arrays. Conversion to float and
normalisation happens *inside* the jitted/pmapped step (see `normalize`), so
only 1 byte per pixel crosses the host -> accelerator link.

Supported datasets:
  * cifar10    - downloaded from the official University of Toronto mirror
  * mnist      - downloaded from the Google CVDF mirror
  * synthetic  - generated procedurally, no network needed
"""

from __future__ import annotations

import gzip
import io
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from .config import DATASETS, DatasetSpec

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


@dataclass
class Dataset:
    x_train: np.ndarray  # uint8 [N, H, W, C]
    y_train: np.ndarray  # int32 [N]
    x_test: np.ndarray
    y_test: np.ndarray
    spec: DatasetSpec
    source: str  # "downloaded" | "cached" | "synthetic" | "synthetic-fallback"


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def _load_cifar10(data_dir: Path):
    archive = _download(CIFAR10_URL, data_dir / "cifar-10-python.tar.gz")
    xs, ys = [], []
    x_test = y_test = None
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if not (name.startswith("data_batch_") or name == "test_batch"):
                continue
            # Official CIFAR-10 python batches are pickles of {bytes: ...}.
            batch = pickle.load(tar.extractfile(member), encoding="bytes")
            data = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = np.asarray(batch[b"labels"], dtype=np.int32)
            if name == "test_batch":
                x_test, y_test = data, labels
            else:
                xs.append(data)
                ys.append(labels)
    return (
        np.concatenate(xs).astype(np.uint8),
        np.concatenate(ys),
        x_test.astype(np.uint8),
        y_test,
    )


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def _load_mnist(data_dir: Path):
    paths = {k: _download(MNIST_BASE + v, data_dir / v) for k, v in MNIST_FILES.items()}
    x_train = _read_idx(paths["x_train"])[..., None]
    x_test = _read_idx(paths["x_test"])[..., None]
    y_train = _read_idx(paths["y_train"]).astype(np.int32)
    y_test = _read_idx(paths["y_test"]).astype(np.int32)
    return x_train, y_train, x_test, y_test


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def _pattern(cls: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Ten visually distinct texture families, returned in [-1, 1]."""
    r = np.sqrt(u**2 + v**2)
    theta = np.arctan2(v, u)
    tau = 2 * np.pi
    if cls == 0:
        return np.sin(tau * 4 * v)
    if cls == 1:
        return np.sin(tau * 4 * u)
    if cls == 2:
        return np.sin(tau * 3 * (u + v))
    if cls == 3:
        return np.sin(tau * 3 * (u - v))
    if cls == 4:
        return np.sign(np.sin(tau * 3 * u) * np.sin(tau * 3 * v))
    if cls == 5:
        return np.sin(tau * 4 * r)
    if cls == 6:
        return np.cos(tau * 4 * u) * np.cos(tau * 4 * v)
    if cls == 7:
        return np.clip(np.exp(-(u**2) / 0.04) + np.exp(-(v**2) / 0.04), 0, 1) * 2 - 1
    if cls == 8:
        return np.sin(8 * r + 2 * theta)
    return np.clip(u * 1.4, -1, 1)


def make_synthetic(n_train: int, n_test: int, seed: int, spec: DatasetSpec):
    h, w, c = spec.image_shape
    rng = np.random.RandomState(seed)
    lin = np.linspace(-1, 1, h, dtype=np.float32)
    uu, vv = np.meshgrid(lin, lin)

    def build(n: int):
        labels = rng.randint(0, spec.num_classes, size=n).astype(np.int32)
        images = np.empty((n, h, w, c), dtype=np.uint8)
        for i, cls in enumerate(labels):
            # random shift + scale jitter, random colour tint, additive noise
            du, dv = rng.uniform(-0.5, 0.5, size=2)
            s = rng.uniform(0.9, 1.1)
            pat = _pattern(int(cls), (uu + du) * s, (vv + dv) * s)
            tint = rng.uniform(0.6, 1.0, size=c)
            img = 0.5 + 0.35 * pat[..., None] * tint
            img = img + rng.normal(0, 0.12, size=img.shape)
            images[i] = np.clip(img * 255, 0, 255).astype(np.uint8)
        return images, labels

    x_train, y_train = build(n_train)
    x_test, y_test = build(n_test)
    return x_train, y_train, x_test, y_test


# --------------------------------------------------------------------------- #
# Public loader
# --------------------------------------------------------------------------- #
def load_dataset(
    name: str,
    data_dir: str = "data",
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    seed: int = 0,
    synthetic_fallback: bool = False,
) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {sorted(DATASETS)}.")
    spec = DATASETS[name]
    root = Path(data_dir)
    source = "synthetic"

    if name == "synthetic":
        arrays = make_synthetic(train_subset or 8192, test_subset or 1024, seed, spec)
    else:
        try:
            cached = any(root.glob("*cifar*")) or any(root.glob("*ubyte*"))
            arrays = _load_cifar10(root) if name == "cifar10" else _load_mnist(root)
            source = "cached" if cached else "downloaded"
        except Exception as exc:  # network blocked, corrupt archive, ...
            if not synthetic_fallback:
                raise RuntimeError(
                    f"Could not load '{name}': {exc}\n"
                    "Check your internet connection, or pass --synthetic-fallback "
                    "(or --dataset synthetic) to train on generated data."
                ) from exc
            print(f"[data] WARNING: {exc}\n[data] falling back to synthetic data")
            spec = DATASETS["synthetic"]
            arrays = make_synthetic(train_subset or 8192, test_subset or 1024, seed, spec)
            source = "synthetic-fallback"

    x_train, y_train, x_test, y_test = arrays
    if train_subset and name != "synthetic" and source != "synthetic-fallback":
        x_train, y_train = x_train[:train_subset], y_train[:train_subset]
    if test_subset and name != "synthetic" and source != "synthetic-fallback":
        x_test, y_test = x_test[:test_subset], y_test[:test_subset]
    return Dataset(x_train, y_train, x_test, y_test, spec, source)


# --------------------------------------------------------------------------- #
# On-device preprocessing + host-side batching
# --------------------------------------------------------------------------- #
def normalize(images_uint8, spec: DatasetSpec):
    """uint8 [..., H, W, C] -> float32, zero-mean / unit-variance per channel."""
    x = images_uint8.astype(jnp.float32) / 255.0
    mean = jnp.asarray(spec.mean, dtype=jnp.float32)
    std = jnp.asarray(spec.std, dtype=jnp.float32)
    return (x - mean) / std


def shard(batch: np.ndarray, n_devices: int) -> np.ndarray:
    """[global_batch, ...] -> [n_devices, per_device_batch, ...] for pmap."""
    return batch.reshape((n_devices, -1) + batch.shape[1:])


def train_batches(
    x: np.ndarray, y: np.ndarray, global_batch: int, n_devices: int, rng: np.random.RandomState
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Shuffled epoch of full batches (the ragged tail is dropped)."""
    idx = rng.permutation(len(x))
    for start in range(0, len(x) - global_batch + 1, global_batch):
        b = idx[start : start + global_batch]
        yield shard(x[b], n_devices), shard(y[b], n_devices)


def eval_batches(
    x: np.ndarray, y: np.ndarray, global_batch: int, n_devices: int
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Every sample exactly once; the last batch is zero-padded and masked."""
    for start in range(0, len(x), global_batch):
        xb, yb = x[start : start + global_batch], y[start : start + global_batch]
        mask = np.ones(len(xb), dtype=np.float32)
        pad = global_batch - len(xb)
        if pad:
            xb = np.concatenate([xb, np.zeros((pad,) + xb.shape[1:], xb.dtype)])
            yb = np.concatenate([yb, np.zeros(pad, yb.dtype)])
            mask = np.concatenate([mask, np.zeros(pad, np.float32)])
        yield shard(xb, n_devices), shard(yb, n_devices), shard(mask, n_devices)
