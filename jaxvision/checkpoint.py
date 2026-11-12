"""Save / load trained weights plus the metadata needed to rebuild the model.

Files written to the artifacts directory:
    checkpoint.msgpack   params + batch_stats (flax msgpack serialisation)
    model_meta.json      model hyper-parameters and dataset description
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import flax.serialization as fser
import jax

from .config import DATASETS, DatasetSpec

CKPT_NAME = "checkpoint.msgpack"
META_NAME = "model_meta.json"


def save_checkpoint(artifacts_dir: str, variables: dict, spec: DatasetSpec, widths, dropout: float) -> None:
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / CKPT_NAME).write_bytes(fser.msgpack_serialize(jax.device_get(variables)))
    meta = {
        "dataset": spec.name,
        "image_shape": list(spec.image_shape),
        "num_classes": spec.num_classes,
        "class_names": list(spec.class_names),
        "mean": list(spec.mean),
        "std": list(spec.std),
        "widths": [int(w) for w in widths],
        "dropout": float(dropout),
    }
    (out / META_NAME).write_text(json.dumps(meta, indent=2))


def checkpoint_exists(artifacts_dir: str) -> bool:
    out = Path(artifacts_dir)
    return (out / CKPT_NAME).exists() and (out / META_NAME).exists()


def load_checkpoint(artifacts_dir: str) -> Tuple[dict, dict, DatasetSpec]:
    """Returns (variables, meta, dataset_spec)."""
    out = Path(artifacts_dir)
    if not checkpoint_exists(artifacts_dir):
        raise FileNotFoundError(
            f"No checkpoint in '{artifacts_dir}'. Train first: python -m jaxvision.train"
        )
    variables = fser.msgpack_restore((out / CKPT_NAME).read_bytes())
    meta = json.loads((out / META_NAME).read_text())
    spec = DatasetSpec(
        name=meta["dataset"],
        image_shape=tuple(meta["image_shape"]),
        num_classes=meta["num_classes"],
        class_names=tuple(meta["class_names"]),
        mean=tuple(meta["mean"]),
        std=tuple(meta["std"]),
        augment=DATASETS[meta["dataset"]].augment,
    )
    return variables, meta, spec
