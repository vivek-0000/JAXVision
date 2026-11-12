"""The CNN, written with Flax Linen.

Architecture (widths = 32, 64, 128 by default):

    [Conv3x3 -> BatchNorm -> ReLU] x2 -> MaxPool -> Dropout     (x len(widths))
    -> global average pool -> Dense(256) -> ReLU -> Dropout -> Dense(classes)

BatchNorm takes an optional `axis_name`. During `pmap` training we pass
axis_name="batch" so batch statistics are averaged across *all* devices
(synchronised BatchNorm) and behave as if computed on the global batch.
"""

from __future__ import annotations

from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class CNN(nn.Module):
    num_classes: int
    widths: Sequence[int] = (32, 64, 128)
    dropout: float = 0.25
    axis_name: Optional[str] = None  # pmap axis for synchronised BatchNorm

    @nn.compact
    def __call__(self, x, train: bool):
        for width in self.widths:
            for _ in range(2):
                x = nn.Conv(width, (3, 3), padding="SAME", use_bias=False)(x)
                x = nn.BatchNorm(
                    use_running_average=not train,
                    momentum=0.9,
                    axis_name=self.axis_name if train else None,
                )(x)
                x = nn.relu(x)
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
            x = nn.Dropout(self.dropout, deterministic=not train)(x)
        x = jnp.mean(x, axis=(1, 2))  # global average pooling
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout, deterministic=not train)(x)
        return nn.Dense(self.num_classes)(x)


def build_model(
    num_classes: int,
    widths: Sequence[int],
    dropout: float,
    axis_name: Optional[str] = None,
) -> CNN:
    return CNN(
        num_classes=num_classes,
        widths=tuple(int(w) for w in widths),
        dropout=float(dropout),
        axis_name=axis_name,
    )


def init_variables(model: CNN, image_shape, seed: int = 0):
    """Initialise params + batch_stats from a dummy batch of one image."""
    dummy = jnp.zeros((1,) + tuple(image_shape), dtype=jnp.float32)
    return model.init(jax.random.PRNGKey(seed), dummy, train=False)


def count_params(tree) -> int:
    return int(sum(np.prod(p.shape) for p in jax.tree.leaves(tree)))
