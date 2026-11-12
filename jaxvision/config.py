"""Static dataset descriptions and the training configuration dataclass."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the rest of the code needs to know about a dataset."""

    name: str
    image_shape: Tuple[int, int, int]  # H, W, C
    num_classes: int
    class_names: Tuple[str, ...]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    augment: bool  # random flip + crop (only sensible for natural images)


CIFAR10 = DatasetSpec(
    name="cifar10",
    image_shape=(32, 32, 3),
    num_classes=10,
    class_names=(
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ),
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616),
    augment=True,
)

MNIST = DatasetSpec(
    name="mnist",
    image_shape=(28, 28, 1),
    num_classes=10,
    class_names=tuple(str(i) for i in range(10)),
    mean=(0.1307,),
    std=(0.3081,),
    augment=False,
)

# Procedurally generated texture dataset. Needs no download, so it is used for
# smoke tests / CI and as an offline fallback.
SYNTHETIC = DatasetSpec(
    name="synthetic",
    image_shape=(32, 32, 3),
    num_classes=10,
    class_names=(
        "h-stripes", "v-stripes", "diag-up", "diag-down", "checker",
        "rings", "dots", "cross", "spiral", "ramp",
    ),
    mean=(0.5, 0.5, 0.5),
    std=(0.25, 0.25, 0.25),
    augment=False,
)

DATASETS = {spec.name: spec for spec in (CIFAR10, MNIST, SYNTHETIC)}


@dataclass
class TrainConfig:
    dataset: str = "cifar10"
    epochs: int = 30
    per_device_batch_size: int = 128  # global batch = this x number of devices
    learning_rate: float = 2e-3
    weight_decay: float = 5e-4
    warmup_epochs: int = 2
    dropout: float = 0.25
    widths: Tuple[int, ...] = (32, 64, 128)
    seed: int = 0
    num_devices: Optional[int] = None  # None = use every local device
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    train_subset: Optional[int] = None  # use only N training images (quick runs)
    test_subset: Optional[int] = None
    synthetic_fallback: bool = False  # fall back to synthetic data if download fails
    skip_benchmark: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["widths"] = list(self.widths)
        return d
