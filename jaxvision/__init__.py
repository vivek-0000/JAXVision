"""JAXVision - distributed image classification with JAX / Flax / Optax.

Importing this package (before JAX itself is imported) lets you *simulate* a
multi-device machine on a plain CPU, which is handy for testing the `pmap`
code path on a laptop:

    python -m jaxvision.train --simulate-devices 8

or:  JAXVISION_SIMULATE_DEVICES=8 python -m jaxvision.train

On a real TPU VM you do NOT pass this flag: JAX discovers the 8 TPU cores itself.
"""

import os
import sys


def _apply_simulated_devices() -> None:
    n = os.environ.get("JAXVISION_SIMULATE_DEVICES")
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == "--simulate-devices" and i + 1 < len(argv):
            n = argv[i + 1]
        elif arg.startswith("--simulate-devices="):
            n = arg.split("=", 1)[1]
    if not n:
        return
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = (
            flags + f" --xla_force_host_platform_device_count={int(n)}"
        ).strip()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


_apply_simulated_devices()

__version__ = "1.0.0"
