"""Fake 8 CPU devices BEFORE jax is imported so the pmap tests exercise real
multi-device code paths on any machine."""

import os
import sys
from pathlib import Path

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
).strip()
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
