import flax.jax_utils as jax_utils
import jax
import jax.numpy as jnp
import numpy as np

from jaxvision.config import SYNTHETIC, TrainConfig
from jaxvision.data import eval_batches, make_synthetic, normalize, shard, train_batches
from jaxvision.model import build_model, count_params, init_variables
from jaxvision.train import create_state, make_eval_step, make_lr_schedule, make_train_step


def test_eight_devices_available():
    assert jax.local_device_count() == 8


def test_model_output_shape_and_params():
    model = build_model(10, (8, 16), 0.1)
    vs = init_variables(model, (32, 32, 3))
    out = model.apply(vs, jnp.zeros((5, 32, 32, 3)), train=False)
    assert out.shape == (5, 10)
    assert count_params(vs["params"]) > 0


def test_shard_shapes():
    x = np.zeros((64, 32, 32, 3))
    assert shard(x, 8).shape == (8, 8, 32, 32, 3)


def test_eval_batches_cover_every_sample_once():
    x = np.zeros((70, 4, 4, 1), np.uint8)
    y = np.zeros(70, np.int32)
    total = sum(float(m.sum()) for _, _, m in eval_batches(x, y, 32, 8))
    assert total == 70  # padding is masked out


def test_train_batches_drop_remainder():
    x = np.zeros((70, 4, 4, 1), np.uint8)
    y = np.zeros(70, np.int32)
    n = sum(1 for _ in train_batches(x, y, 32, 8, np.random.RandomState(0)))
    assert n == 2


def test_synthetic_dataset_is_deterministic():
    a = make_synthetic(16, 4, 3, SYNTHETIC)
    b = make_synthetic(16, 4, 3, SYNTHETIC)
    assert all(np.array_equal(i, j) for i, j in zip(a, b))


def test_normalize_is_zero_centred():
    x = jnp.full((1, 2, 2, 3), 128, jnp.uint8)
    out = normalize(x, SYNTHETIC)
    assert abs(float(out.mean())) < 0.1


def _tiny_setup():
    cfg = TrainConfig(dataset="synthetic", epochs=1, widths=(8, 16), per_device_batch_size=8, dropout=0.1)
    devices = jax.local_devices()
    state = jax_utils.replicate(create_state(cfg, SYNTHETIC, make_lr_schedule(cfg, 10)), devices)
    x, y, _, _ = make_synthetic(64, 8, 0, SYNTHETIC)
    return cfg, devices, state, shard(x, 8), shard(y, 8)


def test_pmap_step_keeps_replicas_identical():
    """The whole point of pmean: after a step every device holds the SAME weights,
    even though each device saw different data."""
    _, devices, state, x, y = _tiny_setup()
    rngs = jax_utils.replicate(jax.random.PRNGKey(0), devices)
    state, metrics = make_train_step(SYNTHETIC)(state, x, y, rngs)
    for leaf in jax.tree.leaves(state.params) + jax.tree.leaves(state.batch_stats):
        arr = np.asarray(leaf)
        assert np.allclose(arr, arr[0:1], atol=1e-6), "replicas diverged"
    assert np.isfinite(np.asarray(metrics["loss"])).all()


def test_training_reduces_loss_and_eval_counts_all():
    _, devices, state, x, y = _tiny_setup()
    rngs = jax_utils.replicate(jax.random.PRNGKey(0), devices)
    p_train = make_train_step(SYNTHETIC)
    first = None
    for _ in range(25):
        state, m = p_train(state, x, y, rngs)
        first = first if first is not None else float(m["loss"][0])
    assert float(m["loss"][0]) < first

    mask = np.ones(y.shape, np.float32)
    out = make_eval_step(SYNTHETIC)(state, x, y, mask)
    assert float(out["count"][0]) == 64
