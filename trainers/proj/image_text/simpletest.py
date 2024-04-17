from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import jax.random as random  # Import jax.random


# # Setting up devices and mesh.
# devices = mesh_utils.create_device_mesh((jax.device_count(),))  # A flat mesh with only a 'batch' dimension
# print(devices)
# mesh = Mesh(devices, axis_names=('batch',))

devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('i', 'j'))
print(devices)
print(mesh)

# Create a device mesh
# devices = jax.devices()
# mesh_axes = ('data',)
# mesh = Mesh(devices, axis_names=mesh_axes)

def _gather_from_device(x, device_id, axis_name="batch"):
    # Gathering data from the specific device using the comparison of axis indices.
    return jax.lax.psum((jax.lax.axis_index(axis_name) == device_id) * x, axis_name)

@partial(shard_map, mesh=mesh, in_specs=(P('batch', None), P('batch', None), None, None),
         out_specs=(None, None))
def chunked_sigmoid_loss_sharded(zimg, ztxt, temperature, bias=0.0):
    """Loss computation adapted for shard_map with cross-device negative loss calculation."""

    # Calculate loss for representations on this shard, which includes positives.
    logits_me = jnp.dot(zimg, ztxt.T)  # (n, D) . (D, n) -> (n, n)
    print('entering ehre1')
    logits_me = logits_me * temperature + bias
    print('entering ehre')
    m1_diag1 = -jnp.ones_like(logits_me) + 2 * jnp.eye(zimg.shape[0])
    loglik_me = jax.nn.log_sigmoid(m1_diag1 * logits_me)
    nll_me = -loglik_me.sum(axis=-1).mean()

    # Function to compute negative loss for other shard data
    def negative_loss(ztxt_other_device):
        logits_ot = jnp.dot(zimg, ztxt_other_device.T)
        logits_ot = logits_ot * temperature + bias
        loglik_ot = jax.nn.log_sigmoid(-logits_ot)
        return -jnp.sum(loglik_ot, axis=-1).mean()

    # Aggregate negative losses across shards
    me = jax.lax.axis_index('batch')
    nll_others = 0
    for device_id in range(jax.device_count()):
        ztxt_other_device = _gather_from_device(ztxt, device_id, "batch")
        if device_id != me:
            nll_others += negative_loss(ztxt_other_device)

    eye = jnp.eye(zimg.shape[0])
    stats = {
        "pos_min_logit": jnp.min(jnp.diag(logits_me)),
        "pos_max_logit": jnp.max(jnp.diag(logits_me)),
        "pos_avg_logit": _avg_pos_logit(logits_me),
        "local_neg_min_logit": jnp.min(logits_me + 1e9 * eye),
        "local_neg_max_logit": jnp.max(logits_me - 1e9 * eye),
        "local_neg_avg_logit": _avg_neg_logit(logits_me),
    }
    return nll_me + nll_others, stats

key = random.PRNGKey(0)  # Initialize a PRNG key
# zimg = random.normal(key, shape=(64, 128))  # Use the random module for normal distribution
# ztxt = random.normal(key, shape=(64, 128))  # Use the random module for normal distribution

zimg = jnp.arange( 8 * 16.).reshape(8, 16)
ztxt = jnp.arange(16 * 32.).reshape(16, 32)

temperature = 0.1
bias = 0.0
print(zimg.shape)
print(ztxt.shape)
loss, stats = chunked_sigmoid_loss_sharded(zimg, ztxt, temperature, bias)
print(loss, stats)
