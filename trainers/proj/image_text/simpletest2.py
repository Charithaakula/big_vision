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

# devices = mesh_utils.create_device_mesh((4, 2))
# mesh = Mesh(devices, axis_names=('i', 'j'))
# print(devices)
# print(mesh)

# Create a device mesh
devices = jax.devices()
mesh_axes = ('batch',)
mesh = Mesh(devices, axis_names=mesh_axes)
print(devices)
print(mesh)

def _gather_from_device(x, device_id, axis_name="batch"):
    me = jax.lax.axis_index('batch').flatten()
    me = me[0]
    noskip = jnp.equal(device_id, me)
    return jax.lax.psum((noskip) * x, axis_name)

# def _gather_from_device(x, device_id, axis_name="batch"):
#   return jax.lax.psum((jax.lax.axis_index(axis_name) == device_id) * x,
#                       axis_name)

@partial(shard_map, mesh=mesh, in_specs=(P('batch', None), P('batch', None)), out_specs=P('batch'))
def chunked_sigmoid_loss_sharded(zimg, ztxt):
    """Loss computation adapted for shard_map with cross-device negative loss calculation."""
    print(zimg.shape)
    print(ztxt.shape)
    # Calculate loss for representations on this shard, which includes positives.
    logits_me = jnp.dot(zimg, ztxt.T)  # (n, D) . (D, n) -> (n, n)
    print('entering ehre1')
    print('entering ehre')
    m1_diag1 = -jnp.ones_like(logits_me) + 2 * jnp.eye(zimg.shape[0])
    loglik_me = jax.nn.log_sigmoid(m1_diag1 * logits_me)
    print('reacing hereree')
    nll_me = -loglik_me.sum(axis=-1).mean()

    # Function to compute negative loss for other shard data
    def negative_loss(ztxt_other_device):
        logits_ot = jnp.dot(zimg, ztxt_other_device.T)
        loglik_ot = jax.nn.log_sigmoid(-logits_ot)
        return -jnp.sum(loglik_ot, axis=-1).mean()
        #return 0

    me = jax.lax.axis_index('batch').flatten()
    print(f'me is {me[0]}')
    me = me[0]
    #me.astype(int)
    #print(type(me))
    nll_others = 0
    for device_id in range(jax.device_count()):
        #print(device_id.dtype)
        ztxt_other_device = _gather_from_device(ztxt, device_id, "batch")
        skip = jnp.not_equal(device_id, me)
        nll_others += skip * negative_loss(ztxt_other_device)

    # eye = jnp.eye(zimg.shape[0])
    # stats = {
    #     "pos_min_logit": jnp.min(jnp.diag(logits_me)),
    #     "pos_max_logit": jnp.max(jnp.diag(logits_me)),
    #     # "pos_avg_logit": _avg_pos_logit(logits_me),
    #     "local_neg_min_logit": jnp.min(logits_me + 1e9 * eye),
    #     "local_neg_max_logit": jnp.max(logits_me - 1e9 * eye),
    #     # "local_neg_avg_logit": _avg_neg_logit(logits_me),
    # }
    print('entering here as well lol')
    val = nll_me + nll_others 
    val = jnp.tile(val, (1))
    #return jnp.array(nll_me) 
    return val
    #return [nll_me]

key = random.PRNGKey(0)  # Initialize a PRNG key
zimg = random.normal(key, shape=(64, 128))  # Use the random module for normal distribution
ztxt = random.normal(key, shape=(64, 128))  # Use the random module for normal distribution

# zimg = jnp.arange( 8 * 16.).reshape(8, 16)
# ztxt = jnp.arange(16 * 32.).reshape(16, 32)
temperature = 0.1
bias = 0.0
print(zimg.shape)
print(ztxt.shape)
loss = chunked_sigmoid_loss_sharded(zimg, ztxt)
print(loss)
