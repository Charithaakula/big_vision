from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('i', 'j'))
print(devices)
print(mesh)
a = jnp.arange( 8 * 16.).reshape(8, 16)
b = jnp.arange(16 * 32.).reshape(16, 32)

@partial(shard_map, mesh=mesh, in_specs=(P('i', 'j'), P('j', None)),
         out_specs=P(None, None))
def matmul_basic(a_block, b_block):
  # a_block: f32[2, 8]
  # b_block: f32[8, 32]
  z_partialsum = jnp.dot(a_block, b_block)
  z_block = jax.lax.psum(z_partialsum, 'j')
  mean_c = jnp.mean(z_block, axis=0)
  mean_c = jnp.mean(mean_c, axis =0)
  
    # Print shape and layout of the output array
  print("Output array shape:", mean_c.shape)
  # with jax.xla_device_mesh(devices):
  #     print("Output array device layout:", jax.xla.xla_device(mean_c))
  print(xf)
  return jnp.tile(mean_c, (1, 1))
  #return mean_c

c = matmul_basic(a, b)  # c: f32[8, 32]
print(c)
