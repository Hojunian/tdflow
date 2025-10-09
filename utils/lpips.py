import jax
from jax import numpy as jnp
from lpips_j.lpips import LPIPS

lpips = LPIPS()
key = jax.random.PRNGKey(0)
x = jnp.zeros((1, 8, 8, 3))
params = lpips.init(key, x, x)

def lpips_loss(x, y):
    x = (x - 0.5) / 0.5
    y = (y - 0.5) / 0.5
    return jnp.mean(lpips.apply(params, x, y))