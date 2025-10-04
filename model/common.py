import math
from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union
from flax import nnx
import jax
import jax.numpy as jnp
from einops import rearrange

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class TimestepEmbedder(nnx.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size: int,
        dtype: Dtype,
        rngs: nnx.Rngs,
        frequency_embedding_size: int = 256,
    ):
        self.mlp = [
            nnx.Linear(frequency_embedding_size, hidden_size, dtype=dtype, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_size, hidden_size, dtype=dtype, rngs=rngs),
        ]

        self.dtype = dtype
        self.frequency_embedding_size = frequency_embedding_size

    def __call__(self, t):
        x = self.timestep_embedding(t)
        for layer in self.mlp:
            x = layer(x)

        return x

    # t is between [0, 1].
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                            These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = t * max_period
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period)
            * jnp.arange(start=0, stop=half, dtype=jnp.float32)
            / half
        )
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        embedding = embedding.astype(self.dtype)
        return embedding


class LabelEmbedder(nnx.Module):
    """
    Embeds class labels into vector representations. Also handles tokeinzation for discrete diffusion.
    """

    def __init__(
        self, num_classes: int, hidden_size: int, dtype: Dtype, rngs: nnx.Rngs
    ):
        self.embedding_table = nnx.Embed(
            num_classes + 1, hidden_size, dtype=dtype, rngs=rngs
        )

    def __call__(self, labels):
        embeddings = self.embedding_table(labels)

        return embeddings


class PatchEmbed(nnx.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        dtype: Dtype,
        rngs: nnx.Rngs,
    ):
        self.proj = nnx.Conv(
            in_channels,
            hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=patch_size,
            padding="VALID",
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array):
        x = self.proj(x)
        x = rearrange(x, "b h w c -> b (h w) c")

        return x


class MLP(nnx.Module):
    def __init__(self, dims: list, dtype: Dtype, rngs: nnx.Rngs):
        assert len(dims) >= 2
        self.layers = []
        depth = len(dims) - 1
        for i in range(depth - 1):
            self.layers.append(nnx.Linear(dims[i], dims[i + 1], dtype=dtype, rngs=rngs))
            self.layers.append(nnx.gelu)
        self.layers.append(
            nnx.Linear(dims[depth - 1], dims[depth], dtype=dtype, rngs=rngs)
        )

    def __call__(self, x: jax.Array):
        for layer in self.layers:
            x = layer(x)

        return x


def modulate(x, shift, scale):
    # scale = jnp.clip(scale, -1, 1)
    return x * (1 + scale[:, None]) + shift[:, None]
