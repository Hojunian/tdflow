import math
from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union
from flax import nnx
import jax
import jax.numpy as jnp
from einops import rearrange

import time

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

from .common import *


# From https://github.com/young-geng/m3ae_public/blob/master/m3ae/model.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    emb = get_1d_sincos_pos_embed_from_grid(
        embed_dim, jnp.arange(length, dtype=jnp.float32)
    )
    return jnp.expand_dims(emb, 0)


def get_2d_sincos_pos_embed(rng, embed_dim, img_size, patch_size):
    # example: embed_dim = 256, length = 16*16
    H = img_size[0] // patch_size
    W = img_size[1] // patch_size

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = jnp.arange(H, dtype=jnp.float32)
    grid_w = jnp.arange(W, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, H, W])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0)  # (1, H*W, D)


class DiTBlock(nnx.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size: int, num_heads: int, dtype: Dtype, rngs: nnx.Rngs):
        self.norm_1 = nnx.LayerNorm(
            hidden_size, use_bias=False, use_scale=False, dtype=dtype, rngs=rngs
        )
        self.norm_2 = nnx.LayerNorm(
            hidden_size, use_bias=False, use_scale=False, dtype=dtype, rngs=rngs
        )

        self.mlp = MLP(
            [hidden_size, hidden_size * 4, hidden_size], dtype=dtype, rngs=rngs
        )
        self.adaLN_modulation = [
            nnx.silu,
            nnx.Linear(hidden_size, hidden_size * 6, kernel_init=nnx.initializers.zeros_init(), dtype=dtype, rngs=rngs),
        ]

        self.qkv = nnx.Linear(
            hidden_size, 3 * hidden_size, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.proj = nnx.Linear(hidden_size, hidden_size, dtype=dtype, rngs=rngs)

        self.num_heads = num_heads
        self.channels_per_head = hidden_size // num_heads

    def __call__(self, x, c):
        # Calculate adaLn modulation parameters.
        for layer in self.adaLN_modulation:
            c = layer(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            c, 6, axis=-1
        )

        # Attention Residual.
        x_norm = self.norm_1(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)

        k, q, v = jnp.split(self.qkv(x_modulated), 3, axis=-1)
        k = jnp.reshape(
            k, (k.shape[0], k.shape[1], self.num_heads, self.channels_per_head)
        )
        q = jnp.reshape(
            q, (q.shape[0], q.shape[1], self.num_heads, self.channels_per_head)
        )
        v = jnp.reshape(
            v, (v.shape[0], v.shape[1], self.num_heads, self.channels_per_head)
        )
        q = q / q.shape[3]  # (1/d) scaling.

        w = jnp.einsum("bqhc,bkhc->bhqk", q, k)  # [B, HW, HW, num_heads]
        w = w.astype(jnp.float32)
        w = nnx.softmax(w, axis=-1)

        y = jnp.einsum("bhqk,bkhc->bqhc", w, v)  # [B, HW, num_heads, channels_per_head]
        y = jnp.reshape(y, x.shape)  # [B, H, W, C] (C = heads * channels_per_head)
        attn_x = self.proj(y)

        x = x + (gate_msa[:, None] * attn_x)

        # MLP Residual.
        x_norm2 = self.norm_2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)

        mlp_x = self.mlp(x_modulated2)
        x = x + (gate_mlp[:, None] * mlp_x)

        return x


class FinalLayer(nnx.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        patch_size: int,
        out_channels: int,
        hidden_size: int,
        dtype: Dtype,
        rngs: nnx.Rngs,
    ):
        self.norm_final = nnx.LayerNorm(
            hidden_size, use_bias=False, use_scale=False, dtype=dtype, rngs=rngs
        )
        self.linear = nnx.Linear(
            hidden_size, patch_size * patch_size * out_channels, kernel_init=nnx.initializers.zeros_init(), dtype=dtype, rngs=rngs
        )
        self.adaLN_modulation = [
            nnx.silu,
            nnx.Linear(hidden_size, hidden_size * 2, kernel_init=nnx.initializers.zeros_init(), dtype=dtype, rngs=rngs),
        ]

    def __call__(self, x, c):
        for layer in self.adaLN_modulation:
            c = layer(c)
        shift, scale = jnp.split(c, 2, axis=-1)

        x = self.norm_final(x)
        x = modulate(x, shift, scale)

        x = self.linear(x)

        return x


class DiT2D(nnx.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        patch_size: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        img_size: tuple,
        in_channels: int,
        rngs: nnx.Rngs,
        dtype: Dtype = jnp.bfloat16,
    ):
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.img_size = img_size
        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size
        self.dtype = dtype

        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size, dtype, rngs)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype, rngs)

        self.blocks = [
            DiTBlock(hidden_size, num_heads, dtype, rngs) for _ in range(depth)
        ]
        self.final_layer = FinalLayer(patch_size, in_channels, hidden_size, dtype, rngs)

    def __call__(self, x, t):
        x = self.x_embedder(x)
        x += get_2d_sincos_pos_embed(
            None, self.hidden_size, self.img_size, self.patch_size
        ).astype(self.dtype)
        c = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)

        x = jnp.reshape(
            x,
            (
                x.shape[0],
                self.num_patches_h,
                self.num_patches_w,
                self.patch_size,
                self.patch_size,
                self.in_channels,
            ),
        )
        x = jnp.einsum("bhwpqc->bhpwqc", x)
        x = rearrange(
            x,
            "B H P W Q C -> B (H P) (W Q) C",
            H=self.num_patches_h,
            W=self.num_patches_w,
        )

        return x
    
class DiT2D_GHM(nnx.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        patch_size: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        img_size: tuple,
        in_channels: int,
        action_dim: int,
        rngs: nnx.Rngs,
        dtype: Dtype = jnp.bfloat16,
    ):
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.img_size = img_size
        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size
        self.dtype = dtype

        self.x_embedder = PatchEmbed(patch_size, 2 * in_channels, hidden_size, dtype, rngs)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype, rngs)
        self.a_embedder = nnx.Linear(action_dim, hidden_size, rngs=rngs)

        self.blocks = [
            DiTBlock(hidden_size, num_heads, dtype, rngs) for _ in range(depth)
        ]
        self.final_layer = FinalLayer(patch_size, in_channels, hidden_size, dtype, rngs)

    def __call__(self, x, x_prev, a, t):
        x = jnp.concatenate([x, x_prev], axis=-1)
        x = self.x_embedder(x)
        x += get_2d_sincos_pos_embed(
            None, self.hidden_size, self.img_size, self.patch_size
        ).astype(self.dtype)
        
        c = self.t_embedder(t)
        c += self.a_embedder(a)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)

        x = jnp.reshape(
            x,
            (
                x.shape[0],
                self.num_patches_h,
                self.num_patches_w,
                self.patch_size,
                self.patch_size,
                self.in_channels,
            ),
        )
        x = jnp.einsum("bhwpqc->bhpwqc", x)
        x = rearrange(
            x,
            "B H P W Q C -> B (H P) (W Q) C",
            H=self.num_patches_h,
            W=self.num_patches_w,
        )

        return x