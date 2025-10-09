from typing import Sequence

import flax.nnx as nnx
import jax.numpy as jnp

from model.common import MLP


class ResnetStack(nnx.Module):
    """ResNet stack module."""

    def __init__(
            self, 
            rngs: nnx.Rngs, 
            in_features,
            out_features,
            num_blocks: int, 
            max_pooling: bool = True,
            deconv: bool = False
        ):
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling
        initializer = nnx.initializers.xavier_uniform()

        self.conv_in = nnx.Conv(
            in_features,
            out_features,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            kernel_init=initializer,
            rngs=rngs,
        )

        self.block_convs1 = [
            nnx.Conv(
                out_features,
                out_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
                rngs=rngs,
            )
            for _ in range(num_blocks)
        ]
        self.block_convs2 = [
            nnx.Conv(
                out_features,
                out_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
                rngs=rngs,
            )
            for _ in range(num_blocks)
        ]
        if deconv:
            self.deconv = nnx.ConvTranspose(
                out_features,
                out_features,
                kernel_size=(2, 2),
                strides=2,
                padding='SAME',
                kernel_init=initializer,
                rngs=rngs,
            )
        else:
            self.deconv = None

    def __call__(self, x):
        conv_out = self.conv_in(x)

        if self.max_pooling:
            conv_out = nnx.max_pool(
                conv_out,
                window_shape=(3, 3),
                strides=(2, 2),
                padding='SAME',
            )
        
        if self.deconv:
            conv_out = self.deconv(conv_out)

        for i in range(self.num_blocks):
            block_input = conv_out
            conv_out = nnx.relu(conv_out)
            conv_out = self.block_convs1[i](conv_out)

            conv_out = nnx.relu(conv_out)
            conv_out = self.block_convs2[i](conv_out)

            conv_out = conv_out + block_input

        return conv_out


class ImpalaEncoder(nnx.Module):
    """IMPALA encoder, with logvar."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        img_size: tuple,
        in_features: int = 3,
        stack_sizes: tuple = (64, 128, 128),  # (16, 32, 32)
        num_blocks: int = 2,    # 1
        dropout_rate: float | None = None,
        mlp_hidden_dims: Sequence[int] = (1024,),    # (512,)
        layer_norm: bool = False,
    ):
        in_head_dim = (img_size[0] // (2 ** len(stack_sizes))) * (img_size[1] // (2 ** len(stack_sizes))) * stack_sizes[-1]
        stack_sizes = (in_features,) + stack_sizes
        self.stack_blocks = [
            ResnetStack(
                rngs=rngs,
                in_features=stack_sizes[i],
                out_features=stack_sizes[i + 1],
                num_blocks=num_blocks,
            )
            for i in range(len(stack_sizes) - 1)
        ]

        if layer_norm:
            self.layer_norm_layer = nnx.LayerNorm()
        else:
            self.layer_norm_layer = None

        if dropout_rate is not None:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        else:
            self.dropout = None

        self.mu_head = MLP(
            (in_head_dim,) + mlp_hidden_dims,
            dtype=jnp.float32,
            rngs=rngs,
        )
        self.logvar_head = MLP(
            (in_head_dim,) + mlp_hidden_dims,
            dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, x):
        # x = x.astype(jnp.float32) / 255.0
        conv_out = x

        for block in self.stack_blocks:
            conv_out = block(conv_out)
            if self.dropout is not None:
                conv_out = self.dropout(conv_out)

        conv_out = nnx.relu(conv_out)
        if self.layer_norm_layer is not None:
            conv_out = self.layer_norm_layer(conv_out)

        out = conv_out.reshape((*x.shape[:-3], -1))

        mu = self.mu_head(out)
        logvar = self.logvar_head(out)
        return mu, logvar

class ImpalaDecoder(nnx.Module):
    """IMPALA decoder."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        img_size: tuple,
        in_features: int = 3,
        stack_sizes: tuple = (64, 128, 128),  # (16, 32, 32)
        num_blocks: int = 2,    # 1
        dropout_rate: float | None = None,
        mlp_hidden_dims: Sequence[int] = (1024,),    # (512,)
        layer_norm: bool = False,
    ):
        self.in_head_size = (img_size[0] // (2 ** len(stack_sizes)), (img_size[1] // (2 ** len(stack_sizes))), stack_sizes[-1]) 
        in_head_dim = (img_size[0] // (2 ** len(stack_sizes))) * (img_size[1] // (2 ** len(stack_sizes))) * stack_sizes[-1]
        stack_sizes = (in_features,) + stack_sizes
        self.stack_blocks = [
            ResnetStack(
                rngs=rngs,
                in_features=stack_sizes[i + 1],
                out_features=stack_sizes[i],
                num_blocks=num_blocks,
                max_pooling=False,
                deconv=True
            )
            for i in range(len(stack_sizes) - 1)
        ]

        if layer_norm:
            self.layer_norm_layer = nnx.LayerNorm()
        else:
            self.layer_norm_layer = None

        if dropout_rate is not None:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        else:
            self.dropout = None

        self.head = MLP(
            mlp_hidden_dims[::-1] + (in_head_dim,),
            dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, z):
        out = self.head(z)

        conv_out = out.reshape((-1, *self.in_head_size))

        conv_out = nnx.relu(conv_out)
        if self.layer_norm_layer is not None:
            conv_out = self.layer_norm_layer(conv_out)
        
        for block in reversed(self.stack_blocks):
            conv_out = block(conv_out)
            if self.dropout is not None:
                conv_out = self.dropout(conv_out)
        
        x = conv_out
        # x = (x * 255.0).astype(jnp.uint8)
        return x

class Encoder(nnx.Module):
    """VAE using IMPALA encoder."""
    def __init__(
        self,
        rngs: nnx.Rngs,
        img_size: tuple = (64, 64),
        in_features: int = 3,
        stack_sizes: tuple = (64, 128, 128),  # (16, 32, 32)
        num_blocks: int = 2,    # 1
        dropout_rate: float | None = None,
        mlp_hidden_dims: Sequence[int] = (1024,),    # (512,)
        layer_norm: bool = False,
    ):
        self.encoder = ImpalaEncoder(
            rngs=rngs,
            img_size=img_size,
            in_features=in_features,
            stack_sizes=stack_sizes,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            mlp_hidden_dims=mlp_hidden_dims,
            layer_norm=layer_norm,
        )
        self.decoder = ImpalaDecoder(
            rngs=rngs,
            img_size=img_size,
            in_features=in_features,
            stack_sizes=stack_sizes,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            mlp_hidden_dims=mlp_hidden_dims,
            layer_norm=layer_norm,
        )
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        y = self.decoder(z)
        return y

    def __call__(self, x):
        mu, _ = self.encoder(x)
        return mu