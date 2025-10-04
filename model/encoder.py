from flax import nnx
import jax
from jax import numpy as jnp
from einops import rearrange
from diffusers import FlaxAutoencoderKL

from model.network import DiT2D

class Encoder(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            from_pretrained = True,
        ):
        module, params = FlaxAutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder = "vae",
        )
        self.module = nnx.bridge.ToNNX(module)
        self.scaling_factor = module.config.scaling_factor
        self.rngs = rngs
        
        # init
        x = jnp.zeros((1, 3, 8, 8))
        nnx.bridge.lazy_init(self.module, x, rngs=rngs)
        if from_pretrained:
            nnx.update(self.module, jax.device_get(params))
    
    def encode(self, images):
        images = (images - 0.5) / 0.5
        images = rearrange(images, "b h w c -> b c h w")
        latent_dist = self.module.encode(images).latent_dist
        latents = latent_dist.sample(self.rngs())
        latents *= self.scaling_factor
        images = rearrange(images, "b c h w -> b h w c")
        return latents, latent_dist

    def decode(self, latents):
        latents = rearrange(latents, "b h w c -> b c h w")
        latents /= self.scaling_factor
        images = self.module.decode(latents).sample
        images = rearrange(images, "b c h w -> b h w c")
        images = images * 0.5 + 0.5
        return images
    
    def __call__(self, images):
        latents, _ = self.encode(images)
        return latents

class EncodedDiT2D(DiT2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rngs = kwargs["rngs"]
        self.encoder = Encoder(
            rngs = rngs,
            from_pretrained = False,
        )
    
    def __call__(self, x, t, a=None):
        x = self.encoder(x)
        x = super().__call__(x, t, a)
        return self.encoder.decode(x)
