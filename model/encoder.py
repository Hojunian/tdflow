from flax import nnx
import jax
from einops import rearrange
from diffusers import FlaxAutoencoderKL

class Encoder(nnx.Module):
    def __init__(self):
        module, params = FlaxAutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder = "vae",
        )
        self.module = module
        self.params = jax.tree_util.tree_map(
            lambda p: nnx.Param(p), params
        )
    
    def encode(self, images):
        params = jax.tree_util.tree_map(
            lambda p: p.value, self.params
        )
        images = rearrange(images, "b h w c -> b c h w")
        latents = self.module.apply(
            {"params": params}, images, method=self.module.encode
        ).latent_dist.sample()
        latents *= self.module.config.scaling_factor
        return latents
    
    def decode(self, latents):
        latents /= self.module.config.scaling_factor
        params = jax.tree_util.tree_map(
            lambda p: p.value, self.params
        )
        images = self.module.apply(
            {"params": params}, latents, method=self.module.decode
        ).sample
        images = rearrange(images, "b c h w -> b h w c")
        return images
    
    def __call__(self, x):
        return self.encode(x)
