from flax import nnx
import jax
from einops import rearrange
from diffusers import FlaxAutoencoderKL

class Encoder(nnx.Module):
    def __init__(
            self,
        ):
        module, params = FlaxAutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder = "vae",
        )
        self.module = module
        self.params = jax.tree_util.tree_map(
            lambda p: nnx.Param(p), params
        )
    
    def encode(self, images, key):
        images = (images - 0.5) / 0.5
        images = rearrange(images, "b h w c -> b c h w")
        latent_dist = self.module.apply(
            {"params": self.params}, images, method=self.module.encode
        ).latent_dist
        latents = latent_dist.sample(key)
        latents *= self.module.config.scaling_factor
        images = rearrange(images, "b c h w -> b h w c")
        return latents, latent_dist

    def decode(self, latents):
        latents = rearrange(latents, "b h w c -> b c h w")
        latents /= self.module.config.scaling_factor
        images = self.module.apply(
            {"params": self.params}, latents, method=self.module.decode
        ).sample
        images = rearrange(images, "b c h w -> b h w c")
        images = images * 0.5 + 0.5
        return images
    
    def __call__(self, images, key):
        latents, _ = self.encode(images, key)
        return latents
