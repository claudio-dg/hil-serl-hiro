from typing import Dict, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_boh(x):
    return print("\033[95m {}\033[00m".format(x))

class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
    """

    encoder: nn.Module
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train=False,
        stop_gradient=False,
        is_encoded=False,
    ) -> jnp.ndarray:
        # encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        # print_green(f"Saved video for camera {camera_key} at {video_path}")
                        print_green(f"image shape PRE {image.shape}")

                        image = rearrange(image, "T H W C -> H W (T C)")

                        print_boh(f"image shape POST {image.shape}")


                    if len(image.shape) == 5:
                        print_green(f"5 --> 6 ")
                        
                        print_green(f"image shape PRE {image.shape}")

                        image = rearrange(image, "B T H W C -> B H W (T C)")
                        print_boh(f"image shape POST {image.shape}")


            image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = jax.lax.stop_gradient(image)

            encoded.append(image)

        encoded = jnp.concatenate(encoded, axis=-1)

        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"]
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    encoded = encoded.reshape(-1)
                if len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
            state = nn.Dense(
                self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform()
            )(state)
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            encoded = jnp.concatenate([encoded, state], axis=-1)

        return encoded
