import jax.numpy as jnp


def mapper_indices_from(total_linear_light_profiles, total_mapper_pixels):
    return jnp.arange(
        total_linear_light_profiles,
        total_linear_light_profiles + total_mapper_pixels,
        dtype=int,
    )
