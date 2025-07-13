import jax.numpy as jnp
import numpy as np

import autoarray as aa

from autolens.lens.tracer import Tracer
from autolens.lens.to_inversion import TracerToInversion

from autolens.fixtures import make_masked_imaging_7x7_no_blur


def mapper_indices_from(model):

    instance = model.instance_from_prior_medians()
    tracer = Tracer(galaxies=instance.galaxies)
    tracer_to_inversion = TracerToInversion(
        dataset=make_masked_imaging_7x7_no_blur(),
        tracer=tracer,
    )

    mapper_indices = aa.util.inversion.param_range_list_from(
        cls=aa.AbstractMapper, linear_obj_list=tracer_to_inversion.linear_obj_list
    )

    return jnp.arange(np.min(mapper_indices), np.max(mapper_indices), dtype=int)
