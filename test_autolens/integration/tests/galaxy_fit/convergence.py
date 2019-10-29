import os
import numpy as np

import autofit as af
import autolens as al
from test import integration_util

test_type = "galaxy_fit"
test_name = "convergence"

test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"


def galaxy_fit_phase():

    pixel_scales = 0.1
    image_shape = (150, 150)

    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    grid = al.grid.uniform(shape_2d=image_shape, pixel_scales=pixel_scales, sub_size=4)

    galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
    )

    convergence = galaxy.convergence_from_grid(galaxies=[galaxy], grid=grid)

    noise_map = al.array.manual_2d
        sub_array_1d=np.ones(convergence.shape), pixel_scales=pixel_scales
    )

    data = al.GalaxyData(
        image=convergence, noise_map=noise_map, pixel_scales=pixel_scales
    )

    phase1 = al.PhaseGalaxy(
        phase_name=test_name + "/",
        galaxies=dict(
            gal=al.GalaxyModel(redshift=0.5, light=al.mp.SphericalIsothermal)
        ),
        use_convergence=True,
        sub_size=4,
        optimizer_class=af.MultiNest,
    )

    phase1.run(galaxy_data=[data])


if __name__ == "__main__":
    galaxy_fit_phase()
