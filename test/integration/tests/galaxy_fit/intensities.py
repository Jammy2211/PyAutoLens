import os
import numpy as np

import autofit as af
import autolens as al
from test.integration import integration_util

test_type = "galaxy_fit"
test_name = "image"

test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"


def galaxy_fit_phase():

    pixel_scale = 0.1
    image_shape = (150, 150)

    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    grid = al.Grid.from_shape_pixel_scale_and_sub_size(
        shape=image_shape, pixel_scale=pixel_scale, sub_size=4
    )

    galaxy = al.Galaxy(
        redshift=0.5,
        light=al.light_profiles.SphericalExponential(
            centre=(0.0, 0.0), intensity=1.0, effective_radius=0.5
        ),
    )

    image = galaxy.profile_image_from_grid(
        galaxies=[galaxy], grid=grid, return_in_2d=True, bypass_decorator=False
    )

    noise_map = al.ScaledSquarePixelArray(
        array=np.ones(image.shape), pixel_scale=pixel_scale
    )

    data = al.GalaxyData(image=image, noise_map=noise_map, pixel_scale=pixel_scale)

    phase1 = al.PhaseGalaxy(
        phase_name=test_name + "/",
        galaxies=dict(
            gal=al.GalaxyModel(
                redshift=0.5, light=al.light_profiles.SphericalExponential
            )
        ),
        use_image=True,
        sub_size=4,
        optimizer_class=af.MultiNest,
    )

    phase1.run(galaxy_data=[data])


if __name__ == "__main__":
    galaxy_fit_phase()
