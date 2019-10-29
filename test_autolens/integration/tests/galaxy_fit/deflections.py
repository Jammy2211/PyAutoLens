import os
import numpy as np

import autofit as af
import autolens as al
from test import integration_util

test_type = "galaxy_fit"
test_name = "deflections"

test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"
config_path = test_path + "config"
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)


def galaxy_fit_phase():

    pixel_scales = 0.1
    image_shape = (150, 150)

    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    grid = al.grid.uniform(shape_2d=image_shape, pixel_scales=pixel_scales, sub_size=4)

    galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
    )

    deflections = galaxy.deflections_from_grid(galaxies=[galaxy], grid=grid)

    noise_map = al.array.manual_2d
        sub_array_1d=np.ones(deflections[:, 0].shape), pixel_scales=pixel_scales
    )

    data_y = al.GalaxyData(
        image=deflections[:, 0], noise_map=noise_map, pixel_scales=pixel_scales
    )
    data_x = al.GalaxyData(
        image=deflections[:, 1], noise_map=noise_map, pixel_scales=pixel_scales
    )

    phase1 = al.PhaseGalaxy(
        phase_name=test_name + "/",
        galaxies=dict(
            gal=al.GalaxyModel(redshift=0.5, light=al.mp.SphericalIsothermal)
        ),
        use_deflections=True,
        sub_size=4,
        optimizer_class=af.MultiNest,
    )

    phase1.run(galaxy_data=[data_y, data_x])


if __name__ == "__main__":
    galaxy_fit_phase()
