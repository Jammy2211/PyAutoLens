import numpy as np
from os import path

import autolens as al

test_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


### Hyper Quantities ###


def test__contribution_map():
    adapt_model_image = al.Array2D.no_mask(values=[[2.0, 4.0, 10.0]], pixel_scales=1.0)
    adapt_galaxy_image = al.Array2D.no_mask(values=[[1.0, 5.0, 8.0]], pixel_scales=1.0)

    hyper_galaxy_0 = al.legacy.HyperGalaxy(contribution_factor=5.0)
    hyper_galaxy_1 = al.legacy.HyperGalaxy(contribution_factor=10.0)

    galaxy_0 = al.legacy.Galaxy(
        redshift=0.5,
        hyper_galaxy=hyper_galaxy_0,
        adapt_model_image=adapt_model_image,
        adapt_galaxy_image=adapt_galaxy_image,
    )

    galaxy_1 = al.legacy.Galaxy(
        redshift=1.0,
        hyper_galaxy=hyper_galaxy_1,
        adapt_model_image=adapt_model_image,
        adapt_galaxy_image=adapt_galaxy_image,
    )

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[galaxy_0, galaxy_1])

    assert (
        tracer.contribution_map
        == tracer.image_plane.contribution_map + tracer.source_plane.contribution_map
    ).all()
    assert (
        tracer.contribution_map_list[0].slim == tracer.image_plane.contribution_map
    ).all()

    assert (
        tracer.contribution_map_list[1].slim == tracer.source_plane.contribution_map
    ).all()

    galaxy_0 = al.legacy.Galaxy(redshift=0.5)

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[galaxy_0, galaxy_1])

    assert (tracer.contribution_map == tracer.source_plane.contribution_map).all()
    assert tracer.contribution_map_list[0] == None

    assert (
        tracer.contribution_map_list[1].slim == tracer.source_plane.contribution_map
    ).all()

    galaxy_1 = al.legacy.Galaxy(redshift=1.0)

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[galaxy_0, galaxy_1])

    assert tracer.contribution_map == None
    assert tracer.contribution_map_list[0] == None

    assert tracer.contribution_map_list[1] == None


def test__hyper_noise_map_list_from(sub_grid_2d_7x7):
    noise_map_1d = al.Array2D.no_mask(values=[[5.0, 3.0, 1.0]], pixel_scales=1.0)

    adapt_model_image = al.Array2D.no_mask(values=[[2.0, 4.0, 10.0]], pixel_scales=1.0)
    adapt_galaxy_image = al.Array2D.no_mask(values=[[1.0, 5.0, 8.0]], pixel_scales=1.0)

    hyper_galaxy_0 = al.legacy.HyperGalaxy(contribution_factor=5.0)
    hyper_galaxy_1 = al.legacy.HyperGalaxy(contribution_factor=10.0)

    galaxy_0 = al.legacy.Galaxy(
        redshift=0.5,
        hyper_galaxy=hyper_galaxy_0,
        adapt_model_image=adapt_model_image,
        adapt_galaxy_image=adapt_galaxy_image,
        hyper_minimum_value=0.0,
    )

    galaxy_1 = al.legacy.Galaxy(
        redshift=1.0,
        hyper_galaxy=hyper_galaxy_1,
        adapt_model_image=adapt_model_image,
        adapt_galaxy_image=adapt_galaxy_image,
        hyper_minimum_value=0.0,
    )

    plane_0 = al.legacy.Plane(redshift=0.5, galaxies=[galaxy_0])
    plane_1 = al.legacy.Plane(redshift=0.5, galaxies=[galaxy_1])
    plane_2 = al.legacy.Plane(redshift=1.0, galaxies=[al.legacy.Galaxy(redshift=0.5)])

    hyper_noise_map_0 = plane_0.hyper_noise_map_from(noise_map=noise_map_1d)
    hyper_noise_map_1 = plane_1.hyper_noise_map_from(noise_map=noise_map_1d)

    tracer = al.legacy.Tracer(
        planes=[plane_0, plane_1, plane_2], cosmology=al.cosmo.Planck15()
    )

    hyper_noise_maps = tracer.hyper_noise_map_list_from(noise_map=noise_map_1d)

    assert (hyper_noise_maps[0].slim == hyper_noise_map_0).all()
    assert (hyper_noise_maps[1].slim == hyper_noise_map_1).all()
    assert (hyper_noise_maps[2].slim == np.zeros(shape=(3, 1))).all()

    hyper_noise_map = tracer.hyper_noise_map_from(noise_map=noise_map_1d)

    assert (hyper_noise_map.slim == hyper_noise_map_0 + hyper_noise_map_1).all()

    tracer = al.legacy.Tracer.from_galaxies(
        galaxies=[galaxy_0, galaxy_1], cosmology=al.cosmo.Planck15()
    )

    hyper_noise_maps = tracer.hyper_noise_map_list_from(noise_map=noise_map_1d)

    assert (hyper_noise_maps[0].slim == hyper_noise_map_0).all()
    assert (hyper_noise_maps[1].slim == hyper_noise_map_1).all()
