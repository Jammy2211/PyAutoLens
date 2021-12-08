from astropy import cosmology as cosmo
import numpy as np
import pytest
import os
from os import path
import shutil
from skimage import measure

import autolens as al

from autoarray.mock.mock import MockPixelization, MockRegularization
from autolens.mock.mock import MockMassProfile

test_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


def test__operate_image__blurred_images_2d_via_psf_from__for_tracer_gives_list_of_planes(
    sub_grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3
):
    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.EllSersic(intensity=1.0),
        mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=2.0))

    plane_0 = al.Plane(redshift=0.5, galaxies=[g0])
    plane_1 = al.Plane(redshift=1.0, galaxies=[g1])

    operate_image = al.OperateImage.from_light_obj(light_obj=plane_0)

    blurred_image_0 = operate_image.blurred_image_2d_via_psf_from(
        grid=sub_grid_2d_7x7, psf=psf_3x3, blurring_grid=blurring_grid_2d_7x7
    )

    source_grid_2d_7x7 = plane_0.traced_grid_from(grid=sub_grid_2d_7x7)
    source_blurring_grid_2d_7x7 = plane_0.traced_grid_from(grid=blurring_grid_2d_7x7)

    operate_image = al.OperateImage.from_light_obj(light_obj=plane_1)

    blurred_image_1 = operate_image.blurred_image_2d_via_psf_from(
        grid=source_grid_2d_7x7, psf=psf_3x3, blurring_grid=source_blurring_grid_2d_7x7
    )

    tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=cosmo.Planck15)

    operate_image = al.OperateImage.from_light_obj(light_obj=tracer)

    blurred_image = operate_image.blurred_image_2d_via_psf_from(
        grid=sub_grid_2d_7x7, psf=psf_3x3, blurring_grid=blurring_grid_2d_7x7
    )

    assert blurred_image.native == pytest.approx(
        blurred_image_0.native + blurred_image_1.native, 1.0e-4
    )

    blurred_image_list = operate_image.blurred_image_2d_list_via_psf_from(
        grid=sub_grid_2d_7x7, psf=psf_3x3, blurring_grid=blurring_grid_2d_7x7
    )

    assert (blurred_image_list[0].slim == blurred_image_0.slim).all()
    assert (blurred_image_list[1].slim == blurred_image_1.slim).all()

    assert (blurred_image_list[0].native == blurred_image_0.native).all()
    assert (blurred_image_list[1].native == blurred_image_1.native).all()


def test__operate_image__blurred_images_2d_via_convolver_from__for_tracer_gives_list_of_planes(
    sub_grid_2d_7x7, blurring_grid_2d_7x7, convolver_7x7
):
    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.EllSersic(intensity=1.0),
        mass_profile=al.mp.SphIsothermal(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=2.0))

    plane_0 = al.Plane(redshift=0.5, galaxies=[g0])
    plane_1 = al.Plane(redshift=1.0, galaxies=[g1])

    operate_image = al.OperateImage.from_light_obj(light_obj=plane_0)

    blurred_image_0 = operate_image.blurred_image_2d_via_convolver_from(
        grid=sub_grid_2d_7x7,
        convolver=convolver_7x7,
        blurring_grid=blurring_grid_2d_7x7,
    )

    source_grid_2d_7x7 = plane_0.traced_grid_from(grid=sub_grid_2d_7x7)
    source_blurring_grid_2d_7x7 = plane_0.traced_grid_from(grid=blurring_grid_2d_7x7)

    operate_image = al.OperateImage.from_light_obj(light_obj=plane_1)

    blurred_image_1 = operate_image.blurred_image_2d_via_convolver_from(
        grid=source_grid_2d_7x7,
        convolver=convolver_7x7,
        blurring_grid=source_blurring_grid_2d_7x7,
    )

    tracer = al.Tracer(planes=[plane_0, plane_1], cosmology=cosmo.Planck15)

    operate_image = al.OperateImage.from_light_obj(light_obj=tracer)

    blurred_image = operate_image.blurred_image_2d_via_convolver_from(
        grid=sub_grid_2d_7x7,
        convolver=convolver_7x7,
        blurring_grid=blurring_grid_2d_7x7,
    )

    assert blurred_image.native == pytest.approx(
        blurred_image_0.native + blurred_image_1.native, 1.0e-4
    )

    blurred_image_list = operate_image.blurred_image_2d_list_via_convolver_from(
        grid=sub_grid_2d_7x7,
        convolver=convolver_7x7,
        blurring_grid=blurring_grid_2d_7x7,
    )

    assert (blurred_image_list[0].slim == blurred_image_0.slim).all()
    assert (blurred_image_list[1].slim == blurred_image_1.slim).all()

    assert (blurred_image_list[0].native == blurred_image_0.native).all()
    assert (blurred_image_list[1].native == blurred_image_1.native).all()


def test__operate_image__visibilities_of_planes_from_grid_and_transformer(
    sub_grid_2d_7x7, transformer_7x7_7
):

    g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.EllSersic(intensity=1.0))
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.EllSersic(intensity=2.0))

    plane_0 = al.Plane(redshift=0.5, galaxies=[g0])
    plane_1 = al.Plane(redshift=0.5, galaxies=[g1])
    plane_2 = al.Plane(redshift=1.0, galaxies=[al.Galaxy(redshift=1.0)])

    operate_image_plane_0 = al.OperateImage.from_light_obj(light_obj=plane_0)
    operate_image_plane_1 = al.OperateImage.from_light_obj(light_obj=plane_1)

    visibilities_0 = operate_image_plane_0.visibilities_via_transformer_from(
        grid=sub_grid_2d_7x7, transformer=transformer_7x7_7
    )

    visibilities_1 = operate_image_plane_1.visibilities_via_transformer_from(
        grid=sub_grid_2d_7x7, transformer=transformer_7x7_7
    )

    tracer = al.Tracer(planes=[plane_0, plane_1, plane_2], cosmology=cosmo.Planck15)

    operate_image_tracer = al.OperateImage.from_light_obj(light_obj=tracer)

    visibilities = operate_image_tracer.visibilities_list_via_transformer_from(
        grid=sub_grid_2d_7x7, transformer=transformer_7x7_7
    )

    assert (visibilities[0] == visibilities_0).all()
    assert (visibilities[1] == visibilities_1).all()


def test__operate_lens__sums_individual_quantities():

    grid = al.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.15)

    sis_0 = al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=0.2)
    sis_1 = al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=0.4)
    sis_2 = al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=0.6)
    sis_3 = al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=0.8)

    galaxy_0 = al.Galaxy(mass_profile_0=sis_0, mass_profile_1=sis_1, redshift=0.5)
    galaxy_1 = al.Galaxy(mass_profile_0=sis_2, mass_profile_1=sis_3, redshift=0.5)

    plane = al.Plane(galaxies=[galaxy_0, galaxy_1])

    tracer = al.Tracer(
        planes=[plane, al.Plane(redshift=1.0, galaxies=None)], cosmology=cosmo.Planck15
    )

    operate_lens_tracer = al.OperateLens.from_mass_obj(mass_obj=tracer)

    einstein_mass = operate_lens_tracer.einstein_mass_angular_from(grid=grid)

    assert einstein_mass == pytest.approx(np.pi * 2.0 ** 2.0, 1.0e-1)
