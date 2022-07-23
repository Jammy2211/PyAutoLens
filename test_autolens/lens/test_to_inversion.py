import numpy as np
import pytest
from os import path

import autolens as al

test_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


def test__lp_linear_func_galaxy_dict_from(sub_grid_2d_7x7, blurring_grid_2d_7x7):

    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_no_pix])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    lp_linear_func_galaxy_dict = tracer_to_inversion.lp_linear_func_galaxy_dict_from(
        grid=sub_grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7
    )

    assert lp_linear_func_galaxy_dict == {}

    lp_linear_0 = al.lp_linear.LightProfileLinear()
    lp_linear_1 = al.lp_linear.LightProfileLinear()
    lp_linear_2 = al.lp_linear.LightProfileLinear()

    galaxy_no_linear = al.Galaxy(redshift=0.5)
    galaxy_linear_0 = al.Galaxy(
        redshift=0.5, lp_linear=lp_linear_0, mass=al.mp.SphIsothermal()
    )

    galaxy_linear_1 = al.Galaxy(
        redshift=1.0, lp_linear=lp_linear_1, mass=al.mp.SphIsothermal()
    )
    galaxy_linear_2 = al.Galaxy(redshift=2.0, lp_linear=lp_linear_2)

    tracer = al.Tracer.from_galaxies(
        galaxies=[galaxy_no_linear, galaxy_linear_0, galaxy_linear_1, galaxy_linear_2]
    )

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    lp_linear_func_galaxy_dict = tracer_to_inversion.lp_linear_func_galaxy_dict_from(
        grid=sub_grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7
    )

    lp_linear_func_list = list(lp_linear_func_galaxy_dict.keys())

    assert lp_linear_func_galaxy_dict[lp_linear_func_list[0]] == galaxy_linear_0
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[1]] == galaxy_linear_1
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[2]] == galaxy_linear_2

    assert lp_linear_func_list[0].light_profile_list[0] == lp_linear_0
    assert lp_linear_func_list[1].light_profile_list[0] == lp_linear_1
    assert lp_linear_func_list[2].light_profile_list[0] == lp_linear_2

    traced_grid_list = tracer.traced_grid_2d_list_from(grid=sub_grid_2d_7x7)

    assert lp_linear_func_list[0].grid == pytest.approx(sub_grid_2d_7x7, 1.0e-4)
    assert lp_linear_func_list[1].grid == pytest.approx(traced_grid_list[1], 1.0e-4)
    assert lp_linear_func_list[2].grid == pytest.approx(traced_grid_list[2], 1.0e-4)


def test__cls_pg_list_from(sub_grid_2d_7x7):
    galaxy_pix = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(mapper=1),
        regularization=al.m.MockRegularization(),
    )
    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_pix])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    assert tracer_to_inversion.cls_pg_list_from(cls=al.pix.Pixelization)[0] == []
    assert (
        tracer_to_inversion.cls_pg_list_from(cls=al.pix.Pixelization)[1][0].mapper == 1
    )

    galaxy_pix_0 = al.Galaxy(
        redshift=0.5,
        pixelization=al.m.MockPixelization(mapper=1),
        regularization=al.m.MockRegularization(),
    )

    galaxy_pix_1 = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(mapper=2),
        regularization=al.m.MockRegularization(),
    )

    galaxy_pix_2 = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(mapper=3),
        regularization=al.m.MockRegularization(),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[galaxy_pix_0, galaxy_pix_1, galaxy_pix_2]
    )

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    assert (
        tracer_to_inversion.cls_pg_list_from(cls=al.pix.Pixelization)[0][0].mapper == 1
    )
    assert (
        tracer_to_inversion.cls_pg_list_from(cls=al.pix.Pixelization)[1][0].mapper == 2
    )
    assert (
        tracer_to_inversion.cls_pg_list_from(cls=al.pix.Pixelization)[1][1].mapper == 3
    )

    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_no_pix])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    assert tracer_to_inversion.cls_pg_list_from(cls=al.pix.Pixelization) == [[]]


def test__hyper_galaxy_image_pg_list(sub_grid_2d_7x7):

    gal = al.Galaxy(redshift=0.5)
    gal_pix = al.Galaxy(
        redshift=0.5,
        pixelization=al.m.MockPixelization(),
        regularization=al.m.MockRegularization(),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[gal, gal])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    assert tracer_to_inversion.hyper_galaxy_image_pg_list == [[]]

    tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal_pix])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    assert tracer_to_inversion.hyper_galaxy_image_pg_list == [[None, None]]

    gal_pix = al.Galaxy(
        redshift=0.5,
        pixelization=al.m.MockPixelization(),
        regularization=al.m.MockRegularization(),
        hyper_galaxy_image=1,
    )

    tracer = al.Tracer.from_galaxies(galaxies=[gal_pix, gal])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    assert tracer_to_inversion.hyper_galaxy_image_pg_list == [[1]]

    gal0 = al.Galaxy(redshift=0.25)
    gal1 = al.Galaxy(redshift=0.75)
    gal2 = al.Galaxy(redshift=1.5)

    gal_pix0 = al.Galaxy(
        redshift=0.5,
        pixelization=al.m.MockPixelization(),
        regularization=al.m.MockRegularization(),
        hyper_galaxy_image=1,
    )

    gal_pix1 = al.Galaxy(
        redshift=2.0,
        pixelization=al.m.MockPixelization(),
        regularization=al.m.MockRegularization(),
        hyper_galaxy_image=2,
    )

    gal_pix2 = al.Galaxy(
        redshift=2.0,
        pixelization=al.m.MockPixelization(),
        regularization=al.m.MockRegularization(),
        hyper_galaxy_image=3,
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[gal0, gal1, gal2, gal_pix0, gal_pix1, gal_pix2]
    )

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    assert tracer_to_inversion.hyper_galaxy_image_pg_list == [[], [1], [], [], [2, 3]]


def test__sparse_image_plane_grid_pg_list_from(sub_grid_2d_7x7):

    # Test Correct Grid

    galaxy_pix = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(
            data_pixelization_grid=np.array([[1.0, 1.0]])
        ),
        regularization=al.m.MockRegularization(),
    )
    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_pix])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    pixelization_grids = tracer_to_inversion.sparse_image_plane_grid_pg_list_from(
        grid=sub_grid_2d_7x7
    )

    assert pixelization_grids[0] == None
    assert (pixelization_grids[1] == np.array([[1.0, 1.0]])).all()

    # Test for extra galaxies

    galaxy_pix0 = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(
            data_pixelization_grid=np.array([[1.0, 1.0]])
        ),
        regularization=al.m.MockRegularization(),
    )

    galaxy_pix1 = al.Galaxy(
        redshift=2.0,
        pixelization=al.m.MockPixelization(
            data_pixelization_grid=np.array([[2.0, 2.0]])
        ),
        regularization=al.m.MockRegularization(),
    )

    galaxy_no_pix_0 = al.Galaxy(redshift=0.25)
    galaxy_no_pix_1 = al.Galaxy(redshift=0.5)
    galaxy_no_pix_2 = al.Galaxy(redshift=1.5)

    tracer = al.Tracer.from_galaxies(
        galaxies=[
            galaxy_pix0,
            galaxy_pix1,
            galaxy_no_pix_0,
            galaxy_no_pix_1,
            galaxy_no_pix_2,
        ]
    )

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    pixelization_grids = tracer_to_inversion.sparse_image_plane_grid_pg_list_from(
        grid=sub_grid_2d_7x7
    )

    assert pixelization_grids[0] == None
    assert pixelization_grids[1] == None
    assert (pixelization_grids[2] == np.array([[1.0, 1.0]])).all()
    assert pixelization_grids[3] == None
    assert (pixelization_grids[4] == np.array([[2.0, 2.0]])).all()


def test__traced_sparse_grid_pg_list_from(sub_grid_2d_7x7):

    # Test Multi plane

    galaxy_no_pix = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=0.5),
    )

    galaxy_pix_0 = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(
            data_pixelization_grid=al.Grid2D.manual_native(
                grid=[[[1.0, 0.0]]], pixel_scales=(1.0, 1.0)
            )
        ),
        regularization=al.m.MockRegularization(),
    )

    galaxy_pix_1 = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(
            data_pixelization_grid=al.Grid2D.manual_native(
                grid=[[[2.0, 0.0]]], pixel_scales=(1.0, 1.0)
            )
        ),
        regularization=al.m.MockRegularization(),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[galaxy_no_pix, galaxy_pix_0, galaxy_pix_1]
    )

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    traced_sparse_grids_list_of_planes, sparse_image_plane_grid_list = tracer_to_inversion.traced_sparse_grid_pg_list_from(
        grid=sub_grid_2d_7x7
    )

    assert traced_sparse_grids_list_of_planes[0] == None
    assert traced_sparse_grids_list_of_planes[1][0] == pytest.approx(
        np.array([[1.0 - 0.5, 0.0]]), 1.0e-4
    )
    assert traced_sparse_grids_list_of_planes[1][1] == pytest.approx(
        np.array([[2.0 - 0.5, 0.0]]), 1.0e-4
    )

    # Test Extra Galaxies

    galaxy_pix_0 = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(
            data_pixelization_grid=al.Grid2D.manual_native(
                grid=[[[1.0, 1.0]]], pixel_scales=(1.0, 1.0)
            )
        ),
        regularization=al.m.MockRegularization(),
    )

    galaxy_pix_1 = al.Galaxy(
        redshift=2.0,
        pixelization=al.m.MockPixelization(
            data_pixelization_grid=al.Grid2D.manual_native(
                grid=[[[2.0, 2.0]]], pixel_scales=(1.0, 1.0)
            )
        ),
        regularization=al.m.MockRegularization(),
    )

    galaxy_no_pix_0 = al.Galaxy(
        redshift=0.25,
        mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=0.5),
    )
    galaxy_no_pix_1 = al.Galaxy(redshift=0.5)
    galaxy_no_pix_2 = al.Galaxy(redshift=1.5)

    tracer = al.Tracer.from_galaxies(
        galaxies=[
            galaxy_pix_0,
            galaxy_pix_1,
            galaxy_no_pix_0,
            galaxy_no_pix_1,
            galaxy_no_pix_2,
        ]
    )

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    traced_sparse_grids_list_of_planes, sparse_image_plane_grid_list = tracer_to_inversion.traced_sparse_grid_pg_list_from(
        grid=sub_grid_2d_7x7
    )

    traced_grid_pix_0 = tracer.traced_grid_2d_list_from(grid=np.array([[1.0, 1.0]]))[2]
    traced_grid_pix_1 = tracer.traced_grid_2d_list_from(grid=np.array([[2.0, 2.0]]))[4]

    assert traced_sparse_grids_list_of_planes[0] == None
    assert traced_sparse_grids_list_of_planes[1] == None
    assert (traced_sparse_grids_list_of_planes[2][0] == traced_grid_pix_0).all()
    assert traced_sparse_grids_list_of_planes[3] == None
    assert (traced_sparse_grids_list_of_planes[4][0] == traced_grid_pix_1).all()


def test__mapper_galaxy_dict_from(sub_grid_2d_7x7):

    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_no_pix])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    mapper_galaxy_dict = tracer_to_inversion.mapper_galaxy_dict_from(
        grid=sub_grid_2d_7x7
    )
    assert mapper_galaxy_dict == {}

    galaxy_no_pix = al.Galaxy(redshift=0.5)
    galaxy_pix_0 = al.Galaxy(
        redshift=0.5,
        pixelization=al.m.MockPixelization(
            mapper=1, data_pixelization_grid=sub_grid_2d_7x7
        ),
        regularization=al.m.MockRegularization(),
    )

    galaxy_pix_1 = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(
            mapper=2, data_pixelization_grid=sub_grid_2d_7x7
        ),
        regularization=al.m.MockRegularization(),
    )
    galaxy_pix_2 = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(
            mapper=3, data_pixelization_grid=sub_grid_2d_7x7
        ),
        regularization=al.m.MockRegularization(),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[galaxy_no_pix, galaxy_pix_0, galaxy_pix_1, galaxy_pix_2]
    )

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    mapper_galaxy_dict = tracer_to_inversion.mapper_galaxy_dict_from(
        grid=sub_grid_2d_7x7
    )

    mapper_list = list(mapper_galaxy_dict.keys())

    assert mapper_galaxy_dict[mapper_list[0]] == galaxy_pix_0
    assert mapper_galaxy_dict[mapper_list[1]] == galaxy_pix_1
    assert mapper_galaxy_dict[mapper_list[2]] == galaxy_pix_2

    assert mapper_list == [1, 2, 3]

    galaxy_no_pix_0 = al.Galaxy(
        redshift=0.25,
        mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=0.5),
    )
    galaxy_no_pix_1 = al.Galaxy(redshift=0.5)
    galaxy_no_pix_2 = al.Galaxy(redshift=1.5)

    galaxy_pix_0 = al.Galaxy(
        redshift=0.75,
        pixelization=al.m.MockPixelization(
            mapper=1, data_pixelization_grid=sub_grid_2d_7x7
        ),
        regularization=al.m.MockRegularization(),
    )
    galaxy_pix_1 = al.Galaxy(
        redshift=2.0,
        pixelization=al.m.MockPixelization(
            mapper=2, data_pixelization_grid=sub_grid_2d_7x7
        ),
        regularization=al.m.MockRegularization(),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[
            galaxy_no_pix_0,
            galaxy_no_pix_1,
            galaxy_no_pix_2,
            galaxy_pix_0,
            galaxy_pix_1,
        ]
    )

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    mapper_galaxy_dict = tracer_to_inversion.mapper_galaxy_dict_from(
        grid=sub_grid_2d_7x7
    )

    mapper_list = list(mapper_galaxy_dict.keys())

    assert mapper_galaxy_dict[mapper_list[0]] == galaxy_pix_0
    assert mapper_galaxy_dict[mapper_list[1]] == galaxy_pix_1

    assert mapper_list == [1, 2]


def test__inversion_imaging_from(sub_grid_2d_7x7, masked_imaging_7x7):

    g_linear = al.Galaxy(redshift=0.5, light_linear=al.lp_linear.EllSersic())

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g_linear])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    inversion = tracer_to_inversion.inversion_imaging_from(
        dataset=masked_imaging_7x7,
        image=masked_imaging_7x7.image,
        noise_map=masked_imaging_7x7.noise_map,
        w_tilde=masked_imaging_7x7.w_tilde,
        settings_pixelization=al.SettingsPixelization(use_border=False),
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert inversion.reconstruction[0] == pytest.approx(0.002310, 1.0e-2)

    pix = al.pix.Rectangular(shape=(3, 3))
    reg = al.reg.Constant(coefficient=0.0)

    g0 = al.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    inversion = tracer_to_inversion.inversion_imaging_from(
        dataset=masked_imaging_7x7,
        image=masked_imaging_7x7.image,
        noise_map=masked_imaging_7x7.noise_map,
        w_tilde=masked_imaging_7x7.w_tilde,
        settings_pixelization=al.SettingsPixelization(use_border=False),
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert inversion.mapped_reconstructed_image == pytest.approx(
        masked_imaging_7x7.image, 1.0e-2
    )


def test__inversion_interferometer_from(sub_grid_2d_7x7, interferometer_7):

    interferometer_7.data = al.Visibilities.ones(shape_slim=(7,))

    g_linear = al.Galaxy(redshift=0.5, light_linear=al.lp_linear.EllSersic())

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g_linear])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    inversion = tracer_to_inversion.inversion_interferometer_from(
        dataset=interferometer_7,
        visibilities=interferometer_7.visibilities,
        noise_map=interferometer_7.noise_map,
        w_tilde=None,
        settings_pixelization=al.SettingsPixelization(use_border=False),
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert inversion.reconstruction[0] == pytest.approx(0.000513447, 1.0e-5)

    pix = al.pix.Rectangular(shape=(7, 7))
    reg = al.reg.Constant(coefficient=0.0)

    g0 = al.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5), g0])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    inversion = tracer_to_inversion.inversion_interferometer_from(
        dataset=interferometer_7,
        visibilities=interferometer_7.visibilities,
        noise_map=interferometer_7.noise_map,
        w_tilde=None,
        settings_pixelization=al.SettingsPixelization(use_border=False),
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert inversion.reconstruction[0] == pytest.approx(-0.2662, 1.0e-4)
