import numpy as np
import pytest
from os import path

import autolens as al

test_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


def test__pixelization_pg_list(sub_grid_2d_7x7):
    galaxy_pix = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(mapper=1),
        regularization=al.m.MockRegularization(),
    )
    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_pix])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    assert tracer_to_inversion.pixelization_pg_list[0] == []
    assert tracer_to_inversion.pixelization_pg_list[1][0].mapper == 1

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

    assert tracer_to_inversion.pixelization_pg_list[0][0].mapper == 1
    assert tracer_to_inversion.pixelization_pg_list[1][0].mapper == 2
    assert tracer_to_inversion.pixelization_pg_list[1][1].mapper == 3

    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_no_pix])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    assert tracer_to_inversion.pixelization_pg_list == [[]]


def test__regularization_pg_list(sub_grid_2d_7x7):

    galaxy_reg = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(),
        regularization=al.m.MockRegularization(regularization_matrix=1),
    )
    galaxy_no_reg = al.Galaxy(redshift=0.5)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_reg, galaxy_reg])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    assert tracer_to_inversion.regularization_pg_list[0] == []
    assert tracer_to_inversion.regularization_pg_list[1][0].regularization_matrix == 1
    assert tracer.regularization_list[0].regularization_matrix == 1

    galaxy_reg_0 = al.Galaxy(
        redshift=0.5,
        pixelization=al.m.MockPixelization(),
        regularization=al.m.MockRegularization(regularization_matrix=1),
    )

    galaxy_reg_1 = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(),
        regularization=al.m.MockRegularization(regularization_matrix=2),
    )

    galaxy_reg_2 = al.Galaxy(
        redshift=1.0,
        pixelization=al.m.MockPixelization(),
        regularization=al.m.MockRegularization(regularization_matrix=3),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[galaxy_reg_0, galaxy_reg_1, galaxy_reg_2]
    )

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    assert tracer_to_inversion.regularization_pg_list[0][0].regularization_matrix == 1
    assert tracer_to_inversion.regularization_pg_list[1][0].regularization_matrix == 2
    assert tracer_to_inversion.regularization_pg_list[1][1].regularization_matrix == 3
    assert tracer.regularization_list[0].regularization_matrix == 1
    assert tracer.regularization_list[1].regularization_matrix == 2
    assert tracer.regularization_list[2].regularization_matrix == 3

    galaxy_no_reg = al.Galaxy(redshift=0.5)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_reg, galaxy_no_reg])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    assert tracer_to_inversion.regularization_pg_list == [[]]


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


def test__light_profile_linear_func_list_from(sub_grid_2d_7x7, blurring_grid_2d_7x7):

    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_no_pix])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    light_profile_linear_func_list = tracer_to_inversion.light_profile_linear_func_list_from(
        grid=sub_grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7
    )

    assert light_profile_linear_func_list == []

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

    light_profile_linear_func_list = tracer_to_inversion.light_profile_linear_func_list_from(
        grid=sub_grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7
    )

    assert light_profile_linear_func_list[0].light_profile == lp_linear_0
    assert light_profile_linear_func_list[1].light_profile == lp_linear_1
    assert light_profile_linear_func_list[2].light_profile == lp_linear_2

    traced_grid_list = tracer.traced_grid_2d_list_from(grid=sub_grid_2d_7x7)

    assert light_profile_linear_func_list[0].grid == pytest.approx(
        sub_grid_2d_7x7, 1.0e-4
    )
    assert light_profile_linear_func_list[1].grid == pytest.approx(
        traced_grid_list[1], 1.0e-4
    )
    assert light_profile_linear_func_list[2].grid == pytest.approx(
        traced_grid_list[2], 1.0e-4
    )


def test__mapper_list_from(sub_grid_2d_7x7):

    galaxy_no_pix = al.Galaxy(redshift=0.5)

    tracer = al.Tracer.from_galaxies(galaxies=[galaxy_no_pix, galaxy_no_pix])

    tracer_to_inversion = al.TracerToInversion(tracer=tracer)

    mappers_of_planes = tracer_to_inversion.mapper_list_from(grid=sub_grid_2d_7x7)
    assert mappers_of_planes == []

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

    mapper_list = tracer_to_inversion.mapper_list_from(grid=sub_grid_2d_7x7)

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

    mapper_list = tracer_to_inversion.mapper_list_from(grid=sub_grid_2d_7x7)

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