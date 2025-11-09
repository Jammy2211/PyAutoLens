import numpy as np
import pytest
from os import path

from autoconf.dictable import from_json, output_to_json
import autofit as af
import autolens as al


grid_simple = al.Grid2DIrregular(values=[(1.0, 2.0)])


def test__has():
    gal_lp = al.Galaxy(redshift=0.5, light_profile=al.LightProfile())
    gal_mp = al.Galaxy(redshift=0.5, mass_profile=al.mp.IsothermalSph())

    tracer = al.Tracer(galaxies=[gal_mp, gal_mp])

    assert tracer.has(cls=al.LightProfile) is False

    tracer = al.Tracer(galaxies=[gal_lp, gal_lp])

    assert tracer.has(cls=al.LightProfile) is True

    tracer = al.Tracer(galaxies=[gal_lp, gal_mp])

    assert tracer.has(cls=al.LightProfile) is True


def test__galaxies_ascending_redshift(grid_2d_7x7):
    g0 = al.Galaxy(redshift=0.5)
    g1 = al.Galaxy(redshift=0.5)

    tracer = al.Tracer(galaxies=[g0, g1])

    assert tracer.galaxies_ascending_redshift == [g0, g1]

    g2 = al.Galaxy(redshift=1.0)
    g3 = al.Galaxy(redshift=1.0)

    tracer = al.Tracer(galaxies=[g0, g1, g2, g3])

    assert tracer.galaxies_ascending_redshift == [g0, g1, g2, g3]

    g4 = al.Galaxy(redshift=0.75)
    g5 = al.Galaxy(redshift=1.5)

    tracer = al.Tracer(galaxies=[g0, g1, g2, g3, g4, g5])

    assert tracer.galaxies_ascending_redshift == [g0, g1, g4, g2, g3, g5]


def test__plane_redshifts():
    g1 = al.Galaxy(redshift=1)
    g2 = al.Galaxy(redshift=2)
    g3 = al.Galaxy(redshift=3)

    tracer = al.Tracer(galaxies=[g1, g2])

    assert tracer.plane_redshifts == [1, 2]

    tracer = al.Tracer(galaxies=[g2, g2, g3, g1, g1])

    assert tracer.plane_redshifts == [1, 2, 3]


def test__planes():
    g1 = al.Galaxy(redshift=1)
    g2 = al.Galaxy(redshift=2)
    g3 = al.Galaxy(redshift=3)

    tracer = al.Tracer(galaxies=[g1, g2])

    assert tracer.planes == [[g1], [g2]]

    tracer = al.Tracer(galaxies=[g2, g2, g3, g1, g1])

    assert tracer.planes == [[g1, g1], [g2, g2], [g3]]


def test__plane_index_via_redshift_from():

    g1 = al.Galaxy(redshift=1)
    g2 = al.Galaxy(redshift=2)
    g3 = al.Galaxy(redshift=3)

    tracer = al.Tracer(galaxies=[g1, g2, g3])

    assert tracer.plane_index_via_redshift_from(redshift=2.0) == 1
    assert tracer.plane_index_via_redshift_from(redshift=3.0) == 2
    assert tracer.plane_index_via_redshift_from(redshift=3.001) == None


def test__upper_plane_index_with_light_profile():
    g0 = al.Galaxy(redshift=0.5)
    g1 = al.Galaxy(redshift=1.0)
    g2 = al.Galaxy(redshift=2.0)
    g3 = al.Galaxy(redshift=3.0)

    g0_lp = al.Galaxy(redshift=0.5, light_profile=al.LightProfile())
    g1_lp = al.Galaxy(redshift=1.0, light_profile=al.LightProfile())
    g2_lp = al.Galaxy(redshift=2.0, light_profile=al.LightProfile())
    g3_lp = al.Galaxy(redshift=3.0, light_profile=al.LightProfile())

    tracer = al.Tracer(galaxies=[g1_lp])

    assert tracer.upper_plane_index_with_light_profile == 0

    tracer = al.Tracer(galaxies=[g0, g1_lp])

    assert tracer.upper_plane_index_with_light_profile == 1

    tracer = al.Tracer(galaxies=[g0_lp, g1_lp, g2_lp])

    assert tracer.upper_plane_index_with_light_profile == 2

    tracer = al.Tracer(galaxies=[g0_lp, g1, g2, g3_lp])

    assert tracer.upper_plane_index_with_light_profile == 3

    tracer = al.Tracer(galaxies=[g0_lp, g1, g2_lp, g3])

    assert tracer.upper_plane_index_with_light_profile == 2


def test__traced_grid_2d_list_from(grid_2d_7x7, grid_2d_7x7_simple):
    g0 = al.Galaxy(redshift=2.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=2.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g2 = al.Galaxy(redshift=0.1, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g3 = al.Galaxy(redshift=3.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g4 = al.Galaxy(redshift=1.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g5 = al.Galaxy(redshift=3.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))

    galaxies = [g0, g1, g2, g3, g4, g5]

    tracer = al.Tracer(galaxies=galaxies, cosmology=al.cosmo.Planck15())

    traced_grid_list = tracer.traced_grid_2d_list_from(
        grid=grid_simple,
    )

    assert traced_grid_list[0][0] == pytest.approx((1.0, 2.0), 1.0e-4)
    assert traced_grid_list[1][0] == pytest.approx((0.58194284, 1.16388568), 1.0e-4)
    assert traced_grid_list[2][0] == pytest.approx((0.22277247, 0.4455449), 1.0e-4)
    assert traced_grid_list[3][0] == pytest.approx((-0.78885438, -1.57770876), 1.0e-4)
    assert len(traced_grid_list) == 4

    traced_grid_list = tracer.traced_grid_2d_list_from(
        grid=grid_simple, plane_index_limit=1
    )

    assert traced_grid_list[0][0] == pytest.approx((1.0, 2.0), 1.0e-4)
    assert traced_grid_list[-1][0] == pytest.approx((0.58194284, 1.16388568), 1.0e-4)
    assert len(traced_grid_list) == 2


def test__grid_2d_at_redshift_from(grid_2d_7x7):
    g0 = al.Galaxy(redshift=2.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=2.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g2 = al.Galaxy(redshift=0.1, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g3 = al.Galaxy(redshift=3.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g4 = al.Galaxy(redshift=1.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g5 = al.Galaxy(redshift=3.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))

    galaxies = [g0, g1, g2, g3, g4, g5]

    tracer = al.Tracer(galaxies=galaxies, cosmology=al.cosmo.Planck15())

    galaxies_plus_extra = [g0, g1, g2, g3, g4, g5, al.Galaxy(redshift=0.5)]
    tracer_plus_extra = al.Tracer(
        galaxies=galaxies_plus_extra, cosmology=al.cosmo.Planck15()
    )
    grid_2d_list_from = tracer_plus_extra.traced_grid_2d_list_from(grid=grid_simple)

    grid_at_redshift = tracer.grid_2d_at_redshift_from(grid=grid_simple, redshift=0.5)

    assert grid_2d_list_from[1] == pytest.approx(grid_at_redshift.array, 1.0e-4)
    assert grid_at_redshift[0] == pytest.approx((0.6273814, 1.2547628), 1.0e-4)

    galaxies_plus_extra = [g0, g1, g2, g3, g4, g5, al.Galaxy(redshift=1.75)]
    tracer_plus_extra = al.Tracer(
        galaxies=galaxies_plus_extra, cosmology=al.cosmo.Planck15()
    )
    grid_2d_list_from = tracer_plus_extra.traced_grid_2d_list_from(grid=grid_simple)

    grid_at_redshift = tracer.grid_2d_at_redshift_from(grid=grid_simple, redshift=1.75)

    assert grid_2d_list_from[2] == pytest.approx(grid_at_redshift.array, 1.0e-4)
    assert grid_at_redshift[0] == pytest.approx((0.27331481161, 0.5466296232), 1.0e-4)

    galaxies_plus_extra = [g0, g1, g2, g3, g4, g5, al.Galaxy(redshift=2.0)]
    tracer_plus_extra = al.Tracer(
        galaxies=galaxies_plus_extra, cosmology=al.cosmo.Planck15()
    )
    grid_2d_list_from = tracer_plus_extra.traced_grid_2d_list_from(grid=grid_simple)

    grid_at_redshift = tracer.grid_2d_at_redshift_from(grid=grid_simple, redshift=2.0)

    assert grid_2d_list_from[2] == pytest.approx(grid_at_redshift.array, 1.0e-4)
    assert grid_at_redshift[0] == pytest.approx((0.222772465, 0.445544931), 1.0e-4)


def test__image_2d_list_from():
    g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.Sersic(intensity=1.0))
    g1 = al.Galaxy(redshift=0.5, light_profile=al.lp.Sersic(intensity=2.0))
    g2 = al.Galaxy(redshift=0.5, light_profile=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    image_list = tracer.image_2d_list_from(grid=grid_simple)

    assert image_list[0][0] == pytest.approx(0.30276535, 1.0e-4)
    assert len(image_list) == 1

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(
        redshift=1.0,
        light_profile=al.lp.Sersic(intensity=2.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=2.0),
    )
    g2 = al.Galaxy(redshift=2.0, light_profile=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    image_list = tracer.image_2d_list_from(grid=grid_simple)

    assert image_list[0][0] == pytest.approx(0.0504608, 1.0e-4)
    assert image_list[1][0] == pytest.approx(0.2517025, 1.0e-4)
    assert image_list[2][0] == pytest.approx(1.8611933, 1.0e-4)
    assert len(image_list) == 3


def test__image_2d_list_from__plane_without_light_profile_is_zeros(
    grid_2d_7x7,
):
    # Planes without light profiles give zeros

    g0 = al.Galaxy(redshift=0.1, light_profile=al.lp.Sersic(intensity=0.1))
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=0.2))
    g2 = al.Galaxy(redshift=2.0)

    tracer = al.Tracer(galaxies=[g0, g1, g2], cosmology=al.cosmo.Planck15())

    tracer_image_of_planes = tracer.image_2d_list_from(grid=grid_2d_7x7)

    assert tracer_image_of_planes[2].shape_native == (7, 7)
    assert (tracer_image_of_planes[2].native == np.zeros((7, 7))).all()


def test__image_2d_from__operated_only_input(grid_2d_7x7, lp_0, lp_operated_0, mp_0):
    galaxy_0 = al.Galaxy(
        redshift=0.5, light=lp_0, light_operated=lp_operated_0, mass=mp_0
    )

    galaxy_1 = al.Galaxy(
        redshift=1.0, light_operated_0=lp_operated_0, light_operated_1=lp_operated_0
    )
    galaxy_2 = al.Galaxy(redshift=2.0)

    tracer = al.Tracer(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    image = tracer.image_2d_from(grid=grid_2d_7x7, operated_only=False)

    assert image[0] == pytest.approx(1.79362512, 1.0e-4)
    assert image[1] == pytest.approx(2.93152589, 1.0e-4)


def test__image_2d_from__sum_of_individual_images(mask_2d_7x7):
    grid_2d_7x7 = al.Grid2D.from_mask(mask=mask_2d_7x7, over_sample_size=2)

    g0 = al.Galaxy(
        redshift=0.1,
        light_profile=al.lp.Sersic(intensity=0.1),
        mass=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=0.2))

    tracer = al.Tracer(galaxies=[g0, g1], cosmology=al.cosmo.Planck15())

    traced_grid_2d_list = tracer.traced_grid_2d_list_from(
        grid=al.Grid2DIrregular(grid_2d_7x7.over_sampled)
    )

    image = g0.image_2d_from(grid=traced_grid_2d_list[0]) + g1.image_2d_from(
        grid=traced_grid_2d_list[1]
    )

    image = grid_2d_7x7.over_sampler.binned_array_2d_from(array=image)

    image_tracer = tracer.image_2d_from(grid=grid_2d_7x7)

    assert image.shape_native == (7, 7)
    assert image == pytest.approx(image_tracer.array, 1.0e-4)


def test__image_2d_via_input_plane_image_from__with_foreground_planes(grid_2d_7x7):
    plane_grid = al.Grid2D.uniform(
        shape_native=(40, 40), pixel_scales=0.3, over_sample_size=1
    )

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.Sersic(intensity=1.0),
        mass=al.mp.IsothermalSph(einstein_radius=0.2),
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=2.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    image_via_light_profile = tracer.image_2d_from(grid=grid_2d_7x7)

    plane_image = g1.image_2d_from(grid=plane_grid)

    image_via_input_plane_image = tracer.image_2d_via_input_plane_image_from(
        grid=grid_2d_7x7,
        plane_image=plane_image,
        plane_index=-1,
        include_other_planes=True,
    )

    assert image_via_light_profile[0] == pytest.approx(
        image_via_input_plane_image[0], 1.0e-2
    )


def test__image_2d_via_input_plane_image_from__without_foreground_planes(
    grid_2d_7x7,
):
    grid_2d_7x7 = al.Grid2D(
        values=grid_2d_7x7,
        mask=grid_2d_7x7.mask,
        over_sample_size=2,
    )

    g0 = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(einstein_radius=0.2),
        light_profile=al.lp.Sersic(intensity=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=2.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    image_via_light_profile = tracer.image_2d_list_from(grid=grid_2d_7x7)[-1]

    plane_grid = al.Grid2D.uniform(shape_native=(80, 80), pixel_scales=0.125)

    plane_image = g1.image_2d_from(
        grid=plane_grid,
    )

    image_via_input_plane_image = tracer.image_2d_via_input_plane_image_from(
        grid=grid_2d_7x7,
        plane_image=plane_image,
        plane_index=-1,
        include_other_planes=False,
    )

    assert image_via_light_profile[0] == pytest.approx(
        image_via_input_plane_image[0], 1.0e-2
    )


def test__image_2d_via_input_plane_image_from__with_foreground_planes__multi_plane(
    grid_2d_7x7,
):
    plane_grid = al.Grid2D.uniform(
        shape_native=(40, 40), pixel_scales=0.3, over_sample_size=1
    )

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.Sersic(intensity=1.0),
        mass=al.mp.IsothermalSph(einstein_radius=0.2),
    )
    g1 = al.Galaxy(
        redshift=1.0,
        light_profile=al.lp.Sersic(intensity=2.0),
        mass=al.mp.IsothermalSph(einstein_radius=0.2),
    )
    g2 = al.Galaxy(redshift=2.0, light_profile=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    image_via_light_profile = tracer.image_2d_from(grid=grid_2d_7x7)

    plane_image = g2.image_2d_from(grid=plane_grid)

    image_via_input_plane_image = tracer.image_2d_via_input_plane_image_from(
        grid=grid_2d_7x7,
        plane_image=plane_image,
        plane_index=-1,
        include_other_planes=True,
    )

    assert image_via_light_profile[0] == pytest.approx(
        image_via_input_plane_image[0], 1.0e-2
    )

    plane_image = g1.image_2d_from(grid=plane_grid)

    image_via_input_plane_image = tracer.image_2d_via_input_plane_image_from(
        grid=grid_2d_7x7,
        plane_image=plane_image,
        plane_index=1,
        include_other_planes=True,
    )

    assert image_via_light_profile[0] == pytest.approx(
        image_via_input_plane_image[0], 1.0e-2
    )


def test__padded_image_2d_from(grid_2d_7x7):
    padded_grid = grid_2d_7x7.padded_grid_from(kernel_shape_native=(3, 3))

    g0 = al.Galaxy(redshift=0.1, light_profile=al.lp.Sersic(intensity=0.1))
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=0.2))
    g2 = al.Galaxy(redshift=2.0, light_profile=al.lp.Sersic(intensity=0.3))

    padded_g0_image = g0.image_2d_from(grid=padded_grid)

    padded_g1_image = g1.image_2d_from(grid=padded_grid)

    padded_g2_image = g2.image_2d_from(grid=padded_grid)

    tracer = al.Tracer(galaxies=[g0, g1, g2], cosmology=al.cosmo.Planck15())

    padded_tracer_image = tracer.padded_image_2d_from(
        grid=grid_2d_7x7, psf_shape_2d=(3, 3)
    )

    assert padded_tracer_image.shape_native == (9, 9)
    assert padded_tracer_image == pytest.approx(
        padded_g0_image.array + padded_g1_image.array + padded_g2_image.array, 1.0e-4
    )


def test__light_profile_snr__signal_to_noise_via_simulator_correct():
    background_sky_level = 10.0
    exposure_time = 300.0

    grid = al.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    mass = al.mp.IsothermalSph(einstein_radius=1.0)

    sersic = al.lp_snr.Sersic(signal_to_noise_ratio=10.0, effective_radius=0.01)

    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(redshift=0.5, mass=mass),
            al.Galaxy(redshift=1.0, light=sersic),
        ]
    )

    psf = al.Kernel2D.no_mask(values=[[1.0]], pixel_scales=1.0)

    simulator = al.SimulatorImaging(
        psf=psf,
        exposure_time=exposure_time,
        noise_seed=1,
        background_sky_level=background_sky_level,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    assert 8.0 < dataset.signal_to_noise_map.native[0, 1] < 12.0
    assert 8.0 < dataset.signal_to_noise_map.native[1, 0] < 12.0
    assert 8.0 < dataset.signal_to_noise_map.native[1, 2] < 12.0
    assert 8.0 < dataset.signal_to_noise_map.native[2, 1] < 12.0


def test__galaxy_image_2d_dict_from(grid_2d_7x7, mask_2d_7x7):
    grid_2d_7x7 = al.Grid2D.from_mask(mask=mask_2d_7x7, over_sample_size=2)

    g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.Sersic(intensity=1.0))
    g1 = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
        light_profile=al.lp.Sersic(intensity=2.0),
    )

    g2 = al.Galaxy(redshift=0.5, light_profile=al.lp.Sersic(intensity=3.0))

    g3 = al.Galaxy(redshift=1.0, light_profile=al.lp_operated.Gaussian(intensity=5.0))

    g0_image = g0.image_2d_from(grid=grid_2d_7x7)
    g1_image = g1.image_2d_from(grid=grid_2d_7x7)
    g2_image = g2.image_2d_from(grid=grid_2d_7x7)

    tracer = al.Tracer(galaxies=[g3, g1, g0, g2], cosmology=al.cosmo.Planck15())

    galaxy_image_2d_dict = tracer.galaxy_image_2d_dict_from(grid=grid_2d_7x7)

    assert (galaxy_image_2d_dict[g0] == g0_image).all()
    assert (galaxy_image_2d_dict[g1] == g1_image).all()
    assert (galaxy_image_2d_dict[g2] == g2_image).all()
    assert galaxy_image_2d_dict[g3][0] == pytest.approx(4.429962294298565, 1.0e-4)

    galaxy_image_2d_dict = tracer.galaxy_image_2d_dict_from(
        grid=grid_2d_7x7, operated_only=True
    )

    assert (galaxy_image_2d_dict[g0] == np.zeros(shape=(9,))).all()
    assert (galaxy_image_2d_dict[g1] == np.zeros(shape=(9,))).all()
    assert (galaxy_image_2d_dict[g2] == np.zeros(shape=(9,))).all()
    assert galaxy_image_2d_dict[g3][0] == pytest.approx(4.429962294298565, 1.0e-4)

    galaxy_image_2d_dict = tracer.galaxy_image_2d_dict_from(
        grid=grid_2d_7x7, operated_only=False
    )

    assert (galaxy_image_2d_dict[g0] == g0_image).all()
    assert (galaxy_image_2d_dict[g1] == g1_image).all()
    assert (galaxy_image_2d_dict[g2] == g2_image).all()
    assert (galaxy_image_2d_dict[g3] == np.zeros(shape=(9,))).all()
    assert galaxy_image_2d_dict[g2][0] == pytest.approx(0.5244575148617125, 1.0e-4)


def test__convergence_2d_from(grid_2d_7x7):
    g0 = al.Galaxy(redshift=0.5, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=0.5, mass_profile=al.mp.IsothermalSph(einstein_radius=2.0))
    g2 = al.Galaxy(redshift=1.0, mass_profile=al.mp.IsothermalSph(einstein_radius=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    convergence = tracer.convergence_2d_from(grid=grid_simple)

    assert convergence == pytest.approx(1.34164079, 1.0e-4)

    # No Galaxy with mass profile

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=0.5)])

    assert (
        tracer.convergence_2d_from(grid=grid_2d_7x7).native == np.zeros(shape=(7, 7))
    ).all()


def test__potential_2d_from(grid_2d_7x7):
    g0 = al.Galaxy(redshift=0.5, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=0.5, mass_profile=al.mp.IsothermalSph(einstein_radius=2.0))
    g2 = al.Galaxy(redshift=1.0, mass_profile=al.mp.IsothermalSph(einstein_radius=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    potential = tracer.potential_2d_from(grid=grid_simple)

    assert potential == pytest.approx(13.4164078, 1.0e-4)

    # No Galaxy with mass profile

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=0.5)])

    assert (
        tracer.potential_2d_from(grid=grid_2d_7x7).native == np.zeros(shape=(7, 7))
    ).all()


def test__time_delays_from():

    grid = al.Grid2DIrregular(values=[(0.7, 0.5), (1.0, 1.0)])

    mp = al.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    g0 = al.Galaxy(redshift=0.2, mass=mp)
    g1 = al.Galaxy(redshift=0.7)

    tracer = al.Tracer(galaxies=[g0, g1])

    time_delay = tracer.time_delays_from(grid=grid)

    assert time_delay == pytest.approx(np.array([8.52966247, -29.0176387]), 1.0e-4)


def test__deflections_of_planes_summed_from(grid_2d_7x7):
    g0 = al.Galaxy(redshift=0.5, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=0.5, mass_profile=al.mp.IsothermalSph(einstein_radius=2.0))
    g2 = al.Galaxy(redshift=1.0, mass_profile=al.mp.IsothermalSph(einstein_radius=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    deflections = tracer.deflections_of_planes_summed_from(grid=grid_simple)

    assert deflections[0] == pytest.approx((2.68328157, 5.36656315), 1.0e-4)

    # Single plane case

    g0 = al.Galaxy(redshift=0.5, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))

    tracer = al.Tracer(galaxies=[g0])

    deflections = tracer.deflections_of_planes_summed_from(grid=grid_simple)

    assert deflections[0] == pytest.approx((0.4472135954999, 0.8944271909), 1.0e-4)

    # No Galaxy With Mass Profile

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=0.5)])

    tracer_deflections = tracer.deflections_of_planes_summed_from(grid=grid_2d_7x7)

    assert (tracer_deflections.native[:, :, 0] == np.zeros(shape=(7, 7))).all()
    assert (tracer_deflections.native[:, :, 1] == np.zeros(shape=(7, 7))).all()


def test__extract_attribute():
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)])

    values = tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="value")

    assert values == None

    g0 = al.Galaxy(
        redshift=0.5, mp_0=al.m.MockMassProfile(value=0.9, value1=(1.0, 1.0))
    )
    g1 = al.Galaxy(
        redshift=0.5, mp_0=al.m.MockMassProfile(value=0.8, value1=(2.0, 2.0))
    )

    tracer = al.Tracer(galaxies=[g0, g1])

    values = tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="value")

    assert values.in_list == [0.9, 0.8]

    values = tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="value1")

    assert values.in_list == [(1.0, 1.0), (2.0, 2.0)]

    g2 = al.Galaxy(
        redshift=0.5,
        mp_0=al.m.MockMassProfile(value=0.7),
        mp_1=al.m.MockMassProfile(value=0.6),
    )

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    values = tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="value")

    assert values.in_list == [0.9, 0.8, 0.7, 0.6]

    tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="incorrect_value")


def test__extract_attributes_of_plane():
    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)])

    values = tracer.extract_attributes_of_planes(
        cls=al.mp.MassProfile, attr_name="value"
    )

    assert values == [None, None]

    g0 = al.Galaxy(
        redshift=0.5, mp_0=al.m.MockMassProfile(value=0.9, value1=(1.0, 1.0))
    )
    g1 = al.Galaxy(
        redshift=0.75, mp_0=al.m.MockMassProfile(value=0.8, value1=(2.0, 2.0))
    )
    g2 = al.Galaxy(
        redshift=1.0,
        mp_0=al.m.MockMassProfile(value=0.7),
        mp_1=al.m.MockMassProfile(value=0.6),
    )

    tracer = al.Tracer(galaxies=[g0, g1])

    values = tracer.extract_attributes_of_planes(
        cls=al.mp.MassProfile, attr_name="value"
    )

    print(values)

    assert values[0].in_list == [0.9]
    assert values[1].in_list == [0.8]

    values = tracer.extract_attributes_of_planes(
        cls=al.mp.MassProfile, attr_name="value1"
    )

    assert values[0].in_list == [(1.0, 1.0)]
    assert values[1].in_list == [(2.0, 2.0)]

    tracer = al.Tracer(galaxies=[g0, g1, al.Galaxy(redshift=0.25), g2])

    values = tracer.extract_attributes_of_planes(
        cls=al.mp.MassProfile, attr_name="value", filter_nones=False
    )

    assert values[0] == None
    assert values[1] == 0.9
    assert values[2] == 0.8
    assert values[3].in_list == [0.7, 0.6]

    values = tracer.extract_attributes_of_planes(
        cls=al.mp.MassProfile, attr_name="value", filter_nones=True
    )

    assert values[0] == 0.9
    assert values[1] == 0.8
    assert values[2].in_list == [0.7, 0.6]

    tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="incorrect_value")


def test__extract_attributes_of_galaxies():
    g0 = al.Galaxy(
        redshift=0.5, mp_0=al.m.MockMassProfile(value=0.9, value1=(1.0, 1.0))
    )
    g1 = al.Galaxy(
        redshift=0.5, mp_0=al.m.MockMassProfile(value=0.8, value1=(2.0, 2.0))
    )
    g2 = al.Galaxy(
        redshift=0.5,
        mp_0=al.m.MockMassProfile(value=0.7),
        mp_1=al.m.MockMassProfile(value=0.6),
    )

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)])

    values = tracer.extract_attributes_of_galaxies(
        cls=al.mp.MassProfile, attr_name="value"
    )

    assert values == [None, None]

    tracer = al.Tracer(galaxies=[g0, g1])

    values = tracer.extract_attributes_of_galaxies(
        cls=al.mp.MassProfile, attr_name="value"
    )

    assert values[0].in_list == [0.9]
    assert values[1].in_list == [0.8]

    values = tracer.extract_attributes_of_galaxies(
        cls=al.mp.MassProfile, attr_name="value1"
    )

    assert values[0].in_list == [(1.0, 1.0)]
    assert values[1].in_list == [(2.0, 2.0)]

    tracer = al.Tracer(galaxies=[g0, g1, al.Galaxy(redshift=0.5), g2])

    values = tracer.extract_attributes_of_galaxies(
        cls=al.mp.MassProfile, attr_name="value", filter_nones=False
    )

    assert values[0].in_list == [0.9]
    assert values[1].in_list == [0.8]
    assert values[2] == None
    assert values[3].in_list == [0.7, 0.6]

    values = tracer.extract_attributes_of_galaxies(
        cls=al.mp.MassProfile, attr_name="value", filter_nones=True
    )

    assert values[0].in_list == [0.9]
    assert values[1].in_list == [0.8]
    assert values[2].in_list == [0.7, 0.6]

    tracer.extract_attribute(cls=al.mp.MassProfile, attr_name="incorrect_value")


def test__extract_profile():
    g0 = al.Galaxy(
        redshift=0.5, mp_0=al.m.MockMassProfile(value=0.9, value1=(1.0, 1.0))
    )
    g1 = al.Galaxy(
        redshift=0.5, mp_1=al.m.MockMassProfile(value=0.8, value1=(2.0, 2.0))
    )
    g2 = al.Galaxy(
        redshift=1.0,
        mp_2=al.m.MockMassProfile(value=0.7),
        mp_3=al.m.MockMassProfile(value=0.6),
    )

    tracer = al.Tracer(galaxies=[g0, g1, g2], cosmology=None)

    profile = tracer.extract_profile(profile_name="mp_0")

    assert profile.value == 0.9

    profile = tracer.extract_profile(profile_name="mp_3")

    assert profile.value == 0.6


def test__extract_plane_index_of_profile():
    g0 = al.Galaxy(
        redshift=0.5, mp_0=al.m.MockMassProfile(value=0.9, value1=(1.0, 1.0))
    )
    g1 = al.Galaxy(
        redshift=0.75, mp_1=al.m.MockMassProfile(value=0.8, value1=(2.0, 2.0))
    )
    g2 = al.Galaxy(
        redshift=1.0,
        mp_2=al.m.MockMassProfile(value=0.7),
        mp_3=al.m.MockMassProfile(value=0.6),
    )

    tracer = al.Tracer(galaxies=[g0, g1, g2], cosmology=None)

    plane_index = tracer.extract_plane_index_of_profile(profile_name="mp_0")

    assert plane_index == 0

    plane_index = tracer.extract_plane_index_of_profile(profile_name="mp_1")

    assert plane_index == 1

    plane_index = tracer.extract_plane_index_of_profile(profile_name="mp_3")

    assert plane_index == 2


def test__sliced_tracer_from(grid_2d_7x7, grid_2d_7x7_simple):
    lens_g0 = al.Galaxy(redshift=0.5)
    source_g0 = al.Galaxy(redshift=2.0)
    los_g0 = al.Galaxy(redshift=0.1)
    los_g1 = al.Galaxy(redshift=0.2)
    los_g2 = al.Galaxy(redshift=0.4)
    los_g3 = al.Galaxy(redshift=0.6)

    tracer = al.Tracer.sliced_tracer_from(
        lens_galaxies=[lens_g0],
        line_of_sight_galaxies=[los_g0, los_g1, los_g2, los_g3],
        source_galaxies=[source_g0],
        planes_between_lenses=[1, 1],
        cosmology=al.cosmo.Planck15(),
    )

    assert tracer.planes[0] == [los_g0, los_g1]
    assert tracer.planes[1] == [lens_g0, los_g2, los_g3]
    assert tracer.planes[2] == [source_g0]


def test__regression__centre_of_profile_in_right_place():
    grid = al.Grid2D.uniform(shape_native=(7, 7), pixel_scales=1.0)

    g0 = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(centre=(1.9999, 0.9999), einstein_radius=1.0),
    )

    tracer = al.Tracer(galaxies=[g0, al.Galaxy(redshift=1.0)])

    convergence = tracer.convergence_2d_from(grid=grid)
    max_indexes = np.unravel_index(
        convergence.native.argmax(), convergence.shape_native
    )
    assert max_indexes == (1, 4)

    potential = tracer.potential_2d_from(grid=grid)
    max_indexes = np.unravel_index(potential.native.argmin(), potential.shape_native)
    assert max_indexes == (1, 4)

    deflections = tracer.deflections_yx_2d_from(grid=grid)
    assert deflections.native[1, 4, 0] > 0
    assert deflections.native[2, 4, 0] < 0
    assert deflections.native[1, 4, 1] > 0
    assert deflections.native[1, 3, 1] < 0


def test__instance_into_tracer__retains_dictionary_access():
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                light=al.lp.SersicSph(intensity=2.0),
                mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            source=al.Galaxy(redshift=1.0),
        )
    )

    instance = model.instance_from_prior_medians()

    tracer = al.Tracer(galaxies=instance.galaxies)

    assert tracer.galaxies.lens.light.intensity == 2.0


def test__output_to_and_load_from_json():
    json_file = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "tracer.json"
    )

    g0 = al.Galaxy(redshift=0.5, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=1.0)

    tracer = al.Tracer(galaxies=[g0, g1], cosmology=al.cosmo.wrap.Planck15())

    output_to_json(tracer, file_path=json_file)

    tracer_from_json = from_json(file_path=json_file)

    assert tracer_from_json.galaxies[0].redshift == 0.5
    assert tracer_from_json.galaxies[1].redshift == 1.0
    assert tracer_from_json.galaxies[0].mass_profile.einstein_radius == 1.0
