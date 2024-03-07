import numpy as np
import pytest
from os import path
from skimage import measure

import autofit as af
import autolens as al
from autoconf.dictable import from_json, output_to_json

grid_simple = al.Grid2DIrregular(values=[(1.0, 2.0)])


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


def test__traced_grid_2d_list_from(sub_grid_2d_7x7, sub_grid_2d_7x7_simple):
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


def test__grid_2d_at_redshift_from(sub_grid_2d_7x7):

    g0 = al.Galaxy(redshift=2.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=2.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g2 = al.Galaxy(redshift=0.1, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g3 = al.Galaxy(redshift=3.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g4 = al.Galaxy(redshift=1.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g5 = al.Galaxy(redshift=3.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))

    galaxies = [g0, g1, g2, g3, g4, g5]

    tracer = al.Tracer(galaxies=galaxies, cosmology=al.cosmo.Planck15())

    grid_at_redshift = tracer.grid_2d_at_redshift_from(grid=grid_simple, redshift=0.5)

    assert grid_at_redshift[0] == pytest.approx((9.73109691, 19.46219382), 1.0e-4)

    grid_at_redshift = tracer.grid_2d_at_redshift_from(grid=grid_simple, redshift=1.75)

    assert grid_at_redshift[0] == pytest.approx((0.65903649, 1.31807298), 1.0e-4)


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
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=2.0), mass_profile=al.mp.IsothermalSph(einstein_radius=2.0),)
    g2 = al.Galaxy(redshift=2.0, light_profile=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    image_list = tracer.image_2d_list_from(grid=grid_simple)

    assert image_list[0][0] == pytest.approx(0.0504608, 1.0e-4)
    assert image_list[1][0] == pytest.approx(0.2517025, 1.0e-4)
    assert image_list[2][0] == pytest.approx(1.8611933, 1.0e-4)
    assert len(image_list) == 3


def test__image_2d_list_from__plane_without_light_profile_is_zeros(sub_grid_2d_7x7, sub_grid_2d_7x7_simple):

    # Planes without light profiles give zeros

    g0 = al.Galaxy(redshift=0.1, light_profile=al.lp.Sersic(intensity=0.1))
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=0.2))
    g2 = al.Galaxy(redshift=2.0)

    tracer = al.Tracer(galaxies=[g0, g1, g2], cosmology=al.cosmo.Planck15())

    tracer_image_of_planes = tracer.image_2d_list_from(grid=sub_grid_2d_7x7)

    assert tracer_image_of_planes[2].shape_native == (7, 7)
    assert (tracer_image_of_planes[2].binned.native == np.zeros((7, 7))).all()


def test__image_2d_from__operated_only_input(
    sub_grid_2d_7x7, lp_0, lp_operated_0, mp_0
):

    galaxy_0 = al.Galaxy(
        redshift=0.5, light=lp_0, light_operated=lp_operated_0, mass=mp_0
    )

    galaxy_1 = al.Galaxy(
        redshift=1.0, light_operated_0=lp_operated_0, light_operated_1=lp_operated_0
    )
    galaxy_2 = al.Galaxy(redshift=2.0)

    tracer = al.Tracer(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    image = tracer.image_2d_from(grid=sub_grid_2d_7x7, operated_only=False)

    assert image[0] == pytest.approx(1.24579051, 1.0e-4)
    assert image[1] == pytest.approx(1.71100813, 1.0e-4)


def test__image_2d_from__sum_of_individual_images(sub_grid_2d_7x7, sub_grid_2d_7x7_simple):

    g0 = al.Galaxy(redshift=0.1, light_profile=al.lp.Sersic(intensity=0.1), mass=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=0.2))

    tracer = al.Tracer(galaxies=[g0, g1], cosmology=al.cosmo.Planck15())

    traced_grid_2d_list_from = tracer.traced_grid_2d_list_from(grid=sub_grid_2d_7x7)

    image = (
            g0.image_2d_from(grid=sub_grid_2d_7x7)
            + g1.image_2d_from(grid=traced_grid_2d_list_from[1])
    )

    image_tracer = tracer.image_2d_from(grid=sub_grid_2d_7x7)

    assert image.shape_native == (7, 7)
    assert image == pytest.approx(image_tracer, 1.0e-4)


def test__image_2d_via_input_plane_image_from__with_foreground_planes(sub_grid_2d_7x7):
    plane_grid = al.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.3, sub_size=4)

    g0 = al.Galaxy(
        redshift=0.5,
        light_profile=al.lp.Sersic(intensity=1.0),
        mass=al.mp.IsothermalSph(einstein_radius=0.2),
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=2.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    image_via_light_profile = tracer.image_2d_from(grid=sub_grid_2d_7x7)

    plane_image = g1.image_2d_from(grid=plane_grid)

    image_via_input_plane_image = tracer.image_2d_via_input_plane_image_from(
        grid=sub_grid_2d_7x7,
        plane_image=plane_image,
        plane_index=-1,
        include_other_planes=True,
    )

    assert image_via_light_profile.binned[0] == pytest.approx(
        image_via_input_plane_image.binned[0], 1.0e-2
    )



def test__image_2d_via_input_plane_image_from__without_foreground_planes(
    sub_grid_2d_7x7,
):
    g0 = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(einstein_radius=0.2),
        light_profile=al.lp.Sersic(intensity=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, light_profile=al.lp.Sersic(intensity=2.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    image_via_light_profile = tracer.image_2d_list_from(grid=sub_grid_2d_7x7)[-1]

    plane_grid = al.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.5, sub_size=4)

    plane_image = g1.image_2d_from(grid=plane_grid)

    image_via_input_plane_image = tracer.image_2d_via_input_plane_image_from(
        grid=sub_grid_2d_7x7,
        plane_image=plane_image,
        plane_index=-1,
        include_other_planes=False,
    )

    assert image_via_light_profile.binned[0] == pytest.approx(
        image_via_input_plane_image.binned[0], 1.0e-2
    )


def test__image_2d_via_input_plane_image_from__with_foreground_planes__multi_plane(
    sub_grid_2d_7x7,
):
    plane_grid = al.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.3, sub_size=4)

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

    image_via_light_profile = tracer.image_2d_from(grid=sub_grid_2d_7x7)

    plane_image = g2.image_2d_from(grid=plane_grid)

    image_via_input_plane_image = tracer.image_2d_via_input_plane_image_from(
        grid=sub_grid_2d_7x7,
        plane_image=plane_image,
        plane_index=-1,
        include_other_planes=True,
    )

    assert image_via_light_profile.binned[0] == pytest.approx(
        image_via_input_plane_image.binned[0], 1.0e-2
    )

    plane_image = g1.image_2d_from(grid=plane_grid)

    image_via_input_plane_image = tracer.image_2d_via_input_plane_image_from(
        grid=sub_grid_2d_7x7,
        plane_image=plane_image,
        plane_index=1,
        include_other_planes=True,
    )

    assert image_via_light_profile.binned[0] == pytest.approx(
        image_via_input_plane_image.binned[0], 1.0e-2
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


def test__galaxy_image_2d_dict_from(sub_grid_2d_7x7):
    g0 = al.Galaxy(redshift=0.5, light_profile=al.lp.Sersic(intensity=1.0))
    g1 = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
        light_profile=al.lp.Sersic(intensity=2.0),
    )

    g2 = al.Galaxy(redshift=0.5, light_profile=al.lp.Sersic(intensity=3.0))

    g3 = al.Galaxy(redshift=1.0, light_profile=al.lp_operated.Gaussian(intensity=5.0))

    g0_image = g0.image_2d_from(grid=sub_grid_2d_7x7)
    g1_image = g1.image_2d_from(grid=sub_grid_2d_7x7)
    g2_image = g2.image_2d_from(grid=sub_grid_2d_7x7)

    g1_deflections = g1.deflections_yx_2d_from(grid=sub_grid_2d_7x7)

    source_grid_2d_7x7 = sub_grid_2d_7x7 - g1_deflections

    g3_image = g3.image_2d_from(grid=source_grid_2d_7x7)

    tracer = al.Tracer(galaxies=[g3, g1, g0, g2], cosmology=al.cosmo.Planck15())

    galaxy_image_2d_dict = tracer.galaxy_image_2d_dict_from(grid=sub_grid_2d_7x7)

    assert (galaxy_image_2d_dict[g0] == g0_image).all()
    assert (galaxy_image_2d_dict[g1] == g1_image).all()
    assert (galaxy_image_2d_dict[g2] == g2_image).all()
    assert (galaxy_image_2d_dict[g3] == g3_image).all()

    galaxy_image_2d_dict = tracer.galaxy_image_2d_dict_from(
        grid=sub_grid_2d_7x7, operated_only=True
    )

    assert (galaxy_image_2d_dict[g0] == np.zeros(shape=(36,))).all()
    assert (galaxy_image_2d_dict[g1] == np.zeros(shape=(36,))).all()
    assert (galaxy_image_2d_dict[g2] == np.zeros(shape=(36,))).all()
    assert (galaxy_image_2d_dict[g3] == g3_image).all()

    galaxy_image_2d_dict = tracer.galaxy_image_2d_dict_from(
        grid=sub_grid_2d_7x7, operated_only=False
    )

    assert (galaxy_image_2d_dict[g0] == g0_image).all()
    assert (galaxy_image_2d_dict[g1] == g1_image).all()
    assert (galaxy_image_2d_dict[g2] == g2_image).all()
    assert (galaxy_image_2d_dict[g3] == np.zeros(shape=(36,))).all()