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