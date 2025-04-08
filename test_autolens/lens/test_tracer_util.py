import numpy as np
import pytest

import autolens as al


def test__traced_grid_2d_list_from(grid_2d_7x7_simple):
    g0 = al.Galaxy(redshift=2.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=2.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g2 = al.Galaxy(redshift=0.1, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g3 = al.Galaxy(redshift=3.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g4 = al.Galaxy(redshift=1.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))
    g5 = al.Galaxy(redshift=3.0, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0))

    galaxies = [g0, g1, g2, g3, g4, g5]

    planes = al.util.tracer.planes_from(galaxies=galaxies)

    traced_grid_list = al.util.tracer.traced_grid_2d_list_from(
        planes=planes, grid=grid_2d_7x7_simple, cosmology=al.cosmo.Planck15()
    )

    # The scaling factors are as follows and were computed independently from the test_autoarray.
    beta_01 = 0.9348
    beta_02 = 0.9839601
    beta_12 = 0.7539734

    val = np.sqrt(2) / 2.0

    assert traced_grid_list[0][0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
    assert traced_grid_list[0][1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

    assert traced_grid_list[1][0] == pytest.approx(
        np.array([(1.0 - beta_01 * val), (1.0 - beta_01 * val)]), 1e-4
    )
    assert traced_grid_list[1][1] == pytest.approx(
        np.array([(1.0 - beta_01 * 1.0), 0.0]), 1e-4
    )

    defl11 = g0.deflections_yx_2d_from(
        grid=al.Grid2DIrregular([[(1.0 - beta_01 * val), (1.0 - beta_01 * val)]])
    )
    defl12 = g0.deflections_yx_2d_from(
        grid=al.Grid2DIrregular([[(1.0 - beta_01 * 1.0), 0.0]])
    )

    assert traced_grid_list[2][0] == pytest.approx(
        np.array(
            [
                (1.0 - beta_02 * val - beta_12 * defl11.array[0, 0]),
                (1.0 - beta_02 * val - beta_12 * defl11.array[0, 1]),
            ]
        ),
        1e-4,
    )
    assert traced_grid_list[2][1] == pytest.approx(
        np.array([(1.0 - beta_02 * 1.0 - beta_12 * defl12.array[0, 0]), 0.0]), 1e-4
    )

    assert traced_grid_list[3][1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

    traced_grid_list = al.util.tracer.traced_grid_2d_list_from(
        planes=planes,
        grid=grid_2d_7x7_simple,
        plane_index_limit=1,
        cosmology=al.cosmo.Planck15(),
    )

    # The scaling factors are as follows and were computed independently from the test_autoarray.
    beta_01 = 0.9348

    val = np.sqrt(2) / 2.0

    assert traced_grid_list[0][0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
    assert traced_grid_list[0][1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

    assert traced_grid_list[1][0] == pytest.approx(
        np.array([(1.0 - beta_01 * val), (1.0 - beta_01 * val)]), 1e-4
    )
    assert traced_grid_list[1][1] == pytest.approx(
        np.array([(1.0 - beta_01 * 1.0), 0.0]), 1e-4
    )

    assert len(traced_grid_list) == 2


def test__grid_2d_at_redshift_from(grid_2d_7x7):
    g0 = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    g1 = al.Galaxy(
        redshift=0.75,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0),
    )
    g2 = al.Galaxy(
        redshift=1.5,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=3.0),
    )
    g3 = al.Galaxy(
        redshift=1.0,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
    )
    g4 = al.Galaxy(redshift=2.0)

    galaxies = [g0, g1, g2, g3, g4]
    planes = al.util.tracer.planes_from(galaxies=galaxies)

    traced_grid_list = al.util.tracer.traced_grid_2d_list_from(
        planes=planes, grid=grid_2d_7x7
    )

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies, grid=grid_2d_7x7, redshift=0.5
    )

    assert grid_at_redshift == pytest.approx(traced_grid_list[0], 1.0e-4)

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies, grid=grid_2d_7x7, redshift=0.75
    )

    assert grid_at_redshift == pytest.approx(traced_grid_list[1].array, 1.0e-4)

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies, grid=grid_2d_7x7, redshift=1.0
    )

    assert grid_at_redshift == pytest.approx(traced_grid_list[2].array, 1.0e-4)

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies, grid=grid_2d_7x7, redshift=1.5
    )

    assert grid_at_redshift == pytest.approx(traced_grid_list[3].array, 1.0e-4)

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies, grid=grid_2d_7x7, redshift=2.0
    )

    assert grid_at_redshift == pytest.approx(traced_grid_list[4].array, 1.0e-4)


def test__grid_2d_at_redshift_from__redshift_between_planes(grid_2d_7x7):
    grid_2d_7x7[0] = al.Grid2DIrregular([[1.0, -1.0]])
    grid_2d_7x7[1] = al.Grid2DIrregular([[1.0, 0.0]])

    g0 = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
    )
    g1 = al.Galaxy(
        redshift=0.75,
        mass_profile=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0),
    )
    g2 = al.Galaxy(redshift=2.0)

    galaxies = [g0, g1, g2]

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies, grid=grid_2d_7x7, redshift=1.9
    )

    assert grid_at_redshift[0][0] == pytest.approx(-1.06587, 1.0e-1)
    assert grid_at_redshift[0][1] == pytest.approx(1.06587, 1.0e-1)
    assert grid_at_redshift[1][0] == pytest.approx(-1.921583, 1.0e-1)
    assert grid_at_redshift[1][1] == pytest.approx(0.0, 1.0e-1)

    grid_at_redshift = al.util.tracer.grid_2d_at_redshift_from(
        galaxies=galaxies,
        grid=grid_2d_7x7.mask.derive_grid.all_false,
        redshift=0.3,
    )

    assert (grid_at_redshift == grid_2d_7x7.mask.derive_grid.all_false).all()
