from astropy import cosmology as cosmo
import numpy as np
import pytest

import autolens as al


class TestTracedGridListFrom:
    def test__x2_planes__no_galaxy__image_and_source_planes_setup__same_coordinates(
        self, sub_grid_2d_7x7
    ):
        galaxies = [al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]

        planes = al.util.plane.planes_via_galaxies_from(galaxies=galaxies)

        traced_grid_list = al.util.ray_tracing.traced_grid_list_from(
            planes=planes, grid=sub_grid_2d_7x7
        )

        assert traced_grid_list[0][0] == pytest.approx(np.array([1.25, -1.25]), 1e-3)
        assert traced_grid_list[0][1] == pytest.approx(np.array([1.25, -0.75]), 1e-3)
        assert traced_grid_list[0][2] == pytest.approx(np.array([0.75, -1.25]), 1e-3)
        assert traced_grid_list[0][3] == pytest.approx(np.array([0.75, -0.75]), 1e-3)

        assert traced_grid_list[1][0] == pytest.approx(np.array([1.25, -1.25]), 1e-3)
        assert traced_grid_list[1][1] == pytest.approx(np.array([1.25, -0.75]), 1e-3)
        assert traced_grid_list[1][2] == pytest.approx(np.array([0.75, -1.25]), 1e-3)
        assert traced_grid_list[1][3] == pytest.approx(np.array([0.75, -0.75]), 1e-3)

    def test__x2_planes__sis_lens__traced_grid_includes_deflections__on_planes_setup(
        self, sub_grid_2d_7x7_simple, gal_x1_mp
    ):
        galaxies = [gal_x1_mp, al.Galaxy(redshift=1.0)]

        planes = al.util.plane.planes_via_galaxies_from(galaxies=galaxies)

        traced_grid_list = al.util.ray_tracing.traced_grid_list_from(
            planes=planes, grid=sub_grid_2d_7x7_simple
        )

        assert traced_grid_list[0][0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert traced_grid_list[0][1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        assert traced_grid_list[0][2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert traced_grid_list[0][3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        assert traced_grid_list[1][0] == pytest.approx(
            np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3
        )
        assert traced_grid_list[1][1] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
        assert traced_grid_list[1][2] == pytest.approx(
            np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3
        )
        assert traced_grid_list[1][3] == pytest.approx(np.array([0.0, 0.0]), 1e-3)

        galaxies = [gal_x1_mp, gal_x1_mp, al.Galaxy(redshift=1.0)]

        planes = al.util.plane.planes_via_galaxies_from(galaxies=galaxies)

        traced_grid_list = al.util.ray_tracing.traced_grid_list_from(
            planes=planes, grid=sub_grid_2d_7x7_simple
        )

        assert traced_grid_list[0][0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert traced_grid_list[0][1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        assert traced_grid_list[0][2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert traced_grid_list[0][3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        assert traced_grid_list[1][0] == pytest.approx(
            np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3
        )
        assert traced_grid_list[1][1] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
        assert traced_grid_list[1][2] == pytest.approx(
            np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3
        )

        assert traced_grid_list[1][3] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)

    def test__x4_planes__grids_are_correct__sis_mass_profile(
        self, sub_grid_2d_7x7_simple
    ):
        g0 = al.Galaxy(
            redshift=2.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        g1 = al.Galaxy(
            redshift=2.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        g2 = al.Galaxy(
            redshift=0.1, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        g3 = al.Galaxy(
            redshift=3.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        g4 = al.Galaxy(
            redshift=1.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        g5 = al.Galaxy(
            redshift=3.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )

        galaxies = [g0, g1, g2, g3, g4, g5]

        planes = al.util.plane.planes_via_galaxies_from(galaxies=galaxies)

        traced_grid_list = al.util.ray_tracing.traced_grid_list_from(
            planes=planes, grid=sub_grid_2d_7x7_simple, cosmology=cosmo.Planck15
        )

        # The scaling factors are as follows and were computed independently from the test_autoarray.
        beta_01 = 0.9348
        beta_02 = 0.9839601
        # Beta_03 = 1.0
        beta_12 = 0.7539734
        # Beta_13 = 1.0
        # Beta_23 = 1.0

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
            grid=np.array([[(1.0 - beta_01 * val), (1.0 - beta_01 * val)]])
        )
        defl12 = g0.deflections_yx_2d_from(
            grid=np.array([[(1.0 - beta_01 * 1.0), 0.0]])
        )

        assert traced_grid_list[2][0] == pytest.approx(
            np.array(
                [
                    (1.0 - beta_02 * val - beta_12 * defl11[0, 0]),
                    (1.0 - beta_02 * val - beta_12 * defl11[0, 1]),
                ]
            ),
            1e-4,
        )
        assert traced_grid_list[2][1] == pytest.approx(
            np.array([(1.0 - beta_02 * 1.0 - beta_12 * defl12[0, 0]), 0.0]), 1e-4
        )

        assert traced_grid_list[3][1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

    def test__upper_plane_limit_removes_final_plane(
        self, sub_grid_2d_7x7_simple, gal_x1_mp
    ):
        galaxies = [gal_x1_mp, al.Galaxy(redshift=1.0)]

        planes = al.util.plane.planes_via_galaxies_from(galaxies=galaxies)

        traced_grid_list = al.util.ray_tracing.traced_grid_list_from(
            planes=planes, grid=sub_grid_2d_7x7_simple, plane_index_limit=0
        )

        assert traced_grid_list[0][0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert traced_grid_list[0][1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        assert traced_grid_list[0][2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert traced_grid_list[0][3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        assert len(traced_grid_list) == 1

        g0 = al.Galaxy(
            redshift=2.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        g1 = al.Galaxy(
            redshift=2.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        g2 = al.Galaxy(
            redshift=0.1, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        g3 = al.Galaxy(
            redshift=3.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        g4 = al.Galaxy(
            redshift=1.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        g5 = al.Galaxy(
            redshift=3.0, mass_profile=al.mp.SphIsothermal(einstein_radius=1.0)
        )

        galaxies = [g0, g1, g2, g3, g4, g5]

        planes = al.util.plane.planes_via_galaxies_from(galaxies=galaxies)

        traced_grid_list = al.util.ray_tracing.traced_grid_list_from(
            planes=planes,
            grid=sub_grid_2d_7x7_simple,
            plane_index_limit=1,
            cosmology=cosmo.Planck15,
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


class TestGridAtRedshift:
    def test__lens_z05_source_z01_redshifts__match_planes_redshifts__gives_same_grids(
        self, sub_grid_2d_7x7
    ):
        g0 = al.Galaxy(
            redshift=0.5,
            mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
        )
        g1 = al.Galaxy(redshift=1.0)

        galaxies = [g0, g1]
        planes = al.util.plane.planes_via_galaxies_from(galaxies=galaxies)

        grid_at_redshift = al.util.ray_tracing.grid_at_redshift_from(
            galaxies=galaxies, grid=sub_grid_2d_7x7, redshift=0.5
        )

        assert (grid_at_redshift == sub_grid_2d_7x7).all()

        grid_at_redshift = al.util.ray_tracing.grid_at_redshift_from(
            galaxies=galaxies, grid=sub_grid_2d_7x7, redshift=1.0
        )

        source_plane_grid = al.util.ray_tracing.traced_grid_list_from(
            planes=planes, grid=sub_grid_2d_7x7
        )[1]

        assert (grid_at_redshift == source_plane_grid).all()

    def test__same_as_above_but_for_multi_tracing(self, sub_grid_2d_7x7):
        g0 = al.Galaxy(
            redshift=0.5,
            mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
        )
        g1 = al.Galaxy(
            redshift=0.75,
            mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0),
        )
        g2 = al.Galaxy(
            redshift=1.5,
            mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=3.0),
        )
        g3 = al.Galaxy(
            redshift=1.0,
            mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=4.0),
        )
        g4 = al.Galaxy(redshift=2.0)

        galaxies = [g0, g1, g2, g3, g4]
        planes = al.util.plane.planes_via_galaxies_from(galaxies=galaxies)

        traced_grid_list = al.util.ray_tracing.traced_grid_list_from(
            planes=planes, grid=sub_grid_2d_7x7
        )

        grid_at_redshift = al.util.ray_tracing.grid_at_redshift_from(
            galaxies=galaxies, grid=sub_grid_2d_7x7, redshift=0.5
        )

        assert grid_at_redshift == pytest.approx(traced_grid_list[0], 1.0e-4)

        grid_at_redshift = al.util.ray_tracing.grid_at_redshift_from(
            galaxies=galaxies, grid=sub_grid_2d_7x7, redshift=0.75
        )

        assert grid_at_redshift == pytest.approx(traced_grid_list[1], 1.0e-4)

        grid_at_redshift = al.util.ray_tracing.grid_at_redshift_from(
            galaxies=galaxies, grid=sub_grid_2d_7x7, redshift=1.0
        )

        assert grid_at_redshift == pytest.approx(traced_grid_list[2], 1.0e-4)

        grid_at_redshift = al.util.ray_tracing.grid_at_redshift_from(
            galaxies=galaxies, grid=sub_grid_2d_7x7, redshift=1.5
        )

        assert grid_at_redshift == pytest.approx(traced_grid_list[3], 1.0e-4)

        grid_at_redshift = al.util.ray_tracing.grid_at_redshift_from(
            galaxies=galaxies, grid=sub_grid_2d_7x7, redshift=2.0
        )

        assert grid_at_redshift == pytest.approx(traced_grid_list[4], 1.0e-4)

    def test__input_redshift_between_two_planes__two_planes_between_earth_and_input_redshift(
        self, sub_grid_2d_7x7
    ):

        sub_grid_2d_7x7[0] = np.array([[1.0, -1.0]])
        sub_grid_2d_7x7[1] = np.array([[1.0, 0.0]])

        g0 = al.Galaxy(
            redshift=0.5,
            mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
        )
        g1 = al.Galaxy(
            redshift=0.75,
            mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0),
        )
        g2 = al.Galaxy(redshift=2.0)

        galaxies = [g0, g1, g2]

        grid_at_redshift = al.util.ray_tracing.grid_at_redshift_from(
            galaxies=galaxies, grid=sub_grid_2d_7x7, redshift=1.9
        )

        assert grid_at_redshift[0][0] == pytest.approx(-1.06587, 1.0e-1)
        assert grid_at_redshift[0][1] == pytest.approx(1.06587, 1.0e-1)
        assert grid_at_redshift[1][0] == pytest.approx(-1.921583, 1.0e-1)
        assert grid_at_redshift[1][1] == pytest.approx(0.0, 1.0e-1)

    def test__input_redshift_before_first_plane__returns_image_plane(
        self, sub_grid_2d_7x7
    ):
        g0 = al.Galaxy(
            redshift=0.5,
            mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
        )
        g1 = al.Galaxy(
            redshift=0.75,
            mass_profile=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0),
        )

        galaxies = [g0, g1]

        grid_at_redshift = al.util.ray_tracing.grid_at_redshift_from(
            galaxies=galaxies,
            grid=sub_grid_2d_7x7.mask.unmasked_grid_sub_1,
            redshift=0.3,
        )

        assert (grid_at_redshift == sub_grid_2d_7x7.mask.unmasked_grid_sub_1).all()
