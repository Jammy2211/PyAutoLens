import numpy as np
import pytest

from autolens.data.array import grids
from autolens.data.array import mask
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy as g
from autolens.lens import plane as pl
from autolens.lens.util import ray_tracing_util
from autolens.lens.util import plane_util

@pytest.fixture(name="grid_stack")
def make_grid_stack():
    ma = mask.Mask(np.array([[True, True, True, True],
                             [True, False, False, True],
                             [True, True, True, True]]), pixel_scale=6.0)

    grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=2,
                                                                                    psf_shape=(3, 3))

    # Manually overwrite a set of cooridnates to make tests of grid_stacks and defledctions straightforward

    grid_stack.regular[0] = np.array([1.0, 1.0])
    grid_stack.regular[1] = np.array([1.0, 0.0])
    grid_stack.sub[0] = np.array([1.0, 1.0])
    grid_stack.sub[1] = np.array([1.0, 0.0])
    grid_stack.sub[2] = np.array([1.0, 1.0])
    grid_stack.sub[3] = np.array([1.0, 0.0])
    grid_stack.sub[4] = np.array([-1.0, 2.0])
    grid_stack.sub[5] = np.array([-1.0, 4.0])
    grid_stack.sub[6] = np.array([1.0, 2.0])
    grid_stack.sub[7] = np.array([1.0, 4.0])
    grid_stack.blurring[0] = np.array([1.0, 0.0])
    grid_stack.blurring[1] = np.array([-6.0, -3.0])
    grid_stack.blurring[2] = np.array([-6.0, 3.0])
    grid_stack.blurring[3] = np.array([-6.0, 9.0])
    grid_stack.blurring[4] = np.array([0.0, -9.0])
    grid_stack.blurring[5] = np.array([0.0, 9.0])
    grid_stack.blurring[6] = np.array([6.0, -9.0])
    grid_stack.blurring[7] = np.array([6.0, -3.0])
    grid_stack.blurring[8] = np.array([6.0, 3.0])
    grid_stack.blurring[9] = np.array([6.0, 9.0])

    return grid_stack

@pytest.fixture(name="padded_grid_stack")
def make_padded_grid_stack():
    ma = mask.Mask(np.array([[True, False]]), pixel_scale=3.0)
    return grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(ma, 2, (3, 3))

@pytest.fixture(name='galaxy_non', scope='function')
def make_galaxy_non():
    return g.Galaxy()

@pytest.fixture(name="galaxy_light")
def make_galaxy_light():
    return g.Galaxy(light_profile=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                      effective_radius=0.6, sersic_index=4.0))

@pytest.fixture(name="galaxy_mass")
def make_galaxy_mass():
    return g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

@pytest.fixture(name='galaxy_mass_x2')
def make_galaxy_mass_x2():
    return g.Galaxy(sis_0=mp.SphericalIsothermal(einstein_radius=1.0),
                    sis_1=mp.SphericalIsothermal(einstein_radius=1.0))


class TestSetupTracedGrid:

    def test__simple_sis_model__deflection_angles(self, grid_stack, galaxy_mass):

        deflections = plane_util.deflections_of_galaxies_from_grid_stack(grid_stack, [galaxy_mass])

        grid_traced = ray_tracing_util.traced_collection_for_deflections(grid_stack, deflections)

        assert grid_traced.regular[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-2)

    def test_two_identical_lenses__deflection_angles_double(self, grid_stack, galaxy_mass):
        deflections = plane_util.deflections_of_galaxies_from_grid_stack(grid_stack, [galaxy_mass, galaxy_mass])

        grid_traced = ray_tracing_util.traced_collection_for_deflections(grid_stack, deflections)

        assert grid_traced.regular[0] == pytest.approx(np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3)

    def test_one_lens_with_double_identical_mass_profiles__deflection_angles_double(self, grid_stack,
                                                                                    galaxy_mass_x2):
        deflections = plane_util.deflections_of_galaxies_from_grid_stack(grid_stack, [galaxy_mass_x2])

        grid_traced = ray_tracing_util.traced_collection_for_deflections(grid_stack, deflections)

        assert grid_traced.regular[0] == pytest.approx(np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3)


class TestGalaxyOrdering:

    def test__3_galaxies_reordered_in_ascending_redshift(self):
        galaxies = [g.Galaxy(redshift=2.0), g.Galaxy(redshift=1.0), g.Galaxy(redshift=0.1)]

        ordered_plane_redshifts = ray_tracing_util.ordered_plane_redshifts_from_galaxies(galaxies=galaxies)

        assert ordered_plane_redshifts == [0.1, 1.0, 2.0]

        ordered_galaxies = ray_tracing_util.galaxies_in_redshift_ordered_planes_from_galaxies(galaxies=galaxies,
                                                                                              plane_redshifts=ordered_plane_redshifts)

        assert ordered_galaxies[0][0].redshift == 0.1
        assert ordered_galaxies[1][0].redshift == 1.0
        assert ordered_galaxies[2][0].redshift == 2.0

    def test_3_galaxies_two_same_redshift_planes_redshift_order_is_size_2_with_redshifts(self):
        galaxies = [g.Galaxy(redshift=1.0), g.Galaxy(redshift=1.0), g.Galaxy(redshift=0.1)]

        ordered_plane_redshifts = ray_tracing_util.ordered_plane_redshifts_from_galaxies(galaxies=galaxies)

        assert ordered_plane_redshifts == [0.1, 1.0]

        ordered_galaxies = ray_tracing_util.galaxies_in_redshift_ordered_planes_from_galaxies(galaxies=galaxies,
                                                                                              plane_redshifts=ordered_plane_redshifts)

        assert ordered_galaxies[0][0].redshift == 0.1
        assert ordered_galaxies[1][0].redshift == 1.0
        assert ordered_galaxies[1][1].redshift == 1.0

    def test__6_galaxies_producing_4_planes(self):
        g0 = g.Galaxy(redshift=1.0)
        g1 = g.Galaxy(redshift=1.0)
        g2 = g.Galaxy(redshift=0.1)
        g3 = g.Galaxy(redshift=1.05)
        g4 = g.Galaxy(redshift=0.95)
        g5 = g.Galaxy(redshift=1.05)

        galaxies = [g0, g1, g2, g3, g4, g5]

        ordered_plane_redshifts = ray_tracing_util.ordered_plane_redshifts_from_galaxies(galaxies=galaxies)

        assert ordered_plane_redshifts == [0.1, 0.95, 1.0, 1.05]

        ordered_galaxies = ray_tracing_util.galaxies_in_redshift_ordered_planes_from_galaxies(galaxies=galaxies,
                                                                                              plane_redshifts=ordered_plane_redshifts)

        assert ordered_galaxies[0][0].redshift == 0.1
        assert ordered_galaxies[1][0].redshift == 0.95
        assert ordered_galaxies[2][0].redshift == 1.0
        assert ordered_galaxies[2][1].redshift == 1.0
        assert ordered_galaxies[3][0].redshift == 1.05
        assert ordered_galaxies[3][1].redshift == 1.05

        assert ordered_galaxies[0] == [g2]
        assert ordered_galaxies[1] == [g4]
        assert ordered_galaxies[2] == [g0, g1]
        assert ordered_galaxies[3] == [g3, g5]


class TestComputeDeflections:

    def test__if_plane_is_last_plane_is_false_else_is_true(self):

        assert ray_tracing_util.compute_deflections_at_next_plane(plane_index=0, total_planes=4) == True
        assert ray_tracing_util.compute_deflections_at_next_plane(plane_index=2, total_planes=4) == True
        assert ray_tracing_util.compute_deflections_at_next_plane(plane_index=3, total_planes=4) == False
        assert ray_tracing_util.compute_deflections_at_next_plane(plane_index=3, total_planes=5) == True

class TestScaledDeflections:

    def test__deflection_stack_is_scaled_by_scaling_factor(self, grid_stack, galaxy_mass):

        plane = pl.Plane(galaxies=[galaxy_mass], grid_stack=grid_stack)

        scaled_deflection_stack = ray_tracing_util.scaled_deflection_stack_from_plane_and_scaling_factor(plane=plane,
                                                                                                    scaling_factor=1.0)

        assert (scaled_deflection_stack.regular == plane.deflection_stack.regular).all()
        assert (scaled_deflection_stack.sub == plane.deflection_stack.sub).all()
        assert (scaled_deflection_stack.blurring == plane.deflection_stack.blurring).all()

        scaled_deflection_stack = ray_tracing_util.scaled_deflection_stack_from_plane_and_scaling_factor(plane=plane,
                                                                                                    scaling_factor=2.0)

        assert (scaled_deflection_stack.regular == 2.0*plane.deflection_stack.regular).all()
        assert (scaled_deflection_stack.sub == 2.0*plane.deflection_stack.sub).all()
        assert (scaled_deflection_stack.blurring == 2.0*plane.deflection_stack.blurring).all()


class TestGridStackDeflections:

    def test__grid_stack_has_deflections_subtracted_from_it(self, grid_stack, galaxy_mass):

        plane = pl.Plane(galaxies=[galaxy_mass], grid_stack=grid_stack)

        deflection_stack = ray_tracing_util.scaled_deflection_stack_from_plane_and_scaling_factor(plane=plane,
                                                                                                    scaling_factor=3.0)

        traced_grid_stack = ray_tracing_util.grid_stack_from_deflection_stack(grid_stack=grid_stack,
                                                                              deflection_stack=deflection_stack)

        assert (traced_grid_stack.regular == grid_stack.regular - deflection_stack.regular).all()
        assert (traced_grid_stack.sub == grid_stack.sub - deflection_stack.sub).all()
        assert (traced_grid_stack.blurring == grid_stack.blurring - deflection_stack.blurring).all()