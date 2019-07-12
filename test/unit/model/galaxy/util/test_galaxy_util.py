import numpy as np
import pytest

from autolens.data.array.util import mapping_util
from autolens.data.array import grids
from autolens.data.array import mask
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy.util import galaxy_util


class TestIntensitiesFromGrid:

    def test__no_galaxies__intensities_returned_as_0s(
            self, grid_stack_7x7):

        grid_stack_7x7.regular = np.array([[1.0, 1.0],
                                         [2.0, 2.0],
                                         [3.0, 3.0]])

        intensities = galaxy_util.intensities_of_galaxies_from_grid(
            grid=grid_stack_7x7.regular, galaxies=[g.Galaxy(redshift=0.5)])

        assert (intensities[0] == np.array([0.0, 0.0])).all()
        assert (intensities[1] == np.array([0.0, 0.0])).all()
        assert (intensities[2] == np.array([0.0, 0.0])).all()

    def test__gal_x1_lp__intensities_returned_as_correct_values(self, grid_stack_7x7, gal_x1_lp):

        grid_stack_7x7.regular = np.array([[1.0, 1.0],
                                         [1.0, 0.0],
                                         [-1.0, 0.0]])

        galaxy_intensities = gal_x1_lp.intensities_from_grid(grid_stack_7x7.regular)

        util_intensities = galaxy_util.intensities_of_galaxies_from_grid(grid=grid_stack_7x7.regular,
                                                                          galaxies=[gal_x1_lp])

        assert (galaxy_intensities == util_intensities).all()

    def test__gal_x1_lp_x2__intensities_double_from_above(self, grid_stack_7x7, gal_x1_lp):
        grid_stack_7x7.regular = np.array([[1.0, 1.0],
                                         [1.0, 0.0],
                                         [-1.0, 0.0]])

        galaxy_intensities = gal_x1_lp.intensities_from_grid(grid_stack_7x7.regular)

        util_intensities = galaxy_util.intensities_of_galaxies_from_grid(grid=grid_stack_7x7.regular,
                                                                          galaxies=[gal_x1_lp, gal_x1_lp])

        assert (2.0 * galaxy_intensities == util_intensities).all()

    def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper(self, grid_stack_7x7, gal_x1_lp):

        intensities = gal_x1_lp.intensities_from_grid(grid_stack_7x7.sub)

        intensities = (intensities[0] + intensities[1] + intensities[2] +
                        intensities[3]) / 4.0

        util_intensities = galaxy_util.intensities_of_galaxies_from_grid(grid=grid_stack_7x7.sub, galaxies=[gal_x1_lp])

        assert util_intensities[0] == intensities

    def test__no_galaxies__grid_shape_of_grid_returned(self, grid_stack_7x7):

        intensities = galaxy_util.intensities_of_galaxies_from_grid(grid=grid_stack_7x7.regular, galaxies=[])

        assert (intensities == np.zeros(shape=grid_stack_7x7.regular.shape[0])).all()

        intensities = galaxy_util.intensities_of_galaxies_from_grid(grid=grid_stack_7x7.sub, galaxies=[])

        assert (intensities == np.zeros(shape=grid_stack_7x7.regular.shape[0])).all()


class TestPotentialFromGrid:

    def test__no_galaxies__potential_returned_as_0s(self, grid_stack_7x7):
        grid_stack_7x7.regular = np.array([[1.0, 1.0],
                                         [2.0, 2.0],
                                         [3.0, 3.0]])

        potential = galaxy_util.potential_of_galaxies_from_grid(
            grid=grid_stack_7x7.regular, galaxies=[g.Galaxy(redshift=0.5)])

        assert (potential[0] == np.array([0.0, 0.0])).all()
        assert (potential[1] == np.array([0.0, 0.0])).all()
        assert (potential[2] == np.array([0.0, 0.0])).all()

    def test__gal_x1_mp__potential_returned_as_correct_values(self, grid_stack_7x7, gal_x1_mp):
        grid_stack_7x7.regular = np.array([[1.0, 1.0],
                                         [1.0, 0.0],
                                         [-1.0, 0.0]])

        galaxy_potential = gal_x1_mp.potential_from_grid(grid_stack_7x7.regular)

        util_potential = galaxy_util.potential_of_galaxies_from_grid(grid=grid_stack_7x7.regular, galaxies=[gal_x1_mp])

        assert (galaxy_potential == util_potential).all()

    def test__gal_x2_mp__potential_double_from_above(self, grid_stack_7x7, gal_x1_mp):
        grid_stack_7x7.regular = np.array([[1.0, 1.0],
                                         [1.0, 0.0],
                                         [-1.0, 0.0]])

        galaxy_potential = gal_x1_mp.potential_from_grid(grid_stack_7x7.regular)

        util_potential = galaxy_util.potential_of_galaxies_from_grid(grid=grid_stack_7x7.regular,
                                                                    galaxies=[gal_x1_mp, gal_x1_mp])

        assert (2.0 * galaxy_potential == util_potential).all()

    def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper(self, grid_stack_7x7, gal_x1_mp):
        potential = gal_x1_mp.potential_from_grid(grid_stack_7x7.sub)

        potential = (potential[0] + potential[1] + potential[2] +
                        potential[3]) / 4.0

        util_potential = galaxy_util.potential_of_galaxies_from_grid(grid=grid_stack_7x7.sub, galaxies=[gal_x1_mp])

        assert util_potential[0] == potential

    def test__no_galaxies__grid_shape_of_grid_returned(self, grid_stack_7x7):

        potential = galaxy_util.potential_of_galaxies_from_grid(grid=grid_stack_7x7.regular, galaxies=[])

        assert (potential == np.zeros(shape=grid_stack_7x7.regular.shape[0])).all()

        potential = galaxy_util.potential_of_galaxies_from_grid(grid=grid_stack_7x7.sub, galaxies=[])

        assert (potential == np.zeros(shape=grid_stack_7x7.regular.shape[0])).all()


class TestDeflectionsFromGrid:

    def test__all_coordinates(self, grid_stack_simple, gal_x1_mp):

        deflections = galaxy_util.deflections_of_galaxies_from_grid_stack(
            grid_stack=grid_stack_simple, galaxies=[gal_x1_mp])

        assert deflections.regular[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
        assert deflections.sub[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
        assert deflections.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        #    assert deflection_stacks.sub.sub_grid_size == 2
        assert deflections.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

    def test__2_identical_lens_galaxies__deflection_angles_double(self, grid_stack_simple, gal_x1_mp):

        deflections = galaxy_util.deflections_of_galaxies_from_grid_stack(
            grid_stack=grid_stack_simple, galaxies=[gal_x1_mp, gal_x1_mp])

        assert deflections.regular[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
        assert deflections.sub[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
        assert deflections.sub[1] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
        #    assert deflection_stacks.sub.sub_grid_size == 2
        assert deflections.blurring[0] == pytest.approx(np.array([2.0, 0.0]), 1e-3)

    def test__1_lens_with_2_mass_profiles__deflection_angles_triple(self, grid_stack_simple, gal_x2_mp):

        deflections = galaxy_util.deflections_of_galaxies_from_grid_stack(
            grid_stack=grid_stack_simple, galaxies=[gal_x2_mp])

        assert deflections.regular[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
        assert deflections.sub[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
        assert deflections.sub[1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
        assert deflections.blurring[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
        
    def test__no_galaxies__grid_shape_of_grid_returned(self, grid_stack_7x7):

        deflections = galaxy_util.deflections_of_galaxies_from_grid(grid=grid_stack_7x7.regular, galaxies=[])

        assert (deflections == np.zeros(shape=(grid_stack_7x7.regular.shape[0], 2))).all()

        deflections = galaxy_util.deflections_of_galaxies_from_grid(grid=grid_stack_7x7.sub, galaxies=[])

        assert (deflections == np.zeros(shape=(grid_stack_7x7.regular.shape[0], 2))).all()

        deflections = galaxy_util.deflections_of_galaxies_from_sub_grid(sub_grid=grid_stack_7x7.sub, galaxies=[])

        assert (deflections == np.zeros(shape=(grid_stack_7x7.sub.shape[0], 2))).all()