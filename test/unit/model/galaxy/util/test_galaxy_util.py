import numpy as np
import pytest

from autolens.data.array.util import mapping_util
from autolens.data.array import grids
from autolens.data.array import mask
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy.util import galaxy_util


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