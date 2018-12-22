import numpy as np
import pytest

from autolens.data.array.util import mapping_util
from autolens.data.array import grids
from autolens.data.array import mask
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy as g
from autolens.lensing.util import plane_util

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


class TestIntensitiesFromGrid:

    def test__no_galaxies__intensities_returned_as_0s(self, grid_stack, galaxy_non):

        grid_stack.regular = np.array([[1.0, 1.0],
                                         [2.0, 2.0],
                                         [3.0, 3.0]])

        intensities = plane_util.intensities_of_galaxies_from_grid(grid=grid_stack.regular, galaxies=[galaxy_non])

        assert (intensities[0] == np.array([0.0, 0.0])).all()
        assert (intensities[1] == np.array([0.0, 0.0])).all()
        assert (intensities[2] == np.array([0.0, 0.0])).all()

    def test__galaxy_light__intensities_returned_as_correct_values(self, grid_stack, galaxy_light):
        grid_stack.regular = np.array([[1.0, 1.0],
                                         [1.0, 0.0],
                                         [-1.0, 0.0]])

        galaxy_intensities = galaxy_light.intensities_from_grid(grid_stack.regular)

        util_intensities = plane_util.intensities_of_galaxies_from_grid(grid=grid_stack.regular,
                                                                          galaxies=[galaxy_light])

        assert (galaxy_intensities == util_intensities).all()

    def test__galaxy_light_x2__intensities_double_from_above(self, grid_stack, galaxy_light):
        grid_stack.regular = np.array([[1.0, 1.0],
                                         [1.0, 0.0],
                                         [-1.0, 0.0]])

        galaxy_intensities = galaxy_light.intensities_from_grid(grid_stack.regular)

        util_intensities = plane_util.intensities_of_galaxies_from_grid(grid=grid_stack.regular,
                                                                          galaxies=[galaxy_light, galaxy_light])

        assert (2.0 * galaxy_intensities == util_intensities).all()

    def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper(self, grid_stack, galaxy_light):
        galaxy_image = galaxy_light.intensities_from_grid(grid_stack.sub)

        galaxy_image = (galaxy_image[0] + galaxy_image[1] + galaxy_image[2] +
                        galaxy_image[3]) / 4.0

        util_intensities = plane_util.intensities_of_galaxies_from_grid(grid=grid_stack.sub, galaxies=[galaxy_light])

        assert util_intensities[0] == galaxy_image


class TestSurfaceDensityFromGrid:

    def test__no_galaxies__surface_density_returned_as_0s(self, grid_stack, galaxy_non):
        grid_stack.regular = np.array([[1.0, 1.0],
                                         [2.0, 2.0],
                                         [3.0, 3.0]])

        surface_density = plane_util.surface_density_of_galaxies_from_grid(grid=grid_stack.regular, galaxies=[galaxy_non])

        assert (surface_density[0] == np.array([0.0, 0.0])).all()
        assert (surface_density[1] == np.array([0.0, 0.0])).all()
        assert (surface_density[2] == np.array([0.0, 0.0])).all()

    def test__galaxy_mass__surface_density_returned_as_correct_values(self, grid_stack, galaxy_mass):
        grid_stack.regular = np.array([[1.0, 1.0],
                                         [1.0, 0.0],
                                         [-1.0, 0.0]])

        galaxy_surface_density = galaxy_mass.surface_density_from_grid(grid_stack.regular)

        util_surface_density = plane_util.surface_density_of_galaxies_from_grid(grid=grid_stack.regular,
                                                                                  galaxies=[galaxy_mass])

        assert (galaxy_surface_density == util_surface_density).all()

    def test__galaxy_mass_x2__surface_density_double_from_above(self, grid_stack, galaxy_mass):
        grid_stack.regular = np.array([[1.0, 1.0],
                                         [1.0, 0.0],
                                         [-1.0, 0.0]])

        galaxy_surface_density = galaxy_mass.surface_density_from_grid(grid_stack.regular)

        util_surface_density = plane_util.surface_density_of_galaxies_from_grid(grid=grid_stack.regular,
                                                                                  galaxies=[galaxy_mass, galaxy_mass])

        assert (2.0 * galaxy_surface_density == util_surface_density).all()

    def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper(self, grid_stack, galaxy_mass):
        galaxy_image = galaxy_mass.surface_density_from_grid(grid_stack.sub)

        galaxy_image = (galaxy_image[0] + galaxy_image[1] + galaxy_image[2] +
                        galaxy_image[3]) / 4.0

        util_surface_density = plane_util.surface_density_of_galaxies_from_grid(grid=grid_stack.sub,
                                                                                  galaxies=[galaxy_mass])

        assert util_surface_density[0] == galaxy_image


class TestPotentialFromGrid:

    def test__no_galaxies__potential_returned_as_0s(self, grid_stack, galaxy_non):
        grid_stack.regular = np.array([[1.0, 1.0],
                                         [2.0, 2.0],
                                         [3.0, 3.0]])

        potential = plane_util.potential_of_galaxies_from_grid(grid=grid_stack.regular, galaxies=[galaxy_non])

        assert (potential[0] == np.array([0.0, 0.0])).all()
        assert (potential[1] == np.array([0.0, 0.0])).all()
        assert (potential[2] == np.array([0.0, 0.0])).all()

    def test__galaxy_mass__potential_returned_as_correct_values(self, grid_stack, galaxy_mass):
        grid_stack.regular = np.array([[1.0, 1.0],
                                         [1.0, 0.0],
                                         [-1.0, 0.0]])

        galaxy_potential = galaxy_mass.potential_from_grid(grid_stack.regular)

        util_potential = plane_util.potential_of_galaxies_from_grid(grid=grid_stack.regular, galaxies=[galaxy_mass])

        assert (galaxy_potential == util_potential).all()

    def test__galaxy_mass_x2__potential_double_from_above(self, grid_stack, galaxy_mass):
        grid_stack.regular = np.array([[1.0, 1.0],
                                         [1.0, 0.0],
                                         [-1.0, 0.0]])

        galaxy_potential = galaxy_mass.potential_from_grid(grid_stack.regular)

        util_potential = plane_util.potential_of_galaxies_from_grid(grid=grid_stack.regular,
                                                                    galaxies=[galaxy_mass, galaxy_mass])

        assert (2.0 * galaxy_potential == util_potential).all()

    def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper(self, grid_stack, galaxy_mass):
        galaxy_image = galaxy_mass.potential_from_grid(grid_stack.sub)

        galaxy_image = (galaxy_image[0] + galaxy_image[1] + galaxy_image[2] +
                        galaxy_image[3]) / 4.0

        util_potential = plane_util.potential_of_galaxies_from_grid(grid=grid_stack.sub, galaxies=[galaxy_mass])

        assert util_potential[0] == galaxy_image


class TestDeflectionsFromGrid:

    def test__all_coordinates(self, grid_stack, galaxy_mass):
        deflections = plane_util.deflections_of_galaxies_from_grid_stack(grid_stack, [galaxy_mass])

        assert deflections.regular[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
        assert deflections.sub[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
        assert deflections.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        #    assert deflection_stacks.sub.sub_grid_size == 2
        assert deflections.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

    def test__2_identical_lens_galaxies__deflection_angles_double(self, grid_stack, galaxy_mass):
        deflections = plane_util.deflections_of_galaxies_from_grid_stack(grid_stack, [galaxy_mass, galaxy_mass])

        assert deflections.regular[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
        assert deflections.sub[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
        assert deflections.sub[1] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
        #    assert deflection_stacks.sub.sub_grid_size == 2
        assert deflections.blurring[0] == pytest.approx(np.array([2.0, 0.0]), 1e-3)

    def test__1_lens_with_2_identical_mass_profiles__deflection_angles_double(self, grid_stack, galaxy_mass_x2):
        deflections = plane_util.deflections_of_galaxies_from_grid_stack(grid_stack, [galaxy_mass_x2])

        assert deflections.regular[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
        assert deflections.sub[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
        assert deflections.sub[1] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
        assert deflections.blurring[0] == pytest.approx(np.array([2.0, 0.0]), 1e-3)


class TestPlaneImageFromGrid:

    def test__3x3_grid__extracts_max_min_coordinates__creates_regular_grid_including_half_pixel_offset_from_edge(self):

        galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))

        grid = np.array([[-1.5, -1.5], [1.5, 1.5]])

        plane_image = plane_util.plane_image_of_galaxies_from_grid(shape=(3, 3), grid=grid, galaxies=[galaxy], buffer=0.0)

        plane_image_galaxy = galaxy.intensities_from_grid(grid=np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                                                           [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                           [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]))

        plane_image_galaxy = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
            array_1d=plane_image_galaxy, shape=(3,3))

        assert (plane_image == plane_image_galaxy).all()

    def test__3x3_grid__extracts_max_min_coordinates__ignores_other_coordinates_more_central(self):

        galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))

        grid = np.array([[-1.5, -1.5], [1.5, 1.5], [0.1, -0.1], [-1.0, 0.6], [1.4, -1.3], [1.5, 1.5]])

        plane_image = plane_util.plane_image_of_galaxies_from_grid(shape=(3, 3), grid=grid, galaxies=[galaxy], buffer=0.0)

        plane_image_galaxy = galaxy.intensities_from_grid(grid=np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                                                           [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                           [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]))

        plane_image_galaxy = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
            array_1d=plane_image_galaxy, shape=(3,3))

        assert (plane_image == plane_image_galaxy).all()

    def test__2x3_grid__shape_change_correct_and_coordinates_shift(self):

        galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))

        grid = np.array([[-1.5, -1.5], [1.5, 1.5]])

        plane_image = plane_util.plane_image_of_galaxies_from_grid(shape=(2, 3), grid=grid, galaxies=[galaxy], buffer=0.0)

        plane_image_galaxy = galaxy.intensities_from_grid(grid=np.array([[-0.75, -1.0], [-0.75, 0.0], [-0.75, 1.0],
                                                                          [0.75, -1.0], [0.75, 0.0], [0.75, 1.0]]))

        plane_image_galaxy = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
            array_1d=plane_image_galaxy, shape=(2,3))

        assert (plane_image == plane_image_galaxy).all()

    def test__3x2_grid__shape_change_correct_and_coordinates_shift(self):

        galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))

        grid = np.array([[-1.5, -1.5], [1.5, 1.5]])

        plane_image = plane_util.plane_image_of_galaxies_from_grid(shape=(3, 2), grid=grid, galaxies=[galaxy], buffer=0.0)

        plane_image_galaxy = galaxy.intensities_from_grid(grid=np.array([[-1.0, -0.75], [-1.0, 0.75],
                                                                          [0.0, -0.75], [0.0, 0.75],
                                                                          [1.0, -0.75], [1.0, 0.75]]))

        plane_image_galaxy = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
            array_1d=plane_image_galaxy, shape=(3,2))

        assert (plane_image == plane_image_galaxy).all()

    def test__3x3_grid__buffer_aligns_two_grids(self):

        galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))

        grid_without_buffer = np.array([[-1.48, -1.48], [1.48, 1.48]])

        plane_image = plane_util.plane_image_of_galaxies_from_grid(shape=(3, 3), grid=grid_without_buffer, galaxies=[galaxy],
                                                                   buffer=0.02)

        plane_image_galaxy = galaxy.intensities_from_grid(grid=np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                                                           [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                           [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]))

        plane_image_galaxy = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
            array_1d=plane_image_galaxy, shape=(3,3))

        assert (plane_image == plane_image_galaxy).all()
