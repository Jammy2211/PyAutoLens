import numpy as np
import pytest

from autolens.imaging import scaled_array as sca
from autolens.imaging import imaging_util
from autolens.imaging import mask as msk
from autolens.galaxy import galaxy as g
from autolens.galaxy import galaxy_data as gd
from autolens.profiles import mass_profiles as mp
from test.mock.mock_galaxy import MockGalaxy

@pytest.fixture(name='scaled_array')
def make_scaled_array():
    return sca.ScaledSquarePixelArray(array=np.ones((4, 4)), pixel_scale=3.0)

@pytest.fixture(name="mask")
def make_mask():
    return msk.Mask(np.array([[True, True, True, True],
                              [True, False, False, True],
                              [True, False, False, True],
                              [True, True, True, True]]), pixel_scale=3.0)

@pytest.fixture(name="galaxy_data")
def make_galaxy_data(scaled_array, mask):
    return gd.GalaxyData(array=scaled_array, noise_map=2.0*np.ones((4,4)), mask=mask)


class TestGalaxyData(object):

    def test__attributes(self, scaled_array, galaxy_data):
        assert scaled_array.pixel_scale == galaxy_data.pixel_scale

    def test__scaled_array_and_mapper(self, galaxy_data):
        assert (galaxy_data == np.ones(4)).all()
        assert (galaxy_data.array == np.ones((4,4))).all()
        assert (galaxy_data.noise_map == 2.0 * np.ones((4,4))).all()
        assert (galaxy_data.mask == np.array([[True, True, True, True],
                                              [True, False, False, True],
                                              [True, False, False, True],
                                              [True, True, True, True]])).all()

    def test__grids(self, galaxy_data):

        assert galaxy_data.grids.image.shape == (4, 2)

        assert (galaxy_data.grids.image == np.array([[1.5, -1.5], [1.5, 1.5],
                                                       [-1.5, -1.5], [-1.5, 1.5]])).all()
        assert (galaxy_data.grids.sub == np.array([[2.0, -2.0], [2.0, -1.0], [1.0, -2.0], [1.0, -1.0],
                                                     [2.0, 1.0], [2.0, 2.0], [1.0, 1.0], [1.0, 2.0],
                                                     [-1.0, -2.0], [-1.0, -1.0], [-2.0, -2.0], [-2.0, -1.0],
                                                     [-1.0, 1.0], [-1.0, 2.0], [-2.0, 1.0], [-2.0, 2.0]])).all()

    def test__unmasked_grids(self, galaxy_data):

        padded_image_util = imaging_util.image_grid_1d_masked_from_mask_and_pixel_scales(mask=np.full((4, 4), False),
                          pixel_scales=galaxy_data.array.pixel_scales)

        assert (galaxy_data.unmasked_grids.image == padded_image_util).all()
        assert galaxy_data.unmasked_grids.image.image_shape == (4, 4)
        assert galaxy_data.unmasked_grids.image.padded_shape == (4, 4)

        padded_sub_util = imaging_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((4, 4), False), pixel_scales=galaxy_data.array.pixel_scales,
            sub_grid_size=galaxy_data.grids.sub.sub_grid_size)

        assert galaxy_data.unmasked_grids.sub == pytest.approx(padded_sub_util, 1e-4)
        assert galaxy_data.unmasked_grids.sub.image_shape == (4, 4)
        assert galaxy_data.unmasked_grids.sub.padded_shape == (4, 4)

    def test__subtract(self, galaxy_data):
        subtracted_image = galaxy_data - np.array([1, 0, 1, 0])
        assert isinstance(subtracted_image, gd.GalaxyData)
        assert subtracted_image.pixel_scale == galaxy_data.pixel_scale

        assert subtracted_image == np.array([0, 1, 0, 1])

    def test__galaxy_data_intensities(self, scaled_array, mask):

        galaxy_data = gd.GalaxyDataIntensities(array=scaled_array, noise_map=2.0*np.ones((4,4)), mask=mask,
                                               sub_grid_size=2)

        assert scaled_array.pixel_scale == galaxy_data.pixel_scale

        assert (galaxy_data[:] == np.ones(4)).all()
        assert (galaxy_data.array == np.ones((4,4))).all()
        assert (galaxy_data.noise_map_ == 2.0 * np.ones((4))).all()
        assert (galaxy_data.mask == np.array([[True, True, True, True],
                                              [True, False, False, True],
                                              [True, False, False, True],
                                              [True, True, True, True]])).all()

        galaxy = MockGalaxy(value=1, shape=4)

        intensities = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                            sub_grid=galaxy_data.grids.sub)

        assert (intensities == np.array([1.0, 1.0, 1.0, 1.0])).all()

        galaxy = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0))

        intensities_gal = galaxy.intensities_from_grid(grid=galaxy_data.grids.sub)
        intensities_gal = galaxy_data.grids.sub.sub_data_to_image(sub_array=intensities_gal)

        intensities_gd = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                               sub_grid=galaxy_data.grids.sub)

        assert (intensities_gal == intensities_gd).all()

    def test__galaxy_data_surface_density(self, scaled_array, mask):

        galaxy_data = gd.GalaxyDataSurfaceDensity(array=scaled_array, noise_map=2.0*np.ones((4,4)), mask=mask,
                                                  sub_grid_size=2)

        assert scaled_array.pixel_scale == galaxy_data.pixel_scale
        assert (galaxy_data == np.ones(4)).all()
        assert (galaxy_data.array == np.ones((4,4))).all()
        assert (galaxy_data.noise_map_ == 2.0 * np.ones((4))).all()
        assert (galaxy_data.mask == np.array([[True, True, True, True],
                                              [True, False, False, True],
                                              [True, False, False, True],
                                              [True, True, True, True]])).all()

        galaxy = MockGalaxy(value=1, shape=4)

        surface_density = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                                sub_grid=galaxy_data.grids.sub)

        assert (surface_density == np.array([1.0, 1.0, 1.0, 1.0])).all()

        surface_density_gal = galaxy.surface_density_from_grid(grid=galaxy_data.grids.sub)
        surface_density_gal = galaxy_data.grids.sub.sub_data_to_image(sub_array=surface_density_gal)

        surface_density_gd = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                                   sub_grid=galaxy_data.grids.sub)

        assert (surface_density_gal == surface_density_gd).all()
        
    def test__galaxy_data_potential(self, scaled_array, mask):

        galaxy_data = gd.GalaxyDataPotential(array=scaled_array, noise_map=2.0*np.ones((4,4)), mask=mask,
                                             sub_grid_size=2)

        assert scaled_array.pixel_scale == galaxy_data.pixel_scale
        assert (galaxy_data == np.ones(4)).all()
        assert (galaxy_data.array == np.ones((4,4))).all()
        assert (galaxy_data.noise_map_ == 2.0 * np.ones((4))).all()
        assert (galaxy_data.mask == np.array([[True, True, True, True],
                                              [True, False, False, True],
                                              [True, False, False, True],
                                              [True, True, True, True]])).all()

        galaxy = MockGalaxy(value=1, shape=4)

        potential = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                                sub_grid=galaxy_data.grids.sub)

        assert (potential == np.array([1.0, 1.0, 1.0, 1.0])).all()

        galaxy = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0))

        potential_gal = galaxy.potential_from_grid(grid=galaxy_data.grids.sub)
        potential_gal = galaxy_data.grids.sub.sub_data_to_image(sub_array=potential_gal)

        potential_gd = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                                   sub_grid=galaxy_data.grids.sub)

        assert (potential_gal == potential_gd).all()
        
    def test__galaxy_data_deflections_y(self, scaled_array, mask):

        galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(einstein_radius=1.0))

        galaxy_data = gd.GalaxyDataDeflectionsY(array=scaled_array, noise_map=2.0*np.ones((4,4)), mask=mask,
                                                sub_grid_size=2)

        assert scaled_array.pixel_scale == galaxy_data.pixel_scale
        assert (galaxy_data == np.ones(4)).all()
        assert (galaxy_data.array == np.ones((4,4))).all()
        assert (galaxy_data.noise_map_ == 2.0 * np.ones((4))).all()
        assert (galaxy_data.mask == np.array([[True, True, True, True],
                                              [True, False, False, True],
                                              [True, False, False, True],
                                              [True, True, True, True]])).all()

        galaxy = MockGalaxy(value=1, shape=4)

        deflections = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                            sub_grid=galaxy_data.grids.sub)

        assert (deflections[:,0] == np.array([1.0, 1.0, 1.0, 1.0])).all()
        assert (deflections[:,1] == np.array([1.0, 1.0, 1.0, 1.0])).all()

        galaxy = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0))

        deflections_gal = galaxy.deflections_from_grid(grid=galaxy_data.grids.sub)
        deflections_gal = np.asarray([galaxy_data.grids.sub.sub_data_to_image(deflections_gal[:, 0]),
                                      galaxy_data.grids.sub.sub_data_to_image(deflections_gal[:, 1])]).T

        deflections_gd = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                               sub_grid=galaxy_data.grids.sub)

        assert (deflections_gal[:,0] == deflections_gd[:,0]).all()
        assert (deflections_gal[:,1] == deflections_gd[:,1]).all()

    def test__galaxy_data_deflections_x(self, scaled_array, mask):

        galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(einstein_radius=1.0))

        galaxy_data = gd.GalaxyDataDeflectionsX(array=scaled_array, noise_map=2.0*np.ones((4,4)), mask=mask,
                                                sub_grid_size=2)

        assert scaled_array.pixel_scale == galaxy_data.pixel_scale
        assert (galaxy_data == np.ones(4)).all()
        assert (galaxy_data.array == np.ones((4,4))).all()
        assert (galaxy_data.noise_map_ == 2.0 * np.ones((4))).all()
        assert (galaxy_data.mask == np.array([[True, True, True, True],
                                              [True, False, False, True],
                                              [True, False, False, True],
                                              [True, True, True, True]])).all()

        galaxy = MockGalaxy(value=1, shape=4)

        deflections = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                            sub_grid=galaxy_data.grids.sub)

        assert (deflections[:,0] == np.array([1.0, 1.0, 1.0, 1.0])).all()
        assert (deflections[:,1] == np.array([1.0, 1.0, 1.0, 1.0])).all()

        galaxy = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0))

        deflections_gal = galaxy.deflections_from_grid(grid=galaxy_data.grids.sub)
        deflections_gal = np.asarray([galaxy_data.grids.sub.sub_data_to_image(deflections_gal[:, 0]),
                                      galaxy_data.grids.sub.sub_data_to_image(deflections_gal[:, 1])]).T

        deflections_gd = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                               sub_grid=galaxy_data.grids.sub)

        assert (deflections_gal[:,0] == deflections_gd[:,0]).all()
        assert (deflections_gal[:,1] == deflections_gd[:,1]).all()