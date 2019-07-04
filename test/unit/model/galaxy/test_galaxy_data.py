import numpy as np
import pytest

from autolens import exc
from autolens.data.array import grids
from autolens.data.array.util import grid_util
from autolens.data.array import scaled_array as sca
from autolens.model.galaxy import galaxy as g, galaxy_data as gd
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

from test.unit.mock.model import mock_galaxy

class TestGalaxyFitData(object):

    def test__image_noise_map_and_mask(self, gal_data_5x5, mask_5x5):
        
        galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, use_intensities=True)

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (galaxy_fit_data.unmasked_image == np.ones((5, 5))).all()
        assert (galaxy_fit_data.unmasked_noise_map == 2.0 * np.ones((5, 5))).all()

        assert (galaxy_fit_data.image_1d == np.ones(9)).all()
        assert (galaxy_fit_data.noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (galaxy_fit_data.mask_2d == np.array([[True, True, True, True, True],
                                                     [True, False, False, False, True],
                                                     [True, False, False, False, True],
                                                     [True, False, False, False, True],
                                                     [True,  True,  True,  True, True]])).all()

        assert (galaxy_fit_data.image_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 1.0, 1.0, 1.0, 0.0],
                                                      [0.0, 1.0, 1.0, 1.0, 0.0],
                                                      [0.0, 1.0, 1.0, 1.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0]])).all()


        assert (galaxy_fit_data.noise_map_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                          [0.0, 2.0, 2.0, 2.0, 0.0],
                                                          [0.0, 2.0, 2.0, 2.0, 0.0],
                                                          [0.0, 2.0, 2.0, 2.0, 0.0],
                                                          [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

    def test__grid_stack(self, gal_data_5x5, mask_5x5, regular_grid_5x5, sub_grid_5x5, blurring_grid_5x5):

        galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, use_intensities=True)

        assert galaxy_fit_data.grid_stack.regular.shape == (9, 2)

        assert (galaxy_fit_data.grid_stack.regular == regular_grid_5x5).all()
        assert (galaxy_fit_data.grid_stack.sub == sub_grid_5x5).all()

    def test__padded_grid_stack(self, gal_data_5x5, mask_5x5):

        galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, use_intensities=True)

        padded_image_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
            mask=np.full((7, 7), False),  pixel_scales=galaxy_fit_data.unmasked_image.pixel_scales)

        assert (galaxy_fit_data.padded_grid_stack.regular == padded_image_util).all()
        assert galaxy_fit_data.padded_grid_stack.regular.image_shape == (5, 5)
        assert galaxy_fit_data.padded_grid_stack.regular.padded_shape == (7, 7)

        padded_sub_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
             mask=np.full((7,7), False), pixel_scales=galaxy_fit_data.unmasked_image.pixel_scales,
            sub_grid_size=galaxy_fit_data.grid_stack.sub.sub_grid_size)

        assert galaxy_fit_data.padded_grid_stack.sub == pytest.approx(padded_sub_util, 1e-4)
        assert galaxy_fit_data.padded_grid_stack.sub.image_shape == (5, 5)
        assert galaxy_fit_data.padded_grid_stack.sub.padded_shape == (7, 7)

    def test__interp_pixel_scale(self, image_5x5, mask_5x5):

        noise_map = sca.ScaledSquarePixelArray(array=2.0 * np.ones((5, 5)), pixel_scale=3.0)
        gal_data_5x5 = gd.GalaxyData(image=image_5x5, noise_map=noise_map, pixel_scale=3.0)
        gal_data_5x5 = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, interp_pixel_scale=1.0, use_intensities=True)

        grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
                        mask=mask_5x5, sub_grid_size=2, psf_shape=(3, 3))
        new_grid_stack = grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(interp_pixel_scale=1.0)

        assert (gal_data_5x5.grid_stack.regular == new_grid_stack.regular).all()
        assert (gal_data_5x5.grid_stack.regular.interpolator.vtx == new_grid_stack.regular.interpolator.vtx).all()
        assert (gal_data_5x5.grid_stack.regular.interpolator.wts == new_grid_stack.regular.interpolator.wts).all()

        assert (gal_data_5x5.grid_stack.sub == new_grid_stack.sub).all()
        assert (gal_data_5x5.grid_stack.sub.interpolator.vtx == new_grid_stack.sub.interpolator.vtx).all()
        assert (gal_data_5x5.grid_stack.sub.interpolator.wts == new_grid_stack.sub.interpolator.wts).all()

        assert (gal_data_5x5.grid_stack.blurring == new_grid_stack.blurring).all()
        assert (gal_data_5x5.grid_stack.blurring.interpolator.vtx == new_grid_stack.blurring.interpolator.vtx).all()
        assert (gal_data_5x5.grid_stack.blurring.interpolator.wts == new_grid_stack.blurring.interpolator.wts).all()

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=mask_5x5, sub_grid_size=2, psf_shape=(3, 3))
        new_padded_grid_stack = \
            padded_grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(interp_pixel_scale=1.0)

        assert (gal_data_5x5.padded_grid_stack.regular == new_padded_grid_stack.regular).all()
        assert (
                    gal_data_5x5.padded_grid_stack.regular.interpolator.vtx == new_padded_grid_stack.regular.interpolator.vtx).all()
        assert (
                    gal_data_5x5.padded_grid_stack.regular.interpolator.wts == new_padded_grid_stack.regular.interpolator.wts).all()

        assert (gal_data_5x5.padded_grid_stack.sub == new_padded_grid_stack.sub).all()
        assert (gal_data_5x5.padded_grid_stack.sub.interpolator.vtx == new_padded_grid_stack.sub.interpolator.vtx).all()
        assert (gal_data_5x5.padded_grid_stack.sub.interpolator.wts == new_padded_grid_stack.sub.interpolator.wts).all()

    def test__gal_data_5x5_intensities(self, gal_data_5x5, mask_5x5):

        galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, use_intensities=True)

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (galaxy_fit_data.unmasked_image == np.ones((5, 5))).all()
        assert (galaxy_fit_data.unmasked_noise_map == 2.0 * np.ones((5, 5))).all()

        assert (galaxy_fit_data.image_1d == np.ones(9)).all()
        assert (galaxy_fit_data.noise_map_1d == 2.0* np.ones(9)).all()
        assert (galaxy_fit_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (galaxy_fit_data.mask_2d == np.array([[True, True,  True,  True, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True,  True,  True,  True, True]])).all()

        assert (galaxy_fit_data.image_2d ==  np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 1.0, 1.0, 1.0, 0.0],
                                                    [0.0, 1.0, 1.0, 1.0, 0.0],
                                                    [0.0, 1.0, 1.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0]])).all()


        assert (galaxy_fit_data.noise_map_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        galaxy = mock_galaxy.MockGalaxy(value=1, shape=36)

        intensities = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=[galaxy], sub_grid=galaxy_fit_data.grid_stack.sub)

        assert (intensities == np.ones(9)).all()

        galaxy = g.Galaxy(redshift=0.5, light=lp.SphericalSersic(intensity=1.0))

        intensities_gal = galaxy.intensities_from_grid(grid=galaxy_fit_data.grid_stack.sub)
        intensities_gal = galaxy_fit_data.grid_stack.sub.array_1d_binned_up_from_sub_array_1d(
            sub_array_1d=intensities_gal)

        intensities_gd = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=[galaxy], sub_grid=galaxy_fit_data.grid_stack.sub)

        assert (intensities_gal == intensities_gd).all()

    def test__gal_data_5x5_convergence(self, gal_data_5x5, mask_5x5):

        galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2,
                                           use_convergence=True)

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (galaxy_fit_data.unmasked_image == np.ones((5, 5))).all()
        assert (galaxy_fit_data.unmasked_noise_map == 2.0 * np.ones((5, 5))).all()

        assert (galaxy_fit_data.image_1d == np.ones(9)).all()
        assert (galaxy_fit_data.noise_map_1d == 2.0* np.ones(9)).all()
        assert (galaxy_fit_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (galaxy_fit_data.mask_2d == np.array([[True, True,  True,  True, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True,  True,  True,  True, True]])).all()

        assert (galaxy_fit_data.image_2d ==  np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 1.0, 1.0, 1.0, 0.0],
                                                    [0.0, 1.0, 1.0, 1.0, 0.0],
                                                    [0.0, 1.0, 1.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0]])).all()


        assert (galaxy_fit_data.noise_map_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        galaxy = mock_galaxy.MockGalaxy(value=1, shape=36)

        convergence = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(galaxies=[galaxy],
                                                                                    sub_grid=galaxy_fit_data.grid_stack.sub)

        assert (convergence == np.ones(9)).all()

        galaxy = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))

        convergence_gal = galaxy.convergence_from_grid(grid=galaxy_fit_data.grid_stack.sub)
        convergence_gal = galaxy_fit_data.grid_stack.sub.array_1d_binned_up_from_sub_array_1d(sub_array_1d=convergence_gal)

        convergence_gd = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(galaxies=[galaxy],
                                                                                       sub_grid=galaxy_fit_data.grid_stack.sub)

        assert (convergence_gal == convergence_gd).all()
        
    def test__gal_data_5x5_potential(self, gal_data_5x5, mask_5x5):

        galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2,
                                           use_potential=True)

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (galaxy_fit_data.unmasked_image == np.ones((5, 5))).all()
        assert (galaxy_fit_data.unmasked_noise_map == 2.0 * np.ones((5, 5))).all()

        assert (galaxy_fit_data.image_1d == np.ones(9)).all()
        assert (galaxy_fit_data.noise_map_1d == 2.0* np.ones(9)).all()
        assert (galaxy_fit_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (galaxy_fit_data.mask_2d == np.array([[True, True,  True,  True, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True,  True,  True,  True, True]])).all()

        assert (galaxy_fit_data.image_2d ==  np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 1.0, 1.0, 1.0, 0.0],
                                                    [0.0, 1.0, 1.0, 1.0, 0.0],
                                                    [0.0, 1.0, 1.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0]])).all()


        assert (galaxy_fit_data.noise_map_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0]])).all()
        galaxy = mock_galaxy.MockGalaxy(value=1, shape=36)

        potential = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(galaxies=[galaxy],
                                                                              sub_grid=galaxy_fit_data.grid_stack.sub)

        assert (potential == np.ones(9)).all()

        galaxy = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))

        potential_gal = galaxy.potential_from_grid(grid=galaxy_fit_data.grid_stack.sub)
        potential_gal = galaxy_fit_data.grid_stack.sub.array_1d_binned_up_from_sub_array_1d(sub_array_1d=potential_gal)

        potential_gd = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(galaxies=[galaxy],
                                                                                 sub_grid=galaxy_fit_data.grid_stack.sub)

        assert (potential_gal == potential_gd).all()
        
    def test__gal_data_5x5_deflections_y(self, gal_data_5x5, mask_5x5):

        galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2,
                                           use_deflections_y=True)
        assert galaxy_fit_data.pixel_scale == 1.0
        assert (galaxy_fit_data.unmasked_image == np.ones((5, 5))).all()
        assert (galaxy_fit_data.unmasked_noise_map == 2.0 * np.ones((5, 5))).all()

        assert (galaxy_fit_data.image_1d == np.ones(9)).all()
        assert (galaxy_fit_data.noise_map_1d == 2.0* np.ones(9)).all()
        assert (galaxy_fit_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (galaxy_fit_data.mask_2d == np.array([[True, True,  True,  True, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True,  True,  True,  True, True]])).all()

        assert (galaxy_fit_data.image_2d ==  np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 1.0, 1.0, 1.0, 0.0],
                                                    [0.0, 1.0, 1.0, 1.0, 0.0],
                                                    [0.0, 1.0, 1.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0]])).all()


        assert (galaxy_fit_data.noise_map_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 2.0, 2.0, 2.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        galaxy = mock_galaxy.MockGalaxy(value=1, shape=36)

        deflections_y = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(galaxies=[galaxy],
                                                                                  sub_grid=galaxy_fit_data.grid_stack.sub)

        assert (deflections_y == np.ones(9)).all()

        galaxy = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))

        deflections_gal = galaxy.deflections_from_grid(grid=galaxy_fit_data.grid_stack.sub)
        deflections_gal = np.asarray([galaxy_fit_data.grid_stack.sub.array_1d_binned_up_from_sub_array_1d(deflections_gal[:, 0]),
                                      galaxy_fit_data.grid_stack.sub.array_1d_binned_up_from_sub_array_1d(deflections_gal[:, 1])]).T

        deflections_gd = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(galaxies=[galaxy],
                                                                                   sub_grid=galaxy_fit_data.grid_stack.sub)

        assert (deflections_gal[:,0] == deflections_gd).all()

    def test__gal_data_5x5_deflections_x(self, gal_data_5x5, mask_5x5):

        galaxy_fit_data = gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2, use_deflections_x=True)

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (galaxy_fit_data.unmasked_image == np.ones((5, 5))).all()
        assert (galaxy_fit_data.unmasked_noise_map == 2.0 * np.ones((5, 5))).all()

        assert (galaxy_fit_data.image_1d == np.ones(9)).all()
        assert (galaxy_fit_data.noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (galaxy_fit_data.mask_2d == np.array([[True, True, True, True, True],
                                                     [True, False, False, False, True],
                                                     [True, False, False, False, True],
                                                     [True, False, False, False, True],
                                                     [True, True, True, True, True]])).all()

        assert (galaxy_fit_data.image_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 1.0, 1.0, 1.0, 0.0],
                                                      [0.0, 1.0, 1.0, 1.0, 0.0],
                                                      [0.0, 1.0, 1.0, 1.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        assert (galaxy_fit_data.noise_map_2d == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                          [0.0, 2.0, 2.0, 2.0, 0.0],
                                                          [0.0, 2.0, 2.0, 2.0, 0.0],
                                                          [0.0, 2.0, 2.0, 2.0, 0.0],
                                                          [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        galaxy = mock_galaxy.MockGalaxy(value=1, shape=36)

        deflections_x = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(galaxies=[galaxy],
                                                                                  sub_grid=galaxy_fit_data.grid_stack.sub)

        assert (deflections_x == np.ones(9)).all()

        galaxy = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))

        deflections_gal = galaxy.deflections_from_grid(grid=galaxy_fit_data.grid_stack.sub)
        deflections_gal = np.asarray([galaxy_fit_data.grid_stack.sub.array_1d_binned_up_from_sub_array_1d(deflections_gal[:, 0]),
                                      galaxy_fit_data.grid_stack.sub.array_1d_binned_up_from_sub_array_1d(deflections_gal[:, 1])]).T

        deflections_gd = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(galaxies=[galaxy],
                                                                                   sub_grid=galaxy_fit_data.grid_stack.sub)

        assert (deflections_gal[:,1] == deflections_gd).all()

    def test__no_use_method__raises_exception(self, image_5x5, mask_5x5):

        gal_data_5x5 = gd.GalaxyData(image=image_5x5, noise_map=2.0*np.ones((5, 5)), pixel_scale=3.0)

        with pytest.raises(exc.GalaxyException):
            gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2)

    def test__multiple_use_methods__raises_exception(self, image_5x5, mask_5x5):

        gal_data_5x5 = gd.GalaxyData(image=image_5x5, noise_map=2.0*np.ones((5, 5)), pixel_scale=3.0)

        with pytest.raises(exc.GalaxyException):
            gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2,
                             use_intensities=True, use_convergence=True)

        with pytest.raises(exc.GalaxyException):
            gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2,
                             use_intensities=True, use_potential=True)

        with pytest.raises(exc.GalaxyException):
            gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2,
                             use_intensities=True, use_deflections_y=True)

        with pytest.raises(exc.GalaxyException):
            gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2,
                             use_intensities=True, use_convergence=True, use_potential=True)

        with pytest.raises(exc.GalaxyException):
                gd.GalaxyFitData(galaxy_data=gal_data_5x5, mask=mask_5x5, sub_grid_size=2,
                                 use_intensities=True, use_convergence=True, use_potential=True, use_deflections_x=True)