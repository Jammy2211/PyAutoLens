import numpy as np
import pytest

from autolens import exc
from autolens.data.array import grids
from autolens.data.array import scaled_array as sca
from autolens.model.galaxy import galaxy as g, galaxy_data as gd
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

from test.unit.mock.model import mock_galaxy


class TestGalaxyFitData(object):
    def test__image_noise_map_and_mask(self, gal_data_7x7, mask_7x7):

        galaxy_fit_data = gd.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=mask_7x7, use_intensities=True
        )

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (galaxy_fit_data.unmasked_image == np.ones((7, 7))).all()
        assert (galaxy_fit_data.unmasked_noise_map == 2.0 * np.ones((7, 7))).all()

        assert (galaxy_fit_data.image_1d == np.ones(9)).all()
        assert (galaxy_fit_data.noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (
            galaxy_fit_data.mask(return_in_2d=True)
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()

        assert (
            galaxy_fit_data.image(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        assert (
            galaxy_fit_data.noise_map(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__grid_stack(
        self, gal_data_7x7, mask_7x7, grid_7x7, sub_grid_7x7, blurring_grid_7x7
    ):

        galaxy_fit_data = gd.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=mask_7x7, use_intensities=True
        )

        assert galaxy_fit_data.grid_stack.regular.shape == (9, 2)

        assert (galaxy_fit_data.grid_stack.regular == grid_7x7).all()
        assert (galaxy_fit_data.grid_stack.sub == sub_grid_7x7).all()

    def test__interp_pixel_scale(self, image_7x7, mask_7x7):

        noise_map = sca.ScaledSquarePixelArray(
            array=2.0 * np.ones((7, 7)), pixel_scale=3.0
        )
        gal_data_7x7 = gd.GalaxyData(
            image=image_7x7, noise_map=noise_map, pixel_scale=3.0
        )
        gal_data_7x7 = gd.GalaxyFitData(
            galaxy_data=gal_data_7x7,
            mask=mask_7x7,
            interp_pixel_scale=1.0,
            use_intensities=True,
        )

        grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=mask_7x7, sub_grid_size=2, psf_shape=(3, 3)
        )
        new_grid_stack = grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(
            interp_pixel_scale=1.0
        )

        assert (gal_data_7x7.grid_stack.regular == new_grid_stack.regular).all()
        assert (
            gal_data_7x7.grid_stack.regular.interpolator.vtx
            == new_grid_stack.regular.interpolator.vtx
        ).all()
        assert (
            gal_data_7x7.grid_stack.regular.interpolator.wts
            == new_grid_stack.regular.interpolator.wts
        ).all()

        assert (gal_data_7x7.grid_stack.sub == new_grid_stack.sub).all()
        assert (
            gal_data_7x7.grid_stack.sub.interpolator.vtx
            == new_grid_stack.sub.interpolator.vtx
        ).all()
        assert (
            gal_data_7x7.grid_stack.sub.interpolator.wts
            == new_grid_stack.sub.interpolator.wts
        ).all()

        assert (gal_data_7x7.grid_stack.blurring == new_grid_stack.blurring).all()
        assert (
            gal_data_7x7.grid_stack.blurring.interpolator.vtx
            == new_grid_stack.blurring.interpolator.vtx
        ).all()
        assert (
            gal_data_7x7.grid_stack.blurring.interpolator.wts
            == new_grid_stack.blurring.interpolator.wts
        ).all()

    def test__gal_data_7x7_intensities(self, gal_data_7x7, mask_7x7):

        galaxy_fit_data = gd.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=mask_7x7, use_intensities=True
        )

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (galaxy_fit_data.unmasked_image == np.ones((7, 7))).all()
        assert (galaxy_fit_data.unmasked_noise_map == 2.0 * np.ones((7, 7))).all()

        assert (galaxy_fit_data.image_1d == np.ones(9)).all()
        assert (galaxy_fit_data.noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (
            galaxy_fit_data.mask(return_in_2d=True)
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()

        assert (
            galaxy_fit_data.image(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        assert (
            galaxy_fit_data.noise_map(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        galaxy = mock_galaxy.MockGalaxy(value=1, shape=36)

        intensities = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=[galaxy], sub_grid=galaxy_fit_data.grid_stack.sub
        )

        assert (intensities == np.ones(9)).all()

        galaxy = g.Galaxy(redshift=0.5, light=lp.SphericalSersic(intensity=1.0))

        intensities_gal = galaxy.intensities_from_grid(
            grid=galaxy_fit_data.grid_stack.sub, return_in_2d=False, return_binned=True
        )

        intensities_gd = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=[galaxy], sub_grid=galaxy_fit_data.grid_stack.sub
        )

        assert (intensities_gal == intensities_gd).all()

    def test__gal_data_7x7_convergence(self, gal_data_7x7, mask_7x7):

        galaxy_fit_data = gd.GalaxyFitData(
            galaxy_data=gal_data_7x7,
            mask=mask_7x7,
            sub_grid_size=2,
            use_convergence=True,
        )

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (galaxy_fit_data.unmasked_image == np.ones((7, 7))).all()
        assert (galaxy_fit_data.unmasked_noise_map == 2.0 * np.ones((7, 7))).all()

        assert (galaxy_fit_data.image_1d == np.ones(9)).all()
        assert (galaxy_fit_data.noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (
            galaxy_fit_data.mask(return_in_2d=True)
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()

        assert (
            galaxy_fit_data.image(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        assert (
            galaxy_fit_data.noise_map(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        galaxy = mock_galaxy.MockGalaxy(value=1, shape=36)

        convergence = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=[galaxy], sub_grid=galaxy_fit_data.grid_stack.sub
        )

        assert (convergence == np.ones(9)).all()

        galaxy = g.Galaxy(
            redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0)
        )

        convergence_gal = galaxy.convergence_from_grid(
            grid=galaxy_fit_data.grid_stack.sub, return_in_2d=False, return_binned=True
        )

        convergence_gd = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=[galaxy], sub_grid=galaxy_fit_data.grid_stack.sub
        )

        assert (convergence_gal == convergence_gd).all()

    def test__gal_data_7x7_potential(self, gal_data_7x7, mask_7x7):

        galaxy_fit_data = gd.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=mask_7x7, sub_grid_size=2, use_potential=True
        )

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (galaxy_fit_data.unmasked_image == np.ones((7, 7))).all()
        assert (galaxy_fit_data.unmasked_noise_map == 2.0 * np.ones((7, 7))).all()

        assert (galaxy_fit_data.image_1d == np.ones(9)).all()
        assert (galaxy_fit_data.noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (
            galaxy_fit_data.mask(return_in_2d=True)
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()

        assert (
            galaxy_fit_data.image(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        assert (
            galaxy_fit_data.noise_map(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        galaxy = mock_galaxy.MockGalaxy(value=1, shape=36)

        potential = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=[galaxy], sub_grid=galaxy_fit_data.grid_stack.sub
        )

        assert (potential == np.ones(9)).all()

        galaxy = g.Galaxy(
            redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0)
        )

        potential_gal = galaxy.potential_from_grid(
            grid=galaxy_fit_data.grid_stack.sub, return_binned=True
        )

        potential_gd = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=[galaxy], sub_grid=galaxy_fit_data.grid_stack.sub
        )

        assert (potential_gal == potential_gd).all()

    def test__gal_data_7x7_deflections_y(self, gal_data_7x7, mask_7x7):

        galaxy_fit_data = gd.GalaxyFitData(
            galaxy_data=gal_data_7x7,
            mask=mask_7x7,
            sub_grid_size=2,
            use_deflections_y=True,
        )
        assert galaxy_fit_data.pixel_scale == 1.0
        assert (galaxy_fit_data.unmasked_image == np.ones((7, 7))).all()
        assert (galaxy_fit_data.unmasked_noise_map == 2.0 * np.ones((7, 7))).all()

        assert (galaxy_fit_data.image_1d == np.ones(9)).all()
        assert (galaxy_fit_data.noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (
            galaxy_fit_data.mask(return_in_2d=True)
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()

        assert (
            galaxy_fit_data.image(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        assert (
            galaxy_fit_data.noise_map(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        galaxy = mock_galaxy.MockGalaxy(value=1, shape=36)

        deflections_y = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=[galaxy], sub_grid=galaxy_fit_data.grid_stack.sub
        )

        assert (deflections_y == np.ones(9)).all()

        galaxy = g.Galaxy(
            redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0)
        )

        deflections_gal = galaxy.deflections_from_grid(
            grid=galaxy_fit_data.grid_stack.sub
        )
        deflections_gal = np.asarray(
            [
                galaxy_fit_data.grid_stack.sub.array_1d_binned_from_sub_array_1d(
                    deflections_gal[:, 0]
                ),
                galaxy_fit_data.grid_stack.sub.array_1d_binned_from_sub_array_1d(
                    deflections_gal[:, 1]
                ),
            ]
        ).T

        deflections_gd = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=[galaxy], sub_grid=galaxy_fit_data.grid_stack.sub
        )

        assert (deflections_gal[:, 0] == deflections_gd).all()

    def test__gal_data_7x7_deflections_x(self, gal_data_7x7, mask_7x7):

        galaxy_fit_data = gd.GalaxyFitData(
            galaxy_data=gal_data_7x7,
            mask=mask_7x7,
            sub_grid_size=2,
            use_deflections_x=True,
        )

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (galaxy_fit_data.unmasked_image == np.ones((7, 7))).all()
        assert (galaxy_fit_data.unmasked_noise_map == 2.0 * np.ones((7, 7))).all()

        assert (galaxy_fit_data.image_1d == np.ones(9)).all()
        assert (galaxy_fit_data.noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data.mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (
            galaxy_fit_data.mask(return_in_2d=True)
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()

        assert (
            galaxy_fit_data.image(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        assert (
            galaxy_fit_data.noise_map(return_in_2d=True)
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        galaxy = mock_galaxy.MockGalaxy(value=1, shape=36)

        deflections_x = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=[galaxy], sub_grid=galaxy_fit_data.grid_stack.sub
        )

        assert (deflections_x == np.ones(9)).all()

        galaxy = g.Galaxy(
            redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0)
        )

        deflections_gal = galaxy.deflections_from_grid(
            grid=galaxy_fit_data.grid_stack.sub
        )
        deflections_gal = np.asarray(
            [
                galaxy_fit_data.grid_stack.sub.array_1d_binned_from_sub_array_1d(
                    deflections_gal[:, 0]
                ),
                galaxy_fit_data.grid_stack.sub.array_1d_binned_from_sub_array_1d(
                    deflections_gal[:, 1]
                ),
            ]
        ).T

        deflections_gd = galaxy_fit_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=[galaxy], sub_grid=galaxy_fit_data.grid_stack.sub
        )

        assert (deflections_gal[:, 1] == deflections_gd).all()

    def test__no_use_method__raises_exception(self, image_7x7, mask_7x7):

        gal_data_7x7 = gd.GalaxyData(
            image=image_7x7, noise_map=2.0 * np.ones((7, 7)), pixel_scale=3.0
        )

        with pytest.raises(exc.GalaxyException):
            gd.GalaxyFitData(galaxy_data=gal_data_7x7, mask=mask_7x7, sub_grid_size=2)

    def test__multiple_use_methods__raises_exception(self, image_7x7, mask_7x7):

        gal_data_7x7 = gd.GalaxyData(
            image=image_7x7, noise_map=2.0 * np.ones((7, 7)), pixel_scale=3.0
        )

        with pytest.raises(exc.GalaxyException):
            gd.GalaxyFitData(
                galaxy_data=gal_data_7x7,
                mask=mask_7x7,
                sub_grid_size=2,
                use_intensities=True,
                use_convergence=True,
            )

        with pytest.raises(exc.GalaxyException):
            gd.GalaxyFitData(
                galaxy_data=gal_data_7x7,
                mask=mask_7x7,
                sub_grid_size=2,
                use_intensities=True,
                use_potential=True,
            )

        with pytest.raises(exc.GalaxyException):
            gd.GalaxyFitData(
                galaxy_data=gal_data_7x7,
                mask=mask_7x7,
                sub_grid_size=2,
                use_intensities=True,
                use_deflections_y=True,
            )

        with pytest.raises(exc.GalaxyException):
            gd.GalaxyFitData(
                galaxy_data=gal_data_7x7,
                mask=mask_7x7,
                sub_grid_size=2,
                use_intensities=True,
                use_convergence=True,
                use_potential=True,
            )

        with pytest.raises(exc.GalaxyException):
            gd.GalaxyFitData(
                galaxy_data=gal_data_7x7,
                mask=mask_7x7,
                sub_grid_size=2,
                use_intensities=True,
                use_convergence=True,
                use_potential=True,
                use_deflections_x=True,
            )
