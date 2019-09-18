import autolens as al
import numpy as np
import pytest


from test.unit.mock.model import mock_galaxy
from autolens import exc


class TestGalaxyFitData(object):
    def test__image_noise_map_and_mask(self, gal_data_7x7, sub_mask_7x7):

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_image=True
        )

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (
            galaxy_fit_data.image(return_in_2d=True, return_masked=False)
            == np.ones((7, 7))
        ).all()
        assert (
            galaxy_fit_data.noise_map(return_in_2d=True, return_masked=False)
            == 2.0 * np.ones((7, 7))
        ).all()

        assert (galaxy_fit_data._image_1d == np.ones(9)).all()
        assert (galaxy_fit_data._noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data._mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (
            galaxy_fit_data.mask
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

    def test__grid(self, gal_data_7x7, sub_mask_7x7, sub_grid_7x7):

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_image=True
        )

        assert (galaxy_fit_data.grid == sub_grid_7x7).all()

    def test__pixel_scale_interpolation_grid(self, image_7x7, sub_mask_7x7):

        noise_map = al.ScaledSquarePixelArray(
            array=2.0 * np.ones((7, 7)), pixel_scale=3.0
        )
        gal_data_7x7 = al.GalaxyData(
            image=image_7x7, noise_map=noise_map, pixel_scale=3.0
        )
        gal_data_7x7 = al.GalaxyFitData(
            galaxy_data=gal_data_7x7,
            mask=sub_mask_7x7,
            pixel_scale_interpolation_grid=1.0,
            use_image=True,
        )

        grid = al.Grid.from_mask(mask=sub_mask_7x7)
        new_grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=1.0)
        assert (gal_data_7x7.grid == new_grid).all()
        assert (gal_data_7x7.grid.interpolator.vtx == new_grid.interpolator.vtx).all()
        assert (gal_data_7x7.grid.interpolator.wts == new_grid.interpolator.wts).all()

    def test__gal_data_7x7_image(self, gal_data_7x7, sub_mask_7x7):

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_image=True
        )

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (
            galaxy_fit_data.image(return_in_2d=True, return_masked=False)
            == np.ones((7, 7))
        ).all()
        assert (
            galaxy_fit_data.noise_map(return_in_2d=True, return_masked=False)
            == 2.0 * np.ones((7, 7))
        ).all()

        assert (galaxy_fit_data._image_1d == np.ones(9)).all()
        assert (galaxy_fit_data._noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data._mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (
            galaxy_fit_data.mask
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

        image = galaxy_fit_data.profile_quantity_from_galaxies(galaxies=[galaxy])

        assert (image == np.ones(9)).all()

        galaxy = al.Galaxy(
            redshift=0.5, light=al.light_profiles.SphericalSersic(intensity=1.0)
        )

        image_gal = galaxy.profile_image_from_grid(
            grid=galaxy_fit_data.grid, return_in_2d=False, return_binned=True
        )

        image_gd = galaxy_fit_data.profile_quantity_from_galaxies(galaxies=[galaxy])

        assert (image_gal == image_gd).all()

    def test__gal_data_7x7_convergence(self, gal_data_7x7, sub_mask_7x7):

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_convergence=True
        )

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (
            galaxy_fit_data.image(return_in_2d=True, return_masked=False)
            == np.ones((7, 7))
        ).all()
        assert (
            galaxy_fit_data.noise_map(return_in_2d=True, return_masked=False)
            == 2.0 * np.ones((7, 7))
        ).all()

        assert (galaxy_fit_data._image_1d == np.ones(9)).all()
        assert (galaxy_fit_data._noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data._mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (
            galaxy_fit_data.mask
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

        convergence = galaxy_fit_data.profile_quantity_from_galaxies(galaxies=[galaxy])

        assert (convergence == np.ones(9)).all()

        galaxy = al.Galaxy(
            redshift=0.5, mass=al.mass_profiles.SphericalIsothermal(einstein_radius=1.0)
        )

        convergence_gal = galaxy.convergence_from_grid(
            grid=galaxy_fit_data.grid, return_in_2d=False, return_binned=True
        )

        convergence_gd = galaxy_fit_data.profile_quantity_from_galaxies(
            galaxies=[galaxy]
        )

        assert (convergence_gal == convergence_gd).all()

    def test__gal_data_7x7_potential(self, gal_data_7x7, sub_mask_7x7):

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_potential=True
        )

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (
            galaxy_fit_data.image(return_in_2d=True, return_masked=False)
            == np.ones((7, 7))
        ).all()
        assert (
            galaxy_fit_data.noise_map(return_in_2d=True, return_masked=False)
            == 2.0 * np.ones((7, 7))
        ).all()

        assert (galaxy_fit_data._image_1d == np.ones(9)).all()
        assert (galaxy_fit_data._noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data._mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (
            galaxy_fit_data.mask
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

        potential = galaxy_fit_data.profile_quantity_from_galaxies(galaxies=[galaxy])

        assert (potential == np.ones(9)).all()

        galaxy = al.Galaxy(
            redshift=0.5, mass=al.mass_profiles.SphericalIsothermal(einstein_radius=1.0)
        )

        potential_gal = galaxy.potential_from_grid(
            grid=galaxy_fit_data.grid, return_in_2d=False, return_binned=True
        )

        potential_gd = galaxy_fit_data.profile_quantity_from_galaxies(galaxies=[galaxy])

        assert (potential_gal == potential_gd).all()

    def test__gal_data_7x7_deflections_y(self, gal_data_7x7, sub_mask_7x7):

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_deflections_y=True
        )
        assert galaxy_fit_data.pixel_scale == 1.0
        assert (
            galaxy_fit_data.image(return_in_2d=True, return_masked=False)
            == np.ones((7, 7))
        ).all()
        assert (
            galaxy_fit_data.noise_map(return_in_2d=True, return_masked=False)
            == 2.0 * np.ones((7, 7))
        ).all()

        assert (galaxy_fit_data._image_1d == np.ones(9)).all()
        assert (galaxy_fit_data._noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data._mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (
            galaxy_fit_data.mask
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

        deflections_y = galaxy_fit_data.profile_quantity_from_galaxies(
            galaxies=[galaxy]
        )

        assert (deflections_y == np.ones(9)).all()

        galaxy = al.Galaxy(
            redshift=0.5, mass=al.mass_profiles.SphericalIsothermal(einstein_radius=1.0)
        )

        deflections_gal = galaxy.deflections_from_grid(grid=galaxy_fit_data.grid)
        deflections_gal = np.asarray(
            [
                galaxy_fit_data.grid.mapping.array_1d_binned_from_sub_array_1d(
                    deflections_gal[:, 0]
                ),
                galaxy_fit_data.grid.mapping.array_1d_binned_from_sub_array_1d(
                    deflections_gal[:, 1]
                ),
            ]
        ).T

        deflections_gd = galaxy_fit_data.profile_quantity_from_galaxies(
            galaxies=[galaxy]
        )

        assert (deflections_gal[:, 0] == deflections_gd).all()

    def test__gal_data_7x7_deflections_x(self, gal_data_7x7, sub_mask_7x7):

        galaxy_fit_data = al.GalaxyFitData(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_deflections_x=True
        )

        assert galaxy_fit_data.pixel_scale == 1.0
        assert (
            galaxy_fit_data.image(return_in_2d=True, return_masked=False)
            == np.ones((7, 7))
        ).all()
        assert (
            galaxy_fit_data.noise_map(return_in_2d=True, return_masked=False)
            == 2.0 * np.ones((7, 7))
        ).all()

        assert (galaxy_fit_data._image_1d == np.ones(9)).all()
        assert (galaxy_fit_data._noise_map_1d == 2.0 * np.ones(9)).all()
        assert (galaxy_fit_data._mask_1d == np.full(fill_value=False, shape=(9))).all()

        assert (
            galaxy_fit_data.mask
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

        deflections_x = galaxy_fit_data.profile_quantity_from_galaxies(
            galaxies=[galaxy]
        )

        assert (deflections_x == np.ones(9)).all()

        galaxy = al.Galaxy(
            redshift=0.5, mass=al.mass_profiles.SphericalIsothermal(einstein_radius=1.0)
        )

        deflections_gal = galaxy.deflections_from_grid(grid=galaxy_fit_data.grid)
        deflections_gal = np.asarray(
            [
                galaxy_fit_data.grid.mapping.array_1d_binned_from_sub_array_1d(
                    deflections_gal[:, 0]
                ),
                galaxy_fit_data.grid.mapping.array_1d_binned_from_sub_array_1d(
                    deflections_gal[:, 1]
                ),
            ]
        ).T

        deflections_gd = galaxy_fit_data.profile_quantity_from_galaxies(
            galaxies=[galaxy]
        )

        assert (deflections_gal[:, 1] == deflections_gd).all()

    def test__no_use_method__raises_exception(self, image_7x7, sub_mask_7x7):

        gal_data_7x7 = al.GalaxyData(
            image=image_7x7, noise_map=2.0 * np.ones((7, 7)), pixel_scale=3.0
        )

        with pytest.raises(exc.GalaxyException):
            al.GalaxyFitData(galaxy_data=gal_data_7x7, mask=sub_mask_7x7)

    def test__multiple_use_methods__raises_exception(self, image_7x7, sub_mask_7x7):

        gal_data_7x7 = al.GalaxyData(
            image=image_7x7, noise_map=2.0 * np.ones((7, 7)), pixel_scale=3.0
        )

        with pytest.raises(exc.GalaxyException):
            al.GalaxyFitData(
                galaxy_data=gal_data_7x7,
                mask=sub_mask_7x7,
                use_image=True,
                use_convergence=True,
            )

        with pytest.raises(exc.GalaxyException):
            al.GalaxyFitData(
                galaxy_data=gal_data_7x7,
                mask=sub_mask_7x7,
                use_image=True,
                use_potential=True,
            )

        with pytest.raises(exc.GalaxyException):
            al.GalaxyFitData(
                galaxy_data=gal_data_7x7,
                mask=sub_mask_7x7,
                use_image=True,
                use_deflections_y=True,
            )

        with pytest.raises(exc.GalaxyException):
            al.GalaxyFitData(
                galaxy_data=gal_data_7x7,
                mask=sub_mask_7x7,
                use_image=True,
                use_convergence=True,
                use_potential=True,
            )

        with pytest.raises(exc.GalaxyException):
            al.GalaxyFitData(
                galaxy_data=gal_data_7x7,
                mask=sub_mask_7x7,
                use_image=True,
                use_convergence=True,
                use_potential=True,
                use_deflections_x=True,
            )
