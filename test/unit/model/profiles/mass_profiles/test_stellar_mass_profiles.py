import numpy as np
import pytest

from autolens import dimensions as dim
from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.model.profiles import mass_profiles as mp

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestSersic(object):
    def test__constructor_and_units(self):

        sersic = mp.EllipticalSersic(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 0.5
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 45.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        sersic = mp.SphericalSersic(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        sersic = mp.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[0.0, 1.5]])
        ) == pytest.approx(4.90657319276, 1e-3)

        sersic = mp.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=6.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[0.0, 1.5]])
        ) == pytest.approx(2.0 * 4.90657319276, 1e-3)

        sersic = mp.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[0.0, 1.5]])
        ) == pytest.approx(2.0 * 4.90657319276, 1e-3)

        sersic = mp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[1.0, 0.0]])
        ) == pytest.approx(5.38066670129, 1e-3)

    def test__deflections_correct_values(self):
        sersic = mp.EllipticalSersic(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        deflections = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(1.1446, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.79374, 1e-3)

        sersic = mp.EllipticalSersic(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        deflections = sersic.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625], [0.1625, 0.1625]])
        )
        assert deflections[0, 0] == pytest.approx(1.1446, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.79374, 1e-3)
        assert deflections[1, 0] == pytest.approx(1.1446, 1e-3)
        assert deflections[1, 1] == pytest.approx(0.79374, 1e-3)

    def test__surfce_density__change_geometry(self):
        sersic_0 = mp.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = mp.SphericalSersic(centre=(1.0, 1.0))
        assert sersic_0.convergence_from_grid(
            grid=np.array([[1.0, 1.0]])
        ) == sersic_1.convergence_from_grid(grid=np.array([[0.0, 0.0]]))

        sersic_0 = mp.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = mp.SphericalSersic(centre=(0.0, 0.0))
        assert sersic_0.convergence_from_grid(
            grid=np.array([[1.0, 0.0]])
        ) == sersic_1.convergence_from_grid(grid=np.array([[0.0, 1.0]]))

        sersic_0 = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        sersic_1 = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        assert sersic_0.convergence_from_grid(
            grid=np.array([[1.0, 0.0]])
        ) == sersic_1.convergence_from_grid(grid=np.array([[0.0, 1.0]]))

    def test__deflections__change_geometry(self):
        sersic_0 = mp.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = mp.SphericalSersic(centre=(1.0, 1.0))
        deflections_0 = sersic_0.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        deflections_1 = sersic_1.deflections_from_grid(grid=np.array([[0.0, 0.0]]))
        assert deflections_0[0, 0] == pytest.approx(-deflections_1[0, 0], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(-deflections_1[0, 1], 1e-5)

        sersic_0 = mp.SphericalSersic(centre=(0.0, 0.0))
        sersic_1 = mp.SphericalSersic(centre=(0.0, 0.0))
        deflections_0 = sersic_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        deflections_1 = sersic_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert deflections_0[0, 0] == pytest.approx(deflections_1[0, 1], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(deflections_1[0, 0], 1e-5)

        sersic_0 = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        sersic_1 = mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        deflections_0 = sersic_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        deflections_1 = sersic_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert deflections_0[0, 0] == pytest.approx(deflections_1[0, 1], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(deflections_1[0, 0], 1e-5)

    def test__spherical_and_elliptical_identical(self):
        elliptical = mp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
        )

        spherical = mp.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
        )

        assert (
            elliptical.convergence_from_grid(grid)
            == spherical.convergence_from_grid(grid)
        ).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid),
            spherical.deflections_from_grid(grid),
        )

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        sersic = mp.EllipticalSersic(
            centre=(-0.7, 0.5),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        sersic = mp.SphericalSersic(
            centre=(-0.7, 0.5),
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__reshape_decorators(self):

        regular_grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=1.0
        )

        sersic = mp.EllipticalSersic()

        convergence = sersic.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = sersic.potential_from_grid(
        #     grid=regular_grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = sersic.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)

        sersic = mp.SphericalSersic()

        convergence = sersic.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = sersic.potential_from_grid(
        #     grid=regular_grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = sersic.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)


class TestExponential(object):
    def test__constructor_and_units(self):
        exponential = mp.EllipticalExponential(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
        )

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], dim.Length)
        assert isinstance(exponential.centre[1], dim.Length)
        assert exponential.centre[0].unit == "arcsec"
        assert exponential.centre[1].unit == "arcsec"

        assert exponential.axis_ratio == 0.5
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 45.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, dim.Luminosity)
        assert exponential.intensity.unit == "eps"

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, dim.Length)
        assert exponential.effective_radius.unit_length == "arcsec"

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.mass_to_light_ratio == 10.0
        assert isinstance(exponential.mass_to_light_ratio, dim.MassOverLuminosity)
        assert exponential.mass_to_light_ratio.unit == "angular / eps"

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        exponential = mp.SphericalExponential(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
        )

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], dim.Length)
        assert isinstance(exponential.centre[1], dim.Length)
        assert exponential.centre[0].unit == "arcsec"
        assert exponential.centre[1].unit == "arcsec"

        assert exponential.axis_ratio == 1.0
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 0.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, dim.Luminosity)
        assert exponential.intensity.unit == "eps"

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, dim.Length)
        assert exponential.effective_radius.unit_length == "arcsec"

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.mass_to_light_ratio == 10.0
        assert isinstance(exponential.mass_to_light_ratio, dim.MassOverLuminosity)
        assert exponential.mass_to_light_ratio.unit == "angular / eps"

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        exponential = mp.EllipticalExponential(
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            mass_to_light_ratio=1.0,
        )
        assert exponential.convergence_from_grid(
            grid=np.array([[1.0, 0.0]])
        ) == pytest.approx(4.9047, 1e-3)

        exponential = mp.EllipticalExponential(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert exponential.convergence_from_grid(
            grid=np.array([[0.0, 1.0]])
        ) == pytest.approx(4.8566, 1e-3)

        exponential = mp.EllipticalExponential(
            axis_ratio=0.5,
            phi=90.0,
            intensity=4.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert exponential.convergence_from_grid(
            grid=np.array([[0.0, 1.0]])
        ) == pytest.approx(2.0 * 4.8566, 1e-3)

        exponential = mp.EllipticalExponential(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=2.0,
        )
        assert exponential.convergence_from_grid(
            grid=np.array([[0.0, 1.0]])
        ) == pytest.approx(2.0 * 4.8566, 1e-3)

        exponential = mp.EllipticalExponential(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert exponential.convergence_from_grid(
            grid=np.array([[0.0, 1.0]])
        ) == pytest.approx(4.8566, 1e-3)

    def test__deflections_correct_values(self):
        exponential = mp.EllipticalExponential(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )
        deflections = exponential.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]])
        )
        assert deflections[0, 0] == pytest.approx(0.90493, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.62569, 1e-3)

        exponential = mp.EllipticalExponential(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )
        deflections = exponential.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]])
        )
        assert deflections[0, 0] == pytest.approx(0.90493, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.62569, 1e-3)

    def test__spherical_and_elliptical_identical(self):
        elliptical = mp.EllipticalExponential(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            mass_to_light_ratio=1.0,
        )

        spherical = mp.SphericalExponential(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            mass_to_light_ratio=1.0,
        )

        assert (
            elliptical.convergence_from_grid(grid)
            == spherical.convergence_from_grid(grid)
        ).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid),
            spherical.deflections_from_grid(grid),
        )

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        exponential = mp.EllipticalExponential(
            centre=(-0.7, 0.5),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = exponential.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5
        )

        interp_deflections_values = exponential.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        exponential = mp.SphericalExponential(
            centre=(-0.7, 0.5),
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = exponential.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5
        )

        interp_deflections_values = exponential.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__reshape_decorators(self):

        regular_grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=1.0
        )

        exponential = mp.EllipticalExponential()

        convergence = exponential.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = exponential.potential_from_grid(
        #     grid=regular_grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = exponential.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)

        exponential = mp.SphericalExponential()

        convergence = exponential.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = exponential.potential_from_grid(
        #     grid=regular_grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = exponential.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)


class TestDevVaucouleurs(object):
    def test__constructor_and_units(self):
        dev_vaucouleurs = mp.EllipticalDevVaucouleurs(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
        )

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], dim.Length)
        assert isinstance(dev_vaucouleurs.centre[1], dim.Length)
        assert dev_vaucouleurs.centre[0].unit == "arcsec"
        assert dev_vaucouleurs.centre[1].unit == "arcsec"

        assert dev_vaucouleurs.axis_ratio == 0.5
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 45.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, dim.Luminosity)
        assert dev_vaucouleurs.intensity.unit == "eps"

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, dim.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == "arcsec"

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.mass_to_light_ratio == 10.0
        assert isinstance(dev_vaucouleurs.mass_to_light_ratio, dim.MassOverLuminosity)
        assert dev_vaucouleurs.mass_to_light_ratio.unit == "angular / eps"

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        dev_vaucouleurs = mp.SphericalDevVaucouleurs(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
        )

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], dim.Length)
        assert isinstance(dev_vaucouleurs.centre[1], dim.Length)
        assert dev_vaucouleurs.centre[0].unit == "arcsec"
        assert dev_vaucouleurs.centre[1].unit == "arcsec"

        assert dev_vaucouleurs.axis_ratio == 1.0
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 0.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, dim.Luminosity)
        assert dev_vaucouleurs.intensity.unit == "eps"

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, dim.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == "arcsec"

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.mass_to_light_ratio == 10.0
        assert isinstance(dev_vaucouleurs.mass_to_light_ratio, dim.MassOverLuminosity)
        assert dev_vaucouleurs.mass_to_light_ratio.unit == "angular / eps"

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        dev = mp.EllipticalDevVaucouleurs(
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            mass_to_light_ratio=1.0,
        )
        assert dev.convergence_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(
            5.6697, 1e-3
        )

        dev = mp.EllipticalDevVaucouleurs(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert dev.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(
            7.4455, 1e-3
        )

        dev = mp.EllipticalDevVaucouleurs(
            axis_ratio=0.5,
            phi=90.0,
            intensity=4.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert dev.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(
            2.0 * 7.4455, 1e-3
        )

        dev = mp.EllipticalDevVaucouleurs(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=2.0,
        )
        assert dev.convergence_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(
            2.0 * 7.4455, 1e-3
        )

        sersic = mp.SphericalDevVaucouleurs(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[0.0, 1.0]])
        ) == pytest.approx(0.351797, 1e-3)

    def test__deflections_correct_values(self):
        dev = mp.EllipticalDevVaucouleurs(
            centre=(0.4, 0.2),
            axis_ratio=0.9,
            phi=10.0,
            intensity=2.0,
            effective_radius=0.8,
            mass_to_light_ratio=3.0,
        )
        deflections = dev.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(-24.528, 1e-3)
        assert deflections[0, 1] == pytest.approx(-3.37605, 1e-3)

    def test__spherical_and_elliptical_identical(self):
        elliptical = mp.EllipticalDevVaucouleurs(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            mass_to_light_ratio=1.0,
        )

        spherical = mp.SphericalDevVaucouleurs(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            mass_to_light_ratio=1.0,
        )

        assert (
            elliptical.convergence_from_grid(grid)
            == spherical.convergence_from_grid(grid)
        ).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)

        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid),
            spherical.deflections_from_grid(grid),
        )

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        dev = mp.EllipticalDevVaucouleurs(
            centre=(-0.7, 0.5),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = dev.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5
        )

        interp_deflections_values = dev.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        dev = mp.SphericalDevVaucouleurs(
            centre=(-0.7, 0.5),
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = dev.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5
        )

        interp_deflections_values = dev.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__reshape_decorators(self):

        regular_grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=1.0
        )

        dev_vaucouleurs = mp.EllipticalDevVaucouleurs()

        convergence = dev_vaucouleurs.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = dev_vaucouleurs.potential_from_grid(
        #     grid=regular_grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = dev_vaucouleurs.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)

        dev_vaucouleurs = mp.SphericalDevVaucouleurs()

        convergence = dev_vaucouleurs.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = dev_vaucouleurs.potential_from_grid(
        #     grid=regular_grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = dev_vaucouleurs.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)


class TestSersicMassRadialGradient(object):
    def test__constructor_and_units(self):
        sersic = mp.EllipticalSersicRadialGradient(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
            mass_to_light_gradient=-1.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 0.5
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 45.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.mass_to_light_gradient == -1.0
        assert isinstance(sersic.mass_to_light_gradient, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        sersic = mp.SphericalSersicRadialGradient(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
            mass_to_light_gradient=-1.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.mass_to_light_gradient == -1.0
        assert isinstance(sersic.mass_to_light_gradient, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1/0.6)**-1.0 = 0.6
        sersic = mp.EllipticalSersicRadialGradient(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[0.0, 1.0]])
        ) == pytest.approx(0.6 * 0.351797, 1e-3)

        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1.5/2.0)**1.0 = 0.75
        sersic = mp.EllipticalSersicRadialGradient(
            axis_ratio=1.0,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=-1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[1.5, 0.0]])
        ) == pytest.approx(0.75 * 4.90657319276, 1e-3)

        sersic = mp.EllipticalSersicRadialGradient(
            axis_ratio=1.0,
            phi=0.0,
            intensity=6.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=-1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[1.5, 0.0]])
        ) == pytest.approx(2.0 * 0.75 * 4.90657319276, 1e-3)

        sersic = mp.EllipticalSersicRadialGradient(
            axis_ratio=1.0,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
            mass_to_light_gradient=-1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[1.5, 0.0]])
        ) == pytest.approx(2.0 * 0.75 * 4.90657319276, 1e-3)

        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = ((0.5*1.41)/2.0)**-1.0 = 2.836
        sersic = mp.EllipticalSersicRadialGradient(
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=np.array([[1.0, 0.0]])
        ) == pytest.approx(2.836879 * 5.38066670129, abs=2e-01)

    def test__deflections_correct_values(self):
        sersic = mp.EllipticalSersicRadialGradient(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        deflections = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(3.60324873535244, 1e-3)
        assert deflections[0, 1] == pytest.approx(2.3638898009652, 1e-3)

        sersic = mp.EllipticalSersicRadialGradient(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=-1.0,
        )
        deflections = sersic.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(0.97806399756448, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.725459334118341, 1e-3)

    def test__compare_to_sersic(self):
        sersic = mp.EllipticalSersicRadialGradient(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=1.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=0.0,
        )
        sersic_deflections = sersic.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]])
        )

        exponential = mp.EllipticalExponential(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )
        exponential_deflections = exponential.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]])
        )

        assert (
            sersic_deflections[0, 0]
            == exponential_deflections[0, 0]
            == pytest.approx(0.90493, 1e-3)
        )
        assert (
            sersic_deflections[0, 1]
            == exponential_deflections[0, 1]
            == pytest.approx(0.62569, 1e-3)
        )

        sersic = mp.EllipticalSersicRadialGradient(
            centre=(0.4, 0.2),
            axis_ratio=0.9,
            phi=10.0,
            intensity=2.0,
            effective_radius=0.8,
            sersic_index=4.0,
            mass_to_light_ratio=3.0,
            mass_to_light_gradient=0.0,
        )
        sersic_deflections = sersic.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]])
        )

        dev = mp.EllipticalDevVaucouleurs(
            centre=(0.4, 0.2),
            axis_ratio=0.9,
            phi=10.0,
            intensity=2.0,
            effective_radius=0.8,
            mass_to_light_ratio=3.0,
        )

        dev_deflections = dev.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

        assert (
            sersic_deflections[0, 0]
            == dev_deflections[0, 0]
            == pytest.approx(-24.528, 1e-3)
        )
        assert (
            sersic_deflections[0, 1]
            == dev_deflections[0, 1]
            == pytest.approx(-3.37605, 1e-3)
        )

        sersic_grad = mp.EllipticalSersicRadialGradient(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=0.0,
        )
        sersic_grad_deflections = sersic_grad.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]])
        )

        sersic = mp.EllipticalSersic(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        sersic_deflections = sersic.deflections_from_grid(
            grid=np.array([[0.1625, 0.1625]])
        )

        assert (
            sersic_grad_deflections[0, 0]
            == sersic_deflections[0, 0]
            == pytest.approx(1.1446, 1e-3)
        )
        assert (
            sersic_grad_deflections[0, 1]
            == sersic_deflections[0, 1]
            == pytest.approx(0.79374, 1e-3)
        )

    def test__spherical_and_elliptical_identical(self):

        elliptical = mp.EllipticalSersicRadialGradient(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        spherical = mp.EllipticalSersicRadialGradient(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        assert (
            elliptical.convergence_from_grid(grid)
            == spherical.convergence_from_grid(grid)
        ).all()
        # assert elliptical.potential_from_grid(grid) == spherical.potential_from_grid(grid)
        assert (
            elliptical.deflections_from_grid(grid)
            == spherical.deflections_from_grid(grid)
        ).all()

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        sersic = mp.EllipticalSersicRadialGradient(
            centre=(-0.7, 0.5),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.5,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):

        sersic = mp.SphericalSersicRadialGradient(
            centre=(-0.7, 0.5),
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.5,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = msk.Mask(mask, pixel_scale=1.0)

        regular = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        regular_with_interp = regular.new_grid_with_interpolator(interp_pixel_scale=0.5)
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular, interp_pixel_scale=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__reshape_decorators(self):

        regular_grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=1.0
        )

        sersic = mp.EllipticalSersicRadialGradient()

        convergence = sersic.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = sersic.potential_from_grid(
        #     grid=regular_grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = sersic.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)

        sersic = mp.SphericalSersicRadialGradient()

        convergence = sersic.convergence_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert convergence.shape == (2, 2)

        # potential = sersic.potential_from_grid(
        #     grid=regular_grid, return_in_2d=True, return_binned=False)
        #
        # assert potential.shape == (2, 2)

        deflections = sersic.deflections_from_grid(
            grid=regular_grid, return_in_2d=True, return_binned=False
        )

        assert deflections.shape == (2, 2, 2)
