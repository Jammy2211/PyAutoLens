from __future__ import division, print_function

import math
import numpy as np
import pytest
import scipy.special

import autofit as af
import autolens as al
from test.unit.mock.model import mock_cosmology


@pytest.fixture(autouse=True)
def reset_config():
    """
    Use configuration from the default path. You may want to change this to set a specific path.
    """
    af.conf.instance = af.conf.default


grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestGaussian:
    def test__constructor_and_units(self):
        gaussian = al.light_profiles.EllipticalGaussian(
            centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0, sigma=0.1
        )

        assert gaussian.centre == (1.0, 2.0)
        assert isinstance(gaussian.centre[0], al.Length)
        assert isinstance(gaussian.centre[1], al.Length)
        assert gaussian.centre[0].unit == "arcsec"
        assert gaussian.centre[1].unit == "arcsec"

        assert gaussian.axis_ratio == 0.5
        assert isinstance(gaussian.axis_ratio, float)

        assert gaussian.phi == 45.0
        assert isinstance(gaussian.phi, float)

        assert gaussian.intensity == 1.0
        assert isinstance(gaussian.intensity, al.Luminosity)
        assert gaussian.intensity.unit == "eps"

        assert gaussian.sigma == 0.1
        assert isinstance(gaussian.sigma, al.Length)
        assert gaussian.sigma.unit_length == "arcsec"

        gaussian = al.light_profiles.SphericalGaussian(
            centre=(1.0, 2.0), intensity=1.0, sigma=0.1
        )

        assert gaussian.centre == (1.0, 2.0)
        assert isinstance(gaussian.centre[0], al.Length)
        assert isinstance(gaussian.centre[1], al.Length)
        assert gaussian.centre[0].unit == "arcsec"
        assert gaussian.centre[1].unit == "arcsec"

        assert gaussian.axis_ratio == 1.0
        assert isinstance(gaussian.axis_ratio, float)

        assert gaussian.phi == 0.0
        assert isinstance(gaussian.phi, float)

        assert gaussian.intensity == 1.0
        assert isinstance(gaussian.intensity, al.Luminosity)
        assert gaussian.intensity.unit == "eps"

        assert gaussian.sigma == 0.1
        assert isinstance(gaussian.sigma, al.Length)
        assert gaussian.sigma.unit_length == "arcsec"

    def test__intensity_as_radius__correct_value(self):
        gaussian = al.light_profiles.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=1.0
        )
        assert gaussian.profile_image_from_grid_radii(grid_radii=1.0) == pytest.approx(
            0.24197, 1e-2
        )

        gaussian = al.light_profiles.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=2.0, sigma=1.0
        )
        assert gaussian.profile_image_from_grid_radii(grid_radii=1.0) == pytest.approx(
            2.0 * 0.24197, 1e-2
        )

        gaussian = al.light_profiles.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=2.0
        )
        assert gaussian.profile_image_from_grid_radii(grid_radii=1.0) == pytest.approx(
            0.1760, 1e-2
        )

        gaussian = al.light_profiles.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=2.0
        )
        assert gaussian.profile_image_from_grid_radii(grid_radii=3.0) == pytest.approx(
            0.0647, 1e-2
        )

    def test__intensity_from_grid__same_values_as_above(self):
        gaussian = al.light_profiles.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=1.0
        )
        assert gaussian.profile_image_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(0.24197, 1e-2)

        gaussian = al.light_profiles.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=2.0, sigma=1.0
        )

        assert gaussian.profile_image_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(2.0 * 0.24197, 1e-2)

        gaussian = al.light_profiles.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=2.0
        )

        assert gaussian.profile_image_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(0.1760, 1e-2)

        gaussian = al.light_profiles.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=2.0
        )

        assert gaussian.profile_image_from_grid(
            grid=np.array([[0.0, 3.0]]), bypass_decorator=True
        ) == pytest.approx(0.0647, 1e-2)

    def test__intensity_from_grid__change_geometry(self):
        gaussian = al.light_profiles.EllipticalGaussian(
            centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=1.0
        )
        assert gaussian.profile_image_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        ) == pytest.approx(0.24197, 1e-2)

        gaussian = al.light_profiles.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, intensity=1.0, sigma=1.0
        )
        assert gaussian.profile_image_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        ) == pytest.approx(0.05399, 1e-2)

        gaussian_0 = al.light_profiles.EllipticalGaussian(
            centre=(-3.0, -0.0), axis_ratio=0.5, phi=0.0, intensity=1.0, sigma=1.0
        )

        gaussian_1 = al.light_profiles.EllipticalGaussian(
            centre=(3.0, 0.0), axis_ratio=0.5, phi=0.0, intensity=1.0, sigma=1.0
        )

        assert gaussian_0.profile_image_from_grid(
            grid=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            return_in_2d=False,
            return_binned=False,
            bypass_decorator=True,
        ) == pytest.approx(
            gaussian_1.profile_image_from_grid(
                grid=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
                bypass_decorator=True,
            ),
            1e-4,
        )

        gaussian_0 = al.light_profiles.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=0.5, phi=180.0, intensity=1.0, sigma=1.0
        )

        gaussian_1 = al.light_profiles.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, intensity=1.0, sigma=1.0
        )

        assert gaussian_0.profile_image_from_grid(
            grid=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            return_in_2d=False,
            return_binned=False,
            bypass_decorator=True,
        ) == pytest.approx(
            gaussian_1.profile_image_from_grid(
                grid=np.array([[0.0, 0.0], [0.0, -1.0], [0.0, 1.0]]),
                bypass_decorator=True,
            ),
            1e-4,
        )

    def test__spherical_and_elliptical_match(self):
        elliptical = al.light_profiles.EllipticalGaussian(
            axis_ratio=1.0, phi=0.0, intensity=3.0, sigma=2.0
        )
        spherical = al.light_profiles.SphericalGaussian(intensity=3.0, sigma=2.0)

        assert (
            elliptical.profile_image_from_grid(grid=grid, bypass_decorator=True)
            == spherical.profile_image_from_grid(grid=grid, bypass_decorator=True)
        ).all()

    def test__reshape_decorators(self):
        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=1.0, sub_size=1
        )

        gaussian = al.light_profiles.EllipticalGaussian()

        image = gaussian.profile_image_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert image.shape == (2, 2)

        gaussian = al.light_profiles.SphericalGaussian()

        image = gaussian.profile_image_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert image.shape == (2, 2)


class TestSersic:
    def test__constructor_and_units(self):
        sersic = al.light_profiles.EllipticalSersic(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], al.Length)
        assert isinstance(sersic.centre[1], al.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 0.5
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 45.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, al.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, al.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        sersic = al.light_profiles.SphericalSersic(
            centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6, sersic_index=4.0
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], al.Length)
        assert isinstance(sersic.centre[1], al.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, al.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, al.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__intensity_at_radius__correct_value(self):
        sersic = al.light_profiles.EllipticalSersic(
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
        )
        assert sersic.profile_image_from_grid_radii(grid_radii=1.0) == pytest.approx(
            0.351797, 1e-3
        )

        sersic = al.light_profiles.EllipticalSersic(
            axis_ratio=1.0,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )
        # 3.0 * exp(-3.67206544592 * (1,5/2.0) ** (1.0 / 2.0)) - 1) = 0.351797
        assert sersic.profile_image_from_grid_radii(grid_radii=1.5) == pytest.approx(
            4.90657319276, 1e-3
        )

    def test__intensity_from_grid__correct_values(self):
        sersic = al.light_profiles.EllipticalSersic(
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )
        assert sersic.profile_image_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        ) == pytest.approx(5.38066670129, 1e-3)

    def test__intensity_from_grid__change_geometry(self):
        sersic_0 = al.light_profiles.EllipticalSersic(
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )

        sersic_1 = al.light_profiles.EllipticalSersic(
            axis_ratio=0.5,
            phi=90.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )

        assert sersic_0.profile_image_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == sersic_1.profile_image_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        )

    def test__spherical_and_elliptical_match(self):
        elliptical = al.light_profiles.EllipticalSersic(
            axis_ratio=1.0,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )

        spherical = al.light_profiles.SphericalSersic(
            intensity=3.0, effective_radius=2.0, sersic_index=2.0
        )

        assert (
            elliptical.profile_image_from_grid(grid=grid, bypass_decorator=True)
            == spherical.profile_image_from_grid(grid=grid, bypass_decorator=True)
        ).all()

    def test__summarize_in_units(self):
        sersic = al.light_profiles.SphericalSersic(
            intensity=3.0, effective_radius=2.0, sersic_index=2.0
        )

        summary_text = sersic.summarize_in_units(
            radii=[al.Length(10.0), al.Length(500.0)],
            prefix="sersic_",
            unit_length="arcsec",
            unit_luminosity="eps",
            whitespace=50,
        )

        i = 0

        assert summary_text[i] == "Light Profile = SphericalSersic\n"
        i += 1
        assert (
            summary_text[i]
            == "sersic_luminosity_within_10.00_arcsec             1.8854e+02 eps"
        )
        i += 1
        assert (
            summary_text[i]
            == "sersic_luminosity_within_500.00_arcsec            1.9573e+02 eps"
        )
        i += 1

    def test__reshape_decorators(self):
        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=1.0, sub_size=1
        )

        sersic = al.light_profiles.EllipticalSersic()

        image = sersic.profile_image_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert image.shape == (2, 2)

        sersic = al.light_profiles.SphericalSersic()

        image = sersic.profile_image_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert image.shape == (2, 2)


class TestExponential:
    def test__constructor_and_units(self):
        exponential = al.light_profiles.EllipticalExponential(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
        )

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], al.Length)
        assert isinstance(exponential.centre[1], al.Length)
        assert exponential.centre[0].unit == "arcsec"
        assert exponential.centre[1].unit == "arcsec"

        assert exponential.axis_ratio == 0.5
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 45.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, al.Luminosity)
        assert exponential.intensity.unit == "eps"

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, al.Length)
        assert exponential.effective_radius.unit_length == "arcsec"

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        exponential = al.light_profiles.SphericalExponential(
            centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6
        )

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], al.Length)
        assert isinstance(exponential.centre[1], al.Length)
        assert exponential.centre[0].unit == "arcsec"
        assert exponential.centre[1].unit == "arcsec"

        assert exponential.axis_ratio == 1.0
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 0.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, al.Luminosity)
        assert exponential.intensity.unit == "eps"

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, al.Length)
        assert exponential.effective_radius.unit_length == "arcsec"

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6

    def test__intensity_at_radius__correct_value(self):
        exponential = al.light_profiles.EllipticalExponential(
            axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6
        )
        assert exponential.profile_image_from_grid_radii(
            grid_radii=1.0
        ) == pytest.approx(0.3266, 1e-3)

        exponential = al.light_profiles.EllipticalExponential(
            axis_ratio=1.0, phi=0.0, intensity=3.0, effective_radius=2.0
        )
        assert exponential.profile_image_from_grid_radii(
            grid_radii=1.5
        ) == pytest.approx(4.5640, 1e-3)

    def test__intensity_from_grid__correct_values(self):
        exponential = al.light_profiles.EllipticalExponential(
            axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0
        )
        assert exponential.profile_image_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        ) == pytest.approx(4.9047, 1e-3)

        exponential = al.light_profiles.EllipticalExponential(
            axis_ratio=0.5, phi=90.0, intensity=2.0, effective_radius=3.0
        )
        assert exponential.profile_image_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(4.8566, 1e-3)

        exponential = al.light_profiles.EllipticalExponential(
            axis_ratio=0.5, phi=90.0, intensity=4.0, effective_radius=3.0
        )
        assert exponential.profile_image_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(2.0 * 4.8566, 1e-3)

    def test__intensity_from_grid__change_geometry(self):
        exponential_0 = al.light_profiles.EllipticalExponential(
            axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0
        )

        exponential_1 = al.light_profiles.EllipticalExponential(
            axis_ratio=0.5, phi=90.0, intensity=3.0, effective_radius=2.0
        )

        assert exponential_0.profile_image_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == exponential_1.profile_image_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        )

    def test__spherical_and_elliptical_match(self):
        elliptical = al.light_profiles.EllipticalExponential(
            axis_ratio=1.0, phi=0.0, intensity=3.0, effective_radius=2.0
        )

        spherical = al.light_profiles.SphericalExponential(
            intensity=3.0, effective_radius=2.0
        )

        assert (
            elliptical.profile_image_from_grid(grid=grid, bypass_decorator=True)
            == spherical.profile_image_from_grid(grid=grid, bypass_decorator=True)
        ).all()

    def test__reshape_decorators(self):
        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=1.0, sub_size=1
        )

        exponential = al.light_profiles.EllipticalExponential()

        image = exponential.profile_image_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert image.shape == (2, 2)

        exponential = al.light_profiles.SphericalExponential()

        image = exponential.profile_image_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert image.shape == (2, 2)


class TestDevVaucouleurs:
    def test__constructor_and_units(self):
        dev_vaucouleurs = al.light_profiles.EllipticalDevVaucouleurs(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
        )

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], al.Length)
        assert isinstance(dev_vaucouleurs.centre[1], al.Length)
        assert dev_vaucouleurs.centre[0].unit == "arcsec"
        assert dev_vaucouleurs.centre[1].unit == "arcsec"

        assert dev_vaucouleurs.axis_ratio == 0.5
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 45.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, al.Luminosity)
        assert dev_vaucouleurs.intensity.unit == "eps"

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, al.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == "arcsec"

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        dev_vaucouleurs = al.light_profiles.SphericalDevVaucouleurs(
            centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6
        )

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], al.Length)
        assert isinstance(dev_vaucouleurs.centre[1], al.Length)
        assert dev_vaucouleurs.centre[0].unit == "arcsec"
        assert dev_vaucouleurs.centre[1].unit == "arcsec"

        assert dev_vaucouleurs.axis_ratio == 1.0
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 0.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, al.Luminosity)
        assert dev_vaucouleurs.intensity.unit == "eps"

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, al.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == "arcsec"

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6

    def test__intensity_at_radius__correct_value(self):
        dev_vaucouleurs = al.light_profiles.EllipticalDevVaucouleurs(
            axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6
        )
        assert dev_vaucouleurs.profile_image_from_grid_radii(
            grid_radii=1.0
        ) == pytest.approx(0.3518, 1e-3)

        dev_vaucouleurs = al.light_profiles.EllipticalDevVaucouleurs(
            axis_ratio=1.0, phi=0.0, intensity=3.0, effective_radius=2.0
        )
        assert dev_vaucouleurs.profile_image_from_grid_radii(
            grid_radii=1.5
        ) == pytest.approx(5.1081, 1e-3)

    def test__intensity_from_grid__correct_values(self):
        dev_vaucouleurs = al.light_profiles.EllipticalDevVaucouleurs(
            axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0
        )
        assert dev_vaucouleurs.profile_image_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        ) == pytest.approx(5.6697, 1e-3)

        dev_vaucouleurs = al.light_profiles.EllipticalDevVaucouleurs(
            axis_ratio=0.5, phi=90.0, intensity=2.0, effective_radius=3.0
        )

        assert dev_vaucouleurs.profile_image_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(7.4455, 1e-3)

        dev_vaucouleurs = al.light_profiles.EllipticalDevVaucouleurs(
            axis_ratio=0.5, phi=90.0, intensity=4.0, effective_radius=3.0
        )
        assert dev_vaucouleurs.profile_image_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == pytest.approx(2.0 * 7.4455, 1e-3)

    def test__intensity_from_grid__change_geometry(self):
        dev_vaucouleurs_0 = al.light_profiles.EllipticalDevVaucouleurs(
            axis_ratio=0.5, phi=0.0, intensity=3.0, effective_radius=2.0
        )

        dev_vaucouleurs_1 = al.light_profiles.EllipticalDevVaucouleurs(
            axis_ratio=0.5, phi=90.0, intensity=3.0, effective_radius=2.0
        )

        assert dev_vaucouleurs_0.profile_image_from_grid(
            grid=np.array([[0.0, 1.0]]), bypass_decorator=True
        ) == dev_vaucouleurs_1.profile_image_from_grid(
            grid=np.array([[1.0, 0.0]]), bypass_decorator=True
        )

    def test__spherical_and_elliptical_match(self):
        elliptical = al.light_profiles.EllipticalDevVaucouleurs(
            axis_ratio=1.0, phi=0.0, intensity=3.0, effective_radius=2.0
        )

        spherical = al.light_profiles.SphericalDevVaucouleurs(
            intensity=3.0, effective_radius=2.0
        )

        assert (
            elliptical.profile_image_from_grid(grid=grid, bypass_decorator=True)
            == spherical.profile_image_from_grid(grid=grid, bypass_decorator=True)
        ).all()

    def test__reshape_decorators(self):
        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=1.0, sub_size=1
        )

        dev_vaucouleurs = al.light_profiles.EllipticalDevVaucouleurs()

        image = dev_vaucouleurs.profile_image_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert image.shape == (2, 2)

        dev_vaucouleurs = al.light_profiles.SphericalDevVaucouleurs()

        image = dev_vaucouleurs.profile_image_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert image.shape == (2, 2)


class TestCoreSersic(object):
    def test__constructor_and_units(self):
        core_sersic = al.light_profiles.EllipticalCoreSersic(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=2.0,
        )

        assert core_sersic.centre == (1.0, 2.0)
        assert isinstance(core_sersic.centre[0], al.Length)
        assert isinstance(core_sersic.centre[1], al.Length)
        assert core_sersic.centre[0].unit == "arcsec"
        assert core_sersic.centre[1].unit == "arcsec"

        assert core_sersic.axis_ratio == 0.5
        assert isinstance(core_sersic.axis_ratio, float)

        assert core_sersic.phi == 45.0
        assert isinstance(core_sersic.phi, float)

        assert core_sersic.intensity == 1.0
        assert isinstance(core_sersic.intensity, al.Luminosity)
        assert core_sersic.intensity.unit == "eps"

        assert core_sersic.effective_radius == 0.6
        assert isinstance(core_sersic.effective_radius, al.Length)
        assert core_sersic.effective_radius.unit_length == "arcsec"

        assert core_sersic.sersic_index == 4.0
        assert isinstance(core_sersic.sersic_index, float)

        assert core_sersic.radius_break == 0.01
        assert isinstance(core_sersic.radius_break, al.Length)
        assert core_sersic.radius_break.unit_length == "arcsec"

        assert core_sersic.intensity_break == 0.1
        assert isinstance(core_sersic.intensity_break, al.Luminosity)
        assert core_sersic.intensity_break.unit == "eps"

        assert core_sersic.gamma == 1.0
        assert isinstance(core_sersic.gamma, float)

        assert core_sersic.alpha == 2.0
        assert isinstance(core_sersic.alpha, float)

        assert core_sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert core_sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        core_sersic = al.light_profiles.SphericalCoreSersic(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=2.0,
        )

        assert core_sersic.centre == (1.0, 2.0)
        assert isinstance(core_sersic.centre[0], al.Length)
        assert isinstance(core_sersic.centre[1], al.Length)
        assert core_sersic.centre[0].unit == "arcsec"
        assert core_sersic.centre[1].unit == "arcsec"

        assert core_sersic.axis_ratio == 1.0
        assert isinstance(core_sersic.axis_ratio, float)

        assert core_sersic.phi == 0.0
        assert isinstance(core_sersic.phi, float)

        assert core_sersic.intensity == 1.0
        assert isinstance(core_sersic.intensity, al.Luminosity)
        assert core_sersic.intensity.unit == "eps"

        assert core_sersic.effective_radius == 0.6
        assert isinstance(core_sersic.effective_radius, al.Length)
        assert core_sersic.effective_radius.unit_length == "arcsec"

        assert core_sersic.sersic_index == 4.0
        assert isinstance(core_sersic.sersic_index, float)

        assert core_sersic.radius_break == 0.01
        assert isinstance(core_sersic.radius_break, al.Length)
        assert core_sersic.radius_break.unit_length == "arcsec"

        assert core_sersic.intensity_break == 0.1
        assert isinstance(core_sersic.intensity_break, al.Luminosity)
        assert core_sersic.intensity_break.unit == "eps"

        assert core_sersic.gamma == 1.0
        assert isinstance(core_sersic.gamma, float)

        assert core_sersic.alpha == 2.0
        assert isinstance(core_sersic.alpha, float)

        assert core_sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert core_sersic.elliptical_effective_radius == 0.6

    def test__intensity_at_radius__correct_value(self):
        core_sersic = al.light_profiles.EllipticalCoreSersic(
            axis_ratio=0.5,
            phi=0.0,
            intensity=1.0,
            effective_radius=5.0,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=1.0,
        )
        assert core_sersic.profile_image_from_grid_radii(0.01) == 0.1

    def test__spherical_and_elliptical_match(self):
        elliptical = al.light_profiles.EllipticalCoreSersic(
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=5.0,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=1.0,
        )

        spherical = al.light_profiles.SphericalCoreSersic(
            intensity=1.0,
            effective_radius=5.0,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=1.0,
        )

        assert (
            elliptical.profile_image_from_grid(grid=grid, bypass_decorator=True)
            == spherical.profile_image_from_grid(grid=grid, bypass_decorator=True)
        ).all()

    def test__reshape_decorators(self):
        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=1.0, sub_size=1
        )

        core_sersic = al.light_profiles.EllipticalCoreSersic()

        image = core_sersic.profile_image_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert image.shape == (2, 2)

        core_sersic = al.light_profiles.SphericalCoreSersic()

        image = core_sersic.profile_image_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert image.shape == (2, 2)


def luminosity_from_radius_and_profile(radius, profile):
    x = profile.sersic_constant * (
        (radius / profile.effective_radius) ** (1.0 / profile.sersic_index)
    )

    return (
        profile.intensity
        * profile.effective_radius ** 2
        * 2
        * math.pi
        * profile.sersic_index
        * (
            (math.e ** profile.sersic_constant)
            / (profile.sersic_constant ** (2 * profile.sersic_index))
        )
        * scipy.special.gamma(2 * profile.sersic_index)
        * scipy.special.gammainc(2 * profile.sersic_index, x)
    )


class TestBlurredProfileImages(object):
    def test__blurred_image_from_grid_and_psf(
        self, sub_grid_7x7, blurring_grid_7x7, psf_3x3, convolver_7x7
    ):

        light_profile = al.light_profiles.EllipticalSersic(intensity=1.0)

        image_1d = light_profile.profile_image_from_grid(
            grid=sub_grid_7x7, return_in_2d=False, return_binned=True
        )

        blurring_image_1d = light_profile.profile_image_from_grid(
            grid=blurring_grid_7x7, return_in_2d=False, return_binned=True
        )

        blurred_image_1d = convolver_7x7.convolve_image(
            image_array=image_1d, blurring_array=blurring_image_1d
        )

        light_profile_blurred_image_1d = light_profile.blurred_profile_image_from_grid_and_psf(
            grid=sub_grid_7x7,
            blurring_grid=blurring_grid_7x7,
            psf=psf_3x3,
            return_in_2d=False,
        )

        assert blurred_image_1d == pytest.approx(light_profile_blurred_image_1d, 1.0e-4)

        blurred_image_2d = sub_grid_7x7.mapping.array_2d_from_array_1d(
            array_1d=blurred_image_1d
        )

        light_profile_blurred_image_2d = light_profile.blurred_profile_image_from_grid_and_psf(
            grid=sub_grid_7x7,
            blurring_grid=blurring_grid_7x7,
            psf=psf_3x3,
            return_in_2d=True,
        )

        assert blurred_image_2d == pytest.approx(light_profile_blurred_image_2d, 1.0e-4)

    def test__blurred_image_from_grid_and_convolver(
        self, sub_grid_7x7, blurring_grid_7x7, convolver_7x7
    ):

        light_profile = al.light_profiles.EllipticalSersic(intensity=1.0)

        image_1d = light_profile.profile_image_from_grid(
            grid=sub_grid_7x7, return_in_2d=False, return_binned=True
        )

        blurring_image_1d = light_profile.profile_image_from_grid(
            grid=blurring_grid_7x7, return_in_2d=False, return_binned=True
        )

        blurred_image_1d = convolver_7x7.convolve_image(
            image_array=image_1d, blurring_array=blurring_image_1d
        )

        convolver_7x7.blurring_mask = None

        light_profile_blurred_image_1d = light_profile.blurred_profile_image_from_grid_and_convolver(
            grid=sub_grid_7x7,
            convolver=convolver_7x7,
            blurring_grid=blurring_grid_7x7,
            return_in_2d=False,
        )

        assert blurred_image_1d == pytest.approx(light_profile_blurred_image_1d, 1.0e-4)

        blurred_image_2d = sub_grid_7x7.mapping.array_2d_from_array_1d(
            array_1d=blurred_image_1d
        )

        light_profile_blurred_image_2d = light_profile.blurred_profile_image_from_grid_and_convolver(
            grid=sub_grid_7x7,
            blurring_grid=blurring_grid_7x7,
            convolver=convolver_7x7,
            return_in_2d=True,
        )

        assert blurred_image_2d == pytest.approx(light_profile_blurred_image_2d, 1.0e-4)


class TestVisibilities(object):
    def test__visibilities_from_grid_and_transformer(
        self, sub_grid_7x7, transformer_7x7_7
    ):
        light_profile = al.light_profiles.EllipticalSersic(intensity=1.0)

        image_1d = light_profile.profile_image_from_grid(
            grid=sub_grid_7x7, return_in_2d=False, return_binned=True
        )

        visibilities = transformer_7x7_7.visibilities_from_image_1d(image_1d=image_1d)

        light_profile_visibilities = light_profile.profile_visibilities_from_grid_and_transformer(
            grid=sub_grid_7x7, transformer=transformer_7x7_7
        )

        assert (visibilities == light_profile_visibilities).all()


class TestLuminosityWithinCircle(object):
    def test__luminosity_in_eps__spherical_sersic_index_2__compare_to_analytic(self):
        sersic = al.light_profiles.SphericalSersic(
            intensity=3.0, effective_radius=2.0, sersic_index=2.0
        )

        radius = al.Length(0.5, "arcsec")

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=radius, profile=sersic
        )

        luminosity_integral = sersic.luminosity_within_circle_in_units(
            radius=0.5, unit_luminosity="eps"
        )

        assert luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

    def test__luminosity_in_eps__spherical_sersic_2__compare_to_grid(self):
        sersic = al.light_profiles.SphericalSersic(
            intensity=3.0, effective_radius=2.0, sersic_index=2.0
        )

        radius = al.Length(1.0, "arcsec")

        luminosity_grid = luminosity_from_radius_and_profile(
            radius=radius, profile=sersic
        )

        luminosity_integral = sersic.luminosity_within_circle_in_units(
            radius=radius, unit_luminosity="eps"
        )

        assert luminosity_grid == pytest.approx(luminosity_integral, 0.02)

    def test__luminosity_units_conversions__uses_exposure_time(self):
        sersic_eps = al.light_profiles.SphericalSersic(
            intensity=al.Luminosity(3.0, "eps"), effective_radius=2.0, sersic_index=1.0
        )

        radius = al.Length(0.5, "arcsec")

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=radius, profile=sersic_eps
        )

        luminosity_integral = sersic_eps.luminosity_within_circle_in_units(
            radius=radius, unit_luminosity="eps", exposure_time=3.0
        )

        # eps -> eps

        assert luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

        # eps -> counts

        luminosity_integral = sersic_eps.luminosity_within_circle_in_units(
            radius=radius, unit_luminosity="counts", exposure_time=3.0
        )

        assert 3.0 * luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

        sersic_counts = al.light_profiles.SphericalSersic(
            intensity=al.Luminosity(3.0, "counts"),
            effective_radius=2.0,
            sersic_index=1.0,
        )

        radius = al.Length(0.5, "arcsec")

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=radius, profile=sersic_counts
        )
        luminosity_integral = sersic_counts.luminosity_within_circle_in_units(
            radius=radius, unit_luminosity="eps", exposure_time=3.0
        )

        # counts -> eps

        assert luminosity_analytic / 3.0 == pytest.approx(luminosity_integral, 1e-3)

        luminosity_integral = sersic_counts.luminosity_within_circle_in_units(
            radius=radius, unit_luminosity="counts", exposure_time=3.0
        )

        # counts -> counts

        assert luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

    def test__radius_units_conversions__light_profile_updates_units_and_computes_correct_luminosity(
        self
    ):
        cosmology = mock_cosmology.MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0)

        sersic_arcsec = al.light_profiles.SphericalSersic(
            centre=(al.Length(0.0, "arcsec"), al.Length(0.0, "arcsec")),
            intensity=al.Luminosity(3.0, "eps"),
            effective_radius=al.Length(2.0, "arcsec"),
            sersic_index=1.0,
        )

        sersic_kpc = al.light_profiles.SphericalSersic(
            centre=(al.Length(0.0, "kpc"), al.Length(0.0, "kpc")),
            intensity=al.Luminosity(3.0, "eps"),
            effective_radius=al.Length(4.0, "kpc"),
            sersic_index=1.0,
        )

        radius = al.Length(0.5, "arcsec")

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=radius, profile=sersic_arcsec
        )

        # arcsec -> arcsec

        luminosity = sersic_arcsec.luminosity_within_circle_in_units(radius=radius)

        assert luminosity_analytic == pytest.approx(luminosity, 1e-3)

        # kpc -> arcsec

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=1.0, profile=sersic_kpc
        )

        luminosity = sersic_kpc.luminosity_within_circle_in_units(
            radius=radius, redshift_profile=0.5, cosmology=cosmology
        )

        assert luminosity_analytic == pytest.approx(luminosity, 1e-3)

        radius = al.Length(0.5, "kpc")

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=radius, profile=sersic_kpc
        )

        # kpc -> kpc

        luminosity = sersic_kpc.luminosity_within_circle_in_units(radius=radius)

        assert luminosity_analytic == pytest.approx(luminosity, 1e-3)

        # kpc -> arcsec

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=0.25, profile=sersic_arcsec
        )

        luminosity = sersic_arcsec.luminosity_within_circle_in_units(
            radius=radius, redshift_profile=0.5, cosmology=cosmology
        )

        assert luminosity_analytic == pytest.approx(luminosity, 1e-3)

        radius = al.Length(2.0, "arcsec")
        luminosity_arcsec = sersic_arcsec.luminosity_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            unit_mass="angular",
            cosmology=cosmology,
        )
        radius = al.Length(4.0, "kpc")
        luminosity_kpc = sersic_arcsec.luminosity_within_circle_in_units(
            radius=radius,
            redshift_profile=0.5,
            unit_mass="angular",
            cosmology=cosmology,
        )
        assert luminosity_arcsec == luminosity_kpc


class TestLuminosityWithinEllipse(object):
    def test__within_ellipse_in_counts__check_multiplies_by_exposure_time(self):

        sersic = al.light_profiles.EllipticalSersic(
            axis_ratio=0.5,
            phi=90.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )

        radius = al.Length(0.5, "arcsec")
        luminosity_grid = 0.0

        xs = np.linspace(-1.8, 1.8, 80)
        ys = np.linspace(-1.8, 1.8, 80)

        edge = xs[1] - xs[0]
        area = edge ** 2

        for x in xs:
            for y in ys:

                eta = sersic.grid_to_elliptical_radii(np.array([[x, y]]))

                if eta < radius:
                    luminosity_grid += sersic.profile_image_from_grid_radii(eta) * area

        luminosity_integral = sersic.luminosity_within_ellipse_in_units(
            major_axis=radius, unit_luminosity="counts", exposure_time=3.0
        )

        assert 3.0 * luminosity_grid[0] == pytest.approx(luminosity_integral, 0.02)


class TestGrids(object):
    def test__grid_to_eccentric_radius(self):
        elliptical = al.light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0)

        assert elliptical.grid_to_eccentric_radii(np.array([[1, 1]])) == pytest.approx(
            elliptical.grid_to_eccentric_radii(np.array([[-1, -1]])), 1e-10
        )

    def test__intensity_from_grid(self):
        elliptical = al.light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0)

        assert elliptical.profile_image_from_grid(
            np.array([[1, 1]]), bypass_decorator=True
        ) == pytest.approx(
            elliptical.profile_image_from_grid(
                np.array([[-1, -1]]), bypass_decorator=True
            ),
            1e-4,
        )
