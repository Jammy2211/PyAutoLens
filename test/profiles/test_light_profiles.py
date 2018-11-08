from __future__ import division, print_function

import math

import numpy as np
import pytest
import scipy.special

from autolens.profiles import light_profiles as lp

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


@pytest.fixture(name='elliptical')
def elliptical_sersic():
    return lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                               sersic_index=4.0)


@pytest.fixture(name='vertical')
def vertical_sersic():
    return lp.EllipticalSersic(axis_ratio=0.5, phi=90.0, intensity=1.0, effective_radius=0.6,
                               sersic_index=4.0)


class TestGaussian:

    def test__constructor(self):
        gaussian = lp.EllipticalGaussian(centre=(1.0, 1.0), axis_ratio=0.5, phi=45.0, intensity=2.0,
                                         sigma=0.1)

        assert gaussian.centre == (1.0, 1.0)
        assert gaussian.axis_ratio == 0.5
        assert gaussian.phi == 45.0
        assert gaussian.intensity == 2.0
        assert gaussian.sigma == 0.1

    def test__intensity_as_radius__correct_value(self):
        gaussian = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                         sigma=1.0)
        assert gaussian.intensities_from_grid_radii(grid_radii=1.0) == pytest.approx(0.24197, 1e-2)

        gaussian = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=2.0,
                                         sigma=1.0)
        assert gaussian.intensities_from_grid_radii(grid_radii=1.0) == pytest.approx(2.0 * 0.24197, 1e-2)

        gaussian = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                         sigma=2.0)
        assert gaussian.intensities_from_grid_radii(grid_radii=1.0) == pytest.approx(0.1760, 1e-2)

        gaussian = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                         sigma=2.0)
        assert gaussian.intensities_from_grid_radii(grid_radii=3.0) == pytest.approx(0.0647, 1e-2)

    def test__intensity_from_grid__same_values_as_above(self):
        gaussian = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                         sigma=1.0)
        assert gaussian.intensities_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.24197, 1e-2)

        gaussian = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=2.0,
                                         sigma=1.0)

        assert gaussian.intensities_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 0.24197, 1e-2)

        gaussian = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                         sigma=2.0)

        assert gaussian.intensities_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(0.1760, 1e-2)

        gaussian = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                         sigma=2.0)

        assert gaussian.intensities_from_grid(grid=np.array([[0.0, 3.0]])) == pytest.approx(0.0647, 1e-2)

    def test__intensity_from_grid__change_geometry(self):

        gaussian = lp.EllipticalGaussian(centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                         sigma=1.0)
        assert gaussian.intensities_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(0.24197, 1e-2)

        gaussian = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, intensity=1.0,
                                         sigma=1.0)
        assert gaussian.intensities_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(0.05399, 1e-2)

        gaussian_0 = lp.EllipticalGaussian(centre=(-3.0, -0.0), axis_ratio=0.5, phi=0.0, intensity=1.0,
                                           sigma=1.0)

        gaussian_1 = lp.EllipticalGaussian(centre=(3.0, 0.0), axis_ratio=0.5, phi=0.0, intensity=1.0,
                                           sigma=1.0)

        assert gaussian_0.intensities_from_grid(grid=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])) == \
               pytest.approx(gaussian_1.intensities_from_grid(grid=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])),
                             1e-4)

        gaussian_0 = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=0.5, phi=180.0, intensity=1.0,
                                           sigma=1.0)

        gaussian_1 = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, intensity=1.0,
                                           sigma=1.0)

        assert gaussian_0.intensities_from_grid(grid=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])) == \
               pytest.approx(gaussian_1.intensities_from_grid(grid=np.array([[0.0, 0.0], [0.0, -1.0], [0.0, 1.0]])),
                             1e-4)

    def test__spherical_and_elliptical_match(self):
        elliptical = lp.EllipticalGaussian(axis_ratio=1.0, phi=0.0, intensity=3.0, sigma=2.0)
        spherical = lp.SphericalGaussian(intensity=3.0, sigma=2.0)

        assert (elliptical.intensities_from_grid(grid) == spherical.intensities_from_grid(grid)).all()


class TestSersic:

    def test__constructor(self):
        sersic = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                     effective_radius=0.6,
                                     sersic_index=4.0)

        assert sersic.centre == (0.0, 0.0)
        assert sersic.axis_ratio == 1.0
        assert sersic.phi == 0.0
        assert sersic.intensity == 1.0
        assert sersic.effective_radius == 0.6
        assert sersic.sersic_index == 4.0
        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__intensity_at_radius__correct_value(self):
        sersic = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                     effective_radius=0.6,
                                     sersic_index=4.0)
        assert sersic.intensities_from_grid_radii(grid_radii=1.0) == pytest.approx(0.351797, 1e-3)

        sersic = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                     effective_radius=2.0,
                                     sersic_index=2.0)
        # 3.0 * exp(-3.67206544592 * (1,5/2.0) ** (1.0 / 2.0)) - 1) = 0.351797
        assert sersic.intensities_from_grid_radii(grid_radii=1.5) == pytest.approx(4.90657319276, 1e-3)

    def test__intensity_from_grid__correct_values(self):
        sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                     effective_radius=2.0,
                                     sersic_index=2.0)
        assert sersic.intensities_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(5.38066670129, 1e-3)

    def test__intensity_from_grid__change_geometry(self):
        sersic_0 = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                       effective_radius=2.0,
                                       sersic_index=2.0)

        sersic_1 = lp.EllipticalSersic(axis_ratio=0.5, phi=90.0, intensity=3.0,
                                       effective_radius=2.0,
                                       sersic_index=2.0)

        assert sersic_0.intensities_from_grid(grid=np.array([[0.0, 1.0]])) == \
               sersic_1.intensities_from_grid(grid=np.array([[1.0, 0.0]]))

    def test__spherical_and_elliptical_match(self):
        elliptical = lp.EllipticalSersic(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                         effective_radius=2.0, sersic_index=2.0)

        spherical = lp.SphericalSersic(intensity=3.0, effective_radius=2.0, sersic_index=2.0)

        assert (elliptical.intensities_from_grid(grid) == spherical.intensities_from_grid(grid)).all()


class TestExponential:

    def test__constructor(self):
        exponential = lp.EllipticalExponential(axis_ratio=0.5, phi=0.0, intensity=1.0,
                                               effective_radius=0.6)

        assert exponential.centre == (0.0, 0.0)
        assert exponential.axis_ratio == 0.5
        assert exponential.phi == 0.0
        assert exponential.intensity == 1.0
        assert exponential.effective_radius == 0.6
        assert exponential.sersic_index == 1.0
        assert exponential.sersic_constant == pytest.approx(1.678378, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6 / math.sqrt(0.5)

    def test__intensity_at_radius__correct_value(self):
        exponential = lp.EllipticalExponential(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                               effective_radius=0.6)
        assert exponential.intensities_from_grid_radii(grid_radii=1.0) == pytest.approx(0.3266, 1e-3)

        exponential = lp.EllipticalExponential(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                               effective_radius=2.0)
        assert exponential.intensities_from_grid_radii(grid_radii=1.5) == pytest.approx(4.5640, 1e-3)

    def test__intensity_from_grid__correct_values(self):
        exponential = lp.EllipticalExponential(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                               effective_radius=2.0)
        assert exponential.intensities_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(4.9047, 1e-3)

        exponential = lp.EllipticalExponential(axis_ratio=0.5, phi=90.0, intensity=2.0,
                                               effective_radius=3.0)
        assert exponential.intensities_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(4.8566, 1e-3)

        exponential = lp.EllipticalExponential(axis_ratio=0.5, phi=90.0, intensity=4.0,
                                               effective_radius=3.0)
        assert exponential.intensities_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 4.8566, 1e-3)

    def test__intensity_from_grid__change_geometry(self):
        exponential_0 = lp.EllipticalExponential(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                                 effective_radius=2.0)

        exponential_1 = lp.EllipticalExponential(axis_ratio=0.5, phi=90.0, intensity=3.0,
                                                 effective_radius=2.0)

        assert exponential_0.intensities_from_grid(grid=np.array([[0.0, 1.0]])) == \
               exponential_1.intensities_from_grid(grid=np.array([[1.0, 0.0]]))

    def test__spherical_and_elliptical_match(self):
        elliptical = lp.EllipticalExponential(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                              effective_radius=2.0)

        spherical = lp.SphericalExponential(intensity=3.0, effective_radius=2.0)

        assert (elliptical.intensities_from_grid(grid) == spherical.intensities_from_grid(grid)).all()


class TestDevVaucouleurs:

    def test__constructor(self):
        dev_vaucouleurs = lp.EllipticalDevVaucouleurs(axis_ratio=0.6, phi=10.0, intensity=2.0,
                                                      effective_radius=0.9,
                                                      centre=(0.0, 0.1))

        assert dev_vaucouleurs.centre == (0.0, 0.1)
        assert dev_vaucouleurs.axis_ratio == 0.6
        assert dev_vaucouleurs.phi == 10.0
        assert dev_vaucouleurs.intensity == 2.0
        assert dev_vaucouleurs.effective_radius == 0.9
        assert dev_vaucouleurs.sersic_index == 4.0
        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.9 / math.sqrt(0.6)

    def test__intensity_at_radius__correct_value(self):
        dev_vaucouleurs = lp.EllipticalDevVaucouleurs(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                      effective_radius=0.6)
        assert dev_vaucouleurs.intensities_from_grid_radii(grid_radii=1.0) == pytest.approx(0.3518, 1e-3)

        dev_vaucouleurs = lp.EllipticalDevVaucouleurs(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                      effective_radius=2.0)
        assert dev_vaucouleurs.intensities_from_grid_radii(grid_radii=1.5) == pytest.approx(5.1081, 1e-3)

    def test__intensity_from_grid__correct_values(self):
        dev_vaucouleurs = lp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                                      effective_radius=2.0)
        assert dev_vaucouleurs.intensities_from_grid(grid=np.array([[1.0, 0.0]])) == pytest.approx(5.6697, 1e-3)

        dev_vaucouleurs = lp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=90.0, intensity=2.0,
                                                      effective_radius=3.0)

        assert dev_vaucouleurs.intensities_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(7.4455, 1e-3)

        dev_vaucouleurs = lp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=90.0, intensity=4.0,
                                                      effective_radius=3.0)
        assert dev_vaucouleurs.intensities_from_grid(grid=np.array([[0.0, 1.0]])) == pytest.approx(2.0 * 7.4455, 1e-3)

    def test__intensity_from_grid__change_geometry(self):
        dev_vaucouleurs_0 = lp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=0.0, intensity=3.0,
                                                        effective_radius=2.0)

        dev_vaucouleurs_1 = lp.EllipticalDevVaucouleurs(axis_ratio=0.5, phi=90.0, intensity=3.0,
                                                        effective_radius=2.0)

        assert dev_vaucouleurs_0.intensities_from_grid(grid=np.array([[0.0, 1.0]])) \
               == dev_vaucouleurs_1.intensities_from_grid(grid=np.array([[1.0, 0.0]]))

    def test__spherical_and_elliptical_match(self):
        elliptical = lp.EllipticalDevVaucouleurs(axis_ratio=1.0, phi=0.0, intensity=3.0,
                                                 effective_radius=2.0)

        spherical = lp.SphericalDevVaucouleurs(intensity=3.0, effective_radius=2.0)

        assert (elliptical.intensities_from_grid(grid) == spherical.intensities_from_grid(grid)).all()


class TestCoreSersic(object):

    def test__constructor(self):
        cored_sersic = lp.EllipticalCoreSersic(axis_ratio=0.5, phi=0.0, intensity=1.0,
                                               effective_radius=5.0, sersic_index=4.0, radius_break=0.01,
                                               intensity_break=0.1, gamma=1.0, alpha=1.0)

        assert cored_sersic.centre == (0.0, 0.0)
        assert cored_sersic.axis_ratio == 0.5
        assert cored_sersic.phi == 0.0
        assert cored_sersic.intensity == 1.0
        assert cored_sersic.effective_radius == 5.0
        assert cored_sersic.sersic_index == 4.0
        assert cored_sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert cored_sersic.radius_break == 0.01
        assert cored_sersic.intensity_break == 0.1
        assert cored_sersic.gamma == 1.0
        assert cored_sersic.alpha == 1.0
        assert cored_sersic.elliptical_effective_radius == 5.0 / math.sqrt(0.5)

    def test__intensity_at_radius__correct_value(self):
        core_sersic = lp.EllipticalCoreSersic(axis_ratio=0.5, phi=0.0, intensity=1.0,
                                              effective_radius=5.0, sersic_index=4.0,
                                              radius_break=0.01,
                                              intensity_break=0.1, gamma=1.0, alpha=1.0)
        assert core_sersic.intensities_from_grid_radii(0.01) == 0.1

    def test__spherical_and_elliptical_match(self):
        elliptical = lp.EllipticalCoreSersic(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                             effective_radius=5.0, sersic_index=4.0,
                                             radius_break=0.01,
                                             intensity_break=0.1, gamma=1.0, alpha=1.0)

        spherical = lp.SphericalCoreSersic(intensity=1.0, effective_radius=5.0, sersic_index=4.0,
                                           radius_break=0.01,
                                           intensity_break=0.1, gamma=1.0, alpha=1.0)

        assert (elliptical.intensities_from_grid(grid) == spherical.intensities_from_grid(grid)).all()


class TestLuminosityIntegral(object):

    def test__within_circle__spherical_exponential__compare_to_analytic(self):

        sersic = lp.SphericalSersic(intensity=3.0, effective_radius=2.0, sersic_index=1.0)

        integral_radius = 5.5

        # Use gamma function for analytic computation of the intensity within a radius=0.5

        x = sersic.sersic_constant * (integral_radius / sersic.effective_radius) ** (1.0 / sersic.sersic_index)

        intensity_analytic = sersic.intensity * sersic.effective_radius ** 2 * 2 * math.pi * sersic.sersic_index * (
                math.e ** sersic.sersic_constant / (
                sersic.sersic_constant ** (2 * sersic.sersic_index))) * scipy.special.gamma(
            2 * sersic.sersic_index) * scipy.special.gammainc(
            2 * sersic.sersic_index, x)

        intensity_integral = sersic.luminosity_within_circle(radius=integral_radius)

        assert intensity_analytic == pytest.approx(intensity_integral, 1e-3)

    def test__within_circle__spherical_sersic_index_2__compare_to_analytic(self):

        sersic = lp.SphericalSersic(intensity=3.0, effective_radius=2.0, sersic_index=2.0)

        integral_radius = 0.5

        # Use gamma function for analytic computation of the intensity within a radius=0.5

        x = sersic.sersic_constant * ((integral_radius / sersic.effective_radius) ** (1.0 / sersic.sersic_index))

        intensity_analytic = sersic.intensity * sersic.effective_radius ** 2 * 2 * math.pi * sersic.sersic_index * \
                             ((math.e ** sersic.sersic_constant) / (
                                     sersic.sersic_constant ** (2 * sersic.sersic_index))) * \
                             scipy.special.gamma(2 * sersic.sersic_index) * scipy.special.gammainc(
            2 * sersic.sersic_index, x)

        intensity_integral = sersic.luminosity_within_circle(radius=0.5)

        assert intensity_analytic == pytest.approx(intensity_integral, 1e-3)

    def test__within_circle__spherical_dev_vaucouleurs__compare_to_analytic(self):

        sersic = lp.SphericalSersic(intensity=3.0, effective_radius=2.0, sersic_index=4.0)

        integral_radius = 0.5

        # Use gamma function for analytic computation of the intensity within a radius=0.5

        x = sersic.sersic_constant * ((integral_radius / sersic.effective_radius) ** (1.0 / sersic.sersic_index))

        intensity_analytic = sersic.intensity * sersic.effective_radius ** 2 * 2 * math.pi * sersic.sersic_index * \
                             ((math.e ** sersic.sersic_constant) / (
                                     sersic.sersic_constant ** (2 * sersic.sersic_index))) * \
                             scipy.special.gamma(2 * sersic.sersic_index) * scipy.special.gammainc(
            2 * sersic.sersic_index, x)

        intensity_integral = sersic.luminosity_within_circle(radius=0.5)

        assert intensity_analytic == pytest.approx(intensity_integral, 1e-3)

    def test__within_circle__spherical_sersic_2__compare_to_grid(self):

        sersic = lp.SphericalSersic(intensity=3.0, effective_radius=2.0, sersic_index=2.0)

        integral_radius = 1.0
        luminosity_tot = 0.0

        xs = np.linspace(-1.5, 1.5, 40)
        ys = np.linspace(-1.5, 1.5, 40)

        edge = xs[1] - xs[0]
        area = edge ** 2

        for x in xs:
            for y in ys:

                eta = math.sqrt(x ** 2 + y ** 2)
                if eta < integral_radius:
                    luminosity_tot += sersic.intensities_from_grid_radii(eta) * area

        intensity_integral = sersic.luminosity_within_circle(radius=integral_radius)

        assert luminosity_tot == pytest.approx(intensity_integral, 0.02)

    def test__within_ellipse__elliptical_sersic_2__compare_to_grid(self):

        sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=90.0, intensity=3.0, effective_radius=2.0,
                                     sersic_index=2.0)

        integral_radius = 0.5
        luminosity_tot = 0.0

        xs = np.linspace(-1.8, 1.8, 80)
        ys = np.linspace(-1.8, 1.8, 80)

        edge = xs[1] - xs[0]
        area = edge ** 2

        for x in xs:
            for y in ys:

                eta = sersic.grid_to_elliptical_radii(np.array([[x, y]]))

                if eta < integral_radius:
                    luminosity_tot += sersic.intensities_from_grid_radii(eta) * area

        intensity_integral = sersic.luminosity_within_ellipse(major_axis=integral_radius)

        assert luminosity_tot[0] == pytest.approx(intensity_integral, 0.02)


class TestGrids(object):

    def test__grid_to_eccentric_radius(self, elliptical):
        assert elliptical.grid_to_eccentric_radii(np.array([[1, 1]])) == pytest.approx(
            elliptical.grid_to_eccentric_radii(np.array([[-1, -1]])), 1e-10)

    def test__intensity_from_grid(self, elliptical):
        assert elliptical.intensities_from_grid(np.array([[1, 1]])) == \
               pytest.approx(elliptical.intensities_from_grid(np.array([[-1, -1]])), 1e-4)
