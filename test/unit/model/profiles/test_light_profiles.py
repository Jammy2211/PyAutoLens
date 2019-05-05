from __future__ import division, print_function

import math

import numpy as np
import pytest
import scipy.special

from autofit import conf
from autolens import dimensions as dim
from autolens.model.profiles import light_profiles as lp

from test.unit.mock.mock_cosmology import MockCosmology

@pytest.fixture(autouse=True)
def reset_config():
    """
    Use configuration from the default path. You may want to change this to set a specific path.
    """
    conf.instance = conf.default


grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


@pytest.fixture(name='elliptical')
def elliptical_sersic():
    return lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                               sersic_index=4.0)

class TestGaussian:

    def test__constructor_and_units(self):
        
        gaussian = lp.EllipticalGaussian(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0, sigma=0.1)

        assert gaussian.centre == (1.0, 2.0)
        assert isinstance(gaussian.centre[0], dim.Length)
        assert isinstance(gaussian.centre[1], dim.Length)
        assert gaussian.centre[0].unit == 'arcsec'
        assert gaussian.centre[1].unit == 'arcsec'

        assert gaussian.axis_ratio == 0.5
        assert isinstance(gaussian.axis_ratio, float)

        assert gaussian.phi == 45.0
        assert isinstance(gaussian.phi, float)

        assert gaussian.intensity == 1.0
        assert isinstance(gaussian.intensity, dim.Luminosity)
        assert gaussian.intensity.unit == 'eps'

        assert gaussian.sigma == 0.1
        assert isinstance(gaussian.sigma, dim.Length)
        assert gaussian.sigma.unit_length == 'arcsec'

        gaussian = lp.SphericalGaussian(centre=(1.0, 2.0), intensity=1.0, sigma=0.1)

        assert gaussian.centre == (1.0, 2.0)
        assert isinstance(gaussian.centre[0], dim.Length)
        assert isinstance(gaussian.centre[1], dim.Length)
        assert gaussian.centre[0].unit == 'arcsec'
        assert gaussian.centre[1].unit == 'arcsec'

        assert gaussian.axis_ratio == 1.0
        assert isinstance(gaussian.axis_ratio, float)

        assert gaussian.phi == 0.0
        assert isinstance(gaussian.phi, float)

        assert gaussian.intensity == 1.0
        assert isinstance(gaussian.intensity, dim.Luminosity)
        assert gaussian.intensity.unit == 'eps'

        assert gaussian.sigma == 0.1
        assert isinstance(gaussian.sigma, dim.Length)
        assert gaussian.sigma.unit_length == 'arcsec'

    def test__intensity_as_radius__correct_value(self):

        gaussian = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=1.0)
        assert gaussian.intensities_from_grid_radii(grid_radii=1.0) == pytest.approx(0.24197, 1e-2)

        gaussian = lp.EllipticalGaussian(centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=2.0,sigma=1.0)
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

    def test__constructor_and_units(self):

        sersic = lp.EllipticalSersic(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                     effective_radius=0.6, sersic_index=4.0)

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == 'arcsec'
        assert sersic.centre[1].unit == 'arcsec'

        assert sersic.axis_ratio == 0.5
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 45.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == 'eps'

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == 'arcsec'

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        sersic = lp.SphericalSersic(centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6, sersic_index=4.0)

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], dim.Length)
        assert isinstance(sersic.centre[1], dim.Length)
        assert sersic.centre[0].unit == 'arcsec'
        assert sersic.centre[1].unit == 'arcsec'

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, dim.Luminosity)
        assert sersic.intensity.unit == 'eps'

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, dim.Length)
        assert sersic.effective_radius.unit_length == 'arcsec'

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

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

    def test__summarize_in_units(self):

        sersic = lp.SphericalSersic(intensity=3.0, effective_radius=2.0, sersic_index=2.0)

        summary_text = sersic.summarize_in_units(radii=[dim.Length(10.0), dim.Length(500.0)], prefix='sersic_',
                                                  unit_length='arcsec', unit_luminosity='eps', whitespace=50)

        index = 0

        assert summary_text[index] == 'Light Profile = SphericalSersic' ; index += 1
        assert summary_text[index] ==  ''  ; index += 1
        assert summary_text[index] == 'sersic_luminosity_within_10.00_arcsec             1.8854e+02 eps' ; index += 1
        assert summary_text[index] == 'sersic_luminosity_within_500.00_arcsec            1.9573e+02 eps' ; index += 1

class TestExponential:

    def test__constructor_and_units(self):

        exponential = lp.EllipticalExponential(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                               effective_radius=0.6)

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], dim.Length)
        assert isinstance(exponential.centre[1], dim.Length)
        assert exponential.centre[0].unit == 'arcsec'
        assert exponential.centre[1].unit == 'arcsec'

        assert exponential.axis_ratio == 0.5
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 45.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, dim.Luminosity)
        assert exponential.intensity.unit == 'eps'

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, dim.Length)
        assert exponential.effective_radius.unit_length == 'arcsec'

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        exponential = lp.SphericalExponential(centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6)

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], dim.Length)
        assert isinstance(exponential.centre[1], dim.Length)
        assert exponential.centre[0].unit == 'arcsec'
        assert exponential.centre[1].unit == 'arcsec'

        assert exponential.axis_ratio == 1.0
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 0.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, dim.Luminosity)
        assert exponential.intensity.unit == 'eps'

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, dim.Length)
        assert exponential.effective_radius.unit_length == 'arcsec'

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6

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

    def test__constructor_and_units(self):

        dev_vaucouleurs = lp.EllipticalDevVaucouleurs(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                                      effective_radius=0.6)

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], dim.Length)
        assert isinstance(dev_vaucouleurs.centre[1], dim.Length)
        assert dev_vaucouleurs.centre[0].unit == 'arcsec'
        assert dev_vaucouleurs.centre[1].unit == 'arcsec'

        assert dev_vaucouleurs.axis_ratio == 0.5
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 45.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, dim.Luminosity)
        assert dev_vaucouleurs.intensity.unit == 'eps'

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, dim.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == 'arcsec'

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        dev_vaucouleurs = lp.SphericalDevVaucouleurs(centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6)

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], dim.Length)
        assert isinstance(dev_vaucouleurs.centre[1], dim.Length)
        assert dev_vaucouleurs.centre[0].unit == 'arcsec'
        assert dev_vaucouleurs.centre[1].unit == 'arcsec'

        assert dev_vaucouleurs.axis_ratio == 1.0
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 0.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, dim.Luminosity)
        assert dev_vaucouleurs.intensity.unit == 'eps'

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, dim.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == 'arcsec'

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6

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

    def test__constructor_and_units(self):

        core_sersic = lp.EllipticalCoreSersic(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                              effective_radius=0.6, sersic_index=4.0, radius_break=0.01,
                                              intensity_break=0.1, gamma=1.0, alpha=2.0)

        assert core_sersic.centre == (1.0, 2.0)
        assert isinstance(core_sersic.centre[0], dim.Length)
        assert isinstance(core_sersic.centre[1], dim.Length)
        assert core_sersic.centre[0].unit == 'arcsec'
        assert core_sersic.centre[1].unit == 'arcsec'

        assert core_sersic.axis_ratio == 0.5
        assert isinstance(core_sersic.axis_ratio, float)

        assert core_sersic.phi == 45.0
        assert isinstance(core_sersic.phi, float)

        assert core_sersic.intensity == 1.0
        assert isinstance(core_sersic.intensity, dim.Luminosity)
        assert core_sersic.intensity.unit == 'eps'

        assert core_sersic.effective_radius == 0.6
        assert isinstance(core_sersic.effective_radius, dim.Length)
        assert core_sersic.effective_radius.unit_length == 'arcsec'

        assert core_sersic.sersic_index == 4.0
        assert isinstance(core_sersic.sersic_index, float)

        assert core_sersic.radius_break == 0.01
        assert isinstance(core_sersic.radius_break, dim.Length)
        assert core_sersic.radius_break.unit_length == 'arcsec'

        assert core_sersic.intensity_break == 0.1
        assert isinstance(core_sersic.intensity_break, dim.Luminosity)
        assert core_sersic.intensity_break.unit == 'eps'

        assert core_sersic.gamma == 1.0
        assert isinstance(core_sersic.gamma, float)

        assert core_sersic.alpha == 2.0
        assert isinstance(core_sersic.alpha, float)

        assert core_sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert core_sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        core_sersic = lp.SphericalCoreSersic(centre=(1.0, 2.0), intensity=1.0,
                                              effective_radius=0.6, sersic_index=4.0, radius_break=0.01,
                                              intensity_break=0.1, gamma=1.0, alpha=2.0)

        assert core_sersic.centre == (1.0, 2.0)
        assert isinstance(core_sersic.centre[0], dim.Length)
        assert isinstance(core_sersic.centre[1], dim.Length)
        assert core_sersic.centre[0].unit == 'arcsec'
        assert core_sersic.centre[1].unit == 'arcsec'

        assert core_sersic.axis_ratio == 1.0
        assert isinstance(core_sersic.axis_ratio, float)

        assert core_sersic.phi == 0.0
        assert isinstance(core_sersic.phi, float)

        assert core_sersic.intensity == 1.0
        assert isinstance(core_sersic.intensity, dim.Luminosity)
        assert core_sersic.intensity.unit == 'eps'

        assert core_sersic.effective_radius == 0.6
        assert isinstance(core_sersic.effective_radius, dim.Length)
        assert core_sersic.effective_radius.unit_length == 'arcsec'

        assert core_sersic.sersic_index == 4.0
        assert isinstance(core_sersic.sersic_index, float)

        assert core_sersic.radius_break == 0.01
        assert isinstance(core_sersic.radius_break, dim.Length)
        assert core_sersic.radius_break.unit_length == 'arcsec'

        assert core_sersic.intensity_break == 0.1
        assert isinstance(core_sersic.intensity_break, dim.Luminosity)
        assert core_sersic.intensity_break.unit == 'eps'

        assert core_sersic.gamma == 1.0
        assert isinstance(core_sersic.gamma, float)

        assert core_sersic.alpha == 2.0
        assert isinstance(core_sersic.alpha, float)

        assert core_sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert core_sersic.elliptical_effective_radius == 0.6

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


def luminosity_from_radius_and_profile(radius, profile):

    x = profile.sersic_constant * ((radius / profile.effective_radius) ** (1.0 / profile.sersic_index))

    return profile.intensity * profile.effective_radius ** 2 * 2 * math.pi * profile.sersic_index * \
           ((math.e ** profile.sersic_constant) / (
                   profile.sersic_constant ** (2 * profile.sersic_index))) * \
           scipy.special.gamma(2 * profile.sersic_index) * scipy.special.gammainc(
        2 * profile.sersic_index, x)


class TestLuminosityWithinCircle(object):

    def test__luminosity_in_eps__spherical_sersic_index_2__compare_to_analytic(self):

        sersic = lp.SphericalSersic(intensity=3.0, effective_radius=2.0, sersic_index=2.0)

        radius = dim.Length(0.5, 'arcsec')

        luminosity_analytic = luminosity_from_radius_and_profile(radius=radius, profile=sersic)

        luminosity_integral = sersic.luminosity_within_circle_in_units(radius=0.5, unit_luminosity='eps')

        assert luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

    def test__luminosity_in_eps__spherical_sersic_2__compare_to_grid(self):

        sersic = lp.SphericalSersic(intensity=3.0, effective_radius=2.0, sersic_index=2.0)

        radius = dim.Length(1.0, 'arcsec')

        luminosity_grid = luminosity_from_radius_and_profile(radius=radius, profile=sersic)

        luminosity_integral = sersic.luminosity_within_circle_in_units(radius=radius, unit_luminosity='eps')

        assert luminosity_grid == pytest.approx(luminosity_integral, 0.02)

    def test__luminosity_units_conversions__uses_exposure_time(self):

        sersic_eps = lp.SphericalSersic(intensity=dim.Luminosity(3.0, 'eps'),
                                        effective_radius=2.0, sersic_index=1.0)

        radius = dim.Length(0.5, 'arcsec')

        luminosity_analytic = luminosity_from_radius_and_profile(radius=radius, profile=sersic_eps)

        luminosity_integral = sersic_eps.luminosity_within_circle_in_units(radius=radius, unit_luminosity='eps',
                                                                           exposure_time=3.0)

        # eps -> eps
    
        assert luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)
        
        # eps -> counts

        luminosity_integral = sersic_eps.luminosity_within_circle_in_units(radius=radius, unit_luminosity='counts', exposure_time=3.0)

        assert 3.0*luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)
        
        sersic_counts = lp.SphericalSersic(intensity=dim.Luminosity(3.0, 'counts'),
                                        effective_radius=2.0, sersic_index=1.0)

        radius = dim.Length(0.5, 'arcsec')

        luminosity_analytic = luminosity_from_radius_and_profile(radius=radius, profile=sersic_counts)
        luminosity_integral = sersic_counts.luminosity_within_circle_in_units(radius=radius, unit_luminosity='eps',
                                                                              exposure_time=3.0)

        # counts -> eps

        assert luminosity_analytic / 3.0 == pytest.approx(luminosity_integral, 1e-3)

        luminosity_integral = sersic_counts.luminosity_within_circle_in_units(radius=radius, unit_luminosity='counts',
                                                                              exposure_time=3.0)

        # counts -> counts

        assert luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

    def test__radius_units_conversions__light_profile_updates_units_and_computes_correct_luminosity(self):

        cosmology = MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0)

        sersic_arcsec = lp.SphericalSersic(centre=(dim.Length(0.0, 'arcsec'), dim.Length(0.0, 'arcsec')), 
                                    intensity=dim.Luminosity(3.0, 'eps'), 
                                    effective_radius=dim.Length(2.0, 'arcsec'),
                                    sersic_index=1.0)

        sersic_kpc = lp.SphericalSersic(centre=(dim.Length(0.0, 'kpc'), dim.Length(0.0, 'kpc')),
                                    intensity=dim.Luminosity(3.0, 'eps'),
                                    effective_radius=dim.Length(4.0, 'kpc'),
                                    sersic_index=1.0)

        radius = dim.Length(0.5, 'arcsec')

        luminosity_analytic = luminosity_from_radius_and_profile(radius=radius, profile=sersic_arcsec)

        # arcsec -> arcsec

        luminosity = sersic_arcsec.luminosity_within_circle_in_units(radius=radius)

        assert luminosity_analytic == pytest.approx(luminosity, 1e-3)

        # kpc -> arcsec

        luminosity_analytic = luminosity_from_radius_and_profile(radius=1.0, profile=sersic_kpc)

        luminosity = sersic_kpc.luminosity_within_circle_in_units(radius=radius, redshift_profile=0.5, cosmology=cosmology)

        assert luminosity_analytic == pytest.approx(luminosity, 1e-3)

        radius = dim.Length(0.5, 'kpc')

        luminosity_analytic = luminosity_from_radius_and_profile(radius=radius, profile=sersic_kpc)

        # kpc -> kpc

        luminosity = sersic_kpc.luminosity_within_circle_in_units(radius=radius)

        assert luminosity_analytic == pytest.approx(luminosity, 1e-3)

        # kpc -> arcsec

        luminosity_analytic = luminosity_from_radius_and_profile(radius=0.25, profile=sersic_arcsec)

        luminosity = sersic_arcsec.luminosity_within_circle_in_units(radius=radius, redshift_profile=0.5,
                                                                     cosmology=cosmology)

        assert luminosity_analytic == pytest.approx(luminosity, 1e-3)

        radius = dim.Length(2.0, 'arcsec')
        luminosity_arcsec = sersic_arcsec.luminosity_within_circle_in_units(radius=radius, redshift_profile=0.5,
                                                                            unit_mass='angular', cosmology=cosmology)
        radius = dim.Length(4.0, 'kpc')
        luminosity_kpc = sersic_arcsec.luminosity_within_circle_in_units(radius=radius, redshift_profile=0.5,
                                                                         unit_mass='angular', cosmology=cosmology)
        assert luminosity_arcsec == luminosity_kpc


class TestLuminosityWithinEllipse(object):

    def test__within_ellipse_in_counts__check_multiplies_by_exposure_time(self):

        sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=90.0, intensity=3.0, effective_radius=2.0,
                                     sersic_index=2.0)

        radius = dim.Length(0.5, 'arcsec')
        luminosity_grid = 0.0

        xs = np.linspace(-1.8, 1.8, 80)
        ys = np.linspace(-1.8, 1.8, 80)

        edge = xs[1] - xs[0]
        area = edge ** 2

        for x in xs:
            for y in ys:

                eta = sersic.grid_to_elliptical_radii(np.array([[x, y]]))

                if eta < radius:
                    luminosity_grid += sersic.intensities_from_grid_radii(eta) * area

        luminosity_integral = sersic.luminosity_within_ellipse_in_units(major_axis=radius, unit_luminosity='counts',
                                                                        exposure_time=3.0)

        assert 3.0*luminosity_grid[0] == pytest.approx(luminosity_integral, 0.02)


class TestGrids(object):

    def test__grid_to_eccentric_radius(self, elliptical):
        assert elliptical.grid_to_eccentric_radii(np.array([[1, 1]])) == pytest.approx(
            elliptical.grid_to_eccentric_radii(np.array([[-1, -1]])), 1e-10)

    def test__intensity_from_grid(self, elliptical):
        assert elliptical.intensities_from_grid(np.array([[1, 1]])) == \
               pytest.approx(elliptical.intensities_from_grid(np.array([[-1, -1]])), 1e-4)
