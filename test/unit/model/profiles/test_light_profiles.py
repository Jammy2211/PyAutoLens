from __future__ import division, print_function

import math

import numpy as np
import pytest
import scipy.special

from autolens import exc
from autofit import conf
from autolens.model.profiles import light_profiles as lp


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


@pytest.fixture(name='vertical')
def vertical_sersic():
    return lp.EllipticalSersic(axis_ratio=0.5, phi=90.0, intensity=1.0, effective_radius=0.6,
                               sersic_index=4.0)


class TestGaussian:

    def test__constructor_and_unit_conversions(self):
        
        gaussian_arcsec = lp.EllipticalGaussian(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0, sigma=0.1)

        assert gaussian_arcsec.centre == (1.0, 2.0)
        assert gaussian_arcsec.axis_ratio == 0.5
        assert gaussian_arcsec.phi == 45.0
        assert gaussian_arcsec.intensity == 1.0
        assert gaussian_arcsec.sigma == 0.1
        assert gaussian_arcsec.units_distance == 'arcsec'
        assert gaussian_arcsec.units_luminosity == 'electrons_per_second'

        gaussian_arcsec = gaussian_arcsec.new_light_profile_with_units_converted(units_distance='arcsec')

        assert gaussian_arcsec.centre == (1.0, 2.0)
        assert gaussian_arcsec.axis_ratio == 0.5
        assert gaussian_arcsec.phi == 45.0
        assert gaussian_arcsec.intensity == 1.0
        assert gaussian_arcsec.sigma == 0.1
        assert gaussian_arcsec.units_distance == 'arcsec'
        assert gaussian_arcsec.units_luminosity == 'electrons_per_second'

        gaussian_kpc = gaussian_arcsec.new_light_profile_with_units_converted(units_distance='kpc', kpc_per_arcsec=2.0)

        assert gaussian_kpc.centre == (2.0, 4.0)
        assert gaussian_kpc.axis_ratio == 0.5
        assert gaussian_kpc.phi == 45.0
        assert gaussian_kpc.intensity == 1.0
        assert gaussian_arcsec.sigma == 0.2
        assert gaussian_kpc.units_distance == 'kpc'
        assert gaussian_arcsec.units_luminosity == 'electrons_per_second'

        gaussian_kpc = gaussian_kpc.new_light_profile_with_units_converted(units_distance='kpc')

        assert gaussian_kpc.centre == (2.0, 4.0)
        assert gaussian_kpc.axis_ratio == 0.5
        assert gaussian_kpc.phi == 45.0
        assert gaussian_kpc.intensity == 1.0
        assert gaussian_arcsec.sigma == 0.2
        assert gaussian_kpc.units_distance == 'kpc'
        assert gaussian_arcsec.units_luminosity == 'electrons_per_second'

        gaussian_arcsec = gaussian_arcsec.new_light_profile_with_units_converted(units_distance='arcsec',
                                                                                 kpc_per_arcsec=2.0)

        assert gaussian_arcsec.centre == (1.0, 2.0)
        assert gaussian_arcsec.axis_ratio == 0.5
        assert gaussian_arcsec.phi == 45.0
        assert gaussian_arcsec.intensity == 1.0
        assert gaussian_arcsec.sigma == 0.1
        assert gaussian_arcsec.units_distance == 'arcsec'
        assert gaussian_arcsec.units_luminosity == 'electrons_per_second'

        gaussian_arcsec = gaussian_arcsec.new_light_profile_with_units_converted(units_luminosity='counts',
                                                                                 exposure_time=10.0)

        assert gaussian_arcsec.centre == (1.0, 2.0)
        assert gaussian_arcsec.axis_ratio == 0.5
        assert gaussian_arcsec.phi == 45.0
        assert gaussian_arcsec.intensity == 10.0
        assert gaussian_arcsec.sigma == 0.1
        assert gaussian_arcsec.units_distance == 'arcsec'
        assert gaussian_arcsec.units_luminosity == 'counts'

    def test__conversion_requires_kpc_per_arcsec_but_does_not_supply_it_raises_error(self):

        gaussian_arcsec = lp.EllipticalGaussian(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0, sigma=0.1)

        with pytest.raises(exc.UnitsException):
            gaussian_arcsec.new_light_profile_with_units_converted(units_distance='kpc')

        gaussian_kpc = lp.EllipticalGaussian(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0, sigma=0.1)
        gaussian_kpc.units_distance = 'kpc'

        with pytest.raises(exc.UnitsException):
            gaussian_kpc.new_light_profile_with_units_converted(units_distance='arcsec')

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


class TestAbstractSersic:

    def test__constructor_and_unit_conversions(self):
        
        sersic_arcsec = lp.AbstractEllipticalSersic(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                            effective_radius=0.6, sersic_index=4.0)

        assert sersic_arcsec.centre == (1.0, 2.0)
        assert sersic_arcsec.axis_ratio == 0.5
        assert sersic_arcsec.phi == 45.0
        assert sersic_arcsec.intensity == 1.0
        assert sersic_arcsec.effective_radius == 0.6
        assert sersic_arcsec.sersic_index == 4.0
        assert sersic_arcsec.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic_arcsec.elliptical_effective_radius == 0.6 / np.sqrt(0.5)
        assert sersic_arcsec.units_distance == 'arcsec'
        assert sersic_arcsec.units_luminosity == 'electrons_per_second'

        sersic_arcsec = sersic_arcsec.new_light_profile_with_units_converted(units_distance='arcsec')

        assert sersic_arcsec.centre == (1.0, 2.0)
        assert sersic_arcsec.axis_ratio == 0.5
        assert sersic_arcsec.phi == 45.0
        assert sersic_arcsec.intensity == 1.0
        assert sersic_arcsec.effective_radius == 0.6
        assert sersic_arcsec.sersic_index == 4.0
        assert sersic_arcsec.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic_arcsec.elliptical_effective_radius == 0.6 / np.sqrt(0.5)
        assert sersic_arcsec.units_distance == 'arcsec'
        assert sersic_arcsec.units_luminosity == 'electrons_per_second'

        sersic_kpc = sersic_arcsec.new_light_profile_with_units_converted(units_distance='kpc', kpc_per_arcsec=2.0)

        assert sersic_kpc.centre == (2.0, 4.0)
        assert sersic_kpc.axis_ratio == 0.5
        assert sersic_kpc.phi == 45.0
        assert sersic_kpc.intensity == 1.0
        assert sersic_kpc.effective_radius == 1.2
        assert sersic_kpc.sersic_index == 4.0
        assert sersic_kpc.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic_kpc.elliptical_effective_radius == 1.2 / np.sqrt(0.5)
        assert sersic_kpc.units_distance == 'kpc'
        assert sersic_arcsec.units_luminosity == 'electrons_per_second'

        sersic_kpc = sersic_kpc.new_light_profile_with_units_converted(units_distance='kpc')

        assert sersic_kpc.centre == (2.0, 4.0)
        assert sersic_kpc.axis_ratio == 0.5
        assert sersic_kpc.phi == 45.0
        assert sersic_kpc.intensity == 1.0
        assert sersic_kpc.effective_radius == 1.2
        assert sersic_kpc.sersic_index == 4.0
        assert sersic_kpc.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic_kpc.elliptical_effective_radius == 1.2 / np.sqrt(0.5)
        assert sersic_kpc.units_distance == 'kpc'
        assert sersic_arcsec.units_luminosity == 'electrons_per_second'

        sersic_arcsec = sersic_arcsec.new_light_profile_with_units_converted(units_distance='arcsec', kpc_per_arcsec=2.0)

        assert sersic_arcsec.centre == (1.0, 2.0)
        assert sersic_arcsec.axis_ratio == 0.5
        assert sersic_arcsec.phi == 45.0
        assert sersic_arcsec.intensity == 1.0
        assert sersic_arcsec.effective_radius == 0.6
        assert sersic_arcsec.sersic_index == 4.0
        assert sersic_arcsec.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic_arcsec.elliptical_effective_radius == 0.6 / np.sqrt(0.5)
        assert sersic_arcsec.units_distance == 'arcsec'
        assert sersic_arcsec.units_luminosity == 'electrons_per_second'

        sersic_arcsec = sersic_arcsec.new_light_profile_with_units_converted(units_luminosity='counts',
                                                                             exposure_time=10.0)

        assert sersic_arcsec.centre == (1.0, 2.0)
        assert sersic_arcsec.axis_ratio == 0.5
        assert sersic_arcsec.phi == 45.0
        assert sersic_arcsec.intensity == 10.0
        assert sersic_arcsec.effective_radius == 0.6
        assert sersic_arcsec.sersic_index == 4.0
        assert sersic_arcsec.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic_arcsec.elliptical_effective_radius == 0.6 / np.sqrt(0.5)
        assert sersic_arcsec.units_distance == 'arcsec'
        assert sersic_arcsec.units_luminosity == 'counts'

    def test__conversion_requires_kpc_per_arcsec_but_does_not_supply_it_raises_error(self):

        sersic_arcsec = lp.EllipticalSersic(centre=(1.0, 2.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                            effective_radius=0.6, sersic_index=4.0)

        with pytest.raises(exc.UnitsException):
            sersic_arcsec.new_light_profile_with_units_converted(units_distance='kpc')

        sersic_kpc = lp.EllipticalSersic(centre=(1.0, 2.0), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                         effective_radius=0.6, sersic_index=4.0)
        sersic_kpc.units_distance = 'kpc'

        with pytest.raises(exc.UnitsException):
            sersic_kpc.new_light_profile_with_units_converted(units_distance='arcsec')


class TestSersic:

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

    def test__constructor_and_unit_conversions(self):

        core_sersic_arcsec = lp.EllipticalCoreSersic(centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0,
                                                     effective_radius=0.6, sersic_index=4.0, radius_break=0.01,
                                                      intensity_break=0.1, gamma=1.0, alpha=2.0)

        assert core_sersic_arcsec.centre == (1.0, 2.0)
        assert core_sersic_arcsec.axis_ratio == 0.5
        assert core_sersic_arcsec.phi == 45.0
        assert core_sersic_arcsec.intensity == 1.0
        assert core_sersic_arcsec.effective_radius == 0.6
        assert core_sersic_arcsec.sersic_index == 4.0
        assert core_sersic_arcsec.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert core_sersic_arcsec.radius_break == 0.01
        assert core_sersic_arcsec.intensity_break == 0.1
        assert core_sersic_arcsec.gamma == 1.0
        assert core_sersic_arcsec.alpha == 2.0
        assert core_sersic_arcsec.elliptical_effective_radius == 0.6 / np.sqrt(0.5)
        assert core_sersic_arcsec.units_distance == 'arcsec'
        assert core_sersic_arcsec.units_luminosity == 'electrons_per_second'

        core_sersic_arcsec = core_sersic_arcsec.new_light_profile_with_units_converted(units_distance='arcsec')

        assert core_sersic_arcsec.centre == (1.0, 2.0)
        assert core_sersic_arcsec.axis_ratio == 0.5
        assert core_sersic_arcsec.phi == 45.0
        assert core_sersic_arcsec.intensity == 1.0
        assert core_sersic_arcsec.effective_radius == 0.6
        assert core_sersic_arcsec.sersic_index == 4.0
        assert core_sersic_arcsec.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert core_sersic_arcsec.radius_break == 0.01
        assert core_sersic_arcsec.intensity_break == 0.1
        assert core_sersic_arcsec.gamma == 1.0
        assert core_sersic_arcsec.alpha == 2.0
        assert core_sersic_arcsec.elliptical_effective_radius == 0.6 / np.sqrt(0.5)
        assert core_sersic_arcsec.units_distance == 'arcsec'
        assert core_sersic_arcsec.units_luminosity == 'electrons_per_second'

        core_sersic_kpc = core_sersic_arcsec.new_light_profile_with_units_converted(units_distance='kpc',
                                                                                    kpc_per_arcsec=2.0)

        assert core_sersic_kpc.centre == (2.0, 4.0)
        assert core_sersic_kpc.axis_ratio == 0.5
        assert core_sersic_kpc.phi == 45.0
        assert core_sersic_kpc.intensity == 1.0
        assert core_sersic_kpc.effective_radius == 1.2
        assert core_sersic_kpc.sersic_index == 4.0
        assert core_sersic_kpc.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert core_sersic_kpc.radius_break == 0.02
        assert core_sersic_kpc.intensity_break == 0.1
        assert core_sersic_kpc.gamma == 1.0
        assert core_sersic_kpc.alpha == 2.0
        assert core_sersic_kpc.elliptical_effective_radius == 1.2 / np.sqrt(0.5)
        assert core_sersic_kpc.units_distance == 'kpc'
        assert core_sersic_arcsec.units_luminosity == 'electrons_per_second'

        core_sersic_kpc = core_sersic_kpc.new_light_profile_with_units_converted(units_distance='kpc')

        assert core_sersic_kpc.centre == (2.0, 4.0)
        assert core_sersic_kpc.axis_ratio == 0.5
        assert core_sersic_kpc.phi == 45.0
        assert core_sersic_kpc.intensity == 1.0
        assert core_sersic_kpc.effective_radius == 1.2
        assert core_sersic_kpc.sersic_index == 4.0
        assert core_sersic_kpc.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert core_sersic_kpc.radius_break == 0.02
        assert core_sersic_kpc.intensity_break == 0.1
        assert core_sersic_kpc.gamma == 1.0
        assert core_sersic_kpc.alpha == 2.0
        assert core_sersic_kpc.elliptical_effective_radius == 1.2 / np.sqrt(0.5)
        assert core_sersic_kpc.units_distance == 'kpc'
        assert core_sersic_arcsec.units_luminosity == 'electrons_per_second'

        core_sersic_arcsec = core_sersic_arcsec.new_light_profile_with_units_converted(units_distance='arcsec',
                                                                                       kpc_per_arcsec=2.0)

        assert core_sersic_arcsec.centre == (1.0, 2.0)
        assert core_sersic_arcsec.axis_ratio == 0.5
        assert core_sersic_arcsec.phi == 45.0
        assert core_sersic_arcsec.intensity == 1.0
        assert core_sersic_arcsec.effective_radius == 0.6
        assert core_sersic_arcsec.sersic_index == 4.0
        assert core_sersic_arcsec.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert core_sersic_arcsec.radius_break == 0.01
        assert core_sersic_arcsec.intensity_break == 0.1
        assert core_sersic_arcsec.gamma == 1.0
        assert core_sersic_arcsec.alpha == 2.0
        assert core_sersic_arcsec.elliptical_effective_radius == 0.6 / np.sqrt(0.5)
        assert core_sersic_arcsec.units_distance == 'arcsec'
        assert core_sersic_arcsec.units_luminosity == 'electrons_per_second'

        core_sersic_arcsec = core_sersic_arcsec.new_light_profile_with_units_converted(units_luminosity='counts',
                                                                                       exposure_time=10.0)

        assert core_sersic_arcsec.centre == (1.0, 2.0)
        assert core_sersic_arcsec.axis_ratio == 0.5
        assert core_sersic_arcsec.phi == 45.0
        assert core_sersic_arcsec.intensity == 10.0
        assert core_sersic_arcsec.effective_radius == 0.6
        assert core_sersic_arcsec.sersic_index == 4.0
        assert core_sersic_arcsec.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert core_sersic_arcsec.radius_break == 0.01
        assert core_sersic_arcsec.intensity_break == 1.0
        assert core_sersic_arcsec.gamma == 1.0
        assert core_sersic_arcsec.alpha == 2.0
        assert core_sersic_arcsec.elliptical_effective_radius == 0.6 / np.sqrt(0.5)
        assert core_sersic_arcsec.units_distance == 'arcsec'
        assert core_sersic_arcsec.units_luminosity == 'counts'

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

    def test__within_circle_in_electrons_per_second__spherical_sersic_index_2__compare_to_analytic(self):

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

    def test__within_circle_in_electrons_per_second__spherical_dev_vaucouleurs__compare_to_analytic(self):

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

    def test__within_circle_in_electrons_per_second__spherical_sersic_2__compare_to_grid(self):

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

    def test__within_ellipse_in_electrons_per_second__elliptical_sersic_2__compare_to_grid(self):

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

    def test__within_circle_in_counts__multiplies_by_exposure_time(self):

        sersic = lp.SphericalSersic(intensity=3.0, effective_radius=2.0, sersic_index=1.0)

        integral_radius = 5.5

        # Use gamma function for analytic computation of the intensity within a radius=0.5

        x = sersic.sersic_constant * (integral_radius / sersic.effective_radius) ** (1.0 / sersic.sersic_index)

        intensity_analytic = sersic.intensity * sersic.effective_radius ** 2 * 2 * math.pi * sersic.sersic_index * (
                math.e ** sersic.sersic_constant / (
                sersic.sersic_constant ** (2 * sersic.sersic_index))) * scipy.special.gamma(
            2 * sersic.sersic_index) * scipy.special.gammainc(
            2 * sersic.sersic_index, x)

        sersic = sersic.new_light_profile_with_units_luminosity_converted(units_luminosity='counts', exposure_time=3.0)

        intensity_integral = sersic.luminosity_within_circle(radius=integral_radius)

        assert 3.0*intensity_analytic == pytest.approx(intensity_integral, 1e-3)

    def test__within_ellipse_in_counts__check_multiplies_by_exposure_time(self):

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

        sersic = sersic.new_light_profile_with_units_luminosity_converted(units_luminosity='counts', exposure_time=3.0)

        intensity_integral = sersic.luminosity_within_ellipse(major_axis=integral_radius)

        assert 3.0*luminosity_tot[0] == pytest.approx(intensity_integral, 0.02)


class TestGrids(object):

    def test__grid_to_eccentric_radius(self, elliptical):
        assert elliptical.grid_to_eccentric_radii(np.array([[1, 1]])) == pytest.approx(
            elliptical.grid_to_eccentric_radii(np.array([[-1, -1]])), 1e-10)

    def test__intensity_from_grid(self, elliptical):
        assert elliptical.intensities_from_grid(np.array([[1, 1]])) == \
               pytest.approx(elliptical.intensities_from_grid(np.array([[-1, -1]])), 1e-4)
