import numpy as np
import pytest
from astropy import cosmology as cosmo

from autolens import exc, dimensions as dim
from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.lens import plane as pl
from autolens.lens.util import lens_util
from autolens.model import cosmology_util
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy.util import galaxy_util
from autolens.model.inversion import pixelizations, regularization
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.unit.mock.mock_imaging import MockBorders
from test.unit.mock.mock_inversion import MockRegularization, MockPixelization
from test.unit.mock.mock_cosmology import MockCosmology

planck = cosmo.Planck15

@pytest.fixture(name="grid_stack")
def make_grid_stack():
    mask = msk.Mask(np.array([[True, True, True, True],
                              [True, False, False, True],
                              [True, True, True, True]]), pixel_scale=6.0)

    grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask, sub_grid_size=2,
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
    mask = msk.Mask(np.array([[True, False]]), pixel_scale=3.0)
    return grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask, 2, (3, 3))


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


class TestAbstractPlane(object):

    class TestCosmology:

        def test__all_cosmological_quantities_match_cosmology_util(self):

            plane = pl.AbstractPlane(redshift=0.1, galaxies=None, cosmology=planck)

            assert plane.arcsec_per_kpc == cosmology_util.arcsec_per_kpc_from_redshift_and_cosmology(
                redshift=0.1, cosmology=planck)

            assert plane.kpc_per_arcsec == \
                   cosmology_util.kpc_per_arcsec_from_redshift_and_cosmology(redshift=0.1, cosmology=planck)

            assert plane.angular_diameter_distance_to_earth_in_units(unit_length='arcsec') == \
                   cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
                       redshift=0.1, cosmology=planck, unit_length='arcsec')


            plane = pl.AbstractPlane(redshift=0.1, galaxies=None, cosmology=planck)

            assert plane.angular_diameter_distance_to_earth_in_units(unit_length='kpc') == \
                   cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
                       redshift=0.1, cosmology=planck, unit_length='kpc')

            plane = pl.AbstractPlane(redshift=1.0, galaxies=None, cosmology=planck)

            assert plane.arcsec_per_kpc == cosmology_util.arcsec_per_kpc_from_redshift_and_cosmology(
                redshift=1.0, cosmology=planck)

            assert plane.kpc_per_arcsec == \
                   cosmology_util.kpc_per_arcsec_from_redshift_and_cosmology(redshift=1.0, cosmology=planck)

            assert plane.angular_diameter_distance_to_earth_in_units(unit_length='arcsec') == \
                   cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
                       redshift=1.0, cosmology=planck, unit_length='arcsec')

            plane = pl.AbstractPlane(redshift=1.0, galaxies=None, cosmology=planck)

            assert plane.angular_diameter_distance_to_earth_in_units(unit_length='kpc') == \
                   cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
                       redshift=1.0, cosmology=planck, unit_length='kpc')

            plane = pl.AbstractPlane(redshift=0.6, galaxies=None)

            assert plane.cosmic_average_density_in_units(unit_length='arcsec', unit_mass='solMass') == \
                   cosmology_util.cosmic_average_density_from_redshift_and_cosmology(
                       redshift=0.6, cosmology=planck, unit_length='arcsec', unit_mass='solMass')

            plane = pl.AbstractPlane(redshift=0.6, galaxies=None, cosmology=planck)

            assert plane.cosmic_average_density_in_units(unit_length='kpc', unit_mass='solMass') == \
                   cosmology_util.cosmic_average_density_from_redshift_and_cosmology(
                       redshift=0.6, cosmology=planck, unit_length='kpc', unit_mass='solMass')

    class TestProperties:

        def test__has_light_profile(self):
            plane = pl.AbstractPlane(galaxies=[g.Galaxy()], redshift=None)
            assert plane.has_light_profile is False

            plane = pl.AbstractPlane(galaxies=[g.Galaxy(light_profile=lp.LightProfile())], redshift=None)
            assert plane.has_light_profile is True

            plane = pl.AbstractPlane(galaxies=[g.Galaxy(light_profile=lp.LightProfile()), g.Galaxy()],
                                     redshift=None)
            assert plane.has_light_profile is True

        def test__has_mass_profile(self):
            plane = pl.AbstractPlane(galaxies=[g.Galaxy()], redshift=None)
            assert plane.has_mass_profile is False

            plane = pl.AbstractPlane(galaxies=[g.Galaxy(light_profile=mp.MassProfile())], redshift=None)
            assert plane.has_mass_profile is True

            plane = pl.AbstractPlane(galaxies=[g.Galaxy(light_profile=mp.MassProfile()), g.Galaxy()],
                                     redshift=None)
            assert plane.has_mass_profile is True

        def test__has_pixelization(self):
            plane = pl.AbstractPlane(galaxies=[g.Galaxy()], redshift=None)
            assert plane.has_pixelization is False

            galaxy_pix = g.Galaxy(pixelization=pixelizations.Pixelization(),
                                  regularization=regularization.Regularization())

            plane = pl.AbstractPlane(galaxies=[galaxy_pix], redshift=None)
            assert plane.has_pixelization is True

            plane = pl.AbstractPlane(galaxies=[galaxy_pix, g.Galaxy()], redshift=None)
            assert plane.has_pixelization is True

        def test__has_regularization(self):
            plane = pl.AbstractPlane(galaxies=[g.Galaxy()], redshift=None)
            assert plane.has_regularization is False

            galaxy_pix = g.Galaxy(pixelization=pixelizations.Pixelization(),
                                  regularization=regularization.Regularization())

            plane = pl.AbstractPlane(galaxies=[galaxy_pix], redshift=None)
            assert plane.has_regularization is True

            plane = pl.AbstractPlane(galaxies=[galaxy_pix, g.Galaxy()], redshift=None)
            assert plane.has_regularization is True

    class TestRegularization:

        def test__no_galaxies_with_pixelizations_in_plane__returns_none(self):
            galaxy_no_pix = g.Galaxy()

            plane = pl.AbstractPlane(galaxies=[galaxy_no_pix], redshift=None)

            assert plane.regularization is None

        def test__1_galaxy_in_plane__it_has_pixelization__returns_mapper(self):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))

            plane = pl.AbstractPlane(galaxies=[galaxy_pix], redshift=None)

            assert plane.regularization.value == 0

        def test__2_galaxies_in_plane__1_has_pixelization__extracts_reconstructor(self):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_no_pix = g.Galaxy()

            plane = pl.AbstractPlane(galaxies=[galaxy_no_pix, galaxy_pix], redshift=None)

            assert plane.regularization.value == 0

        def test__2_galaxies_in_plane__both_have_pixelization__raises_error(self):
            galaxy_pix_0 = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_pix_1 = g.Galaxy(pixelization=MockPixelization(value=2), regularization=MockRegularization(value=0))

            plane = pl.AbstractPlane(galaxies=[galaxy_pix_0, galaxy_pix_1], redshift=None)

            with pytest.raises(exc.PixelizationException):
                print(plane.regularization)

    class TestLuminosities:

        def test__within_circle_different_luminosity_units__same_as_galaxy_luminosities(self):

            g0 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=2.0))

            radius = dim.Length(1.0, 'arcsec')

            g0_luminosity = g0.luminosity_within_circle_in_units(radius=radius, unit_luminosity='eps')
            g1_luminosity = g1.luminosity_within_circle_in_units(radius=radius, unit_luminosity='eps')
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_luminosities = plane.luminosities_of_galaxies_within_circles_in_units(radius=radius, unit_luminosity='eps')

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

            g0_luminosity = g0.luminosity_within_circle_in_units(radius=radius, unit_luminosity='counts', exposure_time=3.0)
            g1_luminosity = g1.luminosity_within_circle_in_units(radius=radius, unit_luminosity='counts', exposure_time=3.0)
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_luminosities = plane.luminosities_of_galaxies_within_circles_in_units(radius=radius, unit_luminosity='counts',
                                                                                        exposure_time=3.0)

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

        def test__within_circle_different_distance_units__same_as_galaxy_luminosities(self):

            g0 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=2.0))

            radius = dim.Length(1.0, 'arcsec')

            g0_luminosity = g0.luminosity_within_circle_in_units(radius=radius)
            g1_luminosity = g1.luminosity_within_circle_in_units(radius=radius)

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_luminosities = plane.luminosities_of_galaxies_within_circles_in_units(radius=radius)

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

            radius = dim.Length(1.0, 'kpc')

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            g0_luminosity = g0.luminosity_within_circle_in_units(radius=radius)
            g1_luminosity = g1.luminosity_within_circle_in_units(radius=radius)

            plane_luminosities = plane.luminosities_of_galaxies_within_circles_in_units(radius=radius)

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

        def test__within_ellipse_different_luminosity_units__same_as_galaxy_luminosities(self):

            g0 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=2.0))

            major_axis = dim.Length(1.0, 'arcsec')

            g0_luminosity = g0.luminosity_within_ellipse_in_units(major_axis=major_axis, unit_luminosity='eps')
            g1_luminosity = g1.luminosity_within_ellipse_in_units(major_axis=major_axis, unit_luminosity='eps')
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_luminosities = plane.luminosities_of_galaxies_within_ellipses_in_units(major_axis=major_axis, unit_luminosity='eps')

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

            g0_luminosity = g0.luminosity_within_ellipse_in_units(major_axis=major_axis, unit_luminosity='counts', exposure_time=3.0)
            g1_luminosity = g1.luminosity_within_ellipse_in_units(major_axis=major_axis, unit_luminosity='counts', exposure_time=3.0)
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_luminosities = plane.luminosities_of_galaxies_within_ellipses_in_units(major_axis=major_axis, unit_luminosity='counts',
                                                                                         exposure_time=3.0)

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

        def test__within_ellipse_different_distance_units__same_as_galaxy_luminosities(self):

            g0 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, luminosity=lp.SphericalSersic(intensity=2.0))

            major_axis = dim.Length(1.0, 'arcsec')

            g0_luminosity = g0.luminosity_within_ellipse_in_units(major_axis=major_axis)
            g1_luminosity = g1.luminosity_within_ellipse_in_units(major_axis=major_axis)
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_luminosities = plane.luminosities_of_galaxies_within_ellipses_in_units(major_axis=major_axis)

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity

            major_axis = dim.Length(1.0, 'kpc')

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)

            g0_luminosity = g0.luminosity_within_ellipse_in_units(major_axis=major_axis)
            g1_luminosity = g1.luminosity_within_ellipse_in_units(major_axis=major_axis)

            plane_luminosities = plane.luminosities_of_galaxies_within_ellipses_in_units(major_axis=major_axis)

            assert plane_luminosities[0] == g0_luminosity
            assert plane_luminosities[1] == g1_luminosity
            
    class TestMasses:

        def test__within_circle_different_mass_units__same_as_galaxy_masses(self):

            g0 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=2.0))

            radius = dim.Length(1.0, 'arcsec')

            g0_mass = g0.mass_within_circle_in_units(radius=radius, unit_mass='angular')
            g1_mass = g1.mass_within_circle_in_units(radius=radius, unit_mass='angular')

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)

            plane_masses = plane.masses_of_galaxies_within_circles_in_units(radius=radius, unit_mass='angular')

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

            g0_mass = g0.mass_within_circle_in_units(radius=radius, unit_mass='solMass',
                                                     redshift_source=1.0)
            g1_mass = g1.mass_within_circle_in_units(radius=radius, unit_mass='solMass',
                                                     redshift_source=1.0)

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)

            plane_masses = plane.masses_of_galaxies_within_circles_in_units(
                radius=radius, unit_mass='solMass', redshift_source=1.0)

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

        def test__within_circle_different_distance_units__same_as_galaxy_masses(self):

            radius = dim.Length(1.0, 'arcsec')

            g0 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=2.0))

            g0_mass = g0.mass_within_circle_in_units(radius=radius, redshift_source=1.0)
            g1_mass = g1.mass_within_circle_in_units(radius=radius, redshift_source=1.0)

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_masses = plane.masses_of_galaxies_within_circles_in_units(radius=radius,
                                                                            redshift_source=1.0)

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

            radius = dim.Length(1.0, 'kpc')

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            g0_mass = g0.mass_within_circle_in_units(radius=radius, redshift_source=1.0,
                                                     kpc_per_arcsec=plane.kpc_per_arcsec)
            g1_mass = g1.mass_within_circle_in_units(radius=radius, redshift_source=1.0,
                                                     kpc_per_arcsec=plane.kpc_per_arcsec)

            plane_masses = plane.masses_of_galaxies_within_circles_in_units(radius=radius,
                                                                            redshift_source=1.0)

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

        def test__within_ellipse_different_mass_units__same_as_galaxy_masses(self):

            g0 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=2.0))

            major_axis = dim.Length(1.0, 'arcsec')

            g0_mass = g0.mass_within_ellipse_in_units(major_axis=major_axis, unit_mass='angular')
            g1_mass = g1.mass_within_ellipse_in_units(major_axis=major_axis, unit_mass='angular')
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_masses = plane.masses_of_galaxies_within_ellipses_in_units(major_axis=major_axis,
                                                                             unit_mass='angular')

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

            g0_mass = g0.mass_within_ellipse_in_units(major_axis=major_axis, unit_mass='solMass',
                                                      redshift_source=1.0)
            g1_mass = g1.mass_within_ellipse_in_units(major_axis=major_axis, unit_mass='solMass',
                                                      redshift_source=1.0)

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_masses = plane.masses_of_galaxies_within_ellipses_in_units(major_axis=major_axis,
                                                                             unit_mass='solMass',
                                                                             redshift_source=1.0)

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

        def test__within_ellipse_different_distance_units__same_as_galaxy_masses(self):

            g0 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=2.0))

            major_axis = dim.Length(1.0, 'arcsec')

            g0_mass = g0.mass_within_ellipse_in_units(major_axis=major_axis,
                                                      redshift_source=1.0)
            g1_mass = g1.mass_within_ellipse_in_units(major_axis=major_axis,
                                                      redshift_source=1.0)
            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            plane_masses = plane.masses_of_galaxies_within_ellipses_in_units(major_axis=major_axis,
                                                                             redshift_source=1.0)

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

            major_axis = dim.Length(1.0, 'kpc')

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=0.5)
            g0_mass = g0.mass_within_ellipse_in_units(major_axis=major_axis, redshift_source=1.0,
                                                      kpc_per_arcsec=plane.kpc_per_arcsec)
            g1_mass = g1.mass_within_ellipse_in_units(major_axis=major_axis, redshift_source=1.0,
                                                      kpc_per_arcsec=plane.kpc_per_arcsec)
            plane_masses = plane.masses_of_galaxies_within_ellipses_in_units(major_axis=major_axis,
                                                                             redshift_source=1.0)

            assert plane_masses[0] == g0_mass
            assert plane_masses[1] == g1_mass

    class TestEinsteinRadiiAndMass:

        def test__plane_has_galaxies_with_sis_profiles__einstein_radius_and_mass_sum_of_sis_profiles(self):

            cosmology = MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0)

            sis_0 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            sis_1 = g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=2.0))

            plane = pl.AbstractPlane(galaxies=[sis_0], redshift=0.5, cosmology=cosmology)

            assert plane.einstein_radius_in_units(unit_length='arcsec') == pytest.approx(1.0, 1.0e-4)
            assert plane.einstein_radius_in_units(unit_length='kpc') == pytest.approx(2.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='angular') == pytest.approx(np.pi, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='solMass', redshift_source=1.0) \
                   == pytest.approx(2.0*np.pi, 1.0e-4)

            plane = pl.AbstractPlane(galaxies=[sis_1], redshift=0.5, cosmology=cosmology)

            assert plane.einstein_radius_in_units(unit_length='arcsec') == pytest.approx(2.0, 1.0e-4)
            assert plane.einstein_radius_in_units(unit_length='kpc') == pytest.approx(4.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='angular') == pytest.approx(np.pi*2.0**2.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='solMass', redshift_source=1.0) == \
                   pytest.approx(2.0*np.pi*2.0**2.0, 1.0e-4)

            plane = pl.AbstractPlane(galaxies=[sis_0, sis_1], redshift=0.5, cosmology=cosmology)

            assert plane.einstein_radius_in_units(unit_length='arcsec') == pytest.approx(3.0, 1.0e-4)
            assert plane.einstein_radius_in_units(unit_length='kpc') == \
                   pytest.approx(2.0*3.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='angular') == pytest.approx(np.pi*(1.0 + 2.0**2.0), 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='solMass', redshift_source=1.0) == \
                   pytest.approx(2.0*np.pi*(1.0 + 2.0**2.0), 1.0e-4)

        def test__include_galaxy_with_no_mass_profile__does_not_impact_einstein_radius_or_mass(self):

            sis_0 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0))
            sis_1 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=2.0))
            g0 = g.Galaxy()

            plane = pl.AbstractPlane(galaxies=[sis_0, g0], redshift=0.5)

            assert plane.einstein_radius_in_units(unit_length='arcsec') == pytest.approx(1.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='angular') == pytest.approx(np.pi, 1.0e-4)

            plane = pl.AbstractPlane(galaxies=[sis_1, g0], redshift=0.5)

            assert plane.einstein_radius_in_units(unit_length='arcsec') == pytest.approx(2.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='angular') == pytest.approx(np.pi*2.0**2.0, 1.0e-4)

            plane = pl.AbstractPlane(galaxies=[sis_0, sis_1, g0], redshift=0.5)

            assert plane.einstein_radius_in_units(unit_length='arcsec') == pytest.approx(3.0, 1.0e-4)
            assert plane.einstein_mass_in_units(unit_mass='angular') == pytest.approx(np.pi*(1.0 + 2.0**2.0), 1.0e-4)

        def test__only_galaxies_without_mass_profiles__einstein_radius_and_mass_are_none(self):
            
            g0 = g.Galaxy()

            plane = pl.AbstractPlane(galaxies=[g0], redshift=0.5)

            assert plane.einstein_radius_in_units() is None
            assert plane.einstein_mass_in_units() is None

            plane = pl.AbstractPlane(galaxies=[g0, g0], redshift=0.5)

            assert plane.einstein_radius_in_units() is None
            assert plane.einstein_mass_in_units() is None

    class TestMassProfileGeometry:

        def test__extract_centres_of_all_mass_profiles_of_all_galaxies(self):

            g0 = g.Galaxy(mass=mp.SphericalIsothermal(centre=(1.0, 1.0)))
            g1 = g.Galaxy(mass=mp.SphericalIsothermal(centre=(2.0, 2.0)))
            g2 = g.Galaxy(mass0=mp.SphericalIsothermal(centre=(3.0, 3.0)),
                          mass1=mp.SphericalIsothermal(centre=(4.0, 4.0)))

            plane = pl.AbstractPlane(galaxies=[g.Galaxy()], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == None

            plane = pl.AbstractPlane(galaxies=[g0], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == [[(1.0, 1.0)]]

            plane = pl.AbstractPlane(galaxies=[g1], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == [[(2.0, 2.0)]]

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == [[(1.0, 1.0)], [(2.0, 2.0)]]

            plane = pl.AbstractPlane(galaxies=[g1, g0], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == [[(2.0, 2.0)], [(1.0, 1.0)]]

            plane = pl.AbstractPlane(galaxies=[g0, g.Galaxy(), g1, g.Galaxy()], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == [[(1.0, 1.0)], [(2.0, 2.0)]]

            plane = pl.AbstractPlane(galaxies=[g0, g.Galaxy(), g1, g.Galaxy(), g2], redshift=None)
            assert plane.centres_of_galaxy_mass_profiles == [[(1.0, 1.0)], [(2.0, 2.0)], [(3.0, 3.0), (4.0, 4.0)]]

        def test__extracts_axis_ratio_of_all_mass_profiles_of_all_galaxies(self):

            g0 = g.Galaxy(mass=mp.EllipticalIsothermal(axis_ratio=0.9))
            g1 = g.Galaxy(mass=mp.EllipticalIsothermal(axis_ratio=0.8))
            g2 = g.Galaxy(mass0=mp.EllipticalIsothermal(axis_ratio=0.7),
                          mass1=mp.EllipticalIsothermal(axis_ratio=0.6))

            plane = pl.AbstractPlane(galaxies=[g.Galaxy()], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == None

            plane = pl.AbstractPlane(galaxies=[g0], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == [[0.9]]

            plane = pl.AbstractPlane(galaxies=[g1], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == [[0.8]]

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == [[0.9], [0.8]]

            plane = pl.AbstractPlane(galaxies=[g1, g0], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == [[0.8], [0.9]]

            plane = pl.AbstractPlane(galaxies=[g0, g.Galaxy(), g1, g.Galaxy()], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == [[0.9], [0.8]]

            plane = pl.AbstractPlane(galaxies=[g0, g.Galaxy(), g1, g.Galaxy(), g2], redshift=None)
            assert plane.axis_ratios_of_galaxy_mass_profiles == [[0.9], [0.8], [0.7, 0.6]]
            
        def test__extracts_phi_of_all_mass_profiles_of_all_galaxies(self):

            g0 = g.Galaxy(mass=mp.EllipticalIsothermal(phi=0.9))
            g1 = g.Galaxy(mass=mp.EllipticalIsothermal(phi=0.8))
            g2 = g.Galaxy(mass0=mp.EllipticalIsothermal(phi=0.7),
                          mass1=mp.EllipticalIsothermal(phi=0.6))

            plane = pl.AbstractPlane(galaxies=[g.Galaxy()], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == None

            plane = pl.AbstractPlane(galaxies=[g0], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == [[0.9]]

            plane = pl.AbstractPlane(galaxies=[g1], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == [[0.8]]

            plane = pl.AbstractPlane(galaxies=[g0, g1], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == [[0.9], [0.8]]

            plane = pl.AbstractPlane(galaxies=[g1, g0], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == [[0.8], [0.9]]

            plane = pl.AbstractPlane(galaxies=[g0, g.Galaxy(), g1, g.Galaxy()], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == [[0.9], [0.8]]

            plane = pl.AbstractPlane(galaxies=[g0, g.Galaxy(), g1, g.Galaxy(), g2], redshift=None)
            assert plane.phis_of_galaxy_mass_profiles == [[0.9], [0.8], [0.7, 0.6]]


class TestAbstractPlaneGridded(object):
    class TestGridLensing:

        def test__grid_stack_setup_for_regular_sub_and_blurring__no_deflections(self, grid_stack, galaxy_mass):
            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_mass], grid_stack=grid_stack, compute_deflections=False,
                                            redshift=None, border=None)

            assert plane.grid_stack.regular == pytest.approx(np.array([[1.0, 1.0], [1.0, 0.0]]), 1e-3)
            assert plane.grid_stack.sub == pytest.approx(np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0],
                                                                   [-1.0, 2.0], [-1.0, 4.0], [1.0, 2.0], [1.0, 4.0]]),
                                                         1e-3)
            assert plane.grid_stack.blurring == pytest.approx(
                np.array([[1.0, 0.0], [-6.0, -3.0], [-6.0, 3.0], [-6.0, 9.0],
                          [0.0, -9.0], [0.0, 9.0],
                          [6.0, -9.0], [6.0, -3.0], [6.0, 3.0], [6.0, 9.0]]), 1e-3)

            assert plane.deflection_stack is None

        def test__same_as_above_but_test_deflections(self, grid_stack, galaxy_mass):
            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_mass], grid_stack=grid_stack, compute_deflections=True,
                                            redshift=None, border=None)

            sub_galaxy_deflections = galaxy_mass.deflections_from_grid(grid_stack.sub)
            blurring_galaxy_deflections = galaxy_mass.deflections_from_grid(grid_stack.blurring)

            assert plane.deflection_stack.regular == pytest.approx(np.array([[0.707, 0.707], [1.0, 0.0]]), 1e-3)
            assert (plane.deflection_stack.sub == sub_galaxy_deflections).all()
            assert (plane.deflection_stack.blurring == blurring_galaxy_deflections).all()

        def test__same_as_above__x2_galaxy_in_plane__or_galaxy_x2_sis__deflections_double(self, grid_stack,
                                                                                          galaxy_mass,
                                                                                          galaxy_mass_x2):
            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_mass_x2], grid_stack=grid_stack, compute_deflections=True,
                                            redshift=None, border=None)

            sub_galaxy_deflections = galaxy_mass_x2.deflections_from_grid(grid_stack.sub)
            blurring_galaxy_deflections = galaxy_mass_x2.deflections_from_grid(grid_stack.blurring)

            assert plane.deflection_stack.regular == pytest.approx(np.array([[2.0 * 0.707, 2.0 * 0.707], [2.0, 0.0]]),
                                                                   1e-3)
            assert (plane.deflection_stack.sub == sub_galaxy_deflections).all()
            assert (plane.deflection_stack.blurring == blurring_galaxy_deflections).all()

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_mass, galaxy_mass], grid_stack=grid_stack,
                                            compute_deflections=True, redshift=None, border=None)

            sub_galaxy_deflections = galaxy_mass.deflections_from_grid(grid_stack.sub)
            blurring_galaxy_deflections = galaxy_mass.deflections_from_grid(grid_stack.blurring)

            assert plane.deflection_stack.regular == pytest.approx(np.array([[2.0 * 0.707, 2.0 * 0.707], [2.0, 0.0]]),
                                                                   1e-3)
            assert (plane.deflection_stack.sub == 2.0 * sub_galaxy_deflections).all()
            assert (plane.deflection_stack.blurring == 2.0 * blurring_galaxy_deflections).all()

        def test__plane_has_no_galaxies__deflections_all_zeros_shape_of_grid_stack(self, grid_stack):
            plane = pl.AbstractGriddedPlane(galaxies=[], grid_stack=grid_stack, border=None,
                                            compute_deflections=True, redshift=None)

            assert (plane.deflection_stack.regular == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (plane.deflection_stack.sub == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])).all()
            assert (plane.deflection_stack.blurring == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                                 [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                                 [0.0, 0.0], [0.0, 0.0]])).all()

    class TestImage:

        def test__image_from_plane__same_as_its_light_profile_image(self, grid_stack, galaxy_light):
            light_profile = galaxy_light.light_profiles[0]

            lp_sub_image = light_profile.intensities_from_grid(grid_stack.sub)

            # Perform sub gridding average manually
            lp_image_pixel_0 = (lp_sub_image[0] + lp_sub_image[1] + lp_sub_image[2] + lp_sub_image[3]) / 4
            lp_image_pixel_1 = (lp_sub_image[4] + lp_sub_image[5] + lp_sub_image[6] + lp_sub_image[7]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_light], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert (plane.image_plane_image_1d[0] == lp_image_pixel_0).all()
            assert (plane.image_plane_image_1d[1] == lp_image_pixel_1).all()
            assert (plane.image_plane_image ==
                    grid_stack.regular.scaled_array_2d_from_array_1d(plane.image_plane_image_1d)).all()

        def test__image_from_plane__same_as_its_galaxy_image(self, grid_stack, galaxy_light):
            galaxy_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack.sub, galaxies=[galaxy_light])

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_light], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.image_plane_image_1d == pytest.approx(galaxy_image, 1.0e-4)

            image_plane_image = grid_stack.regular.scaled_array_2d_from_array_1d(plane.image_plane_image_1d)

            assert plane.image_plane_image == pytest.approx(image_plane_image, 1.0e-4)

        def test_single_multiple_intensity(self, grid_stack):
            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            plane = pl.AbstractGriddedPlane(galaxies=[g0], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)
            assert (plane.image_plane_image_1d_of_galaxies[0] == plane.image_plane_image_1d_of_galaxy(g0)).all()

        def test__image_plane_image_of_galaxies(self, grid_stack):
            # Overwrite one value so intensity in each pixel is different
            grid_stack.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            lp0 = g0.light_profiles[0]
            lp1 = g1.light_profiles[0]

            lp0_sub_image = lp0.intensities_from_grid(grid_stack.sub)
            lp1_sub_image = lp1.intensities_from_grid(grid_stack.sub)

            # Perform sub gridding average manually
            lp0_image_pixel_0 = (lp0_sub_image[0] + lp0_sub_image[1] + lp0_sub_image[2] + lp0_sub_image[3]) / 4
            lp0_image_pixel_1 = (lp0_sub_image[4] + lp0_sub_image[5] + lp0_sub_image[6] + lp0_sub_image[7]) / 4
            lp1_image_pixel_0 = (lp1_sub_image[0] + lp1_sub_image[1] + lp1_sub_image[2] + lp1_sub_image[3]) / 4
            lp1_image_pixel_1 = (lp1_sub_image[4] + lp1_sub_image[5] + lp1_sub_image[6] + lp1_sub_image[7]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.image_plane_image_1d[0] == pytest.approx(lp0_image_pixel_0 + lp1_image_pixel_0, 1.0e-4)
            assert plane.image_plane_image_1d[1] == pytest.approx(lp0_image_pixel_1 + lp1_image_pixel_1, 1.0e-4)

            image_plane_image = grid_stack.regular.scaled_array_2d_from_array_1d(plane.image_plane_image_1d)

            assert plane.image_plane_image == image_plane_image

            assert plane.image_plane_image_1d_of_galaxies[0][0] == lp0_image_pixel_0
            assert plane.image_plane_image_1d_of_galaxies[0][1] == lp0_image_pixel_1
            assert plane.image_plane_image_1d_of_galaxies[1][0] == lp1_image_pixel_0
            assert plane.image_plane_image_1d_of_galaxies[1][1] == lp1_image_pixel_1

        def test__same_as_above__use_multiple_galaxies(self, grid_stack):
            # Overwrite one value so intensity in each pixel is different
            grid_stack.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            g0_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack.sub, galaxies=[g0])
            g1_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack.sub, galaxies=[g1])

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.image_plane_image_1d == pytest.approx(g0_image + g1_image, 1.0e-4)
            assert (plane.image_plane_image ==
                    grid_stack.regular.scaled_array_2d_from_array_1d(plane.image_plane_image_1d)).all()

            assert (plane.image_plane_image_1d_of_galaxies[0] == g0_image).all()
            assert (plane.image_plane_image_1d_of_galaxies[1] == g1_image).all()

        def test__padded_grid_stack_in__image_plane_image_is_padded(self, padded_grid_stack, galaxy_light):
            light_profile = galaxy_light.light_profiles[0]

            lp_sub_image = light_profile.intensities_from_grid(padded_grid_stack.sub)

            # Perform sub gridding average manually
            lp_image_pixel_0 = (lp_sub_image[0] + lp_sub_image[1] + lp_sub_image[2] + lp_sub_image[3]) / 4
            lp_image_pixel_1 = (lp_sub_image[4] + lp_sub_image[5] + lp_sub_image[6] + lp_sub_image[7]) / 4
            lp_image_pixel_2 = (lp_sub_image[8] + lp_sub_image[9] + lp_sub_image[10] + lp_sub_image[11]) / 4
            lp_image_pixel_3 = (lp_sub_image[12] + lp_sub_image[13] + lp_sub_image[14] + lp_sub_image[15]) / 4
            lp_image_pixel_4 = (lp_sub_image[16] + lp_sub_image[17] + lp_sub_image[18] + lp_sub_image[19]) / 4
            lp_image_pixel_5 = (lp_sub_image[20] + lp_sub_image[21] + lp_sub_image[22] + lp_sub_image[23]) / 4
            lp_image_pixel_6 = (lp_sub_image[24] + lp_sub_image[25] + lp_sub_image[26] + lp_sub_image[27]) / 4
            lp_image_pixel_7 = (lp_sub_image[28] + lp_sub_image[29] + lp_sub_image[30] + lp_sub_image[31]) / 4
            lp_image_pixel_8 = (lp_sub_image[32] + lp_sub_image[33] + lp_sub_image[34] + lp_sub_image[35]) / 4
            lp_image_pixel_9 = (lp_sub_image[36] + lp_sub_image[37] + lp_sub_image[38] + lp_sub_image[39]) / 4
            lp_image_pixel_10 = (lp_sub_image[40] + lp_sub_image[41] + lp_sub_image[42] + lp_sub_image[43]) / 4
            lp_image_pixel_11 = (lp_sub_image[44] + lp_sub_image[45] + lp_sub_image[46] + lp_sub_image[47]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_light], grid_stack=padded_grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.image_plane_image_for_simulation.shape == (3, 4)
            assert (plane.image_plane_image_for_simulation[0, 0] == lp_image_pixel_0).all()
            assert (plane.image_plane_image_for_simulation[0, 1] == lp_image_pixel_1).all()
            assert (plane.image_plane_image_for_simulation[0, 2] == lp_image_pixel_2).all()
            assert (plane.image_plane_image_for_simulation[0, 3] == lp_image_pixel_3).all()
            assert (plane.image_plane_image_for_simulation[1, 0] == lp_image_pixel_4).all()
            assert (plane.image_plane_image_for_simulation[1, 1] == lp_image_pixel_5).all()
            assert (plane.image_plane_image_for_simulation[1, 2] == lp_image_pixel_6).all()
            assert (plane.image_plane_image_for_simulation[1, 3] == lp_image_pixel_7).all()
            assert (plane.image_plane_image_for_simulation[2, 0] == lp_image_pixel_8).all()
            assert (plane.image_plane_image_for_simulation[2, 1] == lp_image_pixel_9).all()
            assert (plane.image_plane_image_for_simulation[2, 2] == lp_image_pixel_10).all()
            assert (plane.image_plane_image_for_simulation[2, 3] == lp_image_pixel_11).all()

        def test__plane_has_no_galaxies__image_is_zeros_size_of_unlensed_regular_grid(self, grid_stack):
            plane = pl.AbstractGriddedPlane(galaxies=[], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.image_plane_image.shape == (3, 4)
            assert (plane.image_plane_image[1, 1] == 0.0).all()
            assert (plane.image_plane_image[1, 2] == 0.0).all()

    class TestBlurringImage:

        def test__image_from_plane__same_as_its_light_profile_image(self, grid_stack, galaxy_light):
            light_profile = galaxy_light.light_profiles[0]

            lp_blurring_image = light_profile.intensities_from_grid(grid_stack.blurring)

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_light], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert (plane.image_plane_blurring_image_1d == lp_blurring_image).all()

        def test__same_as_above__use_multiple_galaxies(self, grid_stack):
            # Overwrite one value so intensity in each pixel is different
            grid_stack.blurring[1] = np.array([2.0, 2.0])

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            lp0 = g0.light_profiles[0]
            lp1 = g1.light_profiles[0]

            lp0_blurring_image = lp0.intensities_from_grid(grid_stack.blurring)
            lp1_blurring_image = lp1.intensities_from_grid(grid_stack.blurring)

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert (plane.image_plane_blurring_image_1d == lp0_blurring_image + lp1_blurring_image).all()

        def test__image_from_plane__same_as_its_galaxy_image(self, grid_stack, galaxy_light):
            galaxy_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack.blurring, galaxies=[galaxy_light])

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_light], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert (plane.image_plane_blurring_image_1d == galaxy_image).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, grid_stack):
            # Overwrite one value so intensity in each pixel is different
            grid_stack.blurring[1] = np.array([2.0, 2.0])

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            g0_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack.blurring, galaxies=[g0])
            g1_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack.blurring, galaxies=[g1])

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert (plane.image_plane_blurring_image_1d == g0_image + g1_image).all()

    class TestConvergence:

        def test__convergence_from_plane__same_as_its_mass_profile(self, grid_stack, galaxy_mass):
            mass_profile = galaxy_mass.mass_profiles[0]

            mp_sub_convergence = mass_profile.convergence_from_grid(grid_stack.sub.unlensed_grid)

            # Perform sub gridding average manually
            mp_convergence_pixel_0 = (mp_sub_convergence[0] + mp_sub_convergence[1] +
                                          mp_sub_convergence[2] + mp_sub_convergence[3]) / 4
            mp_convergence_pixel_1 = (mp_sub_convergence[4] + mp_sub_convergence[5] +
                                          mp_sub_convergence[6] + mp_sub_convergence[7]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_mass], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.convergence.shape == (3, 4)
            assert (plane.convergence[1, 1] == mp_convergence_pixel_0).all()
            assert (plane.convergence[1, 2] == mp_convergence_pixel_1).all()

        def test__same_as_above__use_multiple_galaxies(self, grid_stack):
            # Overwrite one value so intensity in each pixel is different
            grid_stack.sub[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0, centre=(1.0, 0.0)))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0, centre=(1.0, 1.0)))

            mp0 = g0.mass_profiles[0]
            mp1 = g1.mass_profiles[0]

            mp0_sub_convergence = mp0.convergence_from_grid(grid=grid_stack.sub.unlensed_grid)
            mp1_sub_convergence = mp1.convergence_from_grid(grid=grid_stack.sub.unlensed_grid)

            # Perform sub gridding average manually
            mp0_convergence_pixel_0 = (mp0_sub_convergence[0] + mp0_sub_convergence[1] +
                                           mp0_sub_convergence[2] + mp0_sub_convergence[3]) / 4
            mp0_convergence_pixel_1 = (mp0_sub_convergence[4] + mp0_sub_convergence[5] +
                                           mp0_sub_convergence[6] + mp0_sub_convergence[7]) / 4
            mp1_convergence_pixel_0 = (mp1_sub_convergence[0] + mp1_sub_convergence[1] +
                                           mp1_sub_convergence[2] + mp1_sub_convergence[3]) / 4
            mp1_convergence_pixel_1 = (mp1_sub_convergence[4] + mp1_sub_convergence[5] +
                                           mp1_sub_convergence[6] + mp1_sub_convergence[7]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.convergence[1, 1] == pytest.approx(mp0_convergence_pixel_0 +
                                                            mp1_convergence_pixel_0, 1.0e-4)
            assert plane.convergence[1, 2] == pytest.approx(mp0_convergence_pixel_1 +
                                                            mp1_convergence_pixel_1, 1.0e-4)

        def test__convergence__same_as_its_galaxy(self, grid_stack, galaxy_mass):
            galaxy_convergence = galaxy_util.convergence_of_galaxies_from_grid(grid_stack.sub.unlensed_grid,
                                                                                   galaxies=[galaxy_mass])

            galaxy_convergence = grid_stack.regular.scaled_array_2d_from_array_1d(galaxy_convergence)

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_mass], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert (plane.convergence == galaxy_convergence).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, grid_stack):
            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            g0_convergence = galaxy_util.convergence_of_galaxies_from_grid(grid_stack.sub.unlensed_grid,
                                                                               galaxies=[g0])
            g1_convergence = galaxy_util.convergence_of_galaxies_from_grid(grid_stack.sub.unlensed_grid,
                                                                               galaxies=[g1])

            g0_convergence = grid_stack.regular.scaled_array_2d_from_array_1d(g0_convergence)
            g1_convergence = grid_stack.regular.scaled_array_2d_from_array_1d(g1_convergence)

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert (plane.convergence == g0_convergence + g1_convergence).all()

        def test__convergence_from_plane__same_as_its_mass_profile__use_padded_grid_stack(self, padded_grid_stack,
                                                                                              galaxy_mass):
            mass_profile = galaxy_mass.mass_profiles[0]

            mp_sub_convergence = mass_profile.convergence_from_grid(padded_grid_stack.sub.unlensed_grid)

            # The padded sub-grid adds 5 pixels around the mask from the top-left which we skip over, thus our
            # first sub-pixel index is 20.
            mp_convergence_pixel_0 = (mp_sub_convergence[20] + mp_sub_convergence[21] +
                                          mp_sub_convergence[22] + mp_sub_convergence[23]) / 4
            mp_convergence_pixel_1 = (mp_sub_convergence[24] + mp_sub_convergence[25] +
                                          mp_sub_convergence[26] + mp_sub_convergence[27]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_mass], grid_stack=padded_grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            # The padded array is trimmed to the same size as the original mask (1x2).
            assert plane.convergence[0, 0] == pytest.approx(mp_convergence_pixel_0, 1.0e-4)
            assert plane.convergence[0, 1] == pytest.approx(mp_convergence_pixel_1, 1.0e-4)

        def test__plane_has_no_galaxies__convergence_is_zeros_size_of_unlensed_regular_grid(self, grid_stack):
            plane = pl.AbstractGriddedPlane(galaxies=[], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.convergence.shape == (3, 4)
            assert (plane.convergence[1, 1] == 0.0).all()
            assert (plane.convergence[1, 2] == 0.0).all()

    class TestPotential:

        def test__potential_from_plane__same_as_its_mass_profile(self, grid_stack, galaxy_mass):
            mass_profile = galaxy_mass.mass_profiles[0]

            mp_sub_potential = mass_profile.potential_from_grid(grid_stack.sub.unlensed_grid)

            # Perform sub gridding average manually
            mp_potential_pixel_0 = (mp_sub_potential[0] + mp_sub_potential[1] + mp_sub_potential[2] + mp_sub_potential[
                3]) / 4
            mp_potential_pixel_1 = (mp_sub_potential[4] + mp_sub_potential[5] + mp_sub_potential[6] + mp_sub_potential[
                7]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_mass], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.potential.shape == (3, 4)
            assert (plane.potential[1, 1] == mp_potential_pixel_0).all()
            assert (plane.potential[1, 2] == mp_potential_pixel_1).all()

        def test__same_as_above__use_multiple_galaxies(self, grid_stack):
            # Overwrite one value so intensity in each pixel is different
            grid_stack.sub.unlensed_grid[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            mp0 = g0.mass_profiles[0]
            mp1 = g1.mass_profiles[0]

            mp0_sub_potential = mp0.potential_from_grid(grid_stack.sub.unlensed_grid)
            mp1_sub_potential = mp1.potential_from_grid(grid_stack.sub.unlensed_grid)

            # Perform sub gridding average manually
            mp0_potential_pixel_0 = (mp0_sub_potential[0] + mp0_sub_potential[1] +
                                     mp0_sub_potential[2] + mp0_sub_potential[3]) / 4
            mp0_potential_pixel_1 = (mp0_sub_potential[4] + mp0_sub_potential[5] +
                                     mp0_sub_potential[6] + mp0_sub_potential[7]) / 4
            mp1_potential_pixel_0 = (mp1_sub_potential[0] + mp1_sub_potential[1] +
                                     mp1_sub_potential[2] + mp1_sub_potential[3]) / 4
            mp1_potential_pixel_1 = (mp1_sub_potential[4] + mp1_sub_potential[5] +
                                     mp1_sub_potential[6] + mp1_sub_potential[7]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.potential[1, 1] == pytest.approx(mp0_potential_pixel_0 +
                                                          mp1_potential_pixel_0, 1.0e-4)
            assert plane.potential[1, 2] == pytest.approx(mp0_potential_pixel_1 +
                                                          mp1_potential_pixel_1, 1.0e-4)

        def test__potential__same_as_its_galaxy(self, grid_stack, galaxy_mass):
            galaxy_potential = galaxy_util.potential_of_galaxies_from_grid(grid_stack.sub.unlensed_grid,
                                                                           galaxies=[galaxy_mass])

            galaxy_potential = grid_stack.regular.scaled_array_2d_from_array_1d(galaxy_potential)

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_mass], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert (plane.potential == galaxy_potential).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, grid_stack):
            # Overwrite one value so intensity in each pixel is different
            grid_stack.sub.unlensed_grid[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            g0_potential = galaxy_util.potential_of_galaxies_from_grid(grid_stack.sub.unlensed_grid, galaxies=[g0])
            g1_potential = galaxy_util.potential_of_galaxies_from_grid(grid_stack.sub.unlensed_grid, galaxies=[g1])

            g0_potential = grid_stack.regular.scaled_array_2d_from_array_1d(g0_potential)
            g1_potential = grid_stack.regular.scaled_array_2d_from_array_1d(g1_potential)

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.potential == pytest.approx(g0_potential + g1_potential, 1.0e-4)

        def test__potential_from_plane__same_as_its_mass_profile__use_padded_grid_stack(self, padded_grid_stack,
                                                                                        galaxy_mass):
            mass_profile = galaxy_mass.mass_profiles[0]

            mp_sub_image = mass_profile.potential_from_grid(padded_grid_stack.sub.unlensed_grid)

            # The padded sub-grid adds 5 pixels around the mask from the top-left which we skip over, thus our
            # first sub-pixel index is 20.
            mp_image_pixel_0 = (mp_sub_image[20] + mp_sub_image[21] + mp_sub_image[22] + mp_sub_image[23]) / 4
            mp_image_pixel_1 = (mp_sub_image[24] + mp_sub_image[25] + mp_sub_image[26] + mp_sub_image[27]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_mass], grid_stack=padded_grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            # The padded array is trimmed to the same size as the original mask (1x2).
            assert plane.potential[0, 0] == pytest.approx(mp_image_pixel_0, 1.0e-4)
            assert plane.potential[0, 1] == pytest.approx(mp_image_pixel_1, 1.0e-4)

        def test__plane_has_no_galaxies__potential_are_zeros_size_of_unlensed_regular_grid(self, grid_stack):
            plane = pl.AbstractGriddedPlane(galaxies=[], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.potential.shape == (3, 4)
            assert (plane.potential[1, 1] == 0.0).all()
            assert (plane.potential[1, 2] == 0.0).all()

    class TestDeflections:

        def test__deflections_from_plane__same_as_its_mass_profile(self, grid_stack, galaxy_mass):
            mp = galaxy_mass.mass_profiles[0]

            mp_sub_image = mp.deflections_from_grid(grid_stack.sub.unlensed_grid)

            # Perform sub gridding average manually
            mp_image_pixel_0x = (mp_sub_image[0, 0] + mp_sub_image[1, 0] + mp_sub_image[2, 0] + mp_sub_image[3, 0]) / 4
            mp_image_pixel_1x = (mp_sub_image[4, 0] + mp_sub_image[5, 0] + mp_sub_image[6, 0] + mp_sub_image[7, 0]) / 4
            mp_image_pixel_0y = (mp_sub_image[0, 1] + mp_sub_image[1, 1] + mp_sub_image[2, 1] + mp_sub_image[3, 1]) / 4
            mp_image_pixel_1y = (mp_sub_image[4, 1] + mp_sub_image[5, 1] + mp_sub_image[6, 1] + mp_sub_image[7, 1]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_mass], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert (plane.deflections_1d[0, 0] == mp_image_pixel_0x).all()
            assert (plane.deflections_1d[0, 1] == mp_image_pixel_0y).all()
            assert (plane.deflections_1d[1, 0] == mp_image_pixel_1x).all()
            assert (plane.deflections_1d[1, 1] == mp_image_pixel_1y).all()

            assert plane.deflections_y.shape == (3, 4)
            assert plane.deflections_x.shape == (3, 4)
            assert (plane.deflections_y ==
                    grid_stack.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 0])).all()
            assert (plane.deflections_x ==
                    grid_stack.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 1])).all()

        def test__same_as_above__use_multiple_galaxies(self, grid_stack):
            # Overwrite one value so intensity in each pixel is different
            grid_stack.sub.unlensed_grid[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            mp0 = g0.mass_profiles[0]
            mp1 = g1.mass_profiles[0]

            mp0_sub_image = mp0.deflections_from_grid(grid_stack.sub.unlensed_grid)
            mp1_sub_image = mp1.deflections_from_grid(grid_stack.sub.unlensed_grid)

            # Perform sub gridding average manually
            mp0_image_pixel_0x = (mp0_sub_image[0, 0] + mp0_sub_image[1, 0] +
                                  mp0_sub_image[2, 0] + mp0_sub_image[3, 0]) / 4
            mp0_image_pixel_1x = (mp0_sub_image[4, 0] + mp0_sub_image[5, 0] +
                                  mp0_sub_image[6, 0] + mp0_sub_image[7, 0]) / 4
            mp0_image_pixel_0y = (mp0_sub_image[0, 1] + mp0_sub_image[1, 1] +
                                  mp0_sub_image[2, 1] + mp0_sub_image[3, 1]) / 4
            mp0_image_pixel_1y = (mp0_sub_image[4, 1] + mp0_sub_image[5, 1] +
                                  mp0_sub_image[6, 1] + mp0_sub_image[7, 1]) / 4

            mp1_image_pixel_0x = (mp1_sub_image[0, 0] + mp1_sub_image[1, 0] +
                                  mp1_sub_image[2, 0] + mp1_sub_image[3, 0]) / 4
            mp1_image_pixel_1x = (mp1_sub_image[4, 0] + mp1_sub_image[5, 0] +
                                  mp1_sub_image[6, 0] + mp1_sub_image[7, 0]) / 4
            mp1_image_pixel_0y = (mp1_sub_image[0, 1] + mp1_sub_image[1, 1] +
                                  mp1_sub_image[2, 1] + mp1_sub_image[3, 1]) / 4
            mp1_image_pixel_1y = (mp1_sub_image[4, 1] + mp1_sub_image[5, 1] +
                                  mp1_sub_image[6, 1] + mp1_sub_image[7, 1]) / 4

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.deflections_1d[0, 0] == pytest.approx(mp0_image_pixel_0x + mp1_image_pixel_0x, 1.0e-4)
            assert plane.deflections_1d[1, 0] == pytest.approx(mp0_image_pixel_1x + mp1_image_pixel_1x, 1.0e-4)
            assert plane.deflections_1d[0, 1] == pytest.approx(mp0_image_pixel_0y + mp1_image_pixel_0y, 1.0e-4)
            assert plane.deflections_1d[1, 1] == pytest.approx(mp0_image_pixel_1y + mp1_image_pixel_1y, 1.0e-4)
            assert (plane.deflections_y ==
                    grid_stack.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 0])).all()
            assert (plane.deflections_x ==
                    grid_stack.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 1])).all()

        def test__deflections__same_as_its_galaxy(self, grid_stack, galaxy_mass):
            galaxy_deflections = galaxy_util.deflections_of_galaxies_from_grid(grid=grid_stack.sub.unlensed_grid,
                                                                               galaxies=[galaxy_mass])

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_mass], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert (plane.deflections_1d == galaxy_deflections).all()
            assert (plane.deflections_y ==
                    grid_stack.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 0])).all()
            assert (plane.deflections_x ==
                    grid_stack.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 1])).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, grid_stack):
            # Overwrite one value so intensity in each pixel is different
            grid_stack.sub.unlensed_grid[5] = np.array([2.0, 2.0])

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))

            g0_deflections = galaxy_util.deflections_of_galaxies_from_grid(grid=grid_stack.sub.unlensed_grid,
                                                                           galaxies=[g0])
            g1_deflections = galaxy_util.deflections_of_galaxies_from_grid(grid=grid_stack.sub.unlensed_grid,
                                                                           galaxies=[g1])

            plane = pl.AbstractGriddedPlane(galaxies=[g0, g1], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.deflections_1d == pytest.approx(g0_deflections + g1_deflections, 1.0e-4)
            assert (plane.deflections_y ==
                    grid_stack.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 0])).all()
            assert (plane.deflections_x ==
                    grid_stack.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 1])).all()

        def test__plane_has_no_galaxies__deflections_are_zeros_size_of_unlensed_regular_grid(self, grid_stack):
            plane = pl.AbstractGriddedPlane(galaxies=[], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert (plane.deflections_1d[0, 0] == 0.0).all()
            assert (plane.deflections_1d[0, 1] == 0.0).all()
            assert (plane.deflections_1d[1, 0] == 0.0).all()
            assert (plane.deflections_1d[1, 1] == 0.0).all()

            assert plane.deflections_y.shape == (3, 4)
            assert plane.deflections_x.shape == (3, 4)
            assert (plane.deflections_y ==
                    grid_stack.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 0])).all()
            assert (plane.deflections_x ==
                    grid_stack.regular.scaled_array_2d_from_array_1d(plane.deflections_1d[:, 1])).all()

    class TestMapper:

        def test__no_galaxies_with_pixelizations_in_plane__returns_none(self, grid_stack):
            galaxy_no_pix = g.Galaxy()

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_no_pix], grid_stack=grid_stack, border=[MockBorders()],
                                            compute_deflections=False, redshift=None)

            assert plane.mapper is None

        def test__1_galaxy_in_plane__it_has_pixelization__returns_mapper(self, grid_stack):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_pix], grid_stack=grid_stack, border=[MockBorders()],
                                            compute_deflections=False, redshift=None)

            assert plane.mapper == 1

        def test__2_galaxies_in_plane__1_has_pixelization__extracts_reconstructor(self, grid_stack):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_no_pix = g.Galaxy()

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_no_pix, galaxy_pix], grid_stack=grid_stack,
                                            border=[MockBorders()], compute_deflections=False, redshift=None)

            assert plane.mapper == 1

        def test__plane_has_no_border__still_returns_mapper(self, grid_stack):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_no_pix = g.Galaxy()

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_no_pix, galaxy_pix], grid_stack=grid_stack, border=None,
                                            compute_deflections=False, redshift=None)

            assert plane.mapper == 1

        def test__2_galaxies_in_plane__both_have_pixelization__raises_error(self, grid_stack):
            galaxy_pix_0 = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_pix_1 = g.Galaxy(pixelization=MockPixelization(value=2), regularization=MockRegularization(value=0))

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy_pix_0, galaxy_pix_1], grid_stack=grid_stack,
                                            border=[MockBorders()], compute_deflections=False, redshift=None)

            with pytest.raises(exc.PixelizationException):
                print(plane.mapper)

    class TestProperties:

        def test__padded_grid_in__tracer_has_padded_gridty(self, grid_stack, padded_grid_stack, galaxy_light):
            plane = pl.AbstractGriddedPlane(grid_stack=grid_stack, galaxies=[galaxy_light], compute_deflections=False,
                                            redshift=None, border=None)
            assert plane.has_padded_grid_stack is False

            plane = pl.AbstractGriddedPlane(grid_stack=padded_grid_stack, galaxies=[galaxy_light],
                                            compute_deflections=False, redshift=None, border=None)
            assert plane.has_padded_grid_stack is True

    class TestPlaneImage:

        def test__3x3_grid__extracts_max_min_coordinates__ignores_other_coordinates_more_central(self, grid_stack):
            grid_stack.regular[1] = np.array([2.0, 2.0])

            galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))

            plane = pl.AbstractGriddedPlane(galaxies=[galaxy], grid_stack=grid_stack, compute_deflections=False,
                                            border=None, redshift=None)

            plane_image_from_func = lens_util.plane_image_of_galaxies_from_grid(shape=(3, 4),
                                                                                grid=grid_stack.regular,
                                                                                galaxies=[galaxy])

            assert (plane_image_from_func == plane.plane_image).all()

        def test__ensure_index_of_plane_image_has_negative_arcseconds_at_start(self, grid_stack):
            # The grid coordinates -2.0 -> 2.0 mean a plane of shape (5,5) has arc second coordinates running over
            # -1.6, -0.8, 0.0, 0.8, 1.6. The origin -1.6, -1.6 of the model_galaxy means its brighest pixel should be
            # index 0 of the 1D grid and (0,0) of the 2d plane datas_.

            mask = msk.Mask(array=np.full((5, 5), False), pixel_scale=1.0)

            grid_stack.regular = grids.RegularGrid(np.array([[-2.0, -2.0], [2.0, 2.0]]), mask=mask)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(centre=(1.6, -1.6), intensity=1.0))
            plane = pl.AbstractGriddedPlane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=False,
                                            border=None, redshift=None)

            assert plane.plane_image.shape == (5, 5)
            assert np.unravel_index(plane.plane_image.argmax(), plane.plane_image.shape) == (0, 0)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(centre=(1.6, 1.6), intensity=1.0))
            plane = pl.AbstractGriddedPlane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=False,
                                            border=None, redshift=None)
            assert np.unravel_index(plane.plane_image.argmax(), plane.plane_image.shape) == (0, 4)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(centre=(-1.6, -1.6), intensity=1.0))
            plane = pl.AbstractGriddedPlane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=False,
                                            border=None, redshift=None)
            assert np.unravel_index(plane.plane_image.argmax(), plane.plane_image.shape) == (4, 0)

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(centre=(-1.6, 1.6), intensity=1.0))
            plane = pl.AbstractGriddedPlane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=False,
                                            border=None, redshift=None)
            assert np.unravel_index(plane.plane_image.argmax(), plane.plane_image.shape) == (4, 4)


class TestPlane(object):

    class TestGalaxies:

        def test__no_galaxies__raises_exception(self):
            with pytest.raises(exc.RayTracingException):
                pl.Plane(galaxies=[], grid_stack=None, compute_deflections=False)

        def test__galaxy_redshifts_gives_list_of_redshifts(self):
            g0 = g.Galaxy(redshift=1.0)
            g1 = g.Galaxy(redshift=1.0)
            g2 = g.Galaxy(redshift=1.0)

            plane = pl.Plane(galaxies=[g0, g1, g2], grid_stack=None, compute_deflections=False)

            assert plane.redshift == 1.0
            assert plane.galaxy_redshifts == [1.0, 1.0, 1.0]

        def test__galaxy_has_no_redshift__returns_none(self):
            g0 = g.Galaxy()

            plane = pl.Plane(galaxies=[g0], grid_stack=None, compute_deflections=False)

            assert plane.redshift is None

        def test__galaxy_has_no_redshift__cosmology_input__raises_exception(self):
            g0 = g.Galaxy()
            g1 = g.Galaxy(redshift=1.0)

            with pytest.raises(exc.RayTracingException):
                pl.Plane(galaxies=[g0, g1], grid_stack=None, compute_deflections=False)

        def test__galaxies_entered_all_have_no_redshifts__no_exception_raised(self):
            g0 = g.Galaxy()
            g1 = g.Galaxy()

            pl.Plane(galaxies=[g0, g1], grid_stack=None, compute_deflections=False)

        def test__galaxies_entered_all_have_same_redshifts__no_exception_raised(self):
            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy(redshift=0.1)

            pl.Plane(galaxies=[g0, g1], grid_stack=None, compute_deflections=False)

        def test__1_galaxy_has_redshift_other_does_not__exception_is_raised(self):
            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy()

            with pytest.raises(exc.RayTracingException):
                pl.Plane(galaxies=[g0, g1], grid_stack=None, compute_deflections=False)

        def test__galaxies_have_different_redshifts__exception_is_raised(self):
            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy(redshift=1.0)

            with pytest.raises(exc.RayTracingException):
                pl.Plane(galaxies=[g0, g1], grid_stack=None, compute_deflections=False)


class TestPlaneSlice(object):
    class TestGalaxies:

        def test__galaxies_have_different_redshifts__plane_redshift_is_input_and_does_not_raise_error(self, grid_stack):
            g0 = g.Galaxy(redshift=0.4)
            g1 = g.Galaxy(redshift=0.5)
            g2 = g.Galaxy(redshift=0.6)

            plane = pl.PlaneSlice(galaxies=[g0, g1, g2], grid_stack=grid_stack, redshift=0.5, compute_deflections=True)

            assert plane.redshift == 0.5

        def test__galaxies_is_empty_list__does_not_raise_error(self, grid_stack):
            plane = pl.PlaneSlice(galaxies=[], grid_stack=grid_stack, redshift=0.5, compute_deflections=True)

            assert plane.redshift == 0.5


class TestPlaneImage:

    def test__compute_xticks_from_regular_grid_correctly(self):
        plane_image = pl.PlaneImage(array=np.ones((3, 3)), pixel_scales=(5.0, 1.0), grid=None)
        assert plane_image.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        plane_image = pl.PlaneImage(array=np.ones((3, 3)), pixel_scales=(5.0, 0.5), grid=None)
        assert plane_image.xticks == pytest.approx(np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3)

        plane_image = pl.PlaneImage(array=np.ones((1, 6)), pixel_scales=(5.0, 1.0), grid=None)
        assert plane_image.xticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-2)

    def test__compute_yticks_from_regular_grid_correctly(self):
        plane_image = pl.PlaneImage(array=np.ones((3, 3)), pixel_scales=(1.0, 5.0), grid=None)
        assert plane_image.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        plane_image = pl.PlaneImage(array=np.ones((3, 3)), pixel_scales=(0.5, 5.0), grid=None)
        assert plane_image.yticks == pytest.approx(np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3)

        plane_image = pl.PlaneImage(array=np.ones((6, 1)), pixel_scales=(1.0, 5.0), grid=None)
        assert plane_image.yticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-2)
