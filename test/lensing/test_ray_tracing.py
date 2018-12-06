import numpy as np
import pytest
from astropy import cosmology as cosmo

from autolens import exc
from autolens.data.array import grids, mask
from autolens.model.inversion import pixelizations, regularization
from autolens.model.galaxy import galaxy as g
from autolens.lensing import plane as pl
from autolens.lensing import ray_tracing
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.mock.mock_inversion import MockRegularization, MockPixelization
from test.mock.mock_imaging import MockBorders


@pytest.fixture(name="data_grids")
def make_data_grids():
    ma = mask.Mask(np.array([[True, True, True, True],
                             [True, False, False, True],
                             [True, True, True, True]]), pixel_scale=6.0)

    data_grids = grids.DataGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=2,
                                                                                psf_shape=(3, 3))

    # Manually overwrite a set of cooridnates to make tests of grids and defledctions straightforward

    data_grids.regular[0] = np.array([1.0, 1.0])
    data_grids.regular[1] = np.array([1.0, 0.0])
    data_grids.sub[0] = np.array([1.0, 1.0])
    data_grids.sub[1] = np.array([1.0, 0.0])
    data_grids.sub[2] = np.array([1.0, 1.0])
    data_grids.sub[3] = np.array([1.0, 0.0])
    data_grids.blurring[0] = np.array([1.0, 0.0])

    return data_grids


@pytest.fixture(name="data_grids_1")
def make_data_grids_1():
    ma = mask.Mask(np.array([[True, True, True, True],
                             [True, False, False, True],
                             [True, True, True, True]]), pixel_scale=12.0)

    data_grids = grids.DataGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=2,
                                                                                psf_shape=(3, 3))

    # Manually overwrite a set of cooridnates to make tests of grids and defledctions straightforward

    data_grids.regular[0] = np.array([2.0, 2.0])
    data_grids.regular[1] = np.array([2.0, 0.0])
    data_grids.sub[0] = np.array([2.0, 2.0])
    data_grids.sub[1] = np.array([2.0, 0.0])
    data_grids.sub[2] = np.array([2.0, 2.0])
    data_grids.sub[3] = np.array([2.0, 0.0])
    data_grids.blurring[0] = np.array([2.0, 0.0])

    return data_grids


@pytest.fixture(name="padded_grids")
def make_padded_grids():
    ma = mask.Mask(np.array([[True, False]]), pixel_scale=3.0)
    return grids.DataGrids.padded_grids_from_mask_sub_grid_size_and_psf_shape(ma, 2, (3, 3))


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


class TestAbstractTracer(object):

    def test__total_planes(self, data_grids):

        tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g.Galaxy()], image_plane_grids=[data_grids])

        assert tracer.total_planes == 1

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[g.Galaxy()],
                                                     image_plane_grids=[data_grids])

        assert tracer.total_planes == 2
        
        tracer = ray_tracing.TracerMulti(galaxies=[g.Galaxy(redshift=1.0), g.Galaxy(redshift=2.0),
                                                   g.Galaxy(redshift=3.0)],
                                         image_plane_grids=[data_grids])

        assert tracer.total_planes == 3

        tracer = ray_tracing.TracerMulti(galaxies=[g.Galaxy(redshift=1.0), g.Galaxy(redshift=2.0),
                                                   g.Galaxy(redshift=1.0)],
                                         image_plane_grids=[data_grids])

        assert tracer.total_planes == 2

    def test_hyper_galaxies_list(self, data_grids):

        tracer = ray_tracing.TracerImageSourcePlanes([g.Galaxy(hyper_galaxy=g.HyperGalaxy())],
                                                     [g.Galaxy(hyper_galaxy=g.HyperGalaxy())],
                                                     image_plane_grids=[data_grids])

        assert tracer.image_plane.hyper_galaxies == [g.HyperGalaxy()]
        assert tracer.source_plane.hyper_galaxies == [g.HyperGalaxy()]

        assert tracer.hyper_galaxies == [g.HyperGalaxy(), g.HyperGalaxy()]

    def test_tracer__hyper_galaxies_with_none_are_filtered(self, data_grids):

        tracer = ray_tracing.TracerImageSourcePlanes([g.Galaxy(hyper_galaxy=g.HyperGalaxy()), g.Galaxy()],
                                                     [g.Galaxy(hyper_galaxy=g.HyperGalaxy()), g.Galaxy(), g.Galaxy()],
                                                     image_plane_grids=[data_grids])

        assert tracer.image_plane.hyper_galaxies == [g.HyperGalaxy(), None]
        assert tracer.source_plane.hyper_galaxies == [g.HyperGalaxy(), None, None]

        assert tracer.hyper_galaxies == [g.HyperGalaxy(), g.HyperGalaxy()]

    def test_multi_tracer(self, data_grids):
        tracer = ray_tracing.TracerMulti(galaxies=[g.Galaxy(hyper_galaxy=g.HyperGalaxy(2), redshift=2),
                                                   g.Galaxy(hyper_galaxy=g.HyperGalaxy(1), redshift=1)],
                                         image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

        assert tracer.hyper_galaxies == [g.HyperGalaxy(1), g.HyperGalaxy(2)]


class TestTracerImagePlane(object):


    class TestImagePlaneImage:

        def test__1_plane__single_plane_tracer(self, data_grids):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0))

            image_plane = pl.Plane(galaxies=[g0, g1, g2], grids=[data_grids], compute_deflections=True)

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1, g2], image_plane_grids=[data_grids])

            assert (tracer.image_plane_images_[0] == image_plane.image_plane_images_[0]).all()

            image_plane_image_2d = data_grids.regular.scaled_array_from_array_1d(image_plane.image_plane_images_[0])
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images[0]).all()

        def test__same_as_above_but_two_grids__returns_two_images(self, data_grids, data_grids_1):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0))

            image_plane = pl.Plane(galaxies=[g0, g1, g2], grids=[data_grids, data_grids_1],
                                   compute_deflections=True)

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1, g2], image_plane_grids=[data_grids,
                                                                                                 data_grids_1])

            assert (tracer.image_plane_images_[0] == image_plane.image_plane_images_[0]).all()

            image_plane_image_2d = data_grids.regular.scaled_array_from_array_1d(image_plane.image_plane_images_[0])
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images[0]).all()

            assert (tracer.image_plane_images_[1] == image_plane.image_plane_images_[1]).all()

            image_plane_image_2d = data_grids.regular.scaled_array_from_array_1d(image_plane.image_plane_images_[1])
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images[1]).all()

    class TestImagePlaneBlurringImages:

        def test__1_plane__single_plane_tracer(self, data_grids):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0))

            image_plane = pl.Plane(galaxies=[g0, g1, g2], grids=[data_grids], compute_deflections=True)

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1, g2], image_plane_grids=[data_grids])

            assert (tracer.image_plane_blurring_images_[0] == image_plane.image_plane_blurring_images_[0]).all()


class TestTracerImageSourcePlanes(object):
    
    class TestSetup:

        def test__image_grid__no_galaxy__image_and_source_planes_setup__same_coordinates(self, data_grids,
                                                                                         galaxy_non):

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_non], source_galaxies=[galaxy_non],
                                                         image_plane_grids=[data_grids])

            assert tracer.image_plane.grids[0].regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.image_plane.deflections[0].regular[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[1] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[2] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[3] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].blurring[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)

            assert tracer.source_plane.grids[0].regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.source_plane.grids[0].sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.source_plane.grids[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.source_plane.grids[0].sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.source_plane.grids[0].sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test__all_data_grids__sis_lens__image_sub_and_blurring_data_grids_on_planes_setup(self, data_grids,
                                                                                                    galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass], source_galaxies=[galaxy_mass],
                                                         image_plane_grids=[data_grids])

            assert tracer.image_plane.grids[0].regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grids[0].blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.image_plane.deflections[0].regular[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[2] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.source_plane.grids[0].regular[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)
            assert tracer.source_plane.grids[0].sub[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)
            assert tracer.source_plane.grids[0].sub[1] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.source_plane.grids[0].sub[2] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)
            assert tracer.source_plane.grids[0].sub[3] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.source_plane.grids[0].blurring[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)

        def test__same_as_above_but_2_sis_lenses__deflections_double(self, data_grids, galaxy_mass):

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass, galaxy_mass],
                                                         source_galaxies=[galaxy_mass], image_plane_grids=[data_grids])

            assert tracer.image_plane.grids[0].regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grids[0].blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.image_plane.deflections[0].regular[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[1] == pytest.approx(np.array([2.0 * 1.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[2] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[3] == pytest.approx(np.array([2.0 * 1.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].blurring[0] == pytest.approx(np.array([2.0 * 1.0, 0.0]), 1e-3)

            assert tracer.source_plane.grids[0].regular[0] == pytest.approx(np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]),
                                                                       1e-3)
            assert tracer.source_plane.grids[0].sub[0] == pytest.approx(np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]),
                                                                     1e-3)
            assert tracer.source_plane.grids[0].sub[1] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            assert tracer.source_plane.grids[0].sub[2] == pytest.approx(np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]),
                                                                     1e-3)
            assert tracer.source_plane.grids[0].sub[3] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            assert tracer.source_plane.grids[0].blurring[0] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)

        def test__x2_image_grid__no_galaxy__image_and_source_planes_setup__same_coordinates(self, data_grids,
                                                                                            data_grids_1,
                                                                                         galaxy_non):

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_non], source_galaxies=[galaxy_non],
                                                         image_plane_grids=[data_grids, data_grids_1])

            assert tracer.image_plane.grids[0].regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids[0].sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.image_plane.deflections[0].regular[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[1] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[2] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].sub[3] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[0].blurring[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)

            assert tracer.source_plane.grids[0].regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.source_plane.grids[0].sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.source_plane.grids[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.source_plane.grids[0].sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.source_plane.grids[0].sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.image_plane.grids[1].regular[0] == pytest.approx(np.array([2.0, 2.0]), 1e-3)
            assert tracer.image_plane.grids[1].sub[0] == pytest.approx(np.array([2.0, 2.0]), 1e-3)
            assert tracer.image_plane.grids[1].sub[1] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
            assert tracer.image_plane.grids[1].sub[2] == pytest.approx(np.array([2.0, 2.0]), 1e-3)
            assert tracer.image_plane.grids[1].sub[3] == pytest.approx(np.array([2.0, 0.0]), 1e-3)

            assert tracer.image_plane.deflections[1].regular[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[1].sub[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[1].sub[1] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[1].sub[2] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[1].sub[3] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections[1].blurring[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)

            assert tracer.source_plane.grids[1].regular[0] == pytest.approx(np.array([2.0, 2.0]), 1e-3)
            assert tracer.source_plane.grids[1].sub[0] == pytest.approx(np.array([2.0, 2.0]), 1e-3)
            assert tracer.source_plane.grids[1].sub[1] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
            assert tracer.source_plane.grids[1].sub[2] == pytest.approx(np.array([2.0, 2.0]), 1e-3)
            assert tracer.source_plane.grids[1].sub[3] == pytest.approx(np.array([2.0, 0.0]), 1e-3)

        def test__grid_attributes_passed(self, data_grids, galaxy_non):

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_non], source_galaxies=[galaxy_non],
                                                         image_plane_grids=[data_grids])

            assert (tracer.image_plane.grids[0].regular.mask == data_grids.regular.mask).all()
            assert (tracer.image_plane.grids[0].sub.mask == data_grids.sub.mask).all()
            assert (tracer.source_plane.grids[0].regular.mask == data_grids.regular.mask).all()
            assert (tracer.source_plane.grids[0].sub.mask == data_grids.sub.mask).all()

    class TestCosmology:

        def test__2_planes__z01_and_z1(self, data_grids):
            
            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy(redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            assert tracer.image_plane.arcsec_per_kpc_proper == pytest.approx(0.525060, 1e-5)
            assert tracer.image_plane.kpc_per_arcsec_proper == pytest.approx(1.904544, 1e-5)
            assert tracer.image_plane.angular_diameter_distance_to_earth == pytest.approx(392840, 1e-5)

            assert tracer.source_plane.arcsec_per_kpc_proper == pytest.approx(0.1214785, 1e-5)
            assert tracer.source_plane.kpc_per_arcsec_proper == pytest.approx(8.231907, 1e-5)
            assert tracer.source_plane.angular_diameter_distance_to_earth == pytest.approx(1697952, 1e-5)

            assert tracer.angular_diameter_distance_from_image_to_source_plane == pytest.approx(1481890.4, 1e-5)

            assert tracer.critical_density_kpc == pytest.approx(4.85e9, 1e-2)
            assert tracer.critical_density_arcsec == pytest.approx(17593241668, 1e-2)

        def test__no_cosmology__returns_none(self, data_grids):

            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy(redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=[data_grids])

            assert tracer.image_plane.arcsec_per_kpc_proper == None
            assert tracer.image_plane.kpc_per_arcsec_proper == None
            assert tracer.image_plane.angular_diameter_distance_to_earth == None

            assert tracer.source_plane.arcsec_per_kpc_proper == None
            assert tracer.source_plane.kpc_per_arcsec_proper == None
            assert tracer.source_plane.angular_diameter_distance_to_earth == None

            assert tracer.angular_diameter_distance_from_image_to_source_plane == None

            assert tracer.critical_density_kpc == None
            assert tracer.critical_density_arcsec == None

    class TestImagePlaneImages:

        def test__galaxy_light__no_mass__image_sum_of_image_and_source_plane(self, data_grids):
            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            image_plane = pl.Plane(galaxies=[g0], grids=[data_grids], compute_deflections=True)
            source_plane = pl.Plane(galaxies=[g1], grids=[data_grids], compute_deflections=False)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=[data_grids])

            image_plane_image_1d = image_plane.image_plane_images_[0] + source_plane.image_plane_images_[0]

            assert (image_plane_image_1d == tracer.image_plane_images_[0]).all()

            image_plane_image_2d = data_grids.regular.scaled_array_from_array_1d(image_plane.image_plane_images_[0]) + \
                                   data_grids.regular.scaled_array_from_array_1d(source_plane.image_plane_images_[0])
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images[0]).all()

        def test__galaxy_light_mass_sis__source_plane_image_includes_deflections(self, data_grids):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0),
                          mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            image_plane = pl.Plane(galaxies=[g0], grids=[data_grids], compute_deflections=True)

            deflections_grid = pl.deflections_from_grid_collection(data_grids, galaxies=[g0])
            source_grids = pl.traced_collection_for_deflections(data_grids, deflections_grid)
            source_plane = pl.Plane(galaxies=[g1], grids=[source_grids], compute_deflections=False)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=[data_grids])

            image_plane_image_1d = image_plane.image_plane_images_[0] + source_plane.image_plane_images_[0]
            assert (image_plane_image_1d == tracer.image_plane_images_[0]).all()

            image_plane_image_2d = data_grids.regular.scaled_array_from_array_1d(image_plane.image_plane_images_[0]) + \
                                   data_grids.regular.scaled_array_from_array_1d(source_plane.image_plane_images_[0])
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images[0]).all()

        def test__image_plane_image_same_as_above(self, data_grids):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0))

            g0_image = pl.intensities_from_grid(data_grids.sub, galaxies=[g0])
            g1_image = pl.intensities_from_grid(data_grids.sub, galaxies=[g1])
            g2_image = pl.intensities_from_grid(data_grids.sub, galaxies=[g2])

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2],
                                                         image_plane_grids=[data_grids])

            assert (tracer.image_plane_images_[0] == g0_image + g1_image + g2_image).all()

            assert (tracer.image_plane_images[0] == data_grids.regular.scaled_array_from_array_1d(g0_image) +
                    data_grids.regular.scaled_array_from_array_1d(g1_image) +
                    data_grids.regular.scaled_array_from_array_1d(g2_image)).all()

        def test__2_planes__returns_image_plane_image_of_each_plane(self, data_grids):
            
            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0),
                          mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            image_plane = pl.Plane(galaxies=[g0], grids=[data_grids], compute_deflections=True)
            source_plane_data_grids = image_plane.trace_grids_to_next_plane()
            source_plane = pl.Plane(galaxies=[g0], grids=source_plane_data_grids, compute_deflections=False)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_plane_grids=[data_grids])

            assert (tracer.image_plane_images_[0] == image_plane.image_plane_images_[0] +
                    source_plane.image_plane_images_[0]).all()

            image_plane_image_2d = data_grids.regular.scaled_array_from_array_1d(image_plane.image_plane_images_[0]) + \
                                   data_grids.regular.scaled_array_from_array_1d(source_plane.image_plane_images_[0])
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images[0]).all()

        def test__same_as_above__x2_grids__returns_x2_images(self, data_grids, data_grids_1):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0),
                          mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            image_plane = pl.Plane(galaxies=[g0], grids=[data_grids, data_grids_1], compute_deflections=True)
            source_plane_grids = image_plane.trace_grids_to_next_plane()
            source_plane = pl.Plane(galaxies=[g0], grids=source_plane_grids, compute_deflections=False)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_plane_grids=[data_grids, data_grids_1])

            assert (tracer.image_plane_images_[0] == image_plane.image_plane_images_[0] +
                    source_plane.image_plane_images_[0]).all()

            image_plane_image_2d = data_grids.regular.scaled_array_from_array_1d(image_plane.image_plane_images_[0]) + \
                                   data_grids.regular.scaled_array_from_array_1d(source_plane.image_plane_images_[0])
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images[0]).all()

            assert (tracer.image_plane_images_[1] == image_plane.image_plane_images_[1] +
                    source_plane.image_plane_images_[1]).all()

            image_plane_image_2d = data_grids.regular.scaled_array_from_array_1d(image_plane.image_plane_images_[1]) + \
                                   data_grids.regular.scaled_array_from_array_1d(source_plane.image_plane_images_[1])
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images[1]).all()

            assert (tracer.image_plane_image == tracer.image_plane_images[0]).all()

        def test__padded_2d_image_from_plane__mapped_correctly(self, padded_grids, galaxy_light, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_light, galaxy_mass],
                                                         source_galaxies=[galaxy_light],
                                                         image_plane_grids=[padded_grids])

            image_plane_image_2d = padded_grids.regular.scaled_array_from_array_1d(tracer.image_plane.image_plane_images_[0]) + \
                                   padded_grids.regular.scaled_array_from_array_1d(tracer.source_plane.image_plane_images_[0])

            assert image_plane_image_2d.shape == (1, 2)
            assert (image_plane_image_2d == tracer.image_plane_images[0]).all()

        def test__padded_2d_image_for_simulation__mapped_correctly_not_trimmed(self, padded_grids, galaxy_light,
                                                                                 galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_light, galaxy_mass],
                                                         source_galaxies=[galaxy_light],
                                                         image_plane_grids=[padded_grids])

            image_plane_image_2d = padded_grids.regular.map_to_2d_keep_padded(tracer.image_plane.image_plane_images_[0]) + \
                                   padded_grids.regular.map_to_2d_keep_padded(tracer.source_plane.image_plane_images_[0])

            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images_for_simulation[0]).all()

            assert (tracer.image_plane_image_for_simulation == tracer.image_plane_images_for_simulation[0]).all()

    class TestImagePlaneBlurringImages:

        def test__galaxy_light__no_mass__image_sum_of_image_and_source_plane(self, data_grids):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            image_plane = pl.Plane(galaxies=[g0], grids=[data_grids], compute_deflections=True)
            source_plane = pl.Plane(galaxies=[g1], grids=[data_grids], compute_deflections=False)

            plane_image_plane_blurring_image = image_plane.image_plane_blurring_images_[0] + \
                                               source_plane.image_plane_blurring_images_[0]

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=[data_grids])

            assert (plane_image_plane_blurring_image == tracer.image_plane_blurring_images_[0]).all()

        def test__galaxy_light_mass_sis__source_plane_image_includes_deflections(self, data_grids):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0),
                          mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            image_plane = pl.Plane(galaxies=[g0], grids=[data_grids], compute_deflections=True)

            deflections_grid = pl.deflections_from_grid_collection(data_grids, galaxies=[g0])
            source_grid = pl.traced_collection_for_deflections(data_grids, deflections_grid)
            source_plane = pl.Plane(galaxies=[g1], grids=[source_grid], compute_deflections=False)

            plane_image_plane_blurring_image = image_plane.image_plane_blurring_images_[0] + \
                                               source_plane.image_plane_blurring_images_[0]

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=[data_grids])

            assert (plane_image_plane_blurring_image == tracer.image_plane_blurring_images_[0]).all()

        def test__same_as_above__x2_grids_in__x2_images_out(self, data_grids, data_grids_1):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0),
                          mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            image_plane = pl.Plane(galaxies=[g0], grids=[data_grids, data_grids_1], compute_deflections=True)

            deflections_grid_0 = pl.deflections_from_grid_collection(data_grids, galaxies=[g0])
            source_grid_0 = pl.traced_collection_for_deflections(data_grids, deflections_grid_0)

            deflections_grid_1 = pl.deflections_from_grid_collection(data_grids_1, galaxies=[g0])
            source_grid_1 = pl.traced_collection_for_deflections(data_grids_1, deflections_grid_1)

            source_plane = pl.Plane(galaxies=[g1], grids=[source_grid_0, source_grid_1], compute_deflections=False)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=[data_grids, data_grids_1])

            plane_image_plane_blurring_image = image_plane.image_plane_blurring_images_[0] + \
                                               source_plane.image_plane_blurring_images_[0]

            assert (plane_image_plane_blurring_image == tracer.image_plane_blurring_images_[0]).all()

            plane_image_plane_blurring_image = image_plane.image_plane_blurring_images_[1] + \
                                               source_plane.image_plane_blurring_images_[1]

            assert (plane_image_plane_blurring_image == tracer.image_plane_blurring_images_[1]).all()

    class TestSurfaceDensity:

        def test__galaxy_mass_sis__no_source_plane_surface_density(self, data_grids):

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy()

            image_plane = pl.Plane(galaxies=[g0], grids=[data_grids], compute_deflections=True)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=[data_grids])

            assert image_plane.surface_density.shape == (3, 4)
            assert (image_plane.surface_density == tracer.surface_density).all()

        def test__galaxy_entered_3_times__both_planes__different_surface_density_for_each(self, data_grids):

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))
            g2 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=3.0))

            g0_surface_density = pl.surface_density_from_grid(data_grids.sub.unlensed_grid, galaxies=[g0])
            g1_surface_density = pl.surface_density_from_grid(data_grids.sub.unlensed_grid, galaxies=[g1])
            g2_surface_density = pl.surface_density_from_grid(data_grids.sub.unlensed_grid, galaxies=[g2])

            g0_surface_density = data_grids.regular.scaled_array_from_array_1d(g0_surface_density)
            g1_surface_density = data_grids.regular.scaled_array_from_array_1d(g1_surface_density)
            g2_surface_density = data_grids.regular.scaled_array_from_array_1d(g2_surface_density)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2],
                                                         image_plane_grids=[data_grids])

            assert (tracer.image_plane.surface_density == g0_surface_density + g1_surface_density).all()
            assert (tracer.source_plane.surface_density == g2_surface_density).all()
            assert (tracer.surface_density == g0_surface_density + g1_surface_density + g2_surface_density).all()

        def test__padded_2d_surface_density_from_plane__mapped_correctly(self, padded_grids, galaxy_mass):

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass], source_galaxies=[g.Galaxy()],
                                                         image_plane_grids=[padded_grids])

            assert tracer.image_plane.surface_density.shape == (1, 2)
            assert tracer.source_plane.surface_density.shape == (1, 2)
            assert (tracer.image_plane.surface_density == tracer.surface_density).all()

    class TestPotential:

        def test__galaxy_mass_sis__source_plane_no_mass_potential_is_ignored(self, data_grids):

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy()

            image_plane = pl.Plane(galaxies=[g0], grids=[data_grids], compute_deflections=True)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=[data_grids])

            assert tracer.potential.shape == (3, 4)
            assert (image_plane.potential == tracer.potential).all()

        def test__galaxy_entered_3_times__different_potential_for_each(self, data_grids):

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))
            g2 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=3.0))

            g0_potential = pl.potential_from_grid(data_grids.sub.unlensed_grid, galaxies=[g0])
            g1_potential = pl.potential_from_grid(data_grids.sub.unlensed_grid, galaxies=[g1])
            g2_potential = pl.potential_from_grid(data_grids.sub.unlensed_grid, galaxies=[g2])

            g0_potential = data_grids.regular.scaled_array_from_array_1d(g0_potential)
            g1_potential = data_grids.regular.scaled_array_from_array_1d(g1_potential)
            g2_potential = data_grids.regular.scaled_array_from_array_1d(g2_potential)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2],
                                                         image_plane_grids=[data_grids])

            assert (tracer.image_plane.potential == g0_potential + g1_potential).all()
            assert (tracer.source_plane.potential == g2_potential).all()
            assert (tracer.potential == g0_potential + g1_potential + g2_potential).all()

        def test__padded_2d_potential_from_plane__mapped_correctly(self, padded_grids, galaxy_mass):

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass], source_galaxies=[g.Galaxy()],
                                                         image_plane_grids=[padded_grids])

            assert tracer.image_plane.potential.shape == (1, 2)
            assert tracer.source_plane.potential.shape == (1, 2)
            assert (tracer.image_plane.potential == tracer.potential).all()
            
    class TestDeflections:

        def test__galaxy_mass_sis__source_plane_no_mass__deflections_is_ignored(self, data_grids):

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy()

            image_plane = pl.Plane(galaxies=[g0], grids=[data_grids], compute_deflections=True)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=[data_grids])

            assert tracer.deflections_y.shape == (3, 4)
            assert (image_plane.deflections_y == tracer.deflections_y).all()
            assert tracer.deflections_x.shape == (3, 4)
            assert (image_plane.deflections_x == tracer.deflections_x).all()

        def test__galaxy_entered_3_times__different_deflections_for_each(self, data_grids):

            g0 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))
            g2 = g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=3.0))

            g0_deflections = pl.deflections_from_grid(data_grids.sub.unlensed_grid, galaxies=[g0])
            g1_deflections = pl.deflections_from_grid(data_grids.sub.unlensed_grid, galaxies=[g1])
            g2_deflections = pl.deflections_from_grid(data_grids.sub.unlensed_grid, galaxies=[g2])

            g0_deflections_y = data_grids.regular.scaled_array_from_array_1d(g0_deflections[:,0])
            g1_deflections_y = data_grids.regular.scaled_array_from_array_1d(g1_deflections[:,0])
            g2_deflections_y = data_grids.regular.scaled_array_from_array_1d(g2_deflections[:,0])

            g0_deflections_x = data_grids.regular.scaled_array_from_array_1d(g0_deflections[:,1])
            g1_deflections_x = data_grids.regular.scaled_array_from_array_1d(g1_deflections[:,1])
            g2_deflections_x = data_grids.regular.scaled_array_from_array_1d(g2_deflections[:,1])

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2],
                                                         image_plane_grids=[data_grids])

            assert (tracer.image_plane.deflections_y == g0_deflections_y + g1_deflections_y).all()
            assert (tracer.source_plane.deflections_y == g2_deflections_y).all()
            assert (tracer.deflections_y == g0_deflections_y + g1_deflections_y + g2_deflections_y).all()

            assert (tracer.image_plane.deflections_x == g0_deflections_x + g1_deflections_x).all()
            assert (tracer.source_plane.deflections_x == g2_deflections_x).all()
            assert (tracer.deflections_x == g0_deflections_x + g1_deflections_x + g2_deflections_x).all()

        def test__padded_2d_deflections_from_plane__mapped_correctly(self, padded_grids, galaxy_mass):

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass], source_galaxies=[g.Galaxy()],
                                                         image_plane_grids=[padded_grids])

            assert tracer.image_plane.deflections_y.shape == (1, 2)
            assert tracer.source_plane.deflections_y.shape == (1, 2)
            assert (tracer.image_plane.deflections_y == tracer.deflections_y).all()

            assert tracer.image_plane.deflections_x.shape == (1, 2)
            assert tracer.source_plane.deflections_x.shape == (1, 2)
            assert (tracer.image_plane.deflections_x == tracer.deflections_x).all()

    class TestMappers:

        def test__no_galaxy_has_pixelization__returns_empty_list(self, data_grids):
            galaxy_no_pix = g.Galaxy()

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_no_pix], source_galaxies=[galaxy_no_pix],
                                                         image_plane_grids=[data_grids], border=[MockBorders()])

            assert tracer.mappers_of_planes == []

        def test__source_galaxy_has_pixelization__returns_mapper(self, data_grids):
            galaxy_pix = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_no_pix = g.Galaxy()

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_no_pix], source_galaxies=[galaxy_pix],
                                                         image_plane_grids=[data_grids], border=[MockBorders()])

            assert tracer.mappers_of_planes[0] == 1

        def test__both_galaxies_have_pixelization__returns_both_mappers(self, data_grids):
            galaxy_pix_0 = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=3))
            galaxy_pix_1 = g.Galaxy(pixelization=MockPixelization(value=2), regularization=MockRegularization(value=4))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_pix_0], source_galaxies=[galaxy_pix_1],
                                                         image_plane_grids=[data_grids], border=[MockBorders()])

            assert tracer.mappers_of_planes[0] == 1
            assert tracer.mappers_of_planes[1] == 2

    class TestRegularizations:

        def test__no_galaxy_has_regularization__returns_empty_list(self, data_grids):
            galaxy_no_reg = g.Galaxy()

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_no_reg], source_galaxies=[galaxy_no_reg],
                                                         image_plane_grids=[data_grids], border=MockBorders())

            assert tracer.regularizations_of_planes == []

        def test__source_galaxy_has_regularization__returns_regularizations(self, data_grids):
            galaxy_reg = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_no_reg = g.Galaxy()

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_no_reg], source_galaxies=[galaxy_reg],
                                                         image_plane_grids=[data_grids], border=MockBorders())

            assert tracer.regularizations_of_planes[0].value == 0

        def test__both_galaxies_have_regularization__returns_both_regularizations(self, data_grids):
            galaxy_reg_0 = g.Galaxy(pixelization=MockPixelization(value=1), regularization=MockRegularization(value=3))
            galaxy_reg_1 = g.Galaxy(pixelization=MockPixelization(value=2), regularization=MockRegularization(value=4))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_reg_0], source_galaxies=[galaxy_reg_1],
                                                         image_plane_grids=[data_grids], border=MockBorders())

            assert tracer.regularizations_of_planes[0].value == 3
            assert tracer.regularizations_of_planes[1].value == 4

    class TestGalaxyMasses:

        def test__masses_with_circle__1_galaxy__consistent_with_galaxy_mass(self, data_grids):

            g0 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0), redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g.Galaxy(redshift=2.0)],
                                                         image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            g0_mass = g0.mass_within_circle(radius=1.0, conversion_factor=tracer.critical_density_arcsec)

            assert tracer.masses_of_image_plane_galaxies_within_circles(radius=1.0)[0] == g0_mass

            g0 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0), redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g.Galaxy(redshift=2.0)],
                                                         image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            g0_mass = g0.mass_within_circle(radius=2.0, conversion_factor=tracer.critical_density_arcsec)

            assert tracer.masses_of_image_plane_galaxies_within_circles(radius=2.0)[0] == g0_mass

        def test__masses_with_circle__2_galaxies__consistent_with_galaxy_masses(self, data_grids):

            g0 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0), redshift=1.0)
            g1 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=2.0), redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g.Galaxy(redshift=2.0)],
                                                         image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            g0_mass = g0.mass_within_circle(radius=1.0, conversion_factor=tracer.critical_density_arcsec)
            g1_mass = g1.mass_within_circle(radius=1.0, conversion_factor=tracer.critical_density_arcsec)

            assert tracer.masses_of_image_plane_galaxies_within_circles(radius=1.0)[0] == g0_mass
            assert tracer.masses_of_image_plane_galaxies_within_circles(radius=1.0)[1] == g1_mass

            g0 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=3.0), redshift=1.5)
            g1 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=4.0), redshift=1.5)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g.Galaxy(redshift=2.0)],
                                                         image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            g0_mass = g0.mass_within_circle(radius=2.0, conversion_factor=tracer.critical_density_arcsec)
            g1_mass = g1.mass_within_circle(radius=2.0, conversion_factor=tracer.critical_density_arcsec)

            assert tracer.masses_of_image_plane_galaxies_within_circles(radius=2.0)[0] == g0_mass
            assert tracer.masses_of_image_plane_galaxies_within_circles(radius=2.0)[1] == g1_mass

        def test__masses_with_ellipse__1_galaxy__consistent_with_galaxy_mass(self, data_grids):

            g0 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0), redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g.Galaxy(redshift=2.0)],
                                                         image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            g0_mass = g0.mass_within_ellipse(major_axis=1.0, conversion_factor=tracer.critical_density_arcsec)

            assert tracer.masses_of_image_plane_galaxies_within_ellipses(major_axis=1.0)[0] == g0_mass

            g0 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0), redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g.Galaxy(redshift=2.0)],
                                                         image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            g0_mass = g0.mass_within_ellipse(major_axis=2.0, conversion_factor=tracer.critical_density_arcsec)

            assert tracer.masses_of_image_plane_galaxies_within_ellipses(major_axis=2.0)[0] == g0_mass

        def test__masses_with_ellipse__2_galaxies__consistent_with_galaxy_masses(self, data_grids):

            g0 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0), redshift=1.0)
            g1 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=2.0), redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g.Galaxy(redshift=2.0)],
                                                         image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            g0_mass = g0.mass_within_ellipse(major_axis=1.0, conversion_factor=tracer.critical_density_arcsec)
            g1_mass = g1.mass_within_ellipse(major_axis=1.0, conversion_factor=tracer.critical_density_arcsec)

            assert tracer.masses_of_image_plane_galaxies_within_ellipses(major_axis=1.0)[0] == g0_mass
            assert tracer.masses_of_image_plane_galaxies_within_ellipses(major_axis=1.0)[1] == g1_mass

            g0 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=3.0), redshift=1.5)
            g1 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=4.0), redshift=1.5)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g.Galaxy(redshift=2.0)],
                                                         image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            g0_mass = g0.mass_within_ellipse(major_axis=2.0, conversion_factor=tracer.critical_density_arcsec)
            g1_mass = g1.mass_within_ellipse(major_axis=2.0, conversion_factor=tracer.critical_density_arcsec)

            assert tracer.masses_of_image_plane_galaxies_within_ellipses(major_axis=2.0)[0] == g0_mass
            assert tracer.masses_of_image_plane_galaxies_within_ellipses(major_axis=2.0)[1] == g1_mass

        def test__circle_and_ellipse__no_cosmology_returns_none(self, data_grids):

            g0 = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0), redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g.Galaxy(redshift=2.0)],
                                                         image_plane_grids=[data_grids])

            assert tracer.masses_of_image_plane_galaxies_within_circles(radius=1.0) == None
            assert tracer.masses_of_image_plane_galaxies_within_ellipses(major_axis=1.0) == None

    class TestImagePlanePixGrid:

        def test__galaxies_have_no_pixelization__no_pix_grid_added(self):

            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False]]), pixel_scale=1.0)

            data_grids = grids.DataGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                     psf_shape=(1, 1))

            galaxy = g.Galaxy()

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy], source_galaxies=[galaxy],
                                                         image_plane_grids=[data_grids])

            assert (tracer.image_plane.grids[0].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.source_plane.grids[0].pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxies_have_other_pixelization__returns_normal_grids(self):

            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False]]), pixel_scale=1.0)

            data_grids = grids.DataGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                     psf_shape=(1, 1))

            galaxy = g.Galaxy(pixelization=pixelizations.Rectangular(shape=(3, 3)),
                              regularization=regularization.Constant())

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy], source_galaxies=[galaxy],
                                                         image_plane_grids=[data_grids])

            assert (tracer.image_plane.grids[0].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.source_plane.grids[0].pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxy_has_pixelization__but_grid_is_padded_grid__returns_normal_grids(self):

            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, True, False]]), pixel_scale=1.0)

            data_grids = grids.DataGrids.padded_grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                            psf_shape=(1, 1))

            galaxy = g.Galaxy(pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                              regularization=regularization.Constant())

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy], source_galaxies=[galaxy],
                                                         image_plane_grids=[data_grids])

            assert (tracer.image_plane.grids[0].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.source_plane.grids[0].pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxy_has_pixelization__returns_grids_with_pix_grid(self):
            
            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, True, False]]), pixel_scale=1.0)

            data_grids = grids.DataGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                     psf_shape=(1, 1))

            galaxy = g.Galaxy(pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                              regularization=regularization.Constant())

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy], source_galaxies=[galaxy],
                                                         image_plane_grids=[data_grids])

            assert (tracer.image_plane.grids[0].regular == data_grids.regular).all()
            assert (tracer.image_plane.grids[0].sub == data_grids.sub).all()
            assert (tracer.image_plane.grids[0].blurring == data_grids.blurring).all()
            assert (tracer.image_plane.grids[0].pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                           [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                           [-1.0, -1.0], [-1.0, 1.0]])).all()

            assert (tracer.source_plane.grids[0].regular == data_grids.regular).all()
            assert (tracer.source_plane.grids[0].sub == data_grids.sub).all()
            assert (tracer.source_plane.grids[0].blurring == data_grids.blurring).all()
            assert (tracer.source_plane.grids[0].pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                           [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                           [-1.0, -1.0], [-1.0, 1.0]])).all()


class TestMultiTracer(object):

    class TestGalaxyOrder:

        def test__3_galaxies_reordered_in_ascending_redshift(self, data_grids):

            tracer = ray_tracing.TracerMulti(galaxies=[g.Galaxy(redshift=2.0), g.Galaxy(redshift=1.0),
                                                       g.Galaxy(redshift=0.1)], image_plane_grids=[data_grids],
                                             cosmology=cosmo.Planck15)

            assert tracer.galaxies_redshift_order[0].redshift == 0.1
            assert tracer.galaxies_redshift_order[1].redshift == 1.0
            assert tracer.galaxies_redshift_order[2].redshift == 2.0

        def test_3_galaxies_two_same_redshift_planes_redshift_order_is_size_2_with_redshifts(self, data_grids):
            tracer = ray_tracing.TracerMulti(galaxies=[g.Galaxy(redshift=1.0), g.Galaxy(redshift=1.0),
                                                       g.Galaxy(redshift=0.1)],
                                             image_plane_grids=[data_grids],
                                             cosmology=cosmo.Planck15)

            assert tracer.galaxies_redshift_order[0].redshift == 0.1
            assert tracer.galaxies_redshift_order[1].redshift == 1.0
            assert tracer.galaxies_redshift_order[2].redshift == 1.0

            assert tracer.planes_redshift_order[0] == 0.1
            assert tracer.planes_redshift_order[1] == 1.0

        def test__6_galaxies_producing_4_planes(self, data_grids):
            g0 = g.Galaxy(redshift=1.0)
            g1 = g.Galaxy(redshift=1.0)
            g2 = g.Galaxy(redshift=0.1)
            g3 = g.Galaxy(redshift=1.05)
            g4 = g.Galaxy(redshift=0.95)
            g5 = g.Galaxy(redshift=1.05)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4, g5],
                                             image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            assert tracer.galaxies_redshift_order[0].redshift == 0.1
            assert tracer.galaxies_redshift_order[1].redshift == 0.95
            assert tracer.galaxies_redshift_order[2].redshift == 1.0
            assert tracer.galaxies_redshift_order[3].redshift == 1.0
            assert tracer.galaxies_redshift_order[4].redshift == 1.05
            assert tracer.galaxies_redshift_order[5].redshift == 1.05

            assert tracer.planes_redshift_order[0] == 0.1
            assert tracer.planes_redshift_order[1] == 0.95
            assert tracer.planes_redshift_order[2] == 1.0
            assert tracer.planes_redshift_order[3] == 1.05

        def test__6_galaxies__plane_galaxies_are_correct(self, data_grids):
            g0 = g.Galaxy(redshift=1.0)
            g1 = g.Galaxy(redshift=1.0)
            g2 = g.Galaxy(redshift=0.1)
            g3 = g.Galaxy(redshift=1.05)
            g4 = g.Galaxy(redshift=0.95)
            g5 = g.Galaxy(redshift=1.05)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4, g5],
                                             image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            assert tracer.planes_galaxies[0] == [g2]
            assert tracer.planes_galaxies[1] == [g4]
            assert tracer.planes_galaxies[2] == [g0, g1]
            assert tracer.planes_galaxies[3] == [g3, g5]

    class TestException:

        def test__no_galaxies_in_tracer__raises_excetion(self, data_grids):
            with pytest.raises(exc.RayTracingException):
                tracer = ray_tracing.TracerMulti(galaxies=[], image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

    class TestCosmology:

        def test__3_planes__z01_z1__and_z2(self, data_grids):

            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy(redshift=1.0)
            g2 = g.Galaxy(redshift=2.0)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                             cosmology=cosmo.Planck15)

            assert tracer.source_plane_index == 2

            assert tracer.arcsec_per_kpc_proper_of_plane(i=0) == pytest.approx(0.525060, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=0) == pytest.approx(1.904544, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth(i=0) == pytest.approx(392840, 1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=0, j=0) == 0.0
            assert tracer.angular_diameter_distance_between_planes(i=0, j=1) == pytest.approx(1481890.4,
                                                                                              1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=0, j=2) == pytest.approx(1626471,
                                                                                              1e-5)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=1) == pytest.approx(0.1214785, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=1) == pytest.approx(8.231907, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth(i=1) == pytest.approx(1697952, 1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=1, j=0) == pytest.approx(-2694346,
                                                                                              1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=1, j=1) == 0.0
            assert tracer.angular_diameter_distance_between_planes(i=1, j=2) == pytest.approx(638544,
                                                                                              1e-5)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=2) == pytest.approx(0.116500, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=2) == pytest.approx(8.58368, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth(i=2) == pytest.approx(1770512, 1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=2, j=0) == pytest.approx(-4435831,
                                                                                              1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=2, j=1) == pytest.approx(-957816)
            assert tracer.angular_diameter_distance_between_planes(i=2, j=2) == 0.0

            assert tracer.critical_density_kpc_between_planes(i=0, j=1) == pytest.approx(4.85e9, 1e-2)
            assert tracer.critical_density_arcsec_between_planes(i=0, j=1) == pytest.approx(17593241668, 1e-2)

            assert tracer.scaling_factor_between_planes(i=0, j=1) == pytest.approx(0.9500, 1e-4)
            assert tracer.scaling_factor_between_planes(i=0, j=2) == pytest.approx(1.0, 1e-4)
            assert tracer.scaling_factor_between_planes(i=1, j=2) == pytest.approx(1.0, 1e-4)

        def test__4_planes__z01_z1_z2_and_z3(self, data_grids):

            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy(redshift=1.0)
            g2 = g.Galaxy(redshift=2.0)
            g3 = g.Galaxy(redshift=3.0)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3], image_plane_grids=[data_grids],
                                             cosmology=cosmo.Planck15)

            assert tracer.source_plane_index == 3

            assert tracer.arcsec_per_kpc_proper_of_plane(i=0) == pytest.approx(0.525060, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=0) == pytest.approx(1.904544, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth(i=0) == pytest.approx(392840, 1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=0, j=0) == 0.0
            assert tracer.angular_diameter_distance_between_planes(i=0, j=1) == pytest.approx(1481890.4,
                                                                                              1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=0, j=2) == pytest.approx(1626471,
                                                                                              1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=0, j=3) == pytest.approx(1519417,
                                                                                              1e-5)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=1) == pytest.approx(0.1214785, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=1) == pytest.approx(8.231907, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth(i=1) == pytest.approx(1697952, 1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=1, j=0) == pytest.approx(-2694346,
                                                                                              1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=1, j=1) == 0.0
            assert tracer.angular_diameter_distance_between_planes(i=1, j=2) == pytest.approx(638544,
                                                                                              1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=1, j=3) == pytest.approx(778472,
                                                                                              1e-5)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=2) == pytest.approx(0.116500, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=2) == pytest.approx(8.58368, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth(i=2) == pytest.approx(1770512, 1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=2, j=0) == pytest.approx(-4435831,
                                                                                              1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=2, j=1) == pytest.approx(-957816)
            assert tracer.angular_diameter_distance_between_planes(i=2, j=2) == 0.0
            assert tracer.angular_diameter_distance_between_planes(i=2, j=3) == pytest.approx(299564)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=3) == pytest.approx(0.12674, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=3) == pytest.approx(7.89009, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth(i=3) == pytest.approx(1627448, 1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=3, j=0) == pytest.approx(-5525155,
                                                                                              1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=3, j=1) == pytest.approx(-1556945,
                                                                                              1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=3, j=2) == pytest.approx(-399419,
                                                                                              1e-5)
            assert tracer.angular_diameter_distance_between_planes(i=3, j=3) == 0.0

            assert tracer.critical_density_kpc_between_planes(i=0, j=1) == pytest.approx(4.85e9, 1e-2)
            assert tracer.critical_density_arcsec_between_planes(i=0, j=1) == pytest.approx(17593241668, 1e-2)

            assert tracer.scaling_factor_between_planes(i=0, j=1) == pytest.approx(0.9348, 1e-4)
            assert tracer.scaling_factor_between_planes(i=0, j=2) == pytest.approx(0.984, 1e-4)
            assert tracer.scaling_factor_between_planes(i=0, j=3) == pytest.approx(1.0, 1e-4)
            assert tracer.scaling_factor_between_planes(i=1, j=2) == pytest.approx(0.754, 1e-4)
            assert tracer.scaling_factor_between_planes(i=1, j=3) == pytest.approx(1.0, 1e-4)
            assert tracer.scaling_factor_between_planes(i=2, j=3) == pytest.approx(1.0, 1e-4)

    class TestRayTracingPlanes:

        def test__6_galaxies__tracer_planes_are_correct(self, data_grids):
            g0 = g.Galaxy(redshift=2.0)
            g1 = g.Galaxy(redshift=2.0)
            g2 = g.Galaxy(redshift=0.1)
            g3 = g.Galaxy(redshift=3.0)
            g4 = g.Galaxy(redshift=1.0)
            g5 = g.Galaxy(redshift=3.0)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4, g5],
                                             image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            assert tracer.planes[0].galaxies == [g2]
            assert tracer.planes[1].galaxies == [g4]
            assert tracer.planes[2].galaxies == [g0, g1]
            assert tracer.planes[3].galaxies == [g3, g5]

        def test__4_planes__coordinate_data_grids_and_deflections_are_correct__sis_mass_profile(self, data_grids):
            import math

            g0 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g2 = g.Galaxy(redshift=0.1, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g3 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g4 = g.Galaxy(redshift=1.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g5 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4, g5],
                                             image_plane_grids=[data_grids], cosmology=cosmo.Planck15)

            # From unit test below:
            # Beta_01 = 0.9348
            # Beta_02 = 0.9840
            # Beta_03 = 1.0
            # Beta_12 = 0.754
            # Beta_13 = 1.0
            # Beta_23 = 1.0

            val = math.sqrt(2) / 2.0

            assert tracer.planes[0].grids[0].regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
            assert tracer.planes[0].grids[0].sub[0] == pytest.approx(np.array([1.0, 1.0]),
                                                                  1e-4)
            assert tracer.planes[0].grids[0].sub[1] == pytest.approx(np.array([1.0, 0.0]),
                                                                  1e-4)
            assert tracer.planes[0].grids[0].blurring[0] == pytest.approx(np.array([1.0, 0.0]),
                                                                       1e-4)
            assert tracer.planes[0].deflections[0].regular[0] == pytest.approx(np.array([val, val]), 1e-4)
            assert tracer.planes[0].deflections[0].sub[0] == pytest.approx(np.array([val, val]), 1e-4)
            assert tracer.planes[0].deflections[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)
            assert tracer.planes[0].deflections[0].blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

            assert tracer.planes[1].grids[0].regular[0] == pytest.approx(
                np.array([(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]), 1e-4)
            assert tracer.planes[1].grids[0].sub[0] == pytest.approx(
                np.array([(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]), 1e-4)
            assert tracer.planes[1].grids[0].sub[1] == pytest.approx(
                np.array([(1.0 - 0.9348 * 1.0), 0.0]), 1e-4)
            assert tracer.planes[1].grids[0].blurring[0] == pytest.approx(
                np.array([(1.0 - 0.9348 * 1.0), 0.0]), 1e-4)

            defl11 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]))
            defl12 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * 1.0), 0.0]]))

            assert tracer.planes[1].deflections[0].regular[0] == pytest.approx(defl11[0], 1e-4)
            assert tracer.planes[1].deflections[0].sub[0] == pytest.approx(defl11[0], 1e-4)
            assert tracer.planes[1].deflections[0].sub[1] == pytest.approx(defl12[0], 1e-4)
            assert tracer.planes[1].deflections[0].blurring[0] == pytest.approx(defl12[0], 1e-4)

            assert tracer.planes[2].grids[0].regular[0] == pytest.approx(
                np.array([(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                          (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]), 1e-4)
            assert tracer.planes[2].grids[0].sub[0] == pytest.approx(
                np.array([(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                          (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]), 1e-4)
            assert tracer.planes[2].grids[0].sub[1] == pytest.approx(
                np.array([(1.0 - 0.9839601 * 1.0 - 0.7539734 * defl12[0, 0]),
                          0.0]), 1e-4)
            assert tracer.planes[2].grids[0].blurring[0] == pytest.approx(
                np.array([(1.0 - 0.9839601 * 1.0 - 0.7539734 * defl12[0, 0]),
                          0.0]), 1e-4)

            # 2 Galaxies in this plane, so multiply by 2.0

            defl21 = 2.0 * g0.deflections_from_grid(
                grid=np.array([[(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                                (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]]))
            defl22 = 2.0 * g0.deflections_from_grid(
                grid=np.array([[(1.0 - 0.9839601 * 1.0 - 0.7539734 * defl12[0, 0]),
                                0.0]]))

            assert tracer.planes[2].deflections[0].regular[0] == pytest.approx(defl21[0], 1e-4)
            assert tracer.planes[2].deflections[0].sub[0] == pytest.approx(defl21[0], 1e-4)
            assert tracer.planes[2].deflections[0].sub[1] == pytest.approx(defl22[0], 1e-4)
            assert tracer.planes[2].deflections[0].blurring[0] == pytest.approx(defl22[0], 1e-4)

            coord1 = (1.0 - tracer.planes[0].deflections[0].regular[0, 0] - tracer.planes[1].deflections[0].regular[
                0, 0] -
                      tracer.planes[2].deflections[0].regular[0, 0])

            coord2 = (1.0 - tracer.planes[0].deflections[0].regular[0, 1] - tracer.planes[1].deflections[0].regular[
                0, 1] -
                      tracer.planes[2].deflections[0].regular[0, 1])

            coord3 = (1.0 - tracer.planes[0].deflections[0].sub[1, 0] -
                      tracer.planes[1].deflections[0].sub[1, 0] -
                      tracer.planes[2].deflections[0].sub[1, 0])

            assert tracer.planes[3].grids[0].regular[0] == pytest.approx(np.array([coord1, coord2]),
                                                                    1e-4)
            assert tracer.planes[3].grids[0].sub[0] == pytest.approx(
                np.array([coord1, coord2]), 1e-4)
            assert tracer.planes[3].grids[0].sub[1] == pytest.approx(np.array([coord3, 0.0]),
                                                                  1e-4)
            assert tracer.planes[3].grids[0].blurring[0] == pytest.approx(np.array([coord3, 0.0]),
                                                                       1e-4)

    class TestImagePlaneImages:

        def test__x1_galaxy_light_no_mass_in_each_plane__image_of_each_plane_is_galaxy_image(self, data_grids):

            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                             cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0], grids=[data_grids], compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1], grids=[data_grids], compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grids=[data_grids], compute_deflections=False)

            image_plane_image = plane_0.image_plane_images[0] + plane_1.image_plane_images[0] + \
                                plane_2.image_plane_images[0]

            assert image_plane_image.shape == (3, 4)
            assert (image_plane_image == tracer.image_plane_images[0]).all()

        def test__galaxy_light_mass_sis__source_plane_image_includes_deflections(self, data_grids):

            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                             cosmology=cosmo.Planck15)

            plane_0 = tracer.planes[0]
            plane_1 = tracer.planes[1]
            plane_2 = tracer.planes[2]

            image_plane_image = plane_0.image_plane_images[0] + plane_1.image_plane_images[0] + plane_2.image_plane_images[0]

            assert image_plane_image.shape == (3, 4)
            assert (image_plane_image == tracer.image_plane_images[0]).all()

        def test__same_as_above_more_galaxies(self, data_grids):

            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))
            g3 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.4))
            g4 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.5))

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4], image_plane_grids=[data_grids],
                                             cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0, g3], grids=[data_grids], compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1, g4], grids=[data_grids], compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grids=[data_grids], compute_deflections=False)

            image_plane_image = plane_0.image_plane_images[0] + plane_1.image_plane_images[0] + \
                                plane_2.image_plane_images[0]

            assert image_plane_image.shape == (3, 4)
            assert (image_plane_image == tracer.image_plane_images[0]).all()

        def test__same_as_above__x2_grids_in__x2_images_out(self, data_grids, data_grids_1):

            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))
            g3 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.4))
            g4 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.5))

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4],
                                             image_plane_grids=[data_grids, data_grids_1],
                                             cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0, g3], grids=[data_grids, data_grids_1], compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1, g4], grids=[data_grids, data_grids_1], compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grids=[data_grids, data_grids_1], compute_deflections=False)

            image_plane_image_0 = plane_0.image_plane_images[0] + plane_1.image_plane_images[0] + \
                                plane_2.image_plane_images[0]

            assert image_plane_image_0.shape == (3, 4)
            assert (image_plane_image_0 == tracer.image_plane_images[0]).all()

            image_plane_image_1 = plane_0.image_plane_images[1] + plane_1.image_plane_images[1] + \
                                plane_2.image_plane_images[1]

            assert image_plane_image_1.shape == (3, 4)
            assert (image_plane_image_1 == tracer.image_plane_images[1]).all()

        def test__padded_2d_image_from_plane__mapped_correctly(self, padded_grids):

            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[padded_grids],
                                             cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0], grids=[padded_grids], compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1], grids=[padded_grids], compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grids=[padded_grids], compute_deflections=False)

            image_plane_image = plane_0.image_plane_images[0] + plane_1.image_plane_images[0] + \
                                plane_2.image_plane_images[0]

            assert image_plane_image.shape == (1, 2)
            assert (image_plane_image == tracer.image_plane_images[0]).all()

        def test__padded_2d_image_for_simulation__mapped_correctly_not_trimmed(self, padded_grids):

            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[padded_grids],
                                             cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0], grids=[padded_grids], compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1], grids=[padded_grids], compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grids=[padded_grids], compute_deflections=False)

            image_plane_image_for_simulation = plane_0.image_plane_images_for_simulation[0] + \
                                               plane_1.image_plane_images_for_simulation[0] + \
                                               plane_2.image_plane_images_for_simulation[0]

            assert image_plane_image_for_simulation.shape == (3,4)
            assert (image_plane_image_for_simulation == tracer.image_plane_images_for_simulation[0]).all()

    class TestImagePlaneBlurringImages:

        def test__x1_galaxy_light_no_mass_in_each_plane__image_of_each_plane_is_galaxy_image(self, data_grids):

            sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                         sersic_index=4.0)

            g0 = g.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = g.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = g.Galaxy(redshift=2.0, light_profile=sersic)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                             cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0], grids=[data_grids], compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1], grids=[data_grids], compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grids=[data_grids], compute_deflections=False)

            image_plane_blurring_image = plane_0.image_plane_blurring_images_[0] + \
                                         plane_1.image_plane_blurring_images_[0] + \
                                         plane_2.image_plane_blurring_images_[0]

            assert (image_plane_blurring_image == tracer.image_plane_blurring_images_[0]).all()

        def test__galaxy_light_mass_sis__source_plane_image_includes_deflections(self, data_grids):

            sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                         sersic_index=4.0)

            sis = mp.SphericalIsothermal(einstein_radius=1.0)

            g0 = g.Galaxy(redshift=0.1, light_profile=sersic, mass_profile=sis)
            g1 = g.Galaxy(redshift=1.0, light_profile=sersic, mass_profile=sis)
            g2 = g.Galaxy(redshift=2.0, light_profile=sersic, mass_profile=sis)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                             cosmology=cosmo.Planck15)

            plane_0 = tracer.planes[0]
            plane_1 = tracer.planes[1]
            plane_2 = tracer.planes[2]

            image_plane_blurring_image = plane_0.image_plane_blurring_images_[0] + \
                                         plane_1.image_plane_blurring_images_[0] + \
                                         plane_2.image_plane_blurring_images_[0]

            assert (image_plane_blurring_image == tracer.image_plane_blurring_images_[0]).all()

        def test__x1_galaxy_light_no_mass_in_each_plane__image_of_each_galaxy_is_galaxy_image(self, data_grids):

            sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                         sersic_index=4.0)

            g0 = g.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = g.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = g.Galaxy(redshift=2.0, light_profile=sersic)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                             cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0], grids=[data_grids], compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1], grids=[data_grids], compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grids=[data_grids], compute_deflections=False)

            image_plane_blurring_image = plane_0.image_plane_blurring_images_[0] + \
                                         plane_1.image_plane_blurring_images_[0] + \
                                         plane_2.image_plane_blurring_images_[0]

            assert (image_plane_blurring_image == tracer.image_plane_blurring_images_[0]).all()

        def test__diffrent_galaxies_light_no_mass_in_each_plane__image_of_each_galaxy_is_galaxy_image(self,
                                                                                                      data_grids):
            sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                         sersic_index=4.0)

            g0 = g.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = g.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = g.Galaxy(redshift=2.0, light_profile=sersic)
            g3 = g.Galaxy(redshift=0.1, light_profile=sersic)
            g4 = g.Galaxy(redshift=1.0, light_profile=sersic)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4], image_plane_grids=[data_grids],
                                             cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0, g3], grids=[data_grids], compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1, g4], grids=[data_grids], compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grids=[data_grids], compute_deflections=False)

            image_plane_blurring_image = plane_0.image_plane_blurring_images_[0] + \
                                         plane_1.image_plane_blurring_images_[0] + \
                                         plane_2.image_plane_blurring_images_[0]

            assert (image_plane_blurring_image == tracer.image_plane_blurring_images_[0]).all()

        def test__same_as_above__x2_grids_in__x2_images_out(self, data_grids, data_grids_1):

            sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                         sersic_index=4.0)

            g0 = g.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = g.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = g.Galaxy(redshift=2.0, light_profile=sersic)
            g3 = g.Galaxy(redshift=0.1, light_profile=sersic)
            g4 = g.Galaxy(redshift=1.0, light_profile=sersic)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4],
                                             image_plane_grids=[data_grids, data_grids_1],
                                             cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0, g3], grids=[data_grids, data_grids_1], compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1, g4], grids=[data_grids, data_grids_1], compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grids=[data_grids, data_grids_1], compute_deflections=False)

            image_plane_blurring_image_0 = plane_0.image_plane_blurring_images_[0] + \
                                           plane_1.image_plane_blurring_images_[0] + \
                                           plane_2.image_plane_blurring_images_[0]

            assert (image_plane_blurring_image_0 == tracer.image_plane_blurring_images_[0]).all()

            image_plane_blurring_image_1 = plane_0.image_plane_blurring_images_[1] + \
                                           plane_1.image_plane_blurring_images_[1] + \
                                           plane_2.image_plane_blurring_images_[1]

            assert (image_plane_blurring_image_1 == tracer.image_plane_blurring_images_[1]).all()

    class TestMappers:

        def test__3_galaxies__non_have_pixelization__returns_none_x3(self, data_grids):
            sis = mp.SphericalIsothermal(einstein_radius=1.0)

            g0 = g.Galaxy(redshift=0.1, mass_profile=sis)
            g1 = g.Galaxy(redshift=1.0, mass_profile=sis)
            g2 = g.Galaxy(redshift=2.0, mass_profile=sis)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                             border=[MockBorders()], cosmology=cosmo.Planck15)

            assert tracer.mappers_of_planes == []

        def test__3_galaxies__1_has_pixelization__returns_none_x2_and_pixelization(self, data_grids):
            sis = mp.SphericalIsothermal(einstein_radius=1.0)

            g0 = g.Galaxy(redshift=0.1, mass_profile=sis)
            g1 = g.Galaxy(redshift=1.0, pixelization=MockPixelization(value=1),
                          regularization=MockRegularization(value=0))
            g2 = g.Galaxy(redshift=2.0, mass_profile=sis)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                             border=[MockBorders()], cosmology=cosmo.Planck15)

            assert tracer.mappers_of_planes[0] == 1

        def test__3_galaxies__all_have_pixelization__returns_pixelizations(self, data_grids):
            sis = mp.SphericalIsothermal(einstein_radius=1.0)

            g0 = g.Galaxy(redshift=0.1, pixelization=MockPixelization(value=0.5),
                          regularization=MockRegularization(value=0), mass_profile=sis)
            g1 = g.Galaxy(redshift=1.0, pixelization=MockPixelization(value=1),
                          regularization=MockRegularization(value=0), mass_profile=sis)
            g2 = g.Galaxy(redshift=2.0, pixelization=MockPixelization(value=2),
                          regularization=MockRegularization(value=0), mass_profile=sis)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                             border=[MockBorders()], cosmology=cosmo.Planck15)

            assert tracer.mappers_of_planes == [0.5, 1, 2]

    class TestRegularizations:

        def test__3_galaxies__non_have_regularization__returns_none_x3(self, data_grids):
            sis = mp.SphericalIsothermal(einstein_radius=1.0)

            g0 = g.Galaxy(redshift=0.1, mass_profile=sis)
            g1 = g.Galaxy(redshift=1.0, mass_profile=sis)
            g2 = g.Galaxy(redshift=2.0, mass_profile=sis)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                             border=MockBorders(), cosmology=cosmo.Planck15)

            assert tracer.regularizations_of_planes == []

        def test__3_galaxies__1_has_regularization__returns_none_x2_and_regularization(self, data_grids):
            sis = mp.SphericalIsothermal(einstein_radius=1.0)

            g0 = g.Galaxy(redshift=0.1, mass_profile=sis)
            g1 = g.Galaxy(redshift=1.0, pixelization=MockPixelization(value=1),
                          regularization=MockRegularization(value=0))
            g2 = g.Galaxy(redshift=2.0, mass_profile=sis)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                             border=MockBorders(), cosmology=cosmo.Planck15)

            assert tracer.regularizations_of_planes[0].value == 0

        def test__3_galaxies__all_have_regularization__returns_regularizations(self, data_grids):
            sis = mp.SphericalIsothermal(einstein_radius=1.0)

            g0 = g.Galaxy(redshift=0.1, pixelization=MockPixelization(value=0.5),
                          regularization=MockRegularization(value=0), mass_profile=sis)
            g1 = g.Galaxy(redshift=1.0, pixelization=MockPixelization(value=1),
                          regularization=MockRegularization(value=1), mass_profile=sis)
            g2 = g.Galaxy(redshift=2.0, pixelization=MockPixelization(value=2),
                          regularization=MockRegularization(value=2), mass_profile=sis)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                             border=MockBorders(), cosmology=cosmo.Planck15)

            assert tracer.regularizations_of_planes[0].value == 0
            assert tracer.regularizations_of_planes[1].value == 1
            assert tracer.regularizations_of_planes[2].value == 2

    class TestImagePlanePixGrid:

        def test__galaxies_have_no_pixelization__no_pix_grid_added(self):

            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False]]), pixel_scale=1.0)

            data_grids = grids.DataGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                     psf_shape=(1, 1))

            tracer = ray_tracing.TracerMulti(galaxies=[g.Galaxy(redshift=2.0), g.Galaxy(redshift=1.0)],
                                             image_plane_grids=[data_grids])

            assert (tracer.planes[0].grids[0].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.planes[1].grids[0].pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxies_have_other_pixelization__returns_normal_grids(self):

            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False]]), pixel_scale=1.0)

            data_grids = grids.DataGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                     psf_shape=(1, 1))

            galaxy = g.Galaxy(pixelization=pixelizations.Rectangular(shape=(3, 3)),
                              regularization=regularization.Constant(), redshift=2.0)

            tracer = ray_tracing.TracerMulti(galaxies=[galaxy, g.Galaxy(redshift=1.0)],
                                             image_plane_grids=[data_grids])

            assert (tracer.planes[0].grids[0].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.planes[1].grids[0].pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxy_has_pixelization__but_grid_is_padded_grid__returns_normal_grids(self):
            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, True, False]]), pixel_scale=1.0)

            data_grids = grids.DataGrids.padded_grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                            psf_shape=(1, 1))

            galaxy = g.Galaxy(pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                              regularization=regularization.Constant(), redshift=2.0)

            tracer = ray_tracing.TracerMulti(galaxies=[galaxy, g.Galaxy(redshift=1.0)],
                                             image_plane_grids=[data_grids])

            assert (tracer.planes[0].grids[0].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.planes[1].grids[0].pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxy_has_pixelization__returns_grids_with_pix_grid(self):

            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, True, False]]), pixel_scale=1.0)

            data_grids = grids.DataGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                     psf_shape=(1, 1))

            galaxy = g.Galaxy(pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                              regularization=regularization.Constant(), redshift=2.0)

            tracer = ray_tracing.TracerMulti(galaxies=[galaxy, g.Galaxy(redshift=1.0)],
                                             image_plane_grids=[data_grids])

            assert (tracer.planes[0].grids[0].regular == data_grids.regular).all()
            assert (tracer.planes[0].grids[0].sub == data_grids.sub).all()
            assert (tracer.planes[0].grids[0].blurring == data_grids.blurring).all()
            assert (tracer.planes[0].grids[0].pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                                 [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                 [-1.0, -1.0], [-1.0, 1.0]])).all()

            assert (tracer.planes[1].grids[0].regular == data_grids.regular).all()
            assert (tracer.planes[1].grids[0].sub == data_grids.sub).all()
            assert (tracer.planes[1].grids[0].blurring == data_grids.blurring).all()
            assert (tracer.planes[1].grids[0].pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                                  [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                  [-1.0, -1.0], [-1.0, 1.0]])).all()

class TestBooleanProperties(object):

    def test__has_galaxy_with_light_profile(self, data_grids):
        gal = g.Galaxy()
        gal_lp = g.Galaxy(light_profile=lp.LightProfile())
        gal_mp = g.Galaxy(mass_profile=mp.SphericalIsothermal())

        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal], [gal], image_plane_grids=[data_grids]).has_light_profile == False
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_mp], [gal_mp], image_plane_grids=[data_grids]).has_light_profile == False
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_lp], [gal_lp], image_plane_grids=[data_grids]).has_light_profile == True
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_lp], [gal], image_plane_grids=[data_grids]).has_light_profile == True
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_lp], [gal_mp], image_plane_grids=[data_grids]).has_light_profile == True

    def test__has_galaxy_with_pixelization(self, data_grids):
        gal = g.Galaxy()
        gal_lp = g.Galaxy(light_profile=lp.LightProfile())
        gal_pix = g.Galaxy(pixelization=pixelizations.Pixelization(), regularization=regularization.Constant())

        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal], [gal], image_plane_grids=[data_grids]).has_pixelization == False
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_lp], [gal_lp], image_plane_grids=[data_grids]).has_pixelization == False
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_pix], [gal_pix], image_plane_grids=[data_grids]).has_pixelization == True
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_pix], [gal], image_plane_grids=[data_grids]).has_pixelization == True
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_pix], [gal_lp], image_plane_grids=[data_grids]).has_pixelization == True

    def test__has_galaxy_with_regularization(self, data_grids):
        gal = g.Galaxy()
        gal_lp = g.Galaxy(light_profile=lp.LightProfile())
        gal_reg = g.Galaxy(pixelization=pixelizations.Pixelization(), regularization=regularization.Constant())

        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal], [gal], image_plane_grids=[data_grids]).has_regularization == False
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_lp], [gal_lp], image_plane_grids=[data_grids]).has_regularization == False
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_reg], [gal_reg], image_plane_grids=[data_grids]).has_regularization == True
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_reg], [gal], image_plane_grids=[data_grids]).has_regularization == True
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_reg], [gal_lp], image_plane_grids=[data_grids]).has_regularization == True

    def test__has_hyper_galaxy(self, data_grids):
        gal = g.Galaxy()
        gal_lp = g.Galaxy(light_profile=lp.LightProfile())
        gal_hyper = g.Galaxy(hyper_galaxy=g.HyperGalaxy())

        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal], [gal], image_plane_grids=[data_grids]).has_hyper_galaxy == False
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_lp], [gal_lp], image_plane_grids=[data_grids]).has_hyper_galaxy == False
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_hyper], [gal_hyper], image_plane_grids=[data_grids]).has_hyper_galaxy == True
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_hyper], [gal], image_plane_grids=[data_grids]).has_hyper_galaxy == True
        assert ray_tracing.TracerImageSourcePlanes \
                   ([gal_hyper], [gal_lp], image_plane_grids=[data_grids]).has_hyper_galaxy == True


class TestTracerImageAndSourcePositions(object):

    class TestSetup:

        def test__x2_positions__no_galaxy__image_and_source_planes_setup__same_positions(self, galaxy_non):
            tracer = ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=[galaxy_non],
                                                                  positions=[np.array([[1.0, 1.0], [-1.0, -1.0]])])

            assert tracer.image_plane.positions[0] == pytest.approx(np.array([[1.0, 1.0], [-1.0, -1.0]]), 1e-3)
            assert tracer.image_plane.deflections[0] == pytest.approx(np.array([[0.0, 0.0], [0.0, 0.0]]), 1e-3)
            assert tracer.source_plane.positions[0] == pytest.approx(np.array([[1.0, 1.0], [-1.0, -1.0]]), 1e-3)

        def test__x2_positions__sis_lens__positions_with_source_plane_deflected(self, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=[galaxy_mass],
                                                                  positions=[np.array([[1.0, 1.0], [-1.0, -1.0]])])

            assert tracer.image_plane.positions[0] == pytest.approx(np.array([[1.0, 1.0], [-1.0, -1.0]]), 1e-3)
            assert tracer.image_plane.deflections[0] == pytest.approx(np.array([[0.707, 0.707], [-0.707, -0.707]]),
                                                                      1e-3)
            assert tracer.source_plane.positions[0] == pytest.approx(np.array([[1.0 - 0.707, 1.0 - 0.707],
                                                                               [-1.0 + 0.707, -1.0 + 0.707]]), 1e-3)

        def test__same_as_above_but_2_sis_lenses__deflections_double(self, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=[galaxy_mass, galaxy_mass],
                                                                  positions=[np.array([[1.0, 1.0], [-1.0, -1.0]])])

            assert tracer.image_plane.positions[0] == pytest.approx(np.array([[1.0, 1.0], [-1.0, -1.0]]), 1e-3)
            assert tracer.image_plane.deflections[0] == pytest.approx(np.array([[1.414, 1.414], [-1.414, -1.414]]),
                                                                      1e-3)
            assert tracer.source_plane.positions[0] == pytest.approx(np.array([[1.0 - 1.414, 1.0 - 1.414],
                                                                               [-1.0 + 1.414, -1.0 + 1.414]]), 1e-3)

        def test__multiple_sets_of_positions_in_different_arrays(self, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=[galaxy_mass],
                                                                  positions=[np.array([[1.0, 1.0], [-1.0, -1.0]]),
                                                                             np.array([[0.5, 0.5]])])

            assert tracer.image_plane.positions[0] == pytest.approx(np.array([[1.0, 1.0], [-1.0, -1.0]]), 1e-3)
            assert tracer.image_plane.deflections[0] == pytest.approx(np.array([[0.707, 0.707], [-0.707, -0.707]]),
                                                                      1e-3)
            assert tracer.source_plane.positions[0] == pytest.approx(np.array([[1.0 - 0.707, 1.0 - 0.707],
                                                                               [-1.0 + 0.707, -1.0 + 0.707]]), 1e-3)

            assert tracer.image_plane.positions[1] == pytest.approx(np.array([[0.5, 0.5]]), 1e-3)
            assert tracer.image_plane.deflections[1] == pytest.approx(np.array([[0.707, 0.707]]), 1e-3)
            assert tracer.source_plane.positions[1] == pytest.approx(np.array([[0.5 - 0.707, 0.5 - 0.707]]), 1e-3)


class TestTracerMultiPositions(object):

    class TestRayTracingPlanes:

        def test__4_planes__coordinate_data_grids_and_deflections_are_correct__sis_mass_profile(self):
            import math

            g0 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g2 = g.Galaxy(redshift=0.1, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g3 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g4 = g.Galaxy(redshift=1.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g5 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            tracer = ray_tracing.TracerMultiPositions(galaxies=[g0, g1, g2, g3, g4, g5],
                                                      positions=[np.array([[1.0, 1.0]])], cosmology=cosmo.Planck15)

            # From unit test below:
            # Beta_01 = 0.9348
            # Beta_02 = 0.9840
            # Beta_03 = 1.0
            # Beta_12 = 0.754
            # Beta_13 = 1.0
            # Beta_23 = 1.0

            val = math.sqrt(2) / 2.0

            assert tracer.planes[0].positions[0] == pytest.approx(np.array([[1.0, 1.0]]), 1e-4)
            assert tracer.planes[0].deflections[0] == pytest.approx(np.array([[val, val]]), 1e-4)

            assert tracer.planes[1].positions[0] == pytest.approx(
                np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]), 1e-4)

            defl11 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]))

            assert tracer.planes[1].deflections[0] == pytest.approx(defl11[[0]], 1e-4)

            assert tracer.planes[2].positions[0] == pytest.approx(
                np.array([[(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                           (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]]), 1e-4)

            # 2 Galaxies in this plane, so multiply by 2.0

            defl21 = 2.0 * g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                                                                    (1.0 - 0.9839601 * val - 0.7539734 * defl11[
                                                                        0, 1])]]))

            assert tracer.planes[2].deflections[0] == pytest.approx(defl21[[0]], 1e-4)

            coord1 = (1.0 - tracer.planes[0].deflections[0][0, 0] - tracer.planes[1].deflections[0][0, 0] -
                      tracer.planes[2].deflections[0][0, 0])

            coord2 = (1.0 - tracer.planes[0].deflections[0][0, 1] - tracer.planes[1].deflections[0][0, 1] -
                      tracer.planes[2].deflections[0][0, 1])

            assert tracer.planes[3].positions[0] == pytest.approx(np.array([[coord1, coord2]]), 1e-4)

        def test__same_as_above_but_multiple_sets_of_positions(self):
            import math

            g0 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g2 = g.Galaxy(redshift=0.1, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g3 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g4 = g.Galaxy(redshift=1.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g5 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            tracer = ray_tracing.TracerMultiPositions(galaxies=[g0, g1, g2, g3, g4, g5],
                                                      positions=[np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]])],
                                                      cosmology=cosmo.Planck15)

            # From unit test below:
            # Beta_01 = 0.9348
            # Beta_02 = 0.9840
            # Beta_03 = 1.0
            # Beta_12 = 0.754
            # Beta_13 = 1.0
            # Beta_23 = 1.0

            val = math.sqrt(2) / 2.0

            assert tracer.planes[0].positions[0] == pytest.approx(np.array([[1.0, 1.0]]), 1e-4)
            assert tracer.planes[0].deflections[0] == pytest.approx(np.array([[val, val]]), 1e-4)

            assert tracer.planes[1].positions[0] == pytest.approx(
                np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]), 1e-4)

            defl11 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]))

            assert tracer.planes[1].deflections[0] == pytest.approx(defl11[[0]], 1e-4)

            assert tracer.planes[2].positions[0] == pytest.approx(
                np.array([[(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                           (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]]), 1e-4)

            # 2 Galaxies in this plane, so multiply by 2.0

            defl21 = 2.0 * g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                                                                    (1.0 - 0.9839601 * val - 0.7539734 * defl11[
                                                                        0, 1])]]))

            assert tracer.planes[2].deflections[0] == pytest.approx(defl21[[0]], 1e-4)

            coord1 = (1.0 - tracer.planes[0].deflections[0][0, 0] - tracer.planes[1].deflections[0][0, 0] -
                      tracer.planes[2].deflections[0][0, 0])

            coord2 = (1.0 - tracer.planes[0].deflections[0][0, 1] - tracer.planes[1].deflections[0][0, 1] -
                      tracer.planes[2].deflections[0][0, 1])

            assert tracer.planes[3].positions[0] == pytest.approx(np.array([[coord1, coord2]]), 1e-4)

            assert tracer.planes[0].positions[1] == pytest.approx(np.array([[1.0, 1.0]]), 1e-4)
            assert tracer.planes[0].deflections[1] == pytest.approx(np.array([[val, val]]), 1e-4)

            assert tracer.planes[1].positions[1] == pytest.approx(
                np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]), 1e-4)

            defl11 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]))

            assert tracer.planes[1].deflections[1] == pytest.approx(defl11[[0]], 1e-4)
            assert tracer.planes[2].positions[1] == pytest.approx(
                np.array([[(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                           (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]]), 1e-4)

            # 2 Galaxies in this plane, so multiply by 2.0

            defl21 = 2.0 * g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                                                                    (1.0 - 0.9839601 * val - 0.7539734 * defl11[
                                                                        0, 1])]]))

            assert tracer.planes[2].deflections[1] == pytest.approx(defl21[[0]], 1e-4)

            coord1 = (1.0 - tracer.planes[0].deflections[1][0, 0] - tracer.planes[1].deflections[1][0, 0] -
                      tracer.planes[2].deflections[1][0, 0])

            coord2 = (1.0 - tracer.planes[0].deflections[1][0, 1] - tracer.planes[1].deflections[1][0, 1] -
                      tracer.planes[2].deflections[1][0, 1])

            assert tracer.planes[3].positions[1] == pytest.approx(np.array([[coord1, coord2]]), 1e-4)
