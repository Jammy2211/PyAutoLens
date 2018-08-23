from autolens import exc
from autolens.analysis import ray_tracing, galaxy
from autolens.profiles import mass_profiles, light_profiles
from autolens.pixelization import pixelization
from astropy import cosmology as cosmo
from autolens.imaging import mask

import pytest
import numpy as np


@pytest.fixture(name="centre_mask")
def make_centre_mask():
    return mask.Mask(np.array([[True, True, True],
                               [True, False, True],
                               [True, True, True]]))


@pytest.fixture(name="grids")
def make_grids(centre_mask):
    grids = mask.GridCollection.from_mask_sub_grid_size_and_blurring_shape(centre_mask, 2, (3, 3))
    grids.image = mask.ImageGrid(np.array([[1.0, 1.0], [1.0, 0.0]]))
    grids.sub = mask.SubGrid(
        np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0],
                  [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]]),
        mask=np.array([[True, True, True],
                       [True, False, True],
                       [True, True, True]]), sub_grid_size=2)
    grids.blurring = mask.ImageGrid(np.array([[1.0, 0.0], [1.0, 0.0]]))
    return grids


@pytest.fixture(name="no_galaxies")
def make_no_galaxies():
    return [galaxy.Galaxy()]


@pytest.fixture(name='galaxy_no_profiles', scope='function')
def make_galaxy_no_profiles():
    return galaxy.Galaxy()


@pytest.fixture(name="galaxy_light_sersic")
def make_galaxy_light_sersic():
    sersic = light_profiles.EllipticalSersicLP(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                               sersic_index=4.0)
    return galaxy.Galaxy(light_profile=sersic)


@pytest.fixture(name="galaxy_mass_sis")
def make_galaxy_mass_sis():
    sis = mass_profiles.SphericalIsothermalMP(einstein_radius=1.0)
    return galaxy.Galaxy(mass_profile=sis)


@pytest.fixture(name='lens_sis_x3')
def make_lens_sis_x3():
    mass_profile = mass_profiles.SphericalIsothermalMP(einstein_radius=1.0)
    return galaxy.Galaxy(mass_profile_1=mass_profile, mass_profile_2=mass_profile,
                         mass_profile_3=mass_profile)


@pytest.fixture(name="sparse_mask")
def make_sparse_mask():
    return mask.SparseMask(np.array([[0, 0]]), 1)


@pytest.fixture(name="sparse_mask")
def make_sparse_mask():
    return None


@pytest.fixture(name="galaxy_light_only")
def make_galaxy_light_only():
    return galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP())


@pytest.fixture(name="galaxy_light_and_mass")
def make_galaxy_light_and_mass():
    return galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(),
                         mass_profile=mass_profiles.SphericalIsothermalMP())


class MockMassProfile(object):

    def __init__(self, value):
        self.value = value


class MockMapping(object):

    def __init__(self):
        pass


class MockPixelization(object):

    def __init__(self, value):
        self.value = value

    # noinspection PyUnusedLocal,PyShadowingNames
    def reconstructor_from_pixelization_and_grids(self, grids, borders, sparse_mask):
        return self.value


class MockBorders(object):

    def __init__(self, image=None, sub=None):
        self.image = image
        self.sub = sub


class TestProperties(object):
    def test_tracer(self, grids):
        tracer = ray_tracing.TracerImageSourcePlanes([galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy())],
                                              [galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy())],
                                              grids)

        assert tracer.image_plane.hyper_galaxies == [galaxy.HyperGalaxy()]
        assert tracer.source_plane.hyper_galaxies == [galaxy.HyperGalaxy()]

        assert tracer.hyper_galaxies == [galaxy.HyperGalaxy(), galaxy.HyperGalaxy()]

    def test_tracer__hyper_galaxies_with_none_are_filtered(self, grids):
        tracer = ray_tracing.TracerImageSourcePlanes([galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy()), galaxy.Galaxy()],
                                              [galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy()), galaxy.Galaxy(), galaxy.Galaxy()],
                                              grids)

        assert tracer.image_plane.hyper_galaxies == [galaxy.HyperGalaxy()]
        assert tracer.source_plane.hyper_galaxies == [galaxy.HyperGalaxy()]

        assert tracer.hyper_galaxies == [galaxy.HyperGalaxy(), galaxy.HyperGalaxy()]

    def test_multi_tracer(self, grids):
        tracer = ray_tracing.TracerMulti([galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy(2), redshift=2),
                                          galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy(1), redshift=1)], grids,
                                         cosmo.Planck15)

        assert tracer.hyper_galaxies == [galaxy.HyperGalaxy(1), galaxy.HyperGalaxy(2)]

    def test_all_with_hyper_galaxies_tracer(self, grids):
        tracer = ray_tracing.TracerImageSourcePlanes([galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy())],
                                              [galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy())],
                                              grids)

        assert tracer.all_with_hyper_galaxies

        tracer = ray_tracing.TracerImageSourcePlanes([galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy())],
                                              [galaxy.Galaxy()],
                                              grids)

        assert not tracer.all_with_hyper_galaxies

    def test_all_with_hyper_galaxies_multi_tracer(self, grids):
        tracer = ray_tracing.TracerMulti([galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy(2), redshift=2),
                                          galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy(1), redshift=1)], grids,
                                         cosmo.Planck15)

        assert tracer.all_with_hyper_galaxies

        tracer = ray_tracing.TracerMulti([galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy(2), redshift=2),
                                          galaxy.Galaxy(redshift=1)], grids,
                                         cosmo.Planck15)

        assert not tracer.all_with_hyper_galaxies


@pytest.fixture(name="light_only_source_plane")
def make_light_only_source_plane(galaxy_light_only, grids):
    return ray_tracing.Plane(galaxies=[galaxy_light_only], grids=grids,
                             compute_deflections=False)


@pytest.fixture(name="light_only_image_plane")
def make_light_only_image_plane(galaxy_light_only, grids):
    return ray_tracing.Plane(galaxies=[galaxy_light_only], grids=grids,
                             compute_deflections=True)


@pytest.fixture(name="light_only_tracer")
def make_light_only_tracer(galaxy_light_only, grids):
    return ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_light_only],
                                        source_galaxies=[galaxy_light_only],
                                        image_plane_grids=grids)


class TestPlane(object):


    class TestBasicSetup:

        def test__sis_lens__grids_and_deflections_setup_for_image_sub_and_blurring_grids(self, grids, galaxy_mass_sis):

            plane = ray_tracing.Plane(galaxies=[galaxy_mass_sis], grids=grids)

            assert plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert plane.grids.sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert plane.grids.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert plane.grids.sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert plane.grids.sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test__same_as_abvove_but_x3_sis_lenses__deflections_triple(self, grids, galaxy_mass_sis):

            lens_plane = ray_tracing.Plane(galaxies=[galaxy_mass_sis, galaxy_mass_sis, galaxy_mass_sis],
                                           grids=grids, compute_deflections=True)

            assert lens_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert lens_plane.deflections.image[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflections.sub[2] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]),1e-3)
            assert lens_plane.deflections.sub[3] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflections.blurring[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test__same_as_above_but_galaxy_is_x3_sis__deflections_tripled_again(self, grids, lens_sis_x3):

            lens_plane = ray_tracing.Plane(galaxies=[lens_sis_x3], grids=grids,
                                           compute_deflections=True)

            assert lens_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert lens_plane.deflections.image[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflections.sub[2] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[3] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflections.blurring[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test__all_grids__complex_mass_model(self, grids):

            power_law = mass_profiles.EllipticalPowerLawMP(centre=(1.0, 4.0), axis_ratio=0.7, phi=30.0,
                                                           einstein_radius=1.0, slope=2.2)

            nfw = mass_profiles.SphericalNFWMP(kappa_s=0.1, scale_radius=5.0)

            lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profile_1=power_law, mass_profile_2=nfw)

            lens_plane = ray_tracing.Plane(galaxies=[lens_galaxy], grids=grids, compute_deflections=True)

            assert lens_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)


    class TestImagesFromPlane:

        def test__image_from_plane__same_as_its_light_profile_image(self, grids, galaxy_light_sersic):

            lp = galaxy_light_sersic.light_profiles[0]

            lp_sub_image = lp.intensity_from_grid(grids.sub)

            # Perform sub gridding average manually
            lp_image_pixel_0 = (lp_sub_image[0] + lp_sub_image[1] + lp_sub_image[2] + lp_sub_image[3]) / 4
            lp_image_pixel_1 = (lp_sub_image[4] + lp_sub_image[5] + lp_sub_image[6] + lp_sub_image[7]) / 4

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=grids)

            assert (plane.image_plane_image[0] == lp_image_pixel_0).all()
            assert (plane.image_plane_image[1] == lp_image_pixel_1).all()
            assert (plane.image_plane_images_of_galaxies[0][0] == lp_image_pixel_0).all()
            assert (plane.image_plane_images_of_galaxies[0][1] == lp_image_pixel_1).all()

        def test__same_as_above__use_multiple_sets_of_coordinates(self, grids, galaxy_light_sersic):

            # Overwrite one value so intensity in each pixel is different
            grids.sub[5] = np.array([2.0, 2.0])

            lp = galaxy_light_sersic.light_profiles[0]

            lp_sub_image = lp.intensity_from_grid(grids.sub)

            # Perform sub gridding average manually
            lp_image_pixel_0 = (lp_sub_image[0] + lp_sub_image[1] + lp_sub_image[2] + lp_sub_image[3]) / 4
            lp_image_pixel_1 = (lp_sub_image[4] + lp_sub_image[5] + lp_sub_image[6] + lp_sub_image[7]) / 4

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=grids)

            assert (plane.image_plane_image[0] == lp_image_pixel_0).all()
            assert (plane.image_plane_image[1] == lp_image_pixel_1).all()
            assert (plane.image_plane_images_of_galaxies[0][0] == lp_image_pixel_0).all()
            assert (plane.image_plane_images_of_galaxies[0][1] == lp_image_pixel_1).all()
            
        def test__same_as_above__use_multiple_galaxies(self, grids):

            # Overwrite one value so intensity in each pixel is different
            grids.sub[5] = np.array([2.0, 2.0])

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))

            lp0 = g0.light_profiles[0]
            lp1 = g1.light_profiles[0]

            lp0_sub_image = lp0.intensity_from_grid(grids.sub)
            lp1_sub_image = lp1.intensity_from_grid(grids.sub)

            # Perform sub gridding average manually
            lp0_image_pixel_0 = (lp0_sub_image[0] + lp0_sub_image[1] + lp0_sub_image[2] + lp0_sub_image[3]) / 4
            lp0_image_pixel_1 = (lp0_sub_image[4] + lp0_sub_image[5] + lp0_sub_image[6] + lp0_sub_image[7]) / 4
            lp1_image_pixel_0 = (lp1_sub_image[0] + lp1_sub_image[1] + lp1_sub_image[2] + lp1_sub_image[3]) / 4
            lp1_image_pixel_1 = (lp1_sub_image[4] + lp1_sub_image[5] + lp1_sub_image[6] + lp1_sub_image[7]) / 4

            plane = ray_tracing.Plane(galaxies=[g0, g1], grids=grids)

            assert (plane.image_plane_image[0] == lp0_image_pixel_0 + lp1_image_pixel_0).all()
            assert (plane.image_plane_image[1] == lp0_image_pixel_1 + lp1_image_pixel_1).all()
            assert (plane.image_plane_images_of_galaxies[0][0] == lp0_image_pixel_0).all()
            assert (plane.image_plane_images_of_galaxies[1][0] == lp1_image_pixel_0).all()
            assert (plane.image_plane_images_of_galaxies[0][1] == lp0_image_pixel_1).all()
            assert (plane.image_plane_images_of_galaxies[1][1] == lp1_image_pixel_1).all()

        def test__image_from_plane__same_as_its_galaxy_image(self, grids, galaxy_light_sersic):

            galaxy_image = ray_tracing.intensities_via_sub_grid(grids.sub, galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=grids)

            assert (plane.image_plane_image == galaxy_image).all()
            assert (plane.image_plane_images_of_galaxies[0] == galaxy_image).all()

        def test__same_as_above_galaxies__use_multiple_sets_of_coordinates(self, grids, galaxy_light_sersic):

            # Overwrite one value so intensity in each pixel is different
            grids.sub[5] = np.array([2.0, 2.0])

            galaxy_image = ray_tracing.intensities_via_sub_grid(grids.sub, galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=grids)

            assert (plane.image_plane_image == galaxy_image).all()
            assert (plane.image_plane_images_of_galaxies[0] == galaxy_image).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, grids):

            # Overwrite one value so intensity in each pixel is different
            grids.sub[5] = np.array([2.0, 2.0])

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))

            g0_image = ray_tracing.intensities_via_sub_grid(grids.sub, galaxies=[g0])
            g1_image = ray_tracing.intensities_via_sub_grid(grids.sub, galaxies=[g1])

            plane = ray_tracing.Plane(galaxies=[g0, g1], grids=grids)

            assert (plane.image_plane_image == g0_image + g1_image).all()
            assert (plane.image_plane_images_of_galaxies[0] == g0_image).all()
            assert (plane.image_plane_images_of_galaxies[1] == g1_image).all()


        def test__same_as_above__galaxy_entered_3_times__diffferent_intensities_for_each(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))
            g2 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=3.0))

            g0_image = ray_tracing.intensities_via_sub_grid(grids.sub, galaxies=[g0])
            g1_image = ray_tracing.intensities_via_sub_grid(grids.sub, galaxies=[g1])
            g2_image = ray_tracing.intensities_via_sub_grid(grids.sub, galaxies=[g2])

            plane = ray_tracing.Plane(galaxies=[g0, g1, g2], grids=grids)

            assert (plane.image_plane_image == g0_image + g1_image + g2_image).all()
            assert (plane.image_plane_images_of_galaxies[0] == g0_image).all()
            assert (plane.image_plane_images_of_galaxies[1] == g1_image).all()
            assert (plane.image_plane_images_of_galaxies[2] == g2_image).all()


    class TestBlurringImageFromPlane:

        def test__image_from_plane__same_as_its_light_profile_image(self, grids, galaxy_light_sersic):
            
            lp = galaxy_light_sersic.light_profiles[0]

            lp_blurring_image = lp.intensity_from_grid(grids.blurring)

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=grids)

            assert (plane.image_plane_blurring_image == lp_blurring_image).all()
            assert (plane.image_plane_blurring_images_of_galaxies[0] == lp_blurring_image).all()

        def test__same_as_above__use_multiple_sets_of_coordinates(self, grids, galaxy_light_sersic):

            # Overwrite one value so intensity in each pixel is different
            grids.blurring[1] = np.array([2.0, 2.0])

            lp = galaxy_light_sersic.light_profiles[0]

            lp_blurring_image = lp.intensity_from_grid(grids.blurring)

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=grids)

            assert (plane.image_plane_blurring_image == lp_blurring_image).all()
            assert (plane.image_plane_blurring_images_of_galaxies[0] == lp_blurring_image).all()

        def test__same_as_above__use_multiple_galaxies(self, grids):

            # Overwrite one value so intensity in each pixel is different
            grids.blurring[1] = np.array([2.0, 2.0])

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))

            lp0 = g0.light_profiles[0]
            lp1 = g1.light_profiles[0]

            lp0_blurring_image = lp0.intensity_from_grid(grids.blurring)
            lp1_blurring_image = lp1.intensity_from_grid(grids.blurring)

            plane = ray_tracing.Plane(galaxies=[g0, g1], grids=grids)

            assert (plane.image_plane_blurring_image == lp0_blurring_image + lp1_blurring_image).all()
            assert (plane.image_plane_blurring_images_of_galaxies[0] == lp0_blurring_image).all()
            assert (plane.image_plane_blurring_images_of_galaxies[1] == lp1_blurring_image).all()

        def test__image_from_plane__same_as_its_galaxy_image(self, grids, galaxy_light_sersic):

            galaxy_image = ray_tracing.intensities_via_grid(grids.blurring, galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=grids)

            assert (plane.image_plane_blurring_image == galaxy_image).all()
            assert (plane.image_plane_blurring_images_of_galaxies[0] == galaxy_image).all()

        def test__same_as_above_galaxies__use_multiple_sets_of_coordinates(self, grids, galaxy_light_sersic):

            # Overwrite one value so intensity in each pixel is different
            grids.blurring[1] = np.array([2.0, 2.0])

            galaxy_image = ray_tracing.intensities_via_grid(grids.blurring, galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=grids)

            assert (plane.image_plane_blurring_image == galaxy_image).all()
            assert (plane.image_plane_blurring_images_of_galaxies[0] == galaxy_image).all()

        def test__same_as_above_galaxies___use_multiple_galaxies(self, grids):

            # Overwrite one value so intensity in each pixel is different
            grids.blurring[1] = np.array([2.0, 2.0])

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))

            g0_image = ray_tracing.intensities_via_grid(grids.blurring, galaxies=[g0])
            g1_image = ray_tracing.intensities_via_grid(grids.blurring, galaxies=[g1])

            plane = ray_tracing.Plane(galaxies=[g0, g1], grids=grids)

            assert (plane.image_plane_blurring_image == g0_image + g1_image).all()
            assert (plane.image_plane_blurring_images_of_galaxies[0] == g0_image).all()
            assert (plane.image_plane_blurring_images_of_galaxies[1] == g1_image).all()

        def test__same_as_above__galaxy_entered_3_times__diffferent_intensities_for_each(self, grids):
            
            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))
            g2 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=3.0))

            g0_image = ray_tracing.intensities_via_grid(grids.blurring, galaxies=[g0])
            g1_image = ray_tracing.intensities_via_grid(grids.blurring, galaxies=[g1])
            g2_image = ray_tracing.intensities_via_grid(grids.blurring, galaxies=[g2])

            plane = ray_tracing.Plane(galaxies=[g0, g1, g2], grids=grids)

            assert (plane.image_plane_blurring_image == g0_image + g1_image + g2_image).all()
            assert (plane.image_plane_blurring_images_of_galaxies[0] == g0_image).all()
            assert (plane.image_plane_blurring_images_of_galaxies[1] == g1_image).all()
            assert (plane.image_plane_blurring_images_of_galaxies[2] == g2_image).all()


    class TestPlaneImage:

        def test__shape_3x3__image_of_plane__same_as_light_profile_on_identicla_uniform_grid(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))

            g0_image = g0.intensity_from_grid(grid=np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                                                 [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                                                 [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]]))

            grids.image = np.array([[-1.5, -1.5], [1.5, 1.5]])

            plane = ray_tracing.Plane(galaxies=[g0], grids=grids)

            assert plane.plane_image(shape=(3,3)) == pytest.approx(g0_image, 1e-4)

        def test__different_shape_and_multiple_galaxies(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))

            g0_image = g0.intensity_from_grid(grid=np.array([[-0.75, -1.0], [-0.75, 0.0], [-0.75, 1.0],
                                                             [0.75, -1.0], [0.75, 0.0], [0.75, 1.0]]))

            g1_image = g1.intensity_from_grid(grid=np.array([[-0.75, -1.0], [-0.75, 0.0], [-0.75, 1.0],
                                                             [0.75, -1.0], [0.75, 0.0], [0.75, 1.0]]))

            grids.image = np.array([[-1.5, -1.5], [1.5, 1.5]])

            plane = ray_tracing.Plane(galaxies=[g0, g1], grids=grids)

            assert plane.plane_image(shape=(2,3)) == pytest.approx(g0_image + g1_image, 1e-4)
            assert plane.plane_images_of_galaxies(shape=(2,3))[0] == pytest.approx(g0_image, 1e-4)
            assert plane.plane_images_of_galaxies(shape=(2,3))[1] == pytest.approx(g1_image, 1e-4)


    class TestReconstructorFromGalaxies:

        def test__no_galaxies_with_pixelizations_in_plane__returns_none(self, grids, sparse_mask):
            galaxy_no_pix = galaxy.Galaxy()

            plane = ray_tracing.Plane(galaxies=[galaxy_no_pix], grids=grids)

            reconstructors = plane.reconstructor_from_plane(MockBorders(), sparse_mask)

            assert reconstructors is None

        def test__1_galaxy_in_plane__it_has_pixelization__extracts_reconstructor(self, grids, sparse_mask):
            galaxy_pix = galaxy.Galaxy(pixelization=MockPixelization(value=1))

            plane = ray_tracing.Plane(galaxies=[galaxy_pix], grids=grids)

            reconstructors = plane.reconstructor_from_plane(MockBorders(), sparse_mask)

            assert reconstructors == 1

        def test__2_galaxies_in_plane__1_has_pixelization__extracts_reconstructor(self, grids, sparse_mask):
            galaxy_pix = galaxy.Galaxy(pixelization=MockPixelization(value=1))
            galaxy_no_pix = galaxy.Galaxy()

            plane = ray_tracing.Plane(galaxies=[galaxy_no_pix, galaxy_pix],
                                      grids=grids)

            reconstructors = plane.reconstructor_from_plane(MockBorders(), sparse_mask)

            assert reconstructors == 1

        def test__2_galaxies_in_plane__both_have_pixelization__raises_error(self, grids, sparse_mask):
            galaxy_pix_0 = galaxy.Galaxy(pixelization=MockPixelization(value=1))
            galaxy_pix_1 = galaxy.Galaxy(pixelization=MockPixelization(value=2))

            plane = ray_tracing.Plane(galaxies=[galaxy_pix_0, galaxy_pix_1],
                                      grids=grids)

            with pytest.raises(exc.PixelizationException):
                plane.reconstructor_from_plane(MockBorders(), sparse_mask)


class TestTracerImageAndSource(object):


    class TestSetup:

        def test__image_grid__no_galaxy__image_and_source_planes_setup__same_coordinates(self, grids, no_galaxies):

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=no_galaxies, source_galaxies=no_galaxies,
                                                  image_plane_grids=grids)

            assert tracer.image_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.deflections.image[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.source_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)

        def test__image_grid__sis_lens__image_grid_with_source_plane_deflected(self, grids, galaxy_mass_sis):

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass_sis], source_galaxies=galaxy_mass_sis,
                                                  image_plane_grids=grids)

            assert tracer.image_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.deflections.image[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert tracer.source_plane.grids.image[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)

        def test__same_as_above_but_2_sis_lenses__deflections_double(self, grids, galaxy_mass_sis):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass_sis, galaxy_mass_sis],
                                                  source_galaxies=galaxy_mass_sis, image_plane_grids=grids)

            assert tracer.image_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.deflections.image[0] == pytest.approx(np.array([1.414, 1.414]), 1e-3)
            assert tracer.source_plane.grids.image[0] == pytest.approx(np.array([1.0 - 1.414, 1.0 - 1.414]), 1e-3)

        def test__all_grids__sis_lens__image_sub_and_blurring_grids_on_planes_setup(self, grids, galaxy_mass_sis):

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass_sis], source_galaxies=galaxy_mass_sis,
                                                  image_plane_grids=grids)

            assert tracer.image_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]),
                                                                                  1e-3)
            assert tracer.image_plane.grids.sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grids.sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grids.sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.image_plane.deflections.image[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert tracer.image_plane.deflections.sub[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert tracer.image_plane.deflections.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections.sub[2] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert tracer.image_plane.deflections.sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.source_plane.grids.image[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)
            assert tracer.source_plane.grids.sub[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)
            assert tracer.source_plane.grids.sub[1] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.source_plane.grids.sub[2] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)
            assert tracer.source_plane.grids.sub[3] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.source_plane.grids.blurring[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)

        def test__no_source_galaxies_in_x2_tracer__raises_excetion(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0),
                               mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))

            with pytest.raises(exc.RayTracingException):
                tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[],
                                                             image_plane_grids=grids)


    class TestImagePlaneImages:

        def test__no_galaxies__image_plane_image__sum_of_image_and_source_plane_images(self, grids, no_galaxies):

            image_plane = ray_tracing.Plane(galaxies=no_galaxies, grids=grids, compute_deflections=True)
            source_plane = ray_tracing.Plane(galaxies=no_galaxies, grids=grids, compute_deflections=False)

            image_plane_image = image_plane.image_plane_image + source_plane.image_plane_image

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=no_galaxies, source_galaxies=no_galaxies, image_plane_grids=grids)

            assert (image_plane_image == tracer.image_plane_image).all()

        def test__galaxy_light__no_mass__image_sum_of_image_and_source_plane(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))

            image_plane = ray_tracing.Plane(galaxies=[g0], grids=grids, compute_deflections=True)
            source_plane = ray_tracing.Plane(galaxies=[g1], grids=grids, compute_deflections=False)

            plane_image_plane_image = image_plane.image_plane_image + source_plane.image_plane_image

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=grids)

            assert (plane_image_plane_image == tracer.image_plane_image).all()

        def test__galaxy_light_sersic_mass_sis__source_plane_image_includes_deflections(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0),
                               mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))

            image_plane = ray_tracing.Plane(galaxies=[g0], grids=grids, compute_deflections=True)
            
            deflections_grid = ray_tracing.deflections_for_grids(grids, galaxies=[g0])
            source_grid = ray_tracing.traced_collection_for_deflections(grids, deflections_grid)
            source_plane = ray_tracing.Plane(galaxies=[g1], grids=source_grid, compute_deflections=False)

            plane_image_plane_image = image_plane.image_plane_image + source_plane.image_plane_image

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1], image_plane_grids=grids)

            assert (plane_image_plane_image == tracer.image_plane_image).all()

        def test__galaxy_entered_3_times__diffferent_intensities_for_each(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))
            g2 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=3.0))

            g0_image = ray_tracing.intensities_via_sub_grid(grids.sub, galaxies=[g0])
            g1_image = ray_tracing.intensities_via_sub_grid(grids.sub, galaxies=[g1])
            g2_image = ray_tracing.intensities_via_sub_grid(grids.sub, galaxies=[g2])

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2], image_plane_grids=grids)

            assert (tracer.image_plane_image == g0_image + g1_image + g2_image).all()
            assert (tracer.image_plane_images_of_galaxies[0] == g0_image).all()
            assert (tracer.image_plane_images_of_galaxies[1] == g1_image).all()
            assert (tracer.image_plane_images_of_galaxies[2] == g2_image).all()

        def test__2_planes__returns_image_plane_image_of_each_plane(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0),
                               mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))

            image_plane = ray_tracing.Plane(galaxies=[g0], grids=grids, compute_deflections=True)
            source_plane_grids = image_plane.trace_to_next_plane()
            source_plane = ray_tracing.Plane(galaxies=[g0], grids=source_plane_grids, compute_deflections=False)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0], image_plane_grids=grids)

            assert (tracer.image_plane_image == image_plane.image_plane_image + source_plane.image_plane_image).all()
            assert (tracer.image_plane_images_of_planes[0] == image_plane.image_plane_image).all()
            assert (tracer.image_plane_images_of_planes[1] == source_plane.image_plane_image).all()

        def test__1_plane__single_plane_tracer(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))
            g2 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=3.0))

            image_plane = ray_tracing.Plane(galaxies=[g0, g1, g2], grids=grids, compute_deflections=True)

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1, g2], image_plane_grids=grids)

            assert (tracer.image_plane_image == image_plane.image_plane_image).all()
            assert (sum(tracer.image_plane_images_of_galaxies) == image_plane.image_plane_image).all()
            assert (tracer.image_plane_images_of_galaxies[0] == image_plane.image_plane_images_of_galaxies[0]).all()
            assert (tracer.image_plane_images_of_galaxies[1] == image_plane.image_plane_images_of_galaxies[1]).all()
            assert (tracer.image_plane_images_of_galaxies[2] == image_plane.image_plane_images_of_galaxies[2]).all()


    class TestImagePlaneBlurringImages:

        def test__no_galaxies__image_plane_blurring_image__sum_of_image_and_source_plane_images(self, grids,
                                                                                                no_galaxies):

            image_plane = ray_tracing.Plane(galaxies=no_galaxies, grids=grids, compute_deflections=True)
            source_plane = ray_tracing.Plane(galaxies=no_galaxies, grids=grids, compute_deflections=False)

            image_plane_blurring_image = image_plane.image_plane_blurring_image + source_plane.image_plane_blurring_image

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=no_galaxies, source_galaxies=no_galaxies, image_plane_grids=grids)

            assert (image_plane_blurring_image == tracer.image_plane_blurring_image).all()

        def test__galaxy_light__no_mass__image_sum_of_image_and_source_plane(self, grids):
            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))

            image_plane = ray_tracing.Plane(galaxies=[g0], grids=grids, compute_deflections=True)
            source_plane = ray_tracing.Plane(galaxies=[g1], grids=grids, compute_deflections=False)

            plane_image_plane_blurring_image = image_plane.image_plane_blurring_image + \
                                               source_plane.image_plane_blurring_image

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1], image_plane_grids=grids)

            assert (plane_image_plane_blurring_image == tracer.image_plane_blurring_image).all()

        def test__galaxy_light_sersic_mass_sis__source_plane_image_includes_deflections(self, grids):
            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0),
                               mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))

            image_plane = ray_tracing.Plane(galaxies=[g0], grids=grids, compute_deflections=True)

            deflections_grid = ray_tracing.deflections_for_grids(grids, galaxies=[g0])
            source_grid = ray_tracing.traced_collection_for_deflections(grids, deflections_grid)
            source_plane = ray_tracing.Plane(galaxies=[g1], grids=source_grid, compute_deflections=False)

            plane_image_plane_blurring_image = image_plane.image_plane_blurring_image + \
                                               source_plane.image_plane_blurring_image

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1], image_plane_grids=grids)

            assert (plane_image_plane_blurring_image == tracer.image_plane_blurring_image).all()

        def test__galaxy_entered_3_times__diffferent_intensities_for_each(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))
            g2 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=3.0))

            g0_image = ray_tracing.intensities_via_grid(grids.blurring, galaxies=[g0])
            g1_image = ray_tracing.intensities_via_grid(grids.blurring, galaxies=[g1])
            g2_image = ray_tracing.intensities_via_grid(grids.blurring, galaxies=[g2])

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2], image_plane_grids=grids)

            assert (tracer.image_plane_blurring_image == g0_image + g1_image + g2_image).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[0] == g0_image).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[1] == g1_image).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[2] == g2_image).all()

        def test__2_planes__returns_image_plane_blurring_image_of_each_plane(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0),
                               mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))

            image_plane = ray_tracing.Plane(galaxies=[g0], grids=grids, compute_deflections=True)
            source_plane_grids = image_plane.trace_to_next_plane()
            source_plane = ray_tracing.Plane(galaxies=[g0], grids=source_plane_grids, compute_deflections=False)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0], image_plane_grids=grids)

            assert (tracer.image_plane_blurring_image == image_plane.image_plane_blurring_image + source_plane.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_planes[0] == image_plane.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_planes[1] == source_plane.image_plane_blurring_image).all()

        def test__1_plane__single_plane_tracer(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))
            g2 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=3.0))

            image_plane = ray_tracing.Plane(galaxies=[g0, g1, g2], grids=grids, compute_deflections=True)

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1, g2], image_plane_grids=grids)

            assert (tracer.image_plane_blurring_image == image_plane.image_plane_blurring_image).all()
            assert (sum(tracer.image_plane_blurring_images_of_galaxies) == image_plane.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[0] ==
                    image_plane.image_plane_blurring_images_of_galaxies[0]).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[1] ==
                    image_plane.image_plane_blurring_images_of_galaxies[1]).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[2] ==
                    image_plane.image_plane_blurring_images_of_galaxies[2]).all()


    class TestPlaneImages:

        def test__galaxy_light_sersic_mass_sis__plane_images_match_galaxy_images(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0),
                               mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))

            g0_image = g0.intensity_from_grid(grid=np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                                                 [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                                                 [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]]))

            grids.image = np.array([[-1.5, -1.5], [1.5, 1.5]])

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=grids)

            g1_image_grid = ray_tracing.uniform_grid_from_lensed_grid(tracer.source_plane.grids.image, shape=(3,3))

            g1_image = g1.intensity_from_grid(grid=g1_image_grid)

            assert (tracer.plane_images_of_planes(shape=(3,3))[0] == g0_image).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] == g1_image).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[0] == tracer.image_plane.plane_image(shape=(3,3))).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] == tracer.source_plane.plane_image(shape=(3,3))).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[0] ==
                    tracer.image_plane.plane_images_of_galaxies(shape=(3,3))[0]).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] ==
                    tracer.source_plane.plane_images_of_galaxies(shape=(3,3))[0]).all()

        def test__same_as_above_but_multiple_galaxies_in_a_plane(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0),
                               mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))

            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0))
            g2 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=3.0))
            g3 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=4.0))

            g0_image = g0.intensity_from_grid(grid=np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                                                 [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                                                 [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]]))

            g1_image = g1.intensity_from_grid(grid=np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                                                 [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                                                 [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]]))

            grids.image = np.array([[-1.5, -1.5], [1.5, 1.5]])

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2, g3],
                                                         image_plane_grids=grids)

            g2_image_grid = ray_tracing.uniform_grid_from_lensed_grid(tracer.source_plane.grids.image, shape=(3,3))

            g2_image = g2.intensity_from_grid(grid=g2_image_grid)
            g3_image = g3.intensity_from_grid(grid=g2_image_grid)

            assert (tracer.plane_images_of_planes(shape=(3,3))[0] == g0_image + g1_image).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] == g2_image + g3_image).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[0] == tracer.image_plane.plane_image(shape=(3,3))).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] == tracer.source_plane.plane_image(shape=(3,3))).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[0] ==
                    tracer.image_plane.plane_images_of_galaxies(shape=(3,3))[0] +
                    tracer.image_plane.plane_images_of_galaxies(shape=(3,3))[1]).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] ==
                    tracer.source_plane.plane_images_of_galaxies(shape=(3,3))[0] +
                    tracer.source_plane.plane_images_of_galaxies(shape=(3,3))[1]).all()


    class TestImageGridsOfPlanes:

        def test__grids_match_plane_grids(self, grids):

            g0 = galaxy.Galaxy(mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))
            g1 = galaxy.Galaxy()

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grids=grids)

            assert (tracer.image_grids_of_planes[0] == tracer.image_plane.grids.image).all()
            assert (tracer.image_grids_of_planes[1] == tracer.source_plane.grids.image).all()


    class TestReconstructorFromGalaxy:

        def test__image_galaxy_has_pixelization__still_returns_none(self, grids, sparse_mask):

            galaxy_pix = galaxy.Galaxy(pixelization=MockPixelization(value=1))
            galaxy_no_pix = galaxy.Galaxy()

            tracing = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_pix], source_galaxies=[galaxy_no_pix],
                                                   image_plane_grids=grids)

            reconstructors = tracing.reconstructors_from_source_plane(MockBorders(), sparse_mask)

            assert reconstructors is None

        def test__source_galaxy_has_pixelization__returns_reconstructor(self, grids, sparse_mask):
            galaxy_pix = galaxy.Galaxy(pixelization=MockPixelization(value=1))
            galaxy_no_pix = galaxy.Galaxy()

            tracing = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_no_pix], source_galaxies=[galaxy_pix],
                                                   image_plane_grids=grids)

            reconstructors = tracing.reconstructors_from_source_plane(MockBorders(), sparse_mask)

            assert reconstructors == 1


class TestTracerGeometry(object):

    def test__2_planes__z01_and_z1(self):
        geometry = ray_tracing.TracerGeometry(redshifts=[0.1, 1.0], cosmology=cosmo.Planck15)

        assert geometry.arcsec_per_kpc(plane_i=0) == pytest.approx(0.525060, 1e-5)
        assert geometry.kpc_per_arcsec(plane_i=0) == pytest.approx(1.904544, 1e-5)

        assert geometry.ang_to_earth(plane_i=0) == pytest.approx(392840, 1e-5)
        assert geometry.ang_between_planes(plane_i=0, plane_j=0) == 0.0
        assert geometry.ang_between_planes(plane_i=0, plane_j=1) == pytest.approx(1481890.4, 1e-5)

        assert geometry.arcsec_per_kpc(plane_i=1) == pytest.approx(0.1214785, 1e-5)
        assert geometry.kpc_per_arcsec(plane_i=1) == pytest.approx(8.231907, 1e-5)

        assert geometry.ang_to_earth(plane_i=1) == pytest.approx(1697952, 1e-5)
        assert geometry.ang_between_planes(plane_i=1, plane_j=0) == pytest.approx(-2694346, 1e-5)
        assert geometry.ang_between_planes(plane_i=1, plane_j=1) == 0.0

        assert geometry.critical_density_kpc(plane_i=0, plane_j=1) == pytest.approx(4.85e9, 1e-2)
        assert geometry.critical_density_arcsec(plane_i=0, plane_j=1) == pytest.approx(17593241668, 1e-2)

    def test__3_planes__z01_z1__and_z2(self):
        geometry = ray_tracing.TracerGeometry(redshifts=[0.1, 1.0, 2.0], cosmology=cosmo.Planck15)

        assert geometry.final_plane == 2

        assert geometry.arcsec_per_kpc(plane_i=0) == pytest.approx(0.525060, 1e-5)
        assert geometry.kpc_per_arcsec(plane_i=0) == pytest.approx(1.904544, 1e-5)

        assert geometry.ang_to_earth(plane_i=0) == pytest.approx(392840, 1e-5)
        assert geometry.ang_between_planes(plane_i=0, plane_j=0) == 0.0
        assert geometry.ang_between_planes(plane_i=0, plane_j=1) == pytest.approx(1481890.4, 1e-5)
        assert geometry.ang_between_planes(plane_i=0, plane_j=2) == pytest.approx(1626471, 1e-5)

        assert geometry.arcsec_per_kpc(plane_i=1) == pytest.approx(0.1214785, 1e-5)
        assert geometry.kpc_per_arcsec(plane_i=1) == pytest.approx(8.231907, 1e-5)

        assert geometry.ang_to_earth(plane_i=1) == pytest.approx(1697952, 1e-5)
        assert geometry.ang_between_planes(plane_i=1, plane_j=0) == pytest.approx(-2694346, 1e-5)
        assert geometry.ang_between_planes(plane_i=1, plane_j=1) == 0.0
        assert geometry.ang_between_planes(plane_i=1, plane_j=2) == pytest.approx(638544, 1e-5)

        assert geometry.arcsec_per_kpc(plane_i=2) == pytest.approx(0.116500, 1e-5)
        assert geometry.kpc_per_arcsec(plane_i=2) == pytest.approx(8.58368, 1e-5)

        assert geometry.ang_to_earth(plane_i=2) == pytest.approx(1770512, 1e-5)
        assert geometry.ang_between_planes(plane_i=2, plane_j=0) == pytest.approx(-4435831, 1e-5)
        assert geometry.ang_between_planes(plane_i=2, plane_j=1) == pytest.approx(-957816)
        assert geometry.ang_between_planes(plane_i=2, plane_j=2) == 0.0

        assert geometry.critical_density_kpc(plane_i=0, plane_j=1) == pytest.approx(4.85e9, 1e-2)
        assert geometry.critical_density_arcsec(plane_i=0, plane_j=1) == pytest.approx(17593241668, 1e-2)

    def test__4_planes__z01_z1_z2_and_z3(self):
        geometry = ray_tracing.TracerGeometry(redshifts=[0.1, 1.0, 2.0, 3.0], cosmology=cosmo.Planck15)

        assert geometry.final_plane == 3

        assert geometry.arcsec_per_kpc(plane_i=0) == pytest.approx(0.525060, 1e-5)
        assert geometry.kpc_per_arcsec(plane_i=0) == pytest.approx(1.904544, 1e-5)

        assert geometry.ang_to_earth(plane_i=0) == pytest.approx(392840, 1e-5)
        assert geometry.ang_between_planes(plane_i=0, plane_j=0) == 0.0
        assert geometry.ang_between_planes(plane_i=0, plane_j=1) == pytest.approx(1481890.4, 1e-5)
        assert geometry.ang_between_planes(plane_i=0, plane_j=2) == pytest.approx(1626471, 1e-5)
        assert geometry.ang_between_planes(plane_i=0, plane_j=3) == pytest.approx(1519417, 1e-5)

        assert geometry.arcsec_per_kpc(plane_i=1) == pytest.approx(0.1214785, 1e-5)
        assert geometry.kpc_per_arcsec(plane_i=1) == pytest.approx(8.231907, 1e-5)

        assert geometry.ang_to_earth(plane_i=1) == pytest.approx(1697952, 1e-5)
        assert geometry.ang_between_planes(plane_i=1, plane_j=0) == pytest.approx(-2694346, 1e-5)
        assert geometry.ang_between_planes(plane_i=1, plane_j=1) == 0.0
        assert geometry.ang_between_planes(plane_i=1, plane_j=2) == pytest.approx(638544, 1e-5)
        assert geometry.ang_between_planes(plane_i=1, plane_j=3) == pytest.approx(778472, 1e-5)

        assert geometry.arcsec_per_kpc(plane_i=2) == pytest.approx(0.116500, 1e-5)
        assert geometry.kpc_per_arcsec(plane_i=2) == pytest.approx(8.58368, 1e-5)

        assert geometry.ang_to_earth(plane_i=2) == pytest.approx(1770512, 1e-5)
        assert geometry.ang_between_planes(plane_i=2, plane_j=0) == pytest.approx(-4435831, 1e-5)
        assert geometry.ang_between_planes(plane_i=2, plane_j=1) == pytest.approx(-957816)
        assert geometry.ang_between_planes(plane_i=2, plane_j=2) == 0.0
        assert geometry.ang_between_planes(plane_i=2, plane_j=3) == pytest.approx(299564)

        assert geometry.arcsec_per_kpc(plane_i=3) == pytest.approx(0.12674, 1e-5)
        assert geometry.kpc_per_arcsec(plane_i=3) == pytest.approx(7.89009, 1e-5)

        assert geometry.ang_to_earth(plane_i=3) == pytest.approx(1627448, 1e-5)
        assert geometry.ang_between_planes(plane_i=3, plane_j=0) == pytest.approx(-5525155, 1e-5)
        assert geometry.ang_between_planes(plane_i=3, plane_j=1) == pytest.approx(-1556945, 1e-5)
        assert geometry.ang_between_planes(plane_i=3, plane_j=2) == pytest.approx(-399419, 1e-5)
        assert geometry.ang_between_planes(plane_i=3, plane_j=3) == 0.0

        assert geometry.critical_density_kpc(plane_i=0, plane_j=1) == pytest.approx(4.85e9, 1e-2)
        assert geometry.critical_density_arcsec(plane_i=0, plane_j=1) == pytest.approx(17593241668, 1e-2)

    def test__scaling_factors__3_planes__z01_z1_and_z2(self):
        geometry = ray_tracing.TracerGeometry(redshifts=[0.1, 1.0, 2.0], cosmology=cosmo.Planck15)

        assert geometry.scaling_factor(plane_i=0, plane_j=1) == pytest.approx(0.9500, 1e-4)
        assert geometry.scaling_factor(plane_i=0, plane_j=2) == pytest.approx(1.0, 1e-4)
        assert geometry.scaling_factor(plane_i=1, plane_j=2) == pytest.approx(1.0, 1e-4)

    def test__scaling_factors__4_planes__z01_z1_z2_and_z3(self):
        geometry = ray_tracing.TracerGeometry(redshifts=[0.1, 1.0, 2.0, 3.0], cosmology=cosmo.Planck15)

        assert geometry.scaling_factor(plane_i=0, plane_j=1) == pytest.approx(0.9348, 1e-4)
        assert geometry.scaling_factor(plane_i=0, plane_j=2) == pytest.approx(0.984, 1e-4)
        assert geometry.scaling_factor(plane_i=0, plane_j=3) == pytest.approx(1.0, 1e-4)
        assert geometry.scaling_factor(plane_i=1, plane_j=2) == pytest.approx(0.754, 1e-4)
        assert geometry.scaling_factor(plane_i=1, plane_j=3) == pytest.approx(1.0, 1e-4)
        assert geometry.scaling_factor(plane_i=2, plane_j=3) == pytest.approx(1.0, 1e-4)


class TestMultiTracer(object):


    class TestException:

        def test__no_galaxies_in_tracer__raises_excetion(self, grids):

            with pytest.raises(exc.RayTracingException):
                tracer = ray_tracing.TracerMulti(galaxies=[],image_plane_grids=grids, cosmology=cosmo.Planck15)


    class TestGalaxyOrder:

        def test__3_galaxies_reordered_in_ascending_redshift(self, grids):
            tracer = ray_tracing.TracerMulti(galaxies=[galaxy.Galaxy(redshift=2.0), galaxy.Galaxy(redshift=1.0),
                                                       galaxy.Galaxy(redshift=0.1)], image_plane_grids=grids, cosmology=cosmo.Planck15)

            assert tracer.galaxies_redshift_order[0].redshift == 0.1
            assert tracer.galaxies_redshift_order[1].redshift == 1.0
            assert tracer.galaxies_redshift_order[2].redshift == 2.0

        def test_3_galaxies_two_same_redshift_planes_redshift_order_is_size_2_with_redshifts(self,
                                                                                             grids):
            tracer = ray_tracing.TracerMulti(galaxies=[galaxy.Galaxy(redshift=1.0), galaxy.Galaxy(redshift=1.0),
                                                       galaxy.Galaxy(redshift=0.1)],
                                             image_plane_grids=grids,
                                             cosmology=cosmo.Planck15)

            assert tracer.galaxies_redshift_order[0].redshift == 0.1
            assert tracer.galaxies_redshift_order[1].redshift == 1.0
            assert tracer.galaxies_redshift_order[2].redshift == 1.0

            assert tracer.planes_redshift_order[0] == 0.1
            assert tracer.planes_redshift_order[1] == 1.0

        def test__6_galaxies_producing_4_planes(self, grids):
            g0 = galaxy.Galaxy(redshift=1.0)
            g1 = galaxy.Galaxy(redshift=1.0)
            g2 = galaxy.Galaxy(redshift=0.1)
            g3 = galaxy.Galaxy(redshift=1.05)
            g4 = galaxy.Galaxy(redshift=0.95)
            g5 = galaxy.Galaxy(redshift=1.05)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4, g5],
                                             image_plane_grids=grids, cosmology=cosmo.Planck15)

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

        def test__6_galaxies__plane_galaxies_are_correct(self, grids):
            g0 = galaxy.Galaxy(redshift=1.0)
            g1 = galaxy.Galaxy(redshift=1.0)
            g2 = galaxy.Galaxy(redshift=0.1)
            g3 = galaxy.Galaxy(redshift=1.05)
            g4 = galaxy.Galaxy(redshift=0.95)
            g5 = galaxy.Galaxy(redshift=1.05)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4, g5],
                                             image_plane_grids=grids, cosmology=cosmo.Planck15)

            assert tracer.planes_galaxies[0] == [g2]
            assert tracer.planes_galaxies[1] == [g4]
            assert tracer.planes_galaxies[2] == [g0, g1]
            assert tracer.planes_galaxies[3] == [g3, g5]


    class TestRayTracingPlanes:

        def test__6_galaxies__tracer_planes_are_correct(self, grids):
            g0 = galaxy.Galaxy(redshift=2.0)
            g1 = galaxy.Galaxy(redshift=2.0)
            g2 = galaxy.Galaxy(redshift=0.1)
            g3 = galaxy.Galaxy(redshift=3.0)
            g4 = galaxy.Galaxy(redshift=1.0)
            g5 = galaxy.Galaxy(redshift=3.0)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4, g5],
                                             image_plane_grids=grids, cosmology=cosmo.Planck15)

            assert tracer.planes[0].galaxies == [g2]
            assert tracer.planes[1].galaxies == [g4]
            assert tracer.planes[2].galaxies == [g0, g1]
            assert tracer.planes[3].galaxies == [g3, g5]

        def test__4_planes__coordinate_grids_and_deflections_are_correct__sis_mass_profile(self,
                                                                                           grids):
            import math

            g0 = galaxy.Galaxy(redshift=2.0, mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))
            g1 = galaxy.Galaxy(redshift=2.0, mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))
            g2 = galaxy.Galaxy(redshift=0.1, mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))
            g3 = galaxy.Galaxy(redshift=3.0, mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))
            g4 = galaxy.Galaxy(redshift=1.0, mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))
            g5 = galaxy.Galaxy(redshift=3.0, mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0))

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4, g5],
                                             image_plane_grids=grids, cosmology=cosmo.Planck15)

            # From unit test below:
            # Beta_01 = 0.9348
            # Beta_02 = 0.9840
            # Beta_03 = 1.0
            # Beta_12 = 0.754
            # Beta_13 = 1.0
            # Beta_23 = 1.0

            val = math.sqrt(2) / 2.0

            assert tracer.planes[0].grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
            assert tracer.planes[0].grids.sub[0] == pytest.approx(np.array([1.0, 1.0]),
                                                                  1e-4)
            assert tracer.planes[0].grids.sub[1] == pytest.approx(np.array([1.0, 0.0]),
                                                                  1e-4)
            assert tracer.planes[0].grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]),
                                                                       1e-4)
            assert tracer.planes[0].deflections.image[0] == pytest.approx(np.array([val, val]), 1e-4)
            assert tracer.planes[0].deflections.sub[0] == pytest.approx(np.array([val, val]), 1e-4)
            assert tracer.planes[0].deflections.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)
            assert tracer.planes[0].deflections.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

            assert tracer.planes[1].grids.image[0] == pytest.approx(
                np.array([(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]), 1e-4)
            assert tracer.planes[1].grids.sub[0] == pytest.approx(
                np.array([(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]), 1e-4)
            assert tracer.planes[1].grids.sub[1] == pytest.approx(
                np.array([(1.0 - 0.9348 * 1.0), 0.0]), 1e-4)
            assert tracer.planes[1].grids.blurring[0] == pytest.approx(
                np.array([(1.0 - 0.9348 * 1.0), 0.0]), 1e-4)

            defl11 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]))
            defl12 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * 1.0), 0.0]]))

            assert tracer.planes[1].deflections.image[0] == pytest.approx(defl11[0], 1e-4)
            assert tracer.planes[1].deflections.sub[0] == pytest.approx(defl11[0], 1e-4)
            assert tracer.planes[1].deflections.sub[1] == pytest.approx(defl12[0], 1e-4)
            assert tracer.planes[1].deflections.blurring[0] == pytest.approx(defl12[0], 1e-4)

            assert tracer.planes[2].grids.image[0] == pytest.approx(
                np.array([(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                          (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]), 1e-4)
            assert tracer.planes[2].grids.sub[0] == pytest.approx(
                np.array([(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                          (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]), 1e-4)
            assert tracer.planes[2].grids.sub[1] == pytest.approx(
                np.array([(1.0 - 0.9839601 * 1.0 - 0.7539734 * defl12[0, 0]),
                          0.0]), 1e-4)
            assert tracer.planes[2].grids.blurring[0] == pytest.approx(
                np.array([(1.0 - 0.9839601 * 1.0 - 0.7539734 * defl12[0, 0]),
                          0.0]), 1e-4)

            # 2 Galaxies in this plane, so multiply by 2.0

            defl21 = 2.0 * g0.deflections_from_grid(
                grid=np.array([[(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                                (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]]))
            defl22 = 2.0 * g0.deflections_from_grid(
                grid=np.array([[(1.0 - 0.9839601 * 1.0 - 0.7539734 * defl12[0, 0]),
                                0.0]]))

            assert tracer.planes[2].deflections.image[0] == pytest.approx(defl21[0], 1e-4)
            assert tracer.planes[2].deflections.sub[0] == pytest.approx(defl21[0], 1e-4)
            assert tracer.planes[2].deflections.sub[1] == pytest.approx(defl22[0], 1e-4)
            assert tracer.planes[2].deflections.blurring[0] == pytest.approx(defl22[0], 1e-4)

            coord1 = (1.0 - tracer.planes[0].deflections.image[0, 0] - tracer.planes[1].deflections.image[
                0, 0] -
                      tracer.planes[2].deflections.image[0, 0])

            coord2 = (1.0 - tracer.planes[0].deflections.image[0, 1] - tracer.planes[1].deflections.image[
                0, 1] -
                      tracer.planes[2].deflections.image[0, 1])

            coord3 = (1.0 - tracer.planes[0].deflections.sub[1, 0] -
                      tracer.planes[1].deflections.sub[1, 0] -
                      tracer.planes[2].deflections.sub[1, 0])

            assert tracer.planes[3].grids.image[0] == pytest.approx(np.array([coord1, coord2]),
                                                                                1e-4)
            assert tracer.planes[3].grids.sub[0] == pytest.approx(
                np.array([coord1, coord2]), 1e-4)
            assert tracer.planes[3].grids.sub[1] == pytest.approx(np.array([coord3, 0.0]),
                                                                  1e-4)
            assert tracer.planes[3].grids.blurring[0] == pytest.approx(np.array([coord3, 0.0]),
                                                                       1e-4)


    class TestImagePlaneImages:

        def test__x1_galaxy_light_no_mass_in_each_plane__image_of_each_plane_is_galaxy_image(self, grids):

            sersic = light_profiles.EllipticalSersicLP(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0)

            g0 = galaxy.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = galaxy.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = galaxy.Galaxy(redshift=2.0, light_profile=sersic)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=grids, cosmology=cosmo.Planck15)

            plane_0 = ray_tracing.Plane(galaxies=[g0], grids=grids, compute_deflections=True)
            plane_1 = ray_tracing.Plane(galaxies=[g1], grids=grids, compute_deflections=True)
            plane_2 = ray_tracing.Plane(galaxies=[g2], grids=grids, compute_deflections=False)

            image_plane_image = plane_0.image_plane_image + plane_1.image_plane_image + plane_2.image_plane_image

            assert (image_plane_image == tracer.image_plane_image).all()
            assert (tracer.image_plane_images_of_planes[0] == plane_0.image_plane_image).all()
            assert (tracer.image_plane_images_of_planes[1] == plane_1.image_plane_image).all()
            assert (tracer.image_plane_images_of_planes[2] == plane_2.image_plane_image).all()

        def test__galaxy_light_sersic_mass_sis__source_plane_image_includes_deflections(self, grids):

            sersic = light_profiles.EllipticalSersicLP(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0)

            sis = mass_profiles.SphericalIsothermalMP(einstein_radius=1.0)

            g0 = galaxy.Galaxy(redshift=0.1, light_profile=sersic, mass_profile=sis)
            g1 = galaxy.Galaxy(redshift=1.0, light_profile=sersic, mass_profile=sis)
            g2 = galaxy.Galaxy(redshift=2.0, light_profile=sersic, mass_profile=sis)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=grids, cosmology=cosmo.Planck15)

            plane_0 = tracer.planes[0]
            plane_1 = tracer.planes[1]
            plane_2 = tracer.planes[2]

            image_plane_image = plane_0.image_plane_image + plane_1.image_plane_image + plane_2.image_plane_image

            assert (image_plane_image == tracer.image_plane_image).all()
            assert (tracer.image_plane_images_of_planes[0] == plane_0.image_plane_image).all()
            assert (tracer.image_plane_images_of_planes[1] == plane_1.image_plane_image).all()
            assert (tracer.image_plane_images_of_planes[2] == plane_2.image_plane_image).all()

        def test__x1_galaxy_light_no_mass_in_each_plane__image_of_each_galaxy_is_galaxy_image(self, grids):

            sersic = light_profiles.EllipticalSersicLP(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0)

            g0 = galaxy.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = galaxy.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = galaxy.Galaxy(redshift=2.0, light_profile=sersic)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=grids,
                                             cosmology=cosmo.Planck15)

            plane_0 = ray_tracing.Plane(galaxies=[g0], grids=grids, compute_deflections=True)
            plane_1 = ray_tracing.Plane(galaxies=[g1], grids=grids, compute_deflections=True)
            plane_2 = ray_tracing.Plane(galaxies=[g2], grids=grids, compute_deflections=False)

            image_plane_image = plane_0.image_plane_image + plane_1.image_plane_image + plane_2.image_plane_image

            assert (image_plane_image == tracer.image_plane_image).all()
            assert (tracer.image_plane_images_of_galaxies[0] == plane_0.image_plane_image).all()
            assert (tracer.image_plane_images_of_galaxies[1] == plane_1.image_plane_image).all()
            assert (tracer.image_plane_images_of_galaxies[2] == plane_2.image_plane_image).all()
            assert (tracer.image_plane_images_of_galaxies[0] == plane_0.image_plane_images_of_galaxies[0]).all()
            assert (tracer.image_plane_images_of_galaxies[1] == plane_1.image_plane_images_of_galaxies[0]).all()
            assert (tracer.image_plane_images_of_galaxies[2] == plane_2.image_plane_images_of_galaxies[0]).all()

        def test__diffrent_galaxies_light_no_mass_in_each_plane__image_of_each_galaxy_is_galaxy_image(self, grids):

            sersic = light_profiles.EllipticalSersicLP(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0)

            g0 = galaxy.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = galaxy.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = galaxy.Galaxy(redshift=2.0, light_profile=sersic)
            g3 = galaxy.Galaxy(redshift=0.1, light_profile=sersic)
            g4 = galaxy.Galaxy(redshift=1.0, light_profile=sersic)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4], image_plane_grids=grids,
                                             cosmology=cosmo.Planck15)

            plane_0 = ray_tracing.Plane(galaxies=[g0, g3], grids=grids, compute_deflections=True)
            plane_1 = ray_tracing.Plane(galaxies=[g1, g4], grids=grids, compute_deflections=True)
            plane_2 = ray_tracing.Plane(galaxies=[g2], grids=grids, compute_deflections=False)

            image_plane_image = plane_0.image_plane_image + plane_1.image_plane_image + plane_2.image_plane_image

            assert (image_plane_image == tracer.image_plane_image).all()
            assert (tracer.image_plane_images_of_galaxies[0] + tracer.image_plane_images_of_galaxies[3]
                    == plane_0.image_plane_image).all()
            assert (tracer.image_plane_images_of_galaxies[1] + tracer.image_plane_images_of_galaxies[4]
                    == plane_1.image_plane_image).all()
            assert (tracer.image_plane_images_of_galaxies[2] == plane_2.image_plane_image).all()

            assert (tracer.image_plane_images_of_galaxies[0] == plane_0.image_plane_images_of_galaxies[0]).all()
            assert (tracer.image_plane_images_of_galaxies[1] == plane_0.image_plane_images_of_galaxies[1]).all()
            assert (tracer.image_plane_images_of_galaxies[2] == plane_1.image_plane_images_of_galaxies[0]).all()
            assert (tracer.image_plane_images_of_galaxies[3] == plane_1.image_plane_images_of_galaxies[1]).all()
            assert (tracer.image_plane_images_of_galaxies[4] == plane_2.image_plane_images_of_galaxies[0]).all()


    class TestImagePlaneBlurringImages:

        def test__x1_galaxy_light_no_mass_in_each_plane__image_of_each_plane_is_galaxy_image(self, grids):

            sersic = light_profiles.EllipticalSersicLP(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0)

            g0 = galaxy.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = galaxy.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = galaxy.Galaxy(redshift=2.0, light_profile=sersic)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=grids, cosmology=cosmo.Planck15)

            plane_0 = ray_tracing.Plane(galaxies=[g0], grids=grids, compute_deflections=True)
            plane_1 = ray_tracing.Plane(galaxies=[g1], grids=grids, compute_deflections=True)
            plane_2 = ray_tracing.Plane(galaxies=[g2], grids=grids, compute_deflections=False)

            image_plane_blurring_image = plane_0.image_plane_blurring_image + plane_1.image_plane_blurring_image + \
                                         plane_2.image_plane_blurring_image

            assert (image_plane_blurring_image == tracer.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_planes[0] == plane_0.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_planes[1] == plane_1.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_planes[2] == plane_2.image_plane_blurring_image).all()

        def test__galaxy_light_sersic_mass_sis__source_plane_image_includes_deflections(self, grids):

            sersic = light_profiles.EllipticalSersicLP(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0)

            sis = mass_profiles.SphericalIsothermalMP(einstein_radius=1.0)

            g0 = galaxy.Galaxy(redshift=0.1, light_profile=sersic, mass_profile=sis)
            g1 = galaxy.Galaxy(redshift=1.0, light_profile=sersic, mass_profile=sis)
            g2 = galaxy.Galaxy(redshift=2.0, light_profile=sersic, mass_profile=sis)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=grids, cosmology=cosmo.Planck15)

            plane_0 = tracer.planes[0]
            plane_1 = tracer.planes[1]
            plane_2 = tracer.planes[2]

            image_plane_blurring_image = plane_0.image_plane_blurring_image + plane_1.image_plane_blurring_image + \
                                         plane_2.image_plane_blurring_image

            assert (image_plane_blurring_image == tracer.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_planes[0] == plane_0.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_planes[1] == plane_1.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_planes[2] == plane_2.image_plane_blurring_image).all()

        def test__x1_galaxy_light_no_mass_in_each_plane__image_of_each_galaxy_is_galaxy_image(self, grids):

            sersic = light_profiles.EllipticalSersicLP(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0)

            g0 = galaxy.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = galaxy.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = galaxy.Galaxy(redshift=2.0, light_profile=sersic)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=grids,
                                             cosmology=cosmo.Planck15)

            plane_0 = ray_tracing.Plane(galaxies=[g0], grids=grids, compute_deflections=True)
            plane_1 = ray_tracing.Plane(galaxies=[g1], grids=grids, compute_deflections=True)
            plane_2 = ray_tracing.Plane(galaxies=[g2], grids=grids, compute_deflections=False)

            image_plane_blurring_image = plane_0.image_plane_blurring_image + plane_1.image_plane_blurring_image + \
                                         plane_2.image_plane_blurring_image

            assert (image_plane_blurring_image == tracer.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[0] == plane_0.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[1] == plane_1.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[2] == plane_2.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[0] == plane_0.image_plane_blurring_images_of_galaxies[0]).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[1] == plane_1.image_plane_blurring_images_of_galaxies[0]).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[2] == plane_2.image_plane_blurring_images_of_galaxies[0]).all()

        def test__diffrent_galaxies_light_no_mass_in_each_plane__image_of_each_galaxy_is_galaxy_image(self, grids):

            sersic = light_profiles.EllipticalSersicLP(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                                       sersic_index=4.0)

            g0 = galaxy.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = galaxy.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = galaxy.Galaxy(redshift=2.0, light_profile=sersic)
            g3 = galaxy.Galaxy(redshift=0.1, light_profile=sersic)
            g4 = galaxy.Galaxy(redshift=1.0, light_profile=sersic)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3, g4], image_plane_grids=grids,
                                             cosmology=cosmo.Planck15)

            plane_0 = ray_tracing.Plane(galaxies=[g0, g3], grids=grids, compute_deflections=True)
            plane_1 = ray_tracing.Plane(galaxies=[g1, g4], grids=grids, compute_deflections=True)
            plane_2 = ray_tracing.Plane(galaxies=[g2], grids=grids, compute_deflections=False)

            image_plane_blurring_image = plane_0.image_plane_blurring_image + plane_1.image_plane_blurring_image + \
                                         plane_2.image_plane_blurring_image

            assert (image_plane_blurring_image == tracer.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[0] + tracer.image_plane_blurring_images_of_galaxies[3]
                    == plane_0.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[1] + tracer.image_plane_blurring_images_of_galaxies[4]
                    == plane_1.image_plane_blurring_image).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[2] == plane_2.image_plane_blurring_image).all()

            assert (tracer.image_plane_blurring_images_of_galaxies[0] ==
                    plane_0.image_plane_blurring_images_of_galaxies[0]).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[1] ==
                    plane_0.image_plane_blurring_images_of_galaxies[1]).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[2] ==
                    plane_1.image_plane_blurring_images_of_galaxies[0]).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[3] ==
                    plane_1.image_plane_blurring_images_of_galaxies[1]).all()
            assert (tracer.image_plane_blurring_images_of_galaxies[4] ==
                    plane_2.image_plane_blurring_images_of_galaxies[0]).all()


    class TestPlaneImages:

        def test__galaxy_light_sersic_mass_sis__x2_plane_images_match_galaxy_images(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0),
                               mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0), redshift=0.1)
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0), redshift=1.0)

            g0_image = g0.intensity_from_grid(grid=np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                                                 [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                                                 [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]]))

            grids.image = np.array([[-1.5, -1.5], [1.5, 1.5]])

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1], image_plane_grids=grids, cosmology=cosmo.Planck15)

            g1_image_grid = ray_tracing.uniform_grid_from_lensed_grid(tracer.planes[1].grids.image, shape=(3,3))

            g1_image = g1.intensity_from_grid(grid=g1_image_grid)

            assert (tracer.plane_images_of_planes(shape=(3,3))[0] == g0_image).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] == g1_image).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[0] == tracer.planes[0].plane_image(shape=(3,3))).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] == tracer.planes[1].plane_image(shape=(3,3))).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[0] ==
                    tracer.planes[0].plane_images_of_galaxies(shape=(3,3))[0]).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] ==
                    tracer.planes[1].plane_images_of_galaxies(shape=(3,3))[0]).all()

        def test__same_as_above_but_multiple_galaxies_in_a_plane(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0),
                               mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0), redshift=0.1)
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0), redshift=0.2)
            g2 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=3.0), redshift=0.3)
            g3 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=4.0), redshift=0.3)

            g0_image = g0.intensity_from_grid(grid=np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                                                 [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                                                 [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]]))

            grids.image = np.array([[-1.5, -1.5], [1.5, 1.5]])

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3], image_plane_grids=grids,
                                             cosmology=cosmo.Planck15)

            g1_image_grid = ray_tracing.uniform_grid_from_lensed_grid(tracer.planes[1].grids.image, shape=(3,3))
            g2_image_grid = ray_tracing.uniform_grid_from_lensed_grid(tracer.planes[2].grids.image, shape=(3,3))

            g1_image = g1.intensity_from_grid(grid=g1_image_grid)
            g2_image = g2.intensity_from_grid(grid=g2_image_grid)
            g3_image = g3.intensity_from_grid(grid=g2_image_grid)

            assert (tracer.plane_images_of_planes(shape=(3,3))[0] == g0_image).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] == g1_image).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[2] == g2_image + g3_image).all()

            assert (tracer.plane_images_of_planes(shape=(3,3))[0] == tracer.planes[0].plane_image(shape=(3,3))).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] == tracer.planes[1].plane_image(shape=(3,3))).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[2] == tracer.planes[2].plane_image(shape=(3,3))).all()

            assert (tracer.plane_images_of_planes(shape=(3,3))[0] ==
                    tracer.planes[0].plane_images_of_galaxies(shape=(3,3))[0]).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] ==
                    tracer.planes[1].plane_images_of_galaxies(shape=(3,3))[0]).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[2] ==
                    tracer.planes[2].plane_images_of_galaxies(shape=(3,3))[0] +
                    tracer.planes[2].plane_images_of_galaxies(shape=(3, 3))[1]).all()

        def test__same_as_above_but_swap_two_galaxy_redshifts__planes_are_reordered(self, grids):

            g0 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=1.0),
                               mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0), redshift=0.1)
            g1 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=2.0), redshift=0.3)
            g2 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=3.0), redshift=0.2)
            g3 = galaxy.Galaxy(light_profile=light_profiles.EllipticalSersicLP(intensity=4.0), redshift=0.3)

            g0_image = g0.intensity_from_grid(grid=np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                                                 [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                                                 [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]]))

            grids.image = np.array([[-1.5, -1.5], [1.5, 1.5]])

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3], image_plane_grids=grids,
                                             cosmology=cosmo.Planck15)

            z03_image_grid = ray_tracing.uniform_grid_from_lensed_grid(tracer.planes[2].grids.image, shape=(3,3))
            z02_image_grid = ray_tracing.uniform_grid_from_lensed_grid(tracer.planes[1].grids.image, shape=(3,3))

            g1_image = g1.intensity_from_grid(grid=z03_image_grid)
            g2_image = g2.intensity_from_grid(grid=z02_image_grid)
            g3_image = g3.intensity_from_grid(grid=z03_image_grid)

            assert (tracer.plane_images_of_planes(shape=(3,3))[0] == g0_image).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] == g2_image).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[2] == g1_image + g3_image).all()

            assert (tracer.plane_images_of_planes(shape=(3,3))[0] == tracer.planes[0].plane_image(shape=(3,3))).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] == tracer.planes[1].plane_image(shape=(3,3))).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[2] == tracer.planes[2].plane_image(shape=(3,3))).all()

            assert (tracer.plane_images_of_planes(shape=(3,3))[0] ==
                    tracer.planes[0].plane_images_of_galaxies(shape=(3,3))[0]).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[1] ==
                    tracer.planes[1].plane_images_of_galaxies(shape=(3,3))[0]).all()
            assert (tracer.plane_images_of_planes(shape=(3,3))[2] ==
                    tracer.planes[2].plane_images_of_galaxies(shape=(3,3))[0] +
                    tracer.planes[2].plane_images_of_galaxies(shape=(3,3))[1]).all()


    class TestImageGridsOfPlanes:

        def test__grids_match_plane_grids(self, grids):

            g0 = galaxy.Galaxy(mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0), redshift=0.1)
            g1 = galaxy.Galaxy(mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0), redshift=0.2)
            g2 = galaxy.Galaxy(mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0), redshift=0.3)
            g3 = galaxy.Galaxy(mass_profile=mass_profiles.SphericalIsothermalMP(einstein_radius=1.0), redshift=0.4)

            tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2, g3], image_plane_grids=grids,
                                             cosmology=cosmo.Planck15)

            assert (tracer.image_grids_of_planes[0] == tracer.planes[0].grids.image).all()
            assert (tracer.image_grids_of_planes[1] == tracer.planes[1].grids.image).all()
            assert (tracer.image_grids_of_planes[2] == tracer.planes[2].grids.image).all()
            assert (tracer.image_grids_of_planes[3] == tracer.planes[3].grids.image).all()


    class TestReconstructorFromGalaxy:

        def test__3_galaxies__non_have_pixelization__returns_none_x3(self, grids, sparse_mask):
            sis = mass_profiles.SphericalIsothermalMP(einstein_radius=1.0)

            g0 = galaxy.Galaxy(redshift=0.1, mass_profile=sis)
            g1 = galaxy.Galaxy(redshift=1.0, mass_profile=sis)
            g2 = galaxy.Galaxy(redshift=2.0, mass_profile=sis)

            tracing = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=grids,
                                              cosmology=cosmo.Planck15)

            reconstructors = tracing.reconstructors_from_planes(MockBorders(), sparse_mask)

            assert reconstructors == [None, None, None]

        def test__3_galaxies__1_has_pixelization__returns_none_x2_and_pixelization(self, grids,
                                                                                   sparse_mask):
            sis = mass_profiles.SphericalIsothermalMP(einstein_radius=1.0)

            g0 = galaxy.Galaxy(redshift=0.1, mass_profile=sis)
            g1 = galaxy.Galaxy(redshift=1.0, pixelization=MockPixelization(value=1), mass_profile=sis)
            g2 = galaxy.Galaxy(redshift=2.0, mass_profile=sis)

            tracing = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=grids,
                                              cosmology=cosmo.Planck15)

            reconstructors = tracing.reconstructors_from_planes(MockBorders(), sparse_mask)

            assert reconstructors == [None, 1, None]

        def test__3_galaxies__all_have_pixelization__returns_pixelizations(self, grids, sparse_mask):
            sis = mass_profiles.SphericalIsothermalMP(einstein_radius=1.0)

            g0 = galaxy.Galaxy(redshift=0.1, pixelization=MockPixelization(value=0.5), mass_profile=sis)
            g1 = galaxy.Galaxy(redshift=1.0, pixelization=MockPixelization(value=1), mass_profile=sis)
            g2 = galaxy.Galaxy(redshift=2.0, pixelization=MockPixelization(value=2), mass_profile=sis)

            tracing = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=grids,
                                              cosmology=cosmo.Planck15)

            reconstructors = tracing.reconstructors_from_planes(MockBorders(), sparse_mask)

            assert reconstructors == [0.5, 1, 2]


class TestDeflectionAnglesViaGalaxy(object):

    def test_all_coordinates(self, grids, galaxy_mass_sis):
        deflections = ray_tracing.deflections_for_grids(grids, [galaxy_mass_sis])

        assert deflections.image[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
        assert deflections.sub[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
        assert deflections.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        #    assert deflections.sub.sub_grid_size == 2
        assert deflections.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

    def test_three_identical_lenses__deflection_angles_triple(self, grids, galaxy_mass_sis):
        deflections = ray_tracing.deflections_for_grids(grids,
                                                        [galaxy_mass_sis, galaxy_mass_sis,
                                                         galaxy_mass_sis])

        assert deflections.image[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
        assert deflections.sub[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
        assert deflections.sub[1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
        #    assert deflections.sub.sub_grid_size == 2
        assert deflections.blurring[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

    def test_one_lens_with_three_identical_mass_profiles__deflection_angles_triple(self, grids,
                                                                                   lens_sis_x3):
        deflections = ray_tracing.deflections_for_grids(grids, [lens_sis_x3])

        assert deflections.image[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
        assert deflections.sub[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
        assert deflections.sub[1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
        #    assert deflections.sub.sub_grid_size == 2
        assert deflections.blurring[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)


class TestIntensityViaGrid:

    def test__no_galaxies__intensities_returned_as_0s(self, grids, galaxy_no_profiles):
        grids.image = np.array([[1.0, 1.0],
                                            [2.0, 2.0],
                                            [3.0, 3.0]])

        intensities = ray_tracing.intensities_via_grid(image_grid=grids.image,
                                                       galaxies=[galaxy_no_profiles])

        assert (intensities[0] == np.array([0.0, 0.0])).all()
        assert (intensities[1] == np.array([0.0, 0.0])).all()
        assert (intensities[2] == np.array([0.0, 0.0])).all()

    def test__galaxy_sersic_light__intensities_returned_as_correct_values(self, grids,
                                                                          galaxy_light_sersic):
        grids.image = np.array([[1.0, 1.0],
                                            [1.0, 0.0],
                                            [-1.0, 0.0]])

        galaxy_intensities = galaxy_light_sersic.intensity_from_grid(grids.image)

        tracer_intensities = ray_tracing.intensities_via_grid(image_grid=grids.image,
                                                              galaxies=[galaxy_light_sersic])

        assert (galaxy_intensities == tracer_intensities).all()

    def test__galaxy_sis_mass_x3__intensities_tripled_from_above(self, grids,
                                                                 galaxy_light_sersic):
        grids.image = np.array([[1.0, 1.0],
                                            [1.0, 0.0],
                                            [-1.0, 0.0]])

        galaxy_intensities = galaxy_light_sersic.intensity_from_grid(grids.image)

        tracer_intensities = ray_tracing.intensities_via_grid(image_grid=grids.image,
                                                              galaxies=[galaxy_light_sersic, galaxy_light_sersic,
                                                                        galaxy_light_sersic])

        assert (3.0 * galaxy_intensities == tracer_intensities).all()


class TestIntensitiesViaSubGrid:

    def test__no_galaxies__intensities_returned_as_0s(self, grids, galaxy_no_profiles):
        intensities = ray_tracing.intensities_via_sub_grid(sub_grid=grids.sub,
                                                           galaxies=[galaxy_no_profiles])

        assert intensities[0] == 0.0

    def test__galaxy_light_sersic__intensities_returned_as_correct_values(self, grids,
                                                                          galaxy_light_sersic):
        galaxy_image = galaxy_light_sersic.intensity_from_grid(grids.sub)

        galaxy_image = (galaxy_image[0] + galaxy_image[1] + galaxy_image[2] +
                            galaxy_image[3]) / 4.0

        tracer_intensities = ray_tracing.intensities_via_sub_grid(sub_grid=grids.sub,
                                                                  galaxies=[galaxy_light_sersic])

        assert tracer_intensities[0] == galaxy_image

    def test__galaxy_light_sersic_x3__deflections_tripled_from_above(self, grids,
                                                                     galaxy_light_sersic):
        galaxy_image = galaxy_light_sersic.intensity_from_grid(grids.sub)

        galaxy_image = (galaxy_image[0] + galaxy_image[1] + galaxy_image[2] +
                            galaxy_image[3]) / 4.0

        tracer_intensities = ray_tracing.intensities_via_sub_grid(sub_grid=grids.sub,
                                                                  galaxies=[galaxy_light_sersic, galaxy_light_sersic,
                                                                            galaxy_light_sersic])

        assert tracer_intensities[0] == pytest.approx(3.0 * galaxy_image, 1e-4)


class TestSetupTracedGrid:

    def test__simple_sis_model__deflection_angles(self, grids, galaxy_mass_sis):
        deflections = ray_tracing.deflections_for_grids(grids, [galaxy_mass_sis])

        grid_traced = ray_tracing.traced_collection_for_deflections(grids, deflections)

        assert grid_traced.image[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-2)

    def test_three_identical_lenses__deflection_angles_triple(self, grids, galaxy_mass_sis):
        deflections = ray_tracing.deflections_for_grids(grids, [galaxy_mass_sis,
                                                                galaxy_mass_sis,
                                                                galaxy_mass_sis])

        grid_traced = ray_tracing.traced_collection_for_deflections(grids, deflections)

        assert grid_traced.image[0] == pytest.approx(np.array([1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]), 1e-3)

    def test_one_lens_with_three_identical_mass_profiles__deflection_angles_triple(self, grids,
                                                                                   lens_sis_x3):
        deflections = ray_tracing.deflections_for_grids(grids, [lens_sis_x3])

        grid_traced = ray_tracing.traced_collection_for_deflections(grids, deflections)

        assert grid_traced.image[0] == pytest.approx(np.array([1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]), 1e-3)


class TestUniformGridFromLensedGrid:

    def test__3x3_grid__extracts_max_min_coordinates__creates_regular_grid_including_half_pixel_offset_from_edge(self):

        grid = np.array([[-1.5, -1.5], [1.5, 1.5]])

        source_plane_grid = ray_tracing.uniform_grid_from_lensed_grid(grid, shape=(3, 3))

        assert (source_plane_grid == np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                               [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                               [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]])).all()

    def test__3x3_grid__extracts_max_min_coordinates__ignores_other_coordinates_more_central(self):

        grid = np.array([[-1.5, -1.5], [1.5, 1.5], [0.1, -0.1], [-1.0, 0.6], [1.4, -1.3], [1.5, 1.5]])

        source_plane_grid = ray_tracing.uniform_grid_from_lensed_grid(grid, shape=(3, 3))

        assert (source_plane_grid == np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                               [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                               [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]])).all()

    def test__2x3_grid__shape_change_correct_and_coordinates_shift(self):

        grid = np.array([[-1.5, -1.5], [1.5, 1.5]])

        source_plane_grid = ray_tracing.uniform_grid_from_lensed_grid(grid, shape=(2, 3))

        assert (source_plane_grid == np.array([[-0.75, -1.0], [-0.75, 0.0], [-0.75, 1.0],
                                                [0.75, -1.0], [0.75, 0.0], [0.75, 1.0]])).all()

    def test__3x2_grid__shape_change_correct_and_coordinates_shift(self):

        grid = np.array([[-1.5, -1.5], [1.5, 1.5]])

        source_plane_grid = ray_tracing.uniform_grid_from_lensed_grid(grid, shape=(3, 2))

        assert (source_plane_grid == np.array([[-1.0, -0.75], [-1.0, 0.75],
                                               [ 0.0, -0.75], [ 0.0, 0.75],
                                               [ 1.0, -0.75], [ 1.0, 0.75]])).all()


class TestBooleanProperties(object):

    def test__has_galaxy_with_light_profile(self, grids):

        gal = galaxy.Galaxy()
        gal_lp = galaxy.Galaxy(light_profile=light_profiles.LightProfile())
        gal_mp = galaxy.Galaxy(mass_profile=mass_profiles.SphericalIsothermalMP())

        assert ray_tracing.TracerImageSourcePlanes([gal], [gal], grids).has_galaxy_with_light_profile == False
        assert ray_tracing.TracerImageSourcePlanes([gal_mp], [gal_mp], grids).has_galaxy_with_light_profile == False
        assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal_lp], grids).has_galaxy_with_light_profile == True
        assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal], grids).has_galaxy_with_light_profile == True
        assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal_mp], grids).has_galaxy_with_light_profile == True
        
    def test__has_galaxy_with_pixelization(self, grids):

        gal = galaxy.Galaxy()
        gal_lp = galaxy.Galaxy(light_profile=light_profiles.LightProfile())
        gal_pix = galaxy.Galaxy(pixelization=pixelization.Pixelization())

        assert ray_tracing.TracerImageSourcePlanes([gal], [gal], grids).has_galaxy_with_pixelization == False
        assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal_lp], grids).has_galaxy_with_pixelization == False
        assert ray_tracing.TracerImageSourcePlanes([gal_pix], [gal_pix], grids).has_galaxy_with_pixelization == True
        assert ray_tracing.TracerImageSourcePlanes([gal_pix], [gal], grids).has_galaxy_with_pixelization == True
        assert ray_tracing.TracerImageSourcePlanes([gal_pix], [gal_lp], grids).has_galaxy_with_pixelization == True
        
    def test__has_hyper_galaxy_with_pixelization(self, grids):

        gal = galaxy.Galaxy()
        gal_lp = galaxy.Galaxy(light_profile=light_profiles.LightProfile())
        gal_hyper = galaxy.Galaxy(hyper_galaxy=galaxy.HyperGalaxy())

        assert ray_tracing.TracerImageSourcePlanes([gal], [gal], grids).has_hyper_galaxy == False
        assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal_lp], grids).has_hyper_galaxy == False
        assert ray_tracing.TracerImageSourcePlanes([gal_hyper], [gal_hyper], grids).has_hyper_galaxy == True
        assert ray_tracing.TracerImageSourcePlanes([gal_hyper], [gal], grids).has_hyper_galaxy == True
        assert ray_tracing.TracerImageSourcePlanes([gal_hyper], [gal_lp], grids).has_hyper_galaxy == True