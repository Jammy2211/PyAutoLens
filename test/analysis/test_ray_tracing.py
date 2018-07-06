from src.analysis import ray_tracing, galaxy
from src.imaging import grids
from src.profiles import mass_profiles, light_profiles

import pytest
import numpy as np


@pytest.fixture(scope='function')
def no_galaxies():
    return [galaxy.Galaxy()]


@pytest.fixture(scope='function')
def galaxy_light_sersic():
    sersic = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                             sersic_index=4.0)
    return galaxy.Galaxy(light_profile=sersic)


@pytest.fixture(scope='function')
def galaxy_mass_sis():
    sis = mass_profiles.SphericalIsothermal(einstein_radius=1.0)
    return galaxy.Galaxy(mass_profile=sis)


@pytest.fixture(scope='function')
def all_grids():
    regular_grid_coords = np.array([[1.0, 1.0]])
    sub_grid_coords = np.array([[[1.0, 1.0]], [[1.0, 0.0]]])
    blurring_grid_coords = np.array([[1.0, 0.0]])

    grid = grids.GridCoordsImage(regular_grid_coords)
    sub_grid = grids.GridCoordsImageSub(sub_grid_coords, grid_size_sub=2)
    blurring_grid = grids.GridCoordsBlurring(blurring_grid_coords)

    all_grids = grids.CoordsCollection(grid, sub_grid, blurring_grid)

    return all_grids


@pytest.fixture(name="galaxy_light_only")
def make_galaxy_light_only():
    return galaxy.Galaxy(light_profile=light_profiles.EllipticalSersic())


@pytest.fixture(name="galaxy_light_and_mass")
def make_galaxy_light_and_mass():
    return galaxy.Galaxy(light_profile=light_profiles.EllipticalSersic(),
                         mass_profile=mass_profiles.SphericalIsothermal())


@pytest.fixture(name='lens_sis_x3')
def make_lens_sis_x3():
    mass_profile = mass_profiles.SphericalIsothermal(einstein_radius=1.0)
    return galaxy.Galaxy(mass_profile_1=mass_profile, mass_profile_2=mass_profile,
                         mass_profile_3=mass_profile)


class TestTraceImageAndSoure(object):
    class TestSetup:

        def test__image_grid__no_galaxy__image_and_source_planes_setup__same_coordinates(self, all_grids, no_galaxies):
            ray_trace = ray_tracing.Tracer(lens_galaxies=no_galaxies, source_galaxies=no_galaxies,
                                           image_plane_grids=all_grids)

            assert ray_trace.image_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace.image_plane.deflections.image[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert ray_trace.source_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)

        def test__image_grid__sis_lens__image_coordinates_are_grid_and_source_plane_is_deflected(self, all_grids,
                                                                                                 galaxy_mass_sis):
            ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_mass_sis], source_galaxies=no_galaxies,
                                           image_plane_grids=all_grids)

            assert ray_trace.image_plane.grids.image == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)
            assert ray_trace.image_plane.deflections.image[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert ray_trace.source_plane.grids.image == pytest.approx(np.array([[1.0 - 0.707, 1.0 - 0.707]]), 1e-3)

        def test__image_grid__2_sis_lenses__same_as_above_but_deflections_double(self, all_grids, galaxy_mass_sis):
            ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_mass_sis, galaxy_mass_sis],
                                           source_galaxies=no_galaxies,
                                           image_plane_grids=all_grids)

            assert ray_trace.image_plane.grids.image == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)
            assert ray_trace.image_plane.deflections.image[0] == pytest.approx(np.array([1.414, 1.414]), 1e-3)
            assert ray_trace.source_plane.grids.image == pytest.approx(np.array([[1.0 - 1.414, 1.0 - 1.414]]), 1e-3)

        def test__all_grids__sis_lens__planes_setup_correctly(self, all_grids, galaxy_mass_sis):
            ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_mass_sis], source_galaxies=no_galaxies,
                                           image_plane_grids=all_grids)

            assert ray_trace.image_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace.image_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace.image_plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert ray_trace.image_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert ray_trace.image_plane.deflections.image[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert ray_trace.image_plane.deflections.sub[0, 0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert ray_trace.image_plane.deflections.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert ray_trace.image_plane.deflections.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert ray_trace.source_plane.grids.image[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)
            assert ray_trace.source_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)
            assert ray_trace.source_plane.grids.sub[1, 0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert ray_trace.source_plane.grids.blurring[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)

    class TestImageFromGalaxies:

        def test__no_galaxies__image_is_sum_of_image_plane_and_source_plane_images(self, all_grids, no_galaxies):
            image_plane = ray_tracing.ImagePlane(galaxies=no_galaxies, grids=all_grids)
            source_plane = ray_tracing.SourcePlane(galaxies=no_galaxies, grids=all_grids)
            plane_image = image_plane.generate_image_of_galaxy_light_profiles() + source_plane.generate_image_of_galaxy_light_profiles()

            ray_trace = ray_tracing.Tracer(lens_galaxies=no_galaxies, source_galaxies=no_galaxies,
                                           image_plane_grids=all_grids)
            ray_trace_image = ray_trace.generate_image_of_galaxy_light_profiles()

            assert (plane_image == ray_trace_image).all()

        def test__galaxy_light_sersic_no_mass__image_is_sum_of_image_plane_and_source_plane_images(self, all_grids,
                                                                                                   galaxy_light_only):
            image_plane = ray_tracing.ImagePlane(galaxies=[galaxy_light_only], grids=all_grids)
            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_light_only], grids=all_grids)
            plane_image = image_plane.generate_image_of_galaxy_light_profiles() + source_plane.generate_image_of_galaxy_light_profiles()

            ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_only],
                                           source_galaxies=[galaxy_light_only], image_plane_grids=all_grids)
            ray_trace_image = ray_trace.generate_image_of_galaxy_light_profiles()

            assert (plane_image == ray_trace_image).all()

        def test__galaxy_light_sersic_mass_sis__source_plane_image_includes_deflections(self, all_grids,
                                                                                        galaxy_light_and_mass):
            image_plane = ray_tracing.ImagePlane(galaxies=[galaxy_light_and_mass], grids=all_grids)
            deflections_grid = all_grids.deflection_grids_for_galaxies(galaxies=[galaxy_light_and_mass])
            source_grid = all_grids.traced_grids_for_deflections(deflections_grid)
            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_light_and_mass], grids=source_grid)
            plane_image = image_plane.generate_image_of_galaxy_light_profiles() + source_plane.generate_image_of_galaxy_light_profiles()

            ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_and_mass],
                                           source_galaxies=[galaxy_light_and_mass],
                                           image_plane_grids=all_grids)
            ray_trace_image = ray_trace.generate_image_of_galaxy_light_profiles()

            assert (plane_image == ray_trace_image).all()

    class TestBlurringImageFromGalaxies:

        def test__no_galaxies__image_is_sum_of_image_plane_and_source_plane_images(self, all_grids, no_galaxies):
            image_plane = ray_tracing.ImagePlane(galaxies=no_galaxies, grids=all_grids)
            source_plane = ray_tracing.SourcePlane(galaxies=no_galaxies, grids=all_grids)
            plane_image = image_plane.generate_blurring_image_of_galaxy_light_profiles() + \
                          source_plane.generate_blurring_image_of_galaxy_light_profiles()

            ray_trace = ray_tracing.Tracer(lens_galaxies=no_galaxies, source_galaxies=no_galaxies,
                                           image_plane_grids=all_grids)
            ray_trace_image = ray_trace.generate_blurring_image_of_galaxy_light_profiles()

            assert (plane_image == ray_trace_image).all()

        def test__galaxy_light_sersic_no_mass__image_is_sum_of_image_plane_and_source_plane_images(self, all_grids,
                                                                                                   galaxy_light_only):
            image_plane = ray_tracing.ImagePlane(galaxies=[galaxy_light_only], grids=all_grids)
            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_light_only], grids=all_grids)
            plane_image = image_plane.generate_blurring_image_of_galaxy_light_profiles() + \
                          source_plane.generate_blurring_image_of_galaxy_light_profiles()

            ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_only],
                                           source_galaxies=[galaxy_light_only],
                                           image_plane_grids=all_grids)
            ray_trace_image = ray_trace.generate_blurring_image_of_galaxy_light_profiles()

            assert (plane_image == ray_trace_image).all()

        def test__galaxy_light_sersic_mass_sis__source_plane_image_includes_deflections(self, all_grids,
                                                                                        galaxy_light_and_mass):
            image_plane = ray_tracing.ImagePlane(galaxies=[galaxy_light_and_mass], grids=all_grids)
            deflections_grid = all_grids.deflection_grids_for_galaxies(galaxies=[galaxy_light_and_mass])
            source_grid = all_grids.traced_grids_for_deflections(deflections_grid)
            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_light_and_mass], grids=source_grid)
            plane_image = image_plane.generate_blurring_image_of_galaxy_light_profiles() + \
                          source_plane.generate_blurring_image_of_galaxy_light_profiles()

            ray_trace = ray_tracing.Tracer(lens_galaxies=[galaxy_light_and_mass],
                                           source_galaxies=[galaxy_light_and_mass],
                                           image_plane_grids=all_grids)
            ray_trace_image = ray_trace.generate_blurring_image_of_galaxy_light_profiles()

            assert (plane_image == ray_trace_image).all()


class TestPlane(object):
    class TestBasicSetup:

        def test__all_grids__sis_lens__coordinates_and_deflections_setup_for_every_grid(self, all_grids,
                                                                                        galaxy_mass_sis):
            plane = ray_tracing.Plane(galaxies=[galaxy_mass_sis], grids=all_grids)

            assert plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

    class TestImageFromGalaxies:

        def test__sersic_light_profile__intensities_equal_to_profile_and_galaxy_values(self, all_grids,
                                                                                       galaxy_light_sersic):
            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity = sersic.intensity_at_coordinates(all_grids.image[0])

            galaxy_intensity = all_grids.image.intensities_via_grid(galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=all_grids)
            image = plane.generate_image_of_galaxy_light_profiles()

            assert (image[0] == profile_intensity == galaxy_intensity).all()

        def test__same_as_above__now_with_multiple_sets_of_coordinates(self, all_grids, galaxy_light_sersic):
            all_grids.image = grids.GridCoordsImage([[1.0, 1.0], [3.0, 3.0], [5.0, -9.0], [-3.2, -5.0]])

            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity_0 = sersic.intensity_at_coordinates(all_grids.image[0])
            profile_intensity_1 = sersic.intensity_at_coordinates(all_grids.image[1])
            profile_intensity_2 = sersic.intensity_at_coordinates(all_grids.image[2])
            profile_intensity_3 = sersic.intensity_at_coordinates(all_grids.image[3])

            galaxy_intensity = all_grids.image.intensities_via_grid(galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=all_grids)
            image = plane.generate_image_of_galaxy_light_profiles()

            assert (image[0] == profile_intensity_0 == galaxy_intensity[0]).all()
            assert (image[1] == profile_intensity_1 == galaxy_intensity[1]).all()
            assert (image[2] == profile_intensity_2 == galaxy_intensity[2]).all()
            assert (image[3] == profile_intensity_3 == galaxy_intensity[3]).all()

        def test__same_as_above__now_galaxy_entered_3_times__intensities_triple(self, galaxy_light_sersic):
            all_grids.image = grids.GridCoordsImage([[1.0, 1.0], [3.0, 3.0], [5.0, -9.0], [-3.2, -5.0]])

            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity_0 = sersic.intensity_at_coordinates(all_grids.image[0])
            profile_intensity_1 = sersic.intensity_at_coordinates(all_grids.image[1])
            profile_intensity_2 = sersic.intensity_at_coordinates(all_grids.image[2])
            profile_intensity_3 = sersic.intensity_at_coordinates(all_grids.image[3])

            galaxy_intensity = all_grids.image.intensities_via_grid(galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic, galaxy_light_sersic, galaxy_light_sersic],
                                      grids=all_grids)
            image = plane.generate_image_of_galaxy_light_profiles()

            assert (image[0] == 3.0 * profile_intensity_0 == 3.0 * galaxy_intensity[0]).all()
            assert (image[1] == 3.0 * profile_intensity_1 == 3.0 * galaxy_intensity[1]).all()
            assert (image[2] == 3.0 * profile_intensity_2 == 3.0 * galaxy_intensity[2]).all()
            assert (image[3] == 3.0 * profile_intensity_3 == 3.0 * galaxy_intensity[3]).all()

    class TestBlurringImageFromGalaxies:

        def test__sersic_light_profile__intensities_equal_to_profile_and_galaxy_values(self, all_grids,
                                                                                       galaxy_light_sersic):
            all_grids.image = grids.GridCoordsImage(np.array([[9.0, 9.0]]))
            all_grids.blurring = grids.GridCoordsBlurring(np.array([[1.0, 1.0]]))

            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity = sersic.intensity_at_coordinates(all_grids.blurring[0])

            blurring_galaxy_intensity = all_grids.blurring.intensities_via_grid(galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=all_grids)
            blurring_image = plane.generate_blurring_image_of_galaxy_light_profiles()

            assert (blurring_image[0] == profile_intensity == blurring_galaxy_intensity).all()

        def test__same_as_above__now_with_multiple_sets_of_coordinates(self, all_grids, galaxy_light_sersic):
            all_grids.image = grids.GridCoordsImage(np.array([[1.0, 1.0], [3.0, 3.0], [5.0, -9.0], [-3.2, -5.0]]))
            all_grids.blurring = grids.GridCoordsBlurring(np.array([[1.0, 1.0], [5.0, 5.0], [-2.0, -9.0], [5.0, 7.0]]))

            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity_0 = sersic.intensity_at_coordinates(all_grids.blurring[0])
            profile_intensity_1 = sersic.intensity_at_coordinates(all_grids.blurring[1])
            profile_intensity_2 = sersic.intensity_at_coordinates(all_grids.blurring[2])
            profile_intensity_3 = sersic.intensity_at_coordinates(all_grids.blurring[3])

            blurring_galaxy_intensity = all_grids.blurring.intensities_via_grid(galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=all_grids)
            blurring_image = plane.generate_blurring_image_of_galaxy_light_profiles()

            assert (blurring_image[0] == profile_intensity_0 == blurring_galaxy_intensity[0]).all()
            assert (blurring_image[1] == profile_intensity_1 == blurring_galaxy_intensity[1]).all()
            assert (blurring_image[2] == profile_intensity_2 == blurring_galaxy_intensity[2]).all()
            assert (blurring_image[3] == profile_intensity_3 == blurring_galaxy_intensity[3]).all()

        def test__same_as_above__now_galaxy_entered_3_times__intensities_triple(self, galaxy_light_sersic):
            all_grids.image = grids.GridCoordsImage(np.array([[1.0, 1.0], [3.0, 3.0], [5.0, -9.0], [-3.2, -5.0]]))
            all_grids.blurring = grids.GridCoordsBlurring(np.array([[1.0, 1.0], [5.0, 5.0], [-2.0, -9.0], [5.0, 7.0]]))

            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity_0 = sersic.intensity_at_coordinates(all_grids.blurring[0])
            profile_intensity_1 = sersic.intensity_at_coordinates(all_grids.blurring[1])
            profile_intensity_2 = sersic.intensity_at_coordinates(all_grids.blurring[2])
            profile_intensity_3 = sersic.intensity_at_coordinates(all_grids.blurring[3])

            blurring_galaxy_intensity = all_grids.blurring.intensities_via_grid(galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic, galaxy_light_sersic, galaxy_light_sersic],
                                      grids=all_grids)
            blurring_image = plane.generate_blurring_image_of_galaxy_light_profiles()

            assert (blurring_image[0] == 3.0 * profile_intensity_0 == 3.0 * blurring_galaxy_intensity[0]).all()
            assert (blurring_image[1] == 3.0 * profile_intensity_1 == 3.0 * blurring_galaxy_intensity[1]).all()
            assert (blurring_image[2] == 3.0 * profile_intensity_2 == 3.0 * blurring_galaxy_intensity[2]).all()
            assert (blurring_image[3] == 3.0 * profile_intensity_3 == 3.0 * blurring_galaxy_intensity[3]).all()


class TestLensPlane(object):
    class TestBasicSetup(object):

        def test__all_grids__sis_lens__coordinates_and_deflections_setup_for_every_grid(self, all_grids,
                                                                                        galaxy_mass_sis):
            lens_plane = ray_tracing.LensPlane(galaxies=[galaxy_mass_sis], grids=all_grids)

            assert lens_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lens_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert lens_plane.deflections.image[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflections.sub[0, 0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflections.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lens_plane.deflections.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test__all_grids__3_identical_sis_lenses__deflections_triple_compared_to_above(self, all_grids,
                                                                                          galaxy_mass_sis):
            lens_plane = ray_tracing.LensPlane(galaxies=[galaxy_mass_sis, galaxy_mass_sis, galaxy_mass_sis],
                                               grids=all_grids)

            assert lens_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert lens_plane.deflections.image[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[0, 0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[1, 0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflections.blurring[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test__all_grids__lens_is_3_identical_sis_profiles__deflections_triple_like_above(self, all_grids,
                                                                                             lens_sis_x3):
            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis_x3], grids=all_grids)

            assert lens_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert lens_plane.deflections.image[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[0, 0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[1, 0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflections.blurring[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        # def test__all_grids__complex_mass_model__deflections_in_grid_are_sum_of_individual_profiles(self, all_grids):
        #     power_law = mass_profiles.EllipticalPowerLaw(centre=(1.0, 4.0), axis_ratio=0.7, phi=30.0,
        #                                                  einstein_radius=1.0, slope=2.2)
        #
        #     nfw = mass_profiles.SphericalNFW(kappa_s=0.1, scale_radius=5.0)
        #
        #     lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profile_1=power_law, mass_profile_2=nfw)
        #
        #     lens_plane = ray_tracing.LensPlane(galaxies=[lens_galaxy], grids=all_grids)
        #
        #     assert lens_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
        #     assert lens_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
        #     assert lens_plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
        #     assert lens_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
        #
        #     defls = power_law.deflections_at_coordinates(all_grids.image[0]) + \
        #             nfw.deflections_at_coordinates(all_grids.image[0])
        #
        #     sub_defls_0 = power_law.deflections_at_coordinates(all_grids.sub[0, 0]) + \
        #                   nfw.deflections_at_coordinates(all_grids.sub[0, 0])
        #
        #     sub_defls_1 = power_law.deflections_at_coordinates(all_grids.sub[1, 0]) + \
        #                   nfw.deflections_at_coordinates(all_grids.sub[1, 0])
        #
        #     blurring_defls = power_law.deflections_at_coordinates(all_grids.blurring[0]) + \
        #                      nfw.deflections_at_coordinates(all_grids.blurring[0])
        #
        #     assert lens_plane.deflections.image[0] == pytest.approx(defls, 1e-3)
        #     assert lens_plane.deflections.sub[0, 0] == pytest.approx(sub_defls_0, 1e-3)
        #     assert lens_plane.deflections.sub[1, 0] == pytest.approx(sub_defls_1, 1e-3)
        #     assert lens_plane.deflections.blurring[0] == pytest.approx(blurring_defls, 1e-3)

    class TestImageFromGalaxies:

        def test__sersic_light_profile__intensities_equal_to_profile_and_galaxy(self, all_grids, galaxy_light_sersic):
            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity = sersic.intensity_at_coordinates(all_grids.image[0])

            galaxy_intensity = all_grids.image.intensities_via_grid(galaxies=[galaxy_light_sersic])

            lens_plane = ray_tracing.LensPlane(galaxies=[galaxy_light_sersic], grids=all_grids)
            image = lens_plane.generate_image_of_galaxy_light_profiles()

            assert (image[0] == profile_intensity == galaxy_intensity).all()


class TestImagePlane(object):
    class TestBasicSetup:

        def test__inheritance_from_lens_plane(self, all_grids, galaxy_mass_sis):
            lens_plane = ray_tracing.LensPlane(galaxies=[galaxy_mass_sis], grids=all_grids)

            image_plane = ray_tracing.ImagePlane(galaxies=[galaxy_mass_sis], grids=all_grids)

            assert (lens_plane.grids.image[0] == image_plane.grids.image[0]).all()
            assert (lens_plane.grids.sub[0, 0] == image_plane.grids.sub[0, 0]).all()
            assert (lens_plane.grids.sub[1, 0] == image_plane.grids.sub[1, 0]).all()
            assert (lens_plane.grids.blurring[0] == image_plane.grids.blurring[0]).all()

            assert (lens_plane.deflections.image[0] == image_plane.deflections.image[0]).all()
            assert (lens_plane.deflections.sub[0, 0] == image_plane.deflections.sub[0, 0]).all()
            assert (lens_plane.deflections.sub[1, 0] == image_plane.deflections.sub[1, 0]).all()
            assert (lens_plane.deflections.blurring[0] == image_plane.deflections.blurring[0]).all()

    class TestImageFromGalaxies:

        def test__sersic_light_profile__intensities_equal_to_individual_profile(self, all_grids, galaxy_light_sersic):
            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity = sersic.intensity_at_coordinates(all_grids.image[0])

            galaxy_intensity = all_grids.image.intensities_via_grid(galaxies=[galaxy_light_sersic])

            image_plane = ray_tracing.ImagePlane(galaxies=[galaxy_light_sersic], grids=all_grids)
            image = image_plane.generate_image_of_galaxy_light_profiles()

            assert (image[0] == profile_intensity == galaxy_intensity).all()


class TestSourcePlane(object):
    class TestBasicSetup:

        def test__all_grids__sis_lens__coordinates_and_deflections_setup_for_every_grid(self, all_grids,
                                                                                        galaxy_mass_sis):
            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_mass_sis], grids=all_grids)

            assert source_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert source_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert source_plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert source_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

    class TestImageFromGalaxies:

        def test__sersic_light_profile__intensities_equal_to_individual_profile(self, all_grids, galaxy_light_sersic):
            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity = sersic.intensity_at_coordinates(all_grids.image[0])

            galaxy_intensity = all_grids.image.intensities_via_grid(galaxies=[galaxy_light_sersic])

            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_light_sersic], grids=all_grids)
            image = source_plane.generate_image_of_galaxy_light_profiles()

            assert (image[0] == profile_intensity == galaxy_intensity).all()
