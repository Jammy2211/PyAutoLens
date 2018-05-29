from auto_lens import ray_tracing
from auto_lens.imaging import grids
from auto_lens import galaxy
from auto_lens.profiles import mass_profiles, light_profiles

import pytest
from astropy import cosmology
import numpy as np

@pytest.fixture(scope='function')
def no_galaxies():
    return [galaxy.Galaxy()]

@pytest.fixture(scope='function')
def galaxy_light_sersic():
    sersic = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                             sersic_index=4.0)
    return galaxy.Galaxy(light_profiles=[sersic])

@pytest.fixture(scope='function')
def galaxy_mass_sis():
    sis = mass_profiles.SphericalIsothermal(einstein_radius=1.0)
    return galaxy.Galaxy(mass_profiles=[sis])

@pytest.fixture(scope='function')
def grid_image():
    regular_grid_coords = np.array([[1.0, 1.0]])
    grid = grids.GridCoordsImage(regular_grid_coords)
    grid_image = grids.GridCoordsCollection(grid)
    return grid_image

@pytest.fixture(scope='function')
def grid_image_and_blurring():

    regular_grid_coords = np.array([[1.0, 1.0]])
    grid_image = grids.GridCoordsImage(regular_grid_coords)

    blurring_grid_coords = np.array([[1.0, 1.0]])
    grid_blurring = grids.GridCoordsImage(blurring_grid_coords)

    grid_image_and_blurring = grids.GridCoordsCollection(image=grid_image, blurring=grid_blurring)
    return grid_image_and_blurring

@pytest.fixture(scope='function')
def all_grids():

    regular_grid_coords = np.array([[1.0, 1.0]])
    sub_grid_coords = np.array([[[1.0, 1.0]], [[1.0, 0.0]]])
    blurring_grid_coords = np.array([[1.0, 0.0]])

    grid = grids.GridCoordsImage(regular_grid_coords)
    sub_grid = grids.GridCoordsImageSub(sub_grid_coords, grid_size_sub=2)
    blurring_grid = grids.GridCoordsBlurring(blurring_grid_coords)

    all_grids = grids.GridCoordsCollection(grid, sub_grid, blurring_grid)

    return all_grids



class TestTraceImageAndSoure(object):


    class TestSetup:

        def test__image_grid__no_galaxy__image_and_source_planes_setup__same_coordinates(self, grid_image, no_galaxies):

            ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=no_galaxies, source_galaxies=no_galaxies,
                                                      image_plane_grids=grid_image)

            assert ray_trace.image_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace.image_plane.deflections.image[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert ray_trace.source_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)

        def test__image_grid__sis_lens__image_coordinates_are_grid_and_source_plane_is_deflected(self, grid_image, galaxy_mass_sis):

            ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[galaxy_mass_sis], source_galaxies=no_galaxies,
                                                      image_plane_grids=grid_image)

            assert ray_trace.image_plane.grids.image == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)
            assert ray_trace.image_plane.deflections.image[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert ray_trace.source_plane.grids.image == pytest.approx(np.array([[1.0-0.707, 1.0-0.707]]), 1e-3)

        def test__image_grid__2_sis_lenses__same_as_above_but_deflections_double(self, grid_image, galaxy_mass_sis):

            ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[galaxy_mass_sis, galaxy_mass_sis], source_galaxies=no_galaxies,
                                                      image_plane_grids=grid_image)

            assert ray_trace.image_plane.grids.image == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)
            assert ray_trace.image_plane.deflections.image[0] == pytest.approx(np.array([1.414, 1.414]), 1e-3)
            assert ray_trace.source_plane.grids.image == pytest.approx(np.array([[1.0-1.414, 1.0-1.414]]), 1e-3)

        def test__all_grids__sis_lens__planes_setup_correctly(self, all_grids, galaxy_mass_sis):

            ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[galaxy_mass_sis], source_galaxies=no_galaxies,
                                                      image_plane_grids=all_grids)

            assert ray_trace.image_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace.image_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace.image_plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert ray_trace.image_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert ray_trace.image_plane.deflections.image[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert ray_trace.image_plane.deflections.sub[0, 0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert ray_trace.image_plane.deflections.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert ray_trace.image_plane.deflections.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert ray_trace.source_plane.grids.image[0] == pytest.approx(np.array([1.0-0.707, 1.0-0.707]), 1e-3)
            assert ray_trace.source_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0-0.707, 1.0-0.707]), 1e-3)
            assert ray_trace.source_plane.grids.sub[1, 0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert ray_trace.source_plane.grids.blurring[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)


    class TestImageFromGalaxies:

        def test__no_galaxies__image_is_sum_of_image_plane_and_source_plane_images(self, grid_image, no_galaxies):

            image_plane = ray_tracing.ImagePlane(galaxies=no_galaxies, grids=grid_image)
            source_plane = ray_tracing.SourcePlane(galaxies=no_galaxies, grids=grid_image)
            plane_image = image_plane.generate_image_of_galaxy_light_profiles() + source_plane.generate_image_of_galaxy_light_profiles()

            ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=no_galaxies, source_galaxies=no_galaxies,
                                                      image_plane_grids=grid_image)
            ray_trace_image = ray_trace.generate_image_of_galaxy_light_profiles()

            assert (plane_image == ray_trace_image).all()

        def test__galaxy_light_sersic_no_mass__image_is_sum_of_image_plane_and_source_plane_images(self, grid_image):

            galaxy_light_only = galaxy.Galaxy(light_profiles=[light_profiles.EllipticalSersic()])

            image_plane = ray_tracing.ImagePlane(galaxies=[galaxy_light_only], grids=grid_image)
            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_light_only], grids=grid_image)
            plane_image = image_plane.generate_image_of_galaxy_light_profiles() + source_plane.generate_image_of_galaxy_light_profiles()

            ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[galaxy_light_only],
                                                        source_galaxies=[galaxy_light_only], image_plane_grids=grid_image)
            ray_trace_image = ray_trace.generate_image_of_galaxy_light_profiles()

            assert (plane_image == ray_trace_image).all()

        def test__galaxy_light_sersic_mass_sis__source_plane_image_includes_deflections(self, grid_image):

            galaxy_light_and_mass = galaxy.Galaxy(light_profiles=[light_profiles.EllipticalSersic()], 
                                    mass_profiles=[mass_profiles.SphericalIsothermal()])

            image_plane = ray_tracing.ImagePlane(galaxies=[galaxy_light_and_mass], grids=grid_image)
            deflections_grid = grid_image.deflection_grids_for_galaxies(galaxies=[galaxy_light_and_mass])
            source_grid = grid_image.traced_grids_for_deflections(deflections_grid)
            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_light_and_mass], grids=source_grid)
            plane_image = image_plane.generate_image_of_galaxy_light_profiles() + source_plane.generate_image_of_galaxy_light_profiles()

            ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[galaxy_light_and_mass],
                                                        source_galaxies=[galaxy_light_and_mass],
                                                        image_plane_grids=grid_image)
            ray_trace_image = ray_trace.generate_image_of_galaxy_light_profiles()

            assert (plane_image == ray_trace_image).all()


    class TestBlurringImageFromGalaxies:

        def test__no_galaxies__image_is_sum_of_image_plane_and_source_plane_images(self, grid_image_and_blurring,
                                                                                   no_galaxies):

            image_plane = ray_tracing.ImagePlane(galaxies=no_galaxies, grids=grid_image_and_blurring)
            source_plane = ray_tracing.SourcePlane(galaxies=no_galaxies, grids=grid_image_and_blurring)
            plane_image = image_plane.generate_blurring_image_of_galaxy_light_profiles() + \
                          source_plane.generate_blurring_image_of_galaxy_light_profiles()

            ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=no_galaxies, source_galaxies=no_galaxies,
                                                      image_plane_grids=grid_image_and_blurring)
            ray_trace_image = ray_trace.generate_blurring_image_of_galaxy_light_profiles()

            assert (plane_image == ray_trace_image).all()

        def test__galaxy_light_sersic_no_mass__image_is_sum_of_image_plane_and_source_plane_images(self,
                                                                                                   grid_image_and_blurring):

            galaxy_light_only = galaxy.Galaxy(light_profiles=[light_profiles.EllipticalSersic()])

            image_plane = ray_tracing.ImagePlane(galaxies=[galaxy_light_only], grids=grid_image_and_blurring)
            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_light_only], grids=grid_image_and_blurring)
            plane_image = image_plane.generate_blurring_image_of_galaxy_light_profiles() + \
                          source_plane.generate_blurring_image_of_galaxy_light_profiles()

            ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[galaxy_light_only],
                                                        source_galaxies=[galaxy_light_only],
                                                      image_plane_grids=grid_image_and_blurring)
            ray_trace_image = ray_trace.generate_blurring_image_of_galaxy_light_profiles()

            assert (plane_image == ray_trace_image).all()

        def test__galaxy_light_sersic_mass_sis__source_plane_image_includes_deflections(self, grid_image_and_blurring):

            galaxy_light_and_mass = galaxy.Galaxy(light_profiles=[light_profiles.EllipticalSersic()],
                                    mass_profiles=[mass_profiles.SphericalIsothermal()])

            image_plane = ray_tracing.ImagePlane(galaxies=[galaxy_light_and_mass], grids=grid_image_and_blurring)
            deflections_grid = grid_image_and_blurring.deflection_grids_for_galaxies(galaxies=[galaxy_light_and_mass])
            source_grid = grid_image_and_blurring.traced_grids_for_deflections(deflections_grid)
            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_light_and_mass], grids=source_grid)
            plane_image = image_plane.generate_blurring_image_of_galaxy_light_profiles() + \
                          source_plane.generate_blurring_image_of_galaxy_light_profiles()

            ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[galaxy_light_and_mass],
                                                        source_galaxies=[galaxy_light_and_mass],
                                                        image_plane_grids=grid_image_and_blurring)
            ray_trace_image = ray_trace.generate_blurring_image_of_galaxy_light_profiles()

            assert (plane_image == ray_trace_image).all()


class TestPlane(object):


    class TestBasicSetup:

        def test__image_grid__sis_lens__coordinates_setup_for_image_grid(self, grid_image, galaxy_mass_sis):

            plane = ray_tracing.Plane(galaxies=[galaxy_mass_sis], grids=grid_image)

            assert plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert plane.grids.sub == None
            assert plane.grids.blurring == None

        def test__all_grids__sis_lens__coordinates_and_deflections_setup_for_every_grid(self, all_grids, galaxy_mass_sis):

            plane = ray_tracing.Plane(galaxies=[galaxy_mass_sis], grids=all_grids)

            assert plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)


    class TestImageFromGalaxies:

        def test__sersic_light_profile__intensities_equal_to_profile_and_galaxy_values(self, galaxy_light_sersic):

            regular_grid_coords = np.array([[1.0, 1.0]])
            grid_image = grids.GridCoordsImage(regular_grid_coords)
            grid_collection = grids.GridCoordsCollection(grid_image)
            
            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity = sersic.intensity_at_coordinates(regular_grid_coords[0])
            
            galaxy_intensity = grid_image.intensities_via_grid(galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=grid_collection)
            image = plane.generate_image_of_galaxy_light_profiles()

            assert (image[0] == profile_intensity == galaxy_intensity).all()

        def test__same_as_above__now_with_multiple_sets_of_coordinates(self, galaxy_light_sersic):

            regular_grid_coords = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, -9.0], [-3.2, -5.0]])
            grid_image = grids.GridCoordsImage(regular_grid_coords)
            grid_collection = grids.GridCoordsCollection(grid_image)

            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity_0 = sersic.intensity_at_coordinates(regular_grid_coords[0])
            profile_intensity_1 = sersic.intensity_at_coordinates(regular_grid_coords[1])
            profile_intensity_2 = sersic.intensity_at_coordinates(regular_grid_coords[2])
            profile_intensity_3 = sersic.intensity_at_coordinates(regular_grid_coords[3])

            galaxy_intensity = grid_image.intensities_via_grid(galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=grid_collection)
            image = plane.generate_image_of_galaxy_light_profiles()

            assert (image[0] == profile_intensity_0 == galaxy_intensity[0]).all()
            assert (image[1] == profile_intensity_1 == galaxy_intensity[1]).all()
            assert (image[2] == profile_intensity_2 == galaxy_intensity[2]).all()
            assert (image[3] == profile_intensity_3 == galaxy_intensity[3]).all()

        def test__same_as_above__now_galaxy_entered_3_times__intensities_triple(self, galaxy_light_sersic):

            regular_grid_coords = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, -9.0], [-3.2, -5.0]])
            grid_image = grids.GridCoordsImage(regular_grid_coords)
            grid_collection = grids.GridCoordsCollection(grid_image)
            
            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity_0 = sersic.intensity_at_coordinates(regular_grid_coords[0])
            profile_intensity_1 = sersic.intensity_at_coordinates(regular_grid_coords[1])
            profile_intensity_2 = sersic.intensity_at_coordinates(regular_grid_coords[2])
            profile_intensity_3 = sersic.intensity_at_coordinates(regular_grid_coords[3])

            galaxy_intensity = grid_image.intensities_via_grid(galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic, galaxy_light_sersic, galaxy_light_sersic], grids=grid_collection)
            image = plane.generate_image_of_galaxy_light_profiles()

            assert (image[0] == 3.0*profile_intensity_0 == 3.0*galaxy_intensity[0]).all()
            assert (image[1] == 3.0*profile_intensity_1 == 3.0*galaxy_intensity[1]).all()
            assert (image[2] == 3.0*profile_intensity_2 == 3.0*galaxy_intensity[2]).all()
            assert (image[3] == 3.0*profile_intensity_3 == 3.0*galaxy_intensity[3]).all()


    class TestBlurringImageFromGalaxies:

        def test__sersic_light_profile__intensities_equal_to_profile_and_galaxy_values(self, galaxy_light_sersic):

            regular_grid_coords = np.array([[9.0, 9.0]])
            grid_image = grids.GridCoordsImage(regular_grid_coords)

            blurring_grid_coords = np.array([[1.0, 1.0]])
            grid_blurring = grids.GridCoordsBlurring(blurring_grid_coords)

            grid_collection = grids.GridCoordsCollection(image=grid_image, blurring=grid_blurring)

            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity = sersic.intensity_at_coordinates(blurring_grid_coords[0])

            blurring_galaxy_intensity = grid_blurring.intensities_via_grid(galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=grid_collection)
            blurring_image = plane.generate_blurring_image_of_galaxy_light_profiles()

            assert (blurring_image[0] == profile_intensity == blurring_galaxy_intensity).all()

        def test__same_as_above__now_with_multiple_sets_of_coordinates(self, galaxy_light_sersic):

            regular_grid_coords = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, -9.0], [-3.2, -5.0]])
            grid_image = grids.GridCoordsImage(regular_grid_coords)

            blurring_grid_coords = np.array([[1.0, 1.0], [5.0, 5.0], [-2.0, -9.0], [5.0, 7.0]])
            grid_blurring = grids.GridCoordsBlurring(blurring_grid_coords)

            grid_collection = grids.GridCoordsCollection(image=grid_image, blurring=grid_blurring)

            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity_0 = sersic.intensity_at_coordinates(blurring_grid_coords[0])
            profile_intensity_1 = sersic.intensity_at_coordinates(blurring_grid_coords[1])
            profile_intensity_2 = sersic.intensity_at_coordinates(blurring_grid_coords[2])
            profile_intensity_3 = sersic.intensity_at_coordinates(blurring_grid_coords[3])

            blurring_galaxy_intensity = grid_blurring.intensities_via_grid(galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic], grids=grid_collection)
            blurring_image = plane.generate_blurring_image_of_galaxy_light_profiles()

            assert (blurring_image[0] == profile_intensity_0 == blurring_galaxy_intensity[0]).all()
            assert (blurring_image[1] == profile_intensity_1 == blurring_galaxy_intensity[1]).all()
            assert (blurring_image[2] == profile_intensity_2 == blurring_galaxy_intensity[2]).all()
            assert (blurring_image[3] == profile_intensity_3 == blurring_galaxy_intensity[3]).all()

        def test__same_as_above__now_galaxy_entered_3_times__intensities_triple(self, galaxy_light_sersic):

            regular_grid_coords = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, -9.0], [-3.2, -5.0]])
            grid_image = grids.GridCoordsImage(regular_grid_coords)

            blurring_grid_coords = np.array([[1.0, 1.0], [5.0, 5.0], [-2.0, -9.0], [5.0, 7.0]])
            grid_blurring = grids.GridCoordsBlurring(blurring_grid_coords)

            grid_collection = grids.GridCoordsCollection(image=grid_image, blurring=grid_blurring)

            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity_0 = sersic.intensity_at_coordinates(blurring_grid_coords[0])
            profile_intensity_1 = sersic.intensity_at_coordinates(blurring_grid_coords[1])
            profile_intensity_2 = sersic.intensity_at_coordinates(blurring_grid_coords[2])
            profile_intensity_3 = sersic.intensity_at_coordinates(blurring_grid_coords[3])

            blurring_galaxy_intensity = grid_blurring.intensities_via_grid(galaxies=[galaxy_light_sersic])

            plane = ray_tracing.Plane(galaxies=[galaxy_light_sersic, galaxy_light_sersic, galaxy_light_sersic],
                                      grids=grid_collection)
            blurring_image = plane.generate_blurring_image_of_galaxy_light_profiles()

            assert (blurring_image[0] == 3.0 * profile_intensity_0 == 3.0 * blurring_galaxy_intensity[0]).all()
            assert (blurring_image[1] == 3.0 * profile_intensity_1 == 3.0 * blurring_galaxy_intensity[1]).all()
            assert (blurring_image[2] == 3.0 * profile_intensity_2 == 3.0 * blurring_galaxy_intensity[2]).all()
            assert (blurring_image[3] == 3.0 * profile_intensity_3 == 3.0 * blurring_galaxy_intensity[3]).all()


class TestLensPlane(object):


    class TestBasicSetup(object):

        def test__image_grid__sis_lens__coordinates_and_deflections_setup_for_image_grid(self, grid_image, galaxy_mass_sis):

            lens_plane = ray_tracing.LensPlane(galaxies=[galaxy_mass_sis], grids=grid_image)

            assert lens_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub == None
            assert lens_plane.grids.blurring == None

            assert lens_plane.deflections.image[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflections.sub == None
            assert lens_plane.deflections.blurring == None

        def test__all_grids__sis_lens__coordinates_and_deflections_setup_for_every_grid(self, all_grids, galaxy_mass_sis):

            lens_plane = ray_tracing.LensPlane(galaxies=[galaxy_mass_sis], grids=all_grids)

            assert lens_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lens_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert lens_plane.deflections.image[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflections.sub[0, 0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflections.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lens_plane.deflections.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test__all_grids__3_identical_sis_lenses__deflections_triple_compared_to_above(self, all_grids, galaxy_mass_sis):

            lens_plane = ray_tracing.LensPlane(galaxies=[galaxy_mass_sis, galaxy_mass_sis, galaxy_mass_sis], grids=all_grids)

            assert lens_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert lens_plane.deflections.image[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[0, 0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[1, 0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflections.blurring[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test__all_grids__lens_is_3_identical_sis_profiles__deflections_triple_like_above(self, all_grids):

            lens_sis_x3 = galaxy.Galaxy(mass_profiles=3*[mass_profiles.SphericalIsothermal(einstein_radius=1.0)])

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis_x3], grids=all_grids)

            assert lens_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert lens_plane.deflections.image[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[0, 0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert lens_plane.deflections.sub[1, 0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflections.blurring[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test__all_grids__complex_mass_model__deflectionsl_in_grid_are_sum_of_individual_profiles(self, all_grids):

            power_law = mass_profiles.EllipticalPowerLaw(centre=(1.0, 4.0), axis_ratio=0.7, phi=30.0,
                                                                    einstein_radius=1.0, slope=2.2)
            
            nfw = mass_profiles.SphericalNFW(kappa_s=0.1, scale_radius=5.0)

            lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profiles=[power_law, nfw])

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_galaxy], grids=all_grids)

            assert lens_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            defls = power_law.deflections_at_coordinates(all_grids.image[0]) + \
                    nfw.deflections_at_coordinates(all_grids.image[0])

            sub_defls_0 = power_law.deflections_at_coordinates(all_grids.sub[0, 0]) + \
                          nfw.deflections_at_coordinates(all_grids.sub[0, 0])

            sub_defls_1 = power_law.deflections_at_coordinates(all_grids.sub[1, 0]) + \
                          nfw.deflections_at_coordinates(all_grids.sub[1, 0])

            blurring_defls = power_law.deflections_at_coordinates(all_grids.blurring[0]) + \
                             nfw.deflections_at_coordinates(all_grids.blurring[0])

            assert lens_plane.deflections.image[0] == pytest.approx(defls, 1e-3)
            assert lens_plane.deflections.sub[0, 0] == pytest.approx(sub_defls_0, 1e-3)
            assert lens_plane.deflections.sub[1, 0] == pytest.approx(sub_defls_1, 1e-3)
            assert lens_plane.deflections.blurring[0] == pytest.approx(blurring_defls, 1e-3)


    class TestImageFromGalaxies:

        def test__sersic_light_profile__intensities_equal_to_profile_and_galaxy(self, galaxy_light_sersic):

            regular_grid_coords = np.array([[1.0, 1.0]])
            grid_image = grids.GridCoordsImage(regular_grid_coords)
            grid_collection = grids.GridCoordsCollection(grid_image)
            
            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity = sersic.intensity_at_coordinates(regular_grid_coords[0])
            
            galaxy_intensity = grid_image.intensities_via_grid(galaxies=[galaxy_light_sersic])

            lens_plane = ray_tracing.LensPlane(galaxies=[galaxy_light_sersic], grids=grid_collection)
            image = lens_plane.generate_image_of_galaxy_light_profiles()

            assert (image[0] == profile_intensity == galaxy_intensity).all()


class TestImagePlane(object):


    class TestBasicSetup:

        def test__inheritance_from_lens_plane(self, all_grids, galaxy_mass_sis):

            lens_plane = ray_tracing.LensPlane(galaxies=[galaxy_mass_sis], grids=all_grids)

            image_plane = ray_tracing.ImagePlane(galaxies=[galaxy_mass_sis], grids=all_grids)

            assert (lens_plane.grids.image[0] == image_plane.grids.image[0]).all()
            assert (lens_plane.grids.sub[0,0] == image_plane.grids.sub[0,0]).all()
            assert (lens_plane.grids.sub[1,0] == image_plane.grids.sub[1,0]).all()
            assert (lens_plane.grids.blurring[0] == image_plane.grids.blurring[0]).all()

            assert (lens_plane.deflections.image[0] == image_plane.deflections.image[0]).all()
            assert (lens_plane.deflections.sub[0, 0] == image_plane.deflections.sub[0, 0]).all()
            assert (lens_plane.deflections.sub[1, 0] == image_plane.deflections.sub[1, 0]).all()
            assert (lens_plane.deflections.blurring[0] == image_plane.deflections.blurring[0]).all()


    class TestImageFromGalaxies:

        def test__sersic_light_profile__intensities_equal_to_individual_profile(self, galaxy_light_sersic):

            regular_grid_coords = np.array([[1.0, 1.0]])
            grid_image = grids.GridCoordsImage(regular_grid_coords)
            grid_collection = grids.GridCoordsCollection(grid_image)

            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity = sersic.intensity_at_coordinates(regular_grid_coords[0])
            
            galaxy_intensity = grid_image.intensities_via_grid(galaxies=[galaxy_light_sersic])

            image_plane = ray_tracing.ImagePlane(galaxies=[galaxy_light_sersic], grids=grid_collection)
            image = image_plane.generate_image_of_galaxy_light_profiles()

            assert (image[0] == profile_intensity == galaxy_intensity).all()


class TestSourcePlane(object):
    
    class TestBasicSetup:

        def test__image_grid__sis_lens__coordinates_and_deflections_setup_for_image_grid(self, grid_image, galaxy_mass_sis):
    
            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_mass_sis], grids=grid_image)
    
            assert source_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert source_plane.grids.sub == None
            assert source_plane.grids.blurring == None
    
        def test__all_grids__sis_lens__coordinates_and_deflections_setup_for_every_grid(self, all_grids, galaxy_mass_sis):
    
            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_mass_sis], grids=all_grids)
    
            assert source_plane.grids.image[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert source_plane.grids.sub[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert source_plane.grids.sub[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert source_plane.grids.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        
        
    class TestImageFromGalaxies:

        def test__sersic_light_profile__intensities_equal_to_individual_profile(self, galaxy_light_sersic):

            regular_grid_coords = np.array([[1.0, 1.0]])
            grid_image = grids.GridCoordsImage(regular_grid_coords)
            grid_collection = grids.GridCoordsCollection(grid_image)

            sersic = galaxy_light_sersic.light_profiles[0]
            profile_intensity = sersic.intensity_at_coordinates(regular_grid_coords[0])

            galaxy_intensity = grid_image.intensities_via_grid(galaxies=[galaxy_light_sersic])

            source_plane = ray_tracing.SourcePlane(galaxies=[galaxy_light_sersic], grids=grid_collection)
            image = source_plane.generate_image_of_galaxy_light_profiles()

            assert (image[0] == profile_intensity == galaxy_intensity).all()