from auto_lens import ray_tracing
from auto_lens.imaging import grids
from auto_lens import galaxy
from auto_lens.profiles import geometry_profiles, mass_profiles, light_profiles

import pytest
from astropy import cosmology
import numpy as np

@pytest.fixture(scope='function')
def no_galaxies():
    return [galaxy.Galaxy()]

@pytest.fixture(scope='function')
def light_sersic():
    sersic = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                             sersic_index=4.0)
    return galaxy.Galaxy(light_profiles=[sersic])

@pytest.fixture(scope='function')
def lens_sis():
    sis = mass_profiles.SphericalIsothermal(einstein_radius=1.0)
    return galaxy.Galaxy(mass_profiles=[sis])

@pytest.fixture(scope='function')
def image_grid():
    grid = np.array([[1.0, 1.0]])
    grid = grids.GridImage(grid)
    image_grid = grids.RayTracingGrids(grid)
    return image_grid

@pytest.fixture(scope='function')
def all_grids():

    grid = np.array([[1.0, 1.0]])
    sub_grid = np.array([[[1.0, 1.0]], [[1.0, 0.0]]])
    blurring_grid = np.array([[1.0, 0.0]])

    grid = grids.GridImage(grid)
    sub_grid = grids.GridImageSub(sub_grid, sub_grid_size=2)
    blurring_grid = grids.GridBlurring(blurring_grid)

    all_grids = grids.RayTracingGrids(grid, sub_grid, blurring_grid)

    return all_grids



class TestTraceImageAndSoure(object):

    def test__image_grid__no_galaxy__image_and_source_planes_setup__same_coordinates(self, image_grid, no_galaxies):

        lensing = ray_tracing.TraceImageAndSource(lens_galaxies=no_galaxies, source_galaxies=no_galaxies,
                                                  image_plane_grids=image_grid)

        assert lensing.image_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert lensing.image_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
        assert lensing.source_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)

    def test__image_grid__sis_lens__image_coordinates_are_grid_and_source_plane_is_deflected(self, image_grid, lens_sis):

        lensing = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_sis], source_galaxies=no_galaxies,
                                                  image_plane_grids=image_grid)

        assert lensing.image_plane.grids.image.grid == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)
        assert lensing.image_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
        assert lensing.source_plane.grids.image.grid == pytest.approx(np.array([[1.0-0.707, 1.0-0.707]]), 1e-3)

    def test__image_grid__2_sis_lenses__same_as_above_but_deflections_double(self, image_grid, lens_sis):

        lensing = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_sis, lens_sis], source_galaxies=no_galaxies,
                                                  image_plane_grids=image_grid)

        assert lensing.image_plane.grids.image.grid == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)
        assert lensing.image_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([1.414, 1.414]), 1e-3)
        assert lensing.source_plane.grids.image.grid == pytest.approx(np.array([[1.0-1.414, 1.0-1.414]]), 1e-3)

    def test__all_grids__sis_lens__planes_setup_correctly(self, all_grids, lens_sis):

        lensing = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_sis], source_galaxies=no_galaxies,
                                                  image_plane_grids=all_grids)

        assert lensing.image_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert lensing.image_plane.grids.sub.grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert lensing.image_plane.grids.sub.grid[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        assert lensing.image_plane.grids.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        assert lensing.image_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
        assert lensing.image_plane.deflection_angles.sub.grid[0, 0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
        assert lensing.image_plane.deflection_angles.sub.grid[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        assert lensing.image_plane.deflection_angles.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        assert lensing.source_plane.grids.image.grid[0] == pytest.approx(np.array([1.0-0.707, 1.0-0.707]), 1e-3)
        assert lensing.source_plane.grids.sub.grid[0, 0] == pytest.approx(np.array([1.0-0.707, 1.0-0.707]), 1e-3)
        assert lensing.source_plane.grids.sub.grid[1, 0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
        assert lensing.source_plane.grids.blurring.grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)


class TestLensPlane(object):


    class TestBasicSetup(object):

        def test__image_grid__sis_lens__coordinates_and_deflections_setup_for_image_grid(self, image_grid, lens_sis):

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis], grids=image_grid)

            assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub == None
            assert lens_plane.grids.blurring == None

            assert lens_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub == None
            assert lens_plane.deflection_angles.blurring == None

        def test__all_grids__sis_lens__coordinates_and_deflections_setup_for_every_grid(self, all_grids, lens_sis):

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis], grids=all_grids)

            assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub.grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub.grid[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lens_plane.grids.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert lens_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub.grid[0,0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub.grid[1,0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lens_plane.deflection_angles.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test__all_grids__3_identical_sis_lenses__deflections_triple_compared_to_above(self, all_grids, lens_sis):

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis, lens_sis, lens_sis], grids=all_grids)

            assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub.grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub.grid[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert lens_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub.grid[0,0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub.grid[1,0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflection_angles.blurring.grid[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test__all_grids__lens_is_3_identical_sis_profiles__deflection_angles_triple_like_above(self, all_grids):

            lens_sis_x3 = galaxy.Galaxy(mass_profiles=3*[mass_profiles.SphericalIsothermal(einstein_radius=1.0)])

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis_x3], grids=all_grids)

            assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub.grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub.grid[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert lens_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub.grid[0,0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub.grid[1,0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflection_angles.blurring.grid[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test__all_grids__complex_mass_model__deflectionsl_in_grid_are_sum_of_individual_profiles(self, all_grids):

            power_law = mass_profiles.EllipticalPowerLaw(centre=(1.0, 4.0), axis_ratio=0.7, phi=30.0,
                                                                    einstein_radius=1.0, slope=2.2)
            
            nfw = mass_profiles.SphericalNFW(kappa_s=0.1, scale_radius=5.0)

            lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profiles=[power_law, nfw])

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_galaxy], grids=all_grids)

            assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub.grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub.grid[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            defls = power_law.deflection_angles_at_coordinates(all_grids.image.grid[0]) + \
                    nfw.deflection_angles_at_coordinates(all_grids.image.grid[0])

            sub_defls_0 = power_law.deflection_angles_at_coordinates(all_grids.sub.grid[0,0]) + \
                          nfw.deflection_angles_at_coordinates(all_grids.sub.grid[0,0])

            sub_defls_1 = power_law.deflection_angles_at_coordinates(all_grids.sub.grid[1,0]) + \
                          nfw.deflection_angles_at_coordinates(all_grids.sub.grid[1,0])

            blurring_defls = power_law.deflection_angles_at_coordinates(all_grids.blurring.grid[0]) + \
                           nfw.deflection_angles_at_coordinates(all_grids.blurring.grid[0])

            assert lens_plane.deflection_angles.image.grid[0] == pytest.approx(defls, 1e-3)
            assert lens_plane.deflection_angles.sub.grid[0,0] == pytest.approx(sub_defls_0, 1e-3)
            assert lens_plane.deflection_angles.sub.grid[1,0] == pytest.approx(sub_defls_1, 1e-3)
            assert lens_plane.deflection_angles.blurring.grid[0] == pytest.approx(blurring_defls, 1e-3)


class TestImagePlane(object):


    class TestBasicSetup:

        def test__inheritance_from_lens_plane(self, all_grids, lens_sis):

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis], grids=all_grids)

            image_plane = ray_tracing.ImagePlane(galaxies=[lens_sis], grids=all_grids)

            assert (lens_plane.grids.image.grid[0] == image_plane.grids.image.grid[0]).all()
            assert (lens_plane.grids.sub.grid[0,0] == image_plane.grids.sub.grid[0,0]).all()
            assert (lens_plane.grids.sub.grid[1,0] == image_plane.grids.sub.grid[1,0]).all()
            assert (lens_plane.grids.blurring.grid[0] == image_plane.grids.blurring.grid[0]).all()

            assert (lens_plane.deflection_angles.image.grid[0] == image_plane.deflection_angles.image.grid[0]).all()
            assert (lens_plane.deflection_angles.sub.grid[0,0] == image_plane.deflection_angles.sub.grid[0,0]).all()
            assert (lens_plane.deflection_angles.sub.grid[1,0] == image_plane.deflection_angles.sub.grid[1,0]).all()
            assert (lens_plane.deflection_angles.blurring.grid[0] == image_plane.deflection_angles.blurring.grid[0]).all()


    class TestModelImages:

        def test__no_sub_grid__sersic_light_profile__image_intensities_equal_to_individual_profile(self, light_sersic):

            grid = np.array([[1.0, 1.0]])
            intensity_value = light_sersic.intensity_grid(grid)

            image_grid = grids.GridImage(grid)
            ray_grids = grids.RayTracingGrids(image_grid)

            image_plane = ray_tracing.ImagePlane(galaxies=[light_sersic], grids=ray_grids)
            image = image_plane.generate_light_profile_image(iterative_sub_grid=False)

            assert (image[0] == intensity_value).all()

        def test__same_as_above___but_with_multiple_sets_of_coordinates(self, light_sersic):

            grid = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, -9.0], [-3.2, -5.0]])
            intensity_values = light_sersic.intensity_grid(grid)

            image_grid = grids.GridImage(grid)
            ray_grids = grids.RayTracingGrids(image_grid)

            image_plane = ray_tracing.ImagePlane(galaxies=[light_sersic], grids=ray_grids)
            image = image_plane.generate_light_profile_image(iterative_sub_grid=False)

            assert (image[0] == intensity_values[0]).all()
            assert (image[1] == intensity_values[1]).all()
            assert (image[2] == intensity_values[2]).all()
            assert (image[3] == intensity_values[3]).all()

        def test__same_as_above__but_galaxy_entered_3_times__intensities_triple(self, light_sersic):

            grid = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, -9.0], [-3.2, -5.0]])
            intensity_values = light_sersic.intensity_grid(grid)

            image_grid = grids.GridImage(grid)
            ray_grids = grids.RayTracingGrids(image_grid)

            image_plane = ray_tracing.ImagePlane(galaxies=[light_sersic, light_sersic, light_sersic], grids=ray_grids)
            image = image_plane.generate_light_profile_image(iterative_sub_grid=False)

            assert (image[0] == intensity_values[0]*3.0).all()
            assert (image[1] == intensity_values[1]*3.0).all()
            assert (image[2] == intensity_values[2]*3.0).all()
            assert (image[3] == intensity_values[3]*3.0).all()


class TestSourcePlane(object):

    def test__image_grid__sis_lens__coordinates_and_deflections_setup_for_image_grid(self, image_grid, lens_sis):

        lens_plane = ray_tracing.SourcePlane(galaxies=[lens_sis], grids=image_grid)

        assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert lens_plane.grids.sub == None
        assert lens_plane.grids.blurring == None

    def test__all_grids__sis_lens__coordinates_and_deflections_setup_for_every_grid(self, all_grids, lens_sis):

        lens_plane = ray_tracing.SourcePlane(galaxies=[lens_sis], grids=all_grids)

        assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert lens_plane.grids.sub.grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert lens_plane.grids.sub.grid[1, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        assert lens_plane.grids.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)