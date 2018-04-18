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
def lens_sis():
    sis = mass_profiles.SphericalIsothermal(einstein_radius=1.0)
    lens_sis = galaxy.Galaxy(mass_profiles=[sis])
    return lens_sis


class TestLensPlane(object):


    class TestBasicSetup(object):

        def test__setup_only_grid__simple_lens(self, lens_sis):

            grid = np.array([[1.0, 1.0]])

            grid = grids.GridImage(grid)

            lens_plane_grid = grids.RayTracingGrids(grid)

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis], grids=lens_plane_grid)

            assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub == None
            assert lens_plane.grids.blurring == None

            assert lens_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub == None
            assert lens_plane.deflection_angles.blurring == None

        def test__setup_all_grid(self, lens_sis):

            grid = np.array([[1.0, 1.0]])
            sub_grid = np.array([[[1.0, 1.0], [1.0, 0.0]]])
            blurring_grid = np.array([[1.0, 0.0]])

            grid = grids.GridImage(grid)
            sub_grid = grids.GridImageSub(sub_grid, sub_grid_size=2)
            blurring_grid = grids.GridBlurring(blurring_grid)

            lens_plane_grid = grids.RayTracingGrids(grid, sub_grid, blurring_grid)

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis], grids=lens_plane_grid)

            assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub.grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub.grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lens_plane.grids.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert lens_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub.grid[0,0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub.grid[0,1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lens_plane.deflection_angles.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test_three_identical_lenses__deflection_angles_triple(self, lens_sis):

            grid = np.array([[1.0, 1.0]])
            sub_grid = np.array([[[1.0, 1.0], [1.0, 0.0]]])
            blurring_grid = np.array([[1.0, 0.0]])

            grid = grids.GridImage(grid)
            sub_grid = grids.GridImageSub(sub_grid, sub_grid_size=2)
            blurring_grid = grids.GridBlurring(blurring_grid)

            lens_plane_grid = grids.RayTracingGrids(grid, sub_grid, blurring_grid)

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis, lens_sis, lens_sis], grids=lens_plane_grid)

            assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub.grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub.grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert lens_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub.grid[0,0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub.grid[0,1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflection_angles.blurring.grid[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test_one_lens_with_three_identical_mass_profiles__deflection_angles_triple(self):

            grid = np.array([[1.0, 1.0]])
            sub_grid = np.array([[[1.0, 1.0], [1.0, 0.0]]])
            blurring_grid = np.array([[1.0, 0.0]])

            grid = grids.GridImage(grid)
            sub_grid = grids.GridImageSub(sub_grid, sub_grid_size=2)
            blurring_grid = grids.GridBlurring(blurring_grid)

            lens_sis_x3 = galaxy.Galaxy(mass_profiles=3*[mass_profiles.SphericalIsothermal(einstein_radius=1.0)])

            lens_plane_grid = grids.RayTracingGrids(grid, sub_grid, blurring_grid)

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis_x3],
                                               grids=lens_plane_grid)

            assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub.grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub.grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert lens_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub.grid[0,0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub.grid[0,1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflection_angles.blurring.grid[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test__complex_mass_model(self):

            grid = np.array([[1.0, 1.0]])
            sub_grid = np.array([[[1.0, 1.0], [1.0, 0.0]]])
            blurring_grid = np.array([[1.0, 0.0]])

            grid = grids.GridImage(grid)
            sub_grid = grids.GridImageSub(sub_grid, sub_grid_size=2)
            blurring_grid = grids.GridBlurring(blurring_grid)

            power_law = mass_profiles.EllipticalPowerLaw(centre=(1.0, 4.0), axis_ratio=0.7, phi=30.0,
                                                                    einstein_radius=1.0, slope=2.2)
            
            nfw = mass_profiles.SphericalNFW(kappa_s=0.1, scale_radius=5.0)

            lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profiles=[power_law, nfw])

            lens_plane_grid = grids.RayTracingGrids(grid, sub_grid, blurring_grid)

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_galaxy], grids=lens_plane_grid)

            assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub.grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub.grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            defls = power_law.deflection_angles_at_coordinates(grid.grid[0]) + \
                    nfw.deflection_angles_at_coordinates(grid.grid[0])

            sub_defls_0 = power_law.deflection_angles_at_coordinates(sub_grid.grid[0,0]) + \
                          nfw.deflection_angles_at_coordinates(sub_grid.grid[0,0])

            sub_defls_1 = power_law.deflection_angles_at_coordinates(sub_grid.grid[0,1]) + \
                          nfw.deflection_angles_at_coordinates(sub_grid.grid[0,1])

            blurring_defls = power_law.deflection_angles_at_coordinates(blurring_grid.grid[0]) + \
                           nfw.deflection_angles_at_coordinates(blurring_grid.grid[0])

            assert lens_plane.deflection_angles.image.grid[0] == pytest.approx(defls, 1e-3)
            assert lens_plane.deflection_angles.sub.grid[0,0] == pytest.approx(sub_defls_0, 1e-3)
            assert lens_plane.deflection_angles.sub.grid[0,1] == pytest.approx(sub_defls_1, 1e-3)
            assert lens_plane.deflection_angles.blurring.grid[0] == pytest.approx(blurring_defls, 1e-3)


class TestImagePlane(object):

    def test__inheritance_from_lens_plane(self, lens_sis):

        grid = np.array([[1.0, 1.0]])
        sub_grid = np.array([[[1.0, 1.0], [1.0, 0.0]]])
        blurring_grid = np.array([[1.0, 0.0]])

        grid = grids.GridImage(grid)
        sub_grid = grids.GridImageSub(sub_grid, sub_grid_size=2)
        blurring_grid = grids.GridBlurring(blurring_grid)

        plane_grid = grids.RayTracingGrids(grid, sub_grid, blurring_grid)

        lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis], grids=plane_grid)

        image_plane = ray_tracing.ImagePlane(galaxies=[lens_sis], grids=plane_grid)

        assert (lens_plane.grids.image.grid[0] == image_plane.grids.image.grid[0]).all()
        assert (lens_plane.grids.sub.grid[0, 0] == image_plane.grids.sub.grid[0, 0]).all()
        assert (lens_plane.grids.sub.grid[0, 1] == image_plane.grids.sub.grid[0, 1]).all()
        assert (lens_plane.grids.blurring.grid[0] == image_plane.grids.blurring.grid[0]).all()

        assert (lens_plane.deflection_angles.image.grid[0] == image_plane.deflection_angles.image.grid[0]).all()
        assert (lens_plane.deflection_angles.sub.grid[0,0] == image_plane.deflection_angles.sub.grid[0,0]).all()
        assert (lens_plane.deflection_angles.sub.grid[0,1] == image_plane.deflection_angles.sub.grid[0,1]).all()
        assert (lens_plane.deflection_angles.blurring.grid[0] == image_plane.deflection_angles.blurring.grid[0]).all()


class TestSourcePlane(object):

    def test__setup_only_grid__simple_lens(self, lens_sis):

        grid = np.array([[1.0, 1.0]])

        grid = grids.GridImage(grid)

        lens_plane_grid = grids.RayTracingGrids(grid)

        lens_plane = ray_tracing.SourcePlane(galaxies=[lens_sis], grids=lens_plane_grid)

        assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert lens_plane.grids.sub == None
        assert lens_plane.grids.blurring == None


    def test__setup_all_grid(self, lens_sis):

        grid = np.array([[1.0, 1.0]])
        sub_grid = np.array([[[1.0, 1.0], [1.0, 0.0]]])
        blurring_grid = np.array([[1.0, 0.0]])

        grid = grids.GridImage(grid)
        sub_grid = grids.GridImageSub(sub_grid, sub_grid_size=2)
        blurring_grid = grids.GridBlurring(blurring_grid)

        lens_plane_grid = grids.RayTracingGrids(grid, sub_grid, blurring_grid)

        lens_plane = ray_tracing.SourcePlane(galaxies=[lens_sis], grids=lens_plane_grid)

        assert lens_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert lens_plane.grids.sub.grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
        assert lens_plane.grids.sub.grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
        assert lens_plane.grids.blurring.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)


class TestTraceImageAndSoure(object):

        def test__grid_only__no_galaxy__image_and_source_plane_setup(self, no_galaxies):

            grid = np.array([[1.0, 0.0]])

            grid = grids.GridImage(grid)

            ray_tracing_grid = grids.RayTracingGrids(grid)

            lensing = ray_tracing.TraceImageAndSource(lens_galaxies=no_galaxies, source_galaxies=no_galaxies,
                                                      image_plane_grids=ray_tracing_grid)

            assert lensing.image_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert lensing.source_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test__lens_galaxy_on__source_plane_is_deflected(self, lens_sis):

            grid = np.array([[1.0, 0.0]])

            grid = grids.GridImage(grid)

            ray_tracing_grid = grids.RayTracingGrids(grid)

            lensing = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_sis], source_galaxies=no_galaxies,
                                                      image_plane_grids=ray_tracing_grid)

            assert lensing.image_plane.grids.image.grid == pytest.approx(np.array([[1.0, 0.0]]), 1e-3)
            assert lensing.image_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.source_plane.grids.image.grid == pytest.approx(np.array([[0.0, 0.0]]), 1e-3)

        def test__two_lens_galaxies__source_plane_deflected_doubles(self, lens_sis):

            grid = np.array([[1.0, 0.0]])

            grid = grids.GridImage(grid)

            ray_tracing_grid = grids.RayTracingGrids(grid)

            lensing = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_sis, lens_sis], source_galaxies=no_galaxies,
                                                      image_plane_grids=ray_tracing_grid)

            assert lensing.image_plane.grids.image.grid == pytest.approx(np.array([[1.0, 0.0]]), 1e-3)
            assert lensing.image_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
            assert lensing.source_plane.grids.image.grid == pytest.approx(np.array([[-1.0, 0.0]]), 1e-3)

        def test__all_grid(self, lens_sis):

            grid = np.array([[1.0, 0.0]])
            sub_grid = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
            blurring_grid = np.array([[-1.0, -1.0]])

            grid = grids.GridImage(grid)
            sub_grid = grids.GridImageSub(sub_grid, sub_grid_size=2)
            blurring_grid = grids.GridBlurring(blurring_grid)

            image_plane_grid = grids.RayTracingGrids(grid, sub_grid, blurring_grid)

            lensing = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_sis], source_galaxies=no_galaxies,
                                                      image_plane_grids=image_plane_grid)

            assert lensing.image_plane.grids.image.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.grids.sub.grid[0,0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.grids.sub.grid[1,0] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
            assert lensing.image_plane.grids.blurring.grid[0] == pytest.approx(np.array([-1.0, -1.0]), 1e-3)

            assert lensing.image_plane.deflection_angles.image.grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.deflection_angles.sub.grid[0,0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.deflection_angles.sub.grid[1,0] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
            assert lensing.image_plane.deflection_angles.blurring.grid[0] == pytest.approx(np.array([-0.707, -0.707]), 1e-3)

            assert lensing.source_plane.grids.image.grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert lensing.source_plane.grids.sub.grid[0,0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert lensing.source_plane.grids.sub.grid[1,0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert lensing.source_plane.grids.blurring.grid[0] == pytest.approx(np.array([-0.293, -0.293]), 1e-3)