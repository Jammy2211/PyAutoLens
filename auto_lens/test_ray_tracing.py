from auto_lens import ray_tracing
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


class TestPlaneCoordinates(object):

    class TestBasicSetup(object):

        def test__image_coordinates_only(self):

            coordinates = np.array([[1.0, 1.0]])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid == None
            assert ray_trace_coordinates.sparse_grid == None
            assert ray_trace_coordinates.blurring_grid == None

        def test__image_and_sub_coordinates(self):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 2.0]]])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates)

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 1] == pytest.approx(np.array([1.0, 2.0]), 1e-5)
            assert ray_trace_coordinates.sparse_grid == None
            assert ray_trace_coordinates.blurring_grid == None

        def test__image_and_sparse_coodinates(self):

            coordinates = np.array([[1.0, 1.0]])
            sparse_coordinates = np.array([[0.0, 0.0]])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates, sparse_grid=sparse_coordinates)

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid == None
            assert ray_trace_coordinates.sparse_grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-5)
            assert ray_trace_coordinates.blurring_grid == None

        def test__image_and_blurring_coordinates(self):

            coordinates = np.array([[1.0, 1.0]])
            blurring_coordinates = np.array([[0.0, 0.0]])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates, blurring_grid=blurring_coordinates)

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid == None
            assert ray_trace_coordinates.sparse_grid == None
            assert ray_trace_coordinates.blurring_grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-5)

        def test_all_coordinates(self):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 2.0]]])
            sparse_coordinates = np.array([[0.0, 0.0]])
            blurring_coordinates = np.array([[-3.0, 3.0]])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates,
                                                                 blurring_coordinates)

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 1] == pytest.approx(np.array([1.0, 2.0]), 1e-5)
            assert ray_trace_coordinates.sparse_grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-5)
            assert ray_trace_coordinates.blurring_grid[0] == pytest.approx(np.array([-3.0, 3.0]), 1e-5)


    class TestDeflectionAnglesForGalaxies(object):

        def test__image_coordinates_only(self, lens_sis):

            coordinates = np.array([[1.0, 1.0]])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates)
            ray_trace_deflection_angles = ray_trace_coordinates.deflection_angles_for_galaxies([lens_sis])

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid == None
            assert ray_trace_coordinates.sparse_grid == None
            assert ray_trace_coordinates.blurring_grid == None

            assert ray_trace_deflection_angles.image_grid == pytest.approx(np.array([[0.707, 0.707]]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid == None
            assert ray_trace_deflection_angles.sparse_grid == None
            assert ray_trace_deflection_angles.blurring_grid == None

        def test__image_and_sub_coordinates(self, lens_sis):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 0.0]]])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates)
            ray_trace_deflection_angles = ray_trace_coordinates.deflection_angles_for_galaxies([lens_sis])

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert ray_trace_coordinates.sparse_grid == None
            assert ray_trace_coordinates.blurring_grid == None

            assert ray_trace_deflection_angles.image_grid == pytest.approx(np.array([[0.707, 0.707]]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0,0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0,1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert ray_trace_deflection_angles.sparse_grid == None
            assert ray_trace_deflection_angles.blurring_grid == None

        def test__image_and_sparse_coodinates(self, lens_sis):

            coordinates = np.array([[1.0, 1.0]])
            sparse_coordinates = np.array([[1.0, 1.0]])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates, sparse_grid=sparse_coordinates)
            ray_trace_deflection_angles = ray_trace_coordinates.deflection_angles_for_galaxies([lens_sis])

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid == None
            assert ray_trace_coordinates.sparse_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.blurring_grid == None

            assert ray_trace_deflection_angles.image_grid == pytest.approx(np.array([[0.707, 0.707]]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid == None
            assert ray_trace_deflection_angles.sparse_grid == pytest.approx(np.array([[0.707, 0.707]]), 1e-3)
            assert ray_trace_deflection_angles.blurring_grid == None

        def test__image_and_blurring_coordinates(self, lens_sis):

            coordinates = np.array([[1.0, 1.0]])
            blurring_coordinates = np.array([[1.0, 0.0]])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates, blurring_grid=blurring_coordinates)
            ray_trace_deflection_angles = ray_trace_coordinates.deflection_angles_for_galaxies([lens_sis])

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid == None
            assert ray_trace_coordinates.sparse_grid == None
            assert ray_trace_coordinates.blurring_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert ray_trace_deflection_angles.image_grid == pytest.approx(np.array([[0.707, 0.707]]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid == None
            assert ray_trace_deflection_angles.sparse_grid == None
            assert ray_trace_deflection_angles.blurring_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test_all_coordinates(self, lens_sis):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 0.0]]])
            sparse_coordinates = np.array([[1.0, 1.0]])
            blurring_coordinates = np.array([[1.0, 0.0]])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates,
                                                                 blurring_coordinates)

            ray_trace_deflection_angles = ray_trace_coordinates.deflection_angles_for_galaxies([lens_sis])

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert ray_trace_coordinates.sparse_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.blurring_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert ray_trace_deflection_angles.image_grid[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0,0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0,1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert ray_trace_deflection_angles.sparse_grid[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert ray_trace_deflection_angles.blurring_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test_three_identical_lenses__deflection_angles_triple(self, lens_sis):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 0.0]]])
            sparse_coordinates = np.array([[1.0, 1.0]])
            blurring_coordinates = np.array([[1.0, 0.0]])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates,
                                                                 blurring_coordinates)

            ray_trace_deflection_angles = ray_trace_coordinates.deflection_angles_for_galaxies([lens_sis, lens_sis, lens_sis])

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert ray_trace_coordinates.sparse_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.blurring_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert ray_trace_deflection_angles.image_grid == pytest.approx(np.array([[3.0*0.707, 3.0*0.707]]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0,0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0,1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert ray_trace_deflection_angles.sparse_grid == pytest.approx(np.array([[3.0*0.707, 3.0*0.707]]), 1e-3)
            assert ray_trace_deflection_angles.blurring_grid[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test_one_lens_with_three_identical_mass_profiles__deflection_angles_triple(self):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 0.0]]])
            sparse_coordinates = np.array([[1.0, 1.0]])
            blurring_coordinates = np.array([[1.0, 0.0]])

            lens_sis_x3 = galaxy.Galaxy(mass_profiles=3*[mass_profiles.SphericalIsothermal(einstein_radius=1.0)])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates,
                                                                 blurring_coordinates)

            ray_trace_deflection_angles = ray_trace_coordinates.deflection_angles_for_galaxies([lens_sis_x3])

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert ray_trace_coordinates.sparse_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.blurring_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert ray_trace_deflection_angles.image_grid == pytest.approx(np.array([[3.0*0.707, 3.0*0.707]]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0,0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0,1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert ray_trace_deflection_angles.sparse_grid == pytest.approx(np.array([[3.0*0.707, 3.0*0.707]]), 1e-3)
            assert ray_trace_deflection_angles.blurring_grid[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test__complex_mass_model(self):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 0.0]]])
            sparse_coordinates = np.array([[1.0, 1.0]])
            blurring_coordinates = np.array([[1.0, 0.0]])

            power_law = mass_profiles.EllipticalPowerLaw(centre=(1.0, 4.0), axis_ratio=0.7, phi=30.0,
                                                                    einstein_radius=1.0, slope=2.2)
            nfw = mass_profiles.SphericalNFW(kappa_s=0.1, scale_radius=5.0)

            lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profiles=[power_law, nfw])

            ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates,
                                                                 blurring_coordinates)

            ray_trace_deflection_angles = ray_trace_coordinates.deflection_angles_for_galaxies([lens_galaxy])

            assert ray_trace_coordinates.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.sub_grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert ray_trace_coordinates.sparse_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert ray_trace_coordinates.blurring_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            defls = power_law.deflection_angles_at_coordinates(coordinates[0]) + \
                    nfw.deflection_angles_at_coordinates(coordinates[0])

            sub_defls_0 = power_law.deflection_angles_at_coordinates(sub_coordinates[0,0]) + \
                          nfw.deflection_angles_at_coordinates(sub_coordinates[0,0])

            sub_defls_1 = power_law.deflection_angles_at_coordinates(sub_coordinates[0,1]) + \
                          nfw.deflection_angles_at_coordinates(sub_coordinates[0,1])

            sparse_defls = power_law.deflection_angles_at_coordinates(sparse_coordinates[0]) + \
                           nfw.deflection_angles_at_coordinates(sparse_coordinates[0])

            blurring_defls = power_law.deflection_angles_at_coordinates(blurring_coordinates[0]) + \
                           nfw.deflection_angles_at_coordinates(blurring_coordinates[0])

            assert ray_trace_deflection_angles.image_grid[0] == pytest.approx(defls, 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0,0] == pytest.approx(sub_defls_0, 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0,1] == pytest.approx(sub_defls_1, 1e-3)
            assert ray_trace_deflection_angles.sparse_grid[0] == pytest.approx(sparse_defls, 1e-3)
            assert ray_trace_deflection_angles.blurring_grid[0] == pytest.approx(blurring_defls, 1e-3)


class TestPlaneDeflectionAngles(object):

    class TestBasicSetup(object):

        def test__image_deflection_angles_only(self):

            deflection_angles = np.array([[1.0, 1.0]])

            ray_trace_deflection_angles = ray_tracing.PlaneDeflectionAngles(deflection_angles)

            assert ray_trace_deflection_angles.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid == None
            assert ray_trace_deflection_angles.sparse_grid == None
            assert ray_trace_deflection_angles.blurring_grid == None

        def test__image_and_sub_deflection_angles(self):

            deflection_angles = np.array([[1.0, 1.0]])
            sub_deflection_angles = np.array([[[1.0, 1.0], [1.0, 2.0]]])

            ray_trace_deflection_angles = ray_tracing.PlaneDeflectionAngles(deflection_angles, sub_deflection_angles)

            assert ray_trace_deflection_angles.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0, 1] == pytest.approx(np.array([1.0, 2.0]), 1e-3)
            assert ray_trace_deflection_angles.sparse_grid == None
            assert ray_trace_deflection_angles.blurring_grid == None

        def test__image_and_sparse_coodinates(self):

            deflection_angles = np.array([[1.0, 1.0]])
            sparse_deflection_angles = np.array([[0.0, 0.0]])

            ray_trace_deflection_angles = ray_tracing.PlaneDeflectionAngles(deflection_angles,
                                                                            sparse_grid=sparse_deflection_angles)

            assert ray_trace_deflection_angles.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid == None
            assert ray_trace_deflection_angles.sparse_grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert ray_trace_deflection_angles.blurring_grid == None

        def test__image_and_blurring_deflection_angles(self):

            deflection_angles = np.array([[1.0, 1.0]])
            blurring_deflection_angles = np.array([[0.0, 0.0]])

            ray_trace_deflection_angles = ray_tracing.PlaneDeflectionAngles(deflection_angles,
                                                                            blurring_grid=blurring_deflection_angles)

            assert ray_trace_deflection_angles.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid == None
            assert ray_trace_deflection_angles.sparse_grid == None
            assert ray_trace_deflection_angles.blurring_grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)

        def test_all_deflection_angles(self):

            deflection_angles = np.array([[1.0, 1.0]])
            sub_deflection_angles = np.array([[[1.0, 1.0], [1.0, 2.0]]])
            sparse_deflection_angles = np.array([[0.0, 0.0]])
            blurring_deflection_angles = np.array([[3.0, 3.0]])

            ray_trace_deflection_angles = ray_tracing.PlaneDeflectionAngles(deflection_angles, sub_deflection_angles, sparse_deflection_angles,
                                                                 blurring_deflection_angles)

            assert ray_trace_deflection_angles.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert ray_trace_deflection_angles.sub_grid[0, 1] == pytest.approx(np.array([1.0, 2.0]), 1e-3)
            assert ray_trace_deflection_angles.sparse_grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert ray_trace_deflection_angles.blurring_grid[0] == pytest.approx(np.array([3.0, 3.0]), 1e-3)


class TestLensPlane(object):

    class TestBasicSetup(object):

        def test__setup_only_coordinates__simple_lens(self, lens_sis):

            coordinates = np.array([[1.0, 1.0]])

            lens_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis], grids=lens_plane_coordinates)

            assert lens_plane.grids.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub_grid == None
            assert lens_plane.grids.sparse_grid == None
            assert lens_plane.grids.blurring_grid == None

            assert lens_plane.deflection_angles.image_grid[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub_grid == None
            assert lens_plane.deflection_angles.sparse_grid == None
            assert lens_plane.deflection_angles.blurring_grid == None

        def test__setup_all_coordinates(self, lens_sis):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 0.0]]])
            sparse_coordinates = np.array([[1.0, 1.0]])
            blurring_coordinates = np.array([[1.0, 0.0]])

            lens_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates,
                                                                   blurring_coordinates)

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis], grids=lens_plane_coordinates)

            assert lens_plane.grids.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.sub_grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lens_plane.grids.sparse_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert lens_plane.grids.blurring_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert lens_plane.deflection_angles.image_grid[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub_grid[0,0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub_grid[0,1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lens_plane.deflection_angles.sparse_grid[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert lens_plane.deflection_angles.blurring_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test_three_identical_lenses__deflection_angles_triple(self, lens_sis):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 0.0]]])
            sparse_coordinates = np.array([[1.0, 1.0]])
            blurring_coordinates = np.array([[1.0, 0.0]])

            lens_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates,
                                                                   blurring_coordinates)

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis, lens_sis, lens_sis], grids=lens_plane_coordinates)

            assert lens_plane.grids.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub_grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.sparse_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.blurring_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert lens_plane.deflection_angles.image_grid[0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub_grid[0,0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub_grid[0,1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflection_angles.sparse_grid[0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.blurring_grid[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test_one_lens_with_three_identical_mass_profiles__deflection_angles_triple(self):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 0.0]]])
            sparse_coordinates = np.array([[1.0, 1.0]])
            blurring_coordinates = np.array([[1.0, 0.0]])

            lens_sis_x3 = galaxy.Galaxy(mass_profiles=3*[mass_profiles.SphericalIsothermal(einstein_radius=1.0)])

            lens_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates,
                                                                   blurring_coordinates)

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis_x3],
                                               grids=lens_plane_coordinates)

            assert lens_plane.grids.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub_grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.sparse_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.blurring_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            assert lens_plane.deflection_angles.image_grid[0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub_grid[0,0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.sub_grid[0,1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert lens_plane.deflection_angles.sparse_grid[0] == pytest.approx(np.array([3.0*0.707, 3.0*0.707]), 1e-3)
            assert lens_plane.deflection_angles.blurring_grid[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test__complex_mass_model(self):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 0.0]]])
            sparse_coordinates = np.array([[1.0, 1.0]])
            blurring_coordinates = np.array([[1.0, 0.0]])

            power_law = mass_profiles.EllipticalPowerLaw(centre=(1.0, 4.0), axis_ratio=0.7, phi=30.0,
                                                                    einstein_radius=1.0, slope=2.2)
            nfw = mass_profiles.SphericalNFW(kappa_s=0.1, scale_radius=5.0)

            lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profiles=[power_law, nfw])

            lens_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates,
                                                                   blurring_coordinates)

            lens_plane = ray_tracing.LensPlane(galaxies=[lens_galaxy], grids=lens_plane_coordinates)

            assert lens_plane.grids.image_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub_grid[0, 0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.sub_grid[0, 1] == pytest.approx(np.array([1.0, 0.0]), 1e-5)
            assert lens_plane.grids.sparse_grid[0] == pytest.approx(np.array([1.0, 1.0]), 1e-5)
            assert lens_plane.grids.blurring_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-5)

            defls = power_law.deflection_angles_at_coordinates(coordinates[0]) + \
                    nfw.deflection_angles_at_coordinates(coordinates[0])

            sub_defls_0 = power_law.deflection_angles_at_coordinates(sub_coordinates[0,0]) + \
                          nfw.deflection_angles_at_coordinates(sub_coordinates[0,0])

            sub_defls_1 = power_law.deflection_angles_at_coordinates(sub_coordinates[0,1]) + \
                          nfw.deflection_angles_at_coordinates(sub_coordinates[0,1])

            sparse_defls = power_law.deflection_angles_at_coordinates(sparse_coordinates[0]) + \
                           nfw.deflection_angles_at_coordinates(sparse_coordinates[0])

            blurring_defls = power_law.deflection_angles_at_coordinates(blurring_coordinates[0]) + \
                           nfw.deflection_angles_at_coordinates(blurring_coordinates[0])

            assert lens_plane.deflection_angles.image_grid[0] == pytest.approx(defls, 1e-3)
            assert lens_plane.deflection_angles.sub_grid[0,0] == pytest.approx(sub_defls_0, 1e-3)
            assert lens_plane.deflection_angles.sub_grid[0,1] == pytest.approx(sub_defls_1, 1e-3)
            assert lens_plane.deflection_angles.sparse_grid[0] == pytest.approx(sparse_defls, 1e-3)
            assert lens_plane.deflection_angles.blurring_grid[0] == pytest.approx(blurring_defls, 1e-3)


class TestImagePlane(object):

    def test__inheritance_from_lens_plane(self, lens_sis):

        coordinates = np.array([[1.0, 1.0]])
        sub_coordinates = np.array([[[1.0, 1.0], [1.0, 0.0]]])
        sparse_coordinates = np.array([[1.0, 1.0]])
        blurring_coordinates = np.array([[1.0, 0.0]])

        plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates,
                                                               blurring_coordinates)

        lens_plane = ray_tracing.LensPlane(galaxies=[lens_sis], grids=plane_coordinates)

        image_plane = ray_tracing.ImagePlane(galaxies=[lens_sis], grids=plane_coordinates)

        assert (lens_plane.grids.image_grid[0] == image_plane.grids.image_grid[0]).all()
        assert (lens_plane.grids.sub_grid[0, 0] == image_plane.grids.sub_grid[0, 0]).all()
        assert (image_plane.grids.sub_grid[0, 1] == image_plane.grids.sub_grid[0, 1]).all()
        assert (lens_plane.grids.sparse_grid[0] == image_plane.grids.sparse_grid[0]).all()
        assert (lens_plane.grids.blurring_grid[0] == image_plane.grids.blurring_grid[0]).all()

        assert (lens_plane.deflection_angles.image_grid[0] == image_plane.deflection_angles.image_grid[0]).all()
        assert (lens_plane.deflection_angles.sub_grid[0,0] == image_plane.deflection_angles.sub_grid[0,0]).all()
        assert (lens_plane.deflection_angles.sub_grid[0,1] == image_plane.deflection_angles.sub_grid[0,1]).all()
        assert (lens_plane.deflection_angles.sparse_grid[0] == image_plane.deflection_angles.sparse_grid[0]).all()
        assert (lens_plane.deflection_angles.blurring_grid[0] == image_plane.deflection_angles.blurring_grid[0]).all()


class TestSourcePlane(object):

    class TestCoordinatesInit(object):

        def test__sets_correct_values(self, no_galaxies):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 2.0]]])
            sparse_coordinates = np.array([0.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert (source_plane.grids.image_grid == np.array([[1.0, 1.0]])).all()
            assert (source_plane.grids.sub_grid == np.array([[[1.0, 1.0], [1.0, 2.0]]])).all()
            assert (source_plane.grids.sparse_grid == np.array([0.0, 0.0])).all()
            assert source_plane.grids.blurring_grid == None

        def test__four_coordinates__correct_source_plane(self, no_galaxies):
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert (source_plane.grids.image_grid == [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]).all()

        def test__four_coordinates_and_offset_centre__doesnt_change_coordinate_values(self, no_galaxies):
            # The centre is used by SourcePlaneGeomtry, but doesn't change the input coordinate values
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert (source_plane.grids.image_grid == np.array(
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])).all()

    class TestCoordinatesToCentre(object):

        def test__source_plane_centre_zeros_by_default__no_shift(self, no_galaxies):
            coordinates = np.array([0.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            coordinates_shift = source_plane.grids.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == 0.0

        def test__source_plane_centre_x_shift__x_shifts(self, no_galaxies):
            coordinates = np.array([0.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(0.5, 0.0))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            coordinates_shift = source_plane.grids.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == 0.0

        def test__source_plane_centre_y_shift__y_shifts(self, no_galaxies):
            coordinates = np.array([0.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(0.0, 0.5))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            coordinates_shift = source_plane.grids.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == -0.5

        def test__source_plane_centre_x_and_y_shift__x_and_y_both_shift(self, no_galaxies):
            coordinates = np.array([0.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(0.5, 0.5))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            coordinates_shift = source_plane.grids.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == -0.5

        def test__source_plane_centre_and_coordinates__correct_shifts(self, no_galaxies):
            coordinates = np.array([0.2, 0.4])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(1.0, 0.5))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            coordinates_shift = source_plane.grids.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == -0.8
            assert coordinates_shift[1] == pytest.approx(-0.1, 1e-5)

    class TestCoordinatesToRadius(object):
        def test__coordinates_overlap_source_plane_analysis__r_is_zero(self, no_galaxies):
            coordinates = np.array([0.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert source_plane.grids.coordinates_to_radius(coordinates) == 0.0

        def test__x_coordinates_is_one__r_is_one(self, no_galaxies):
            coordinates = np.array([1.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert source_plane.grids.coordinates_to_radius(coordinates) == 1.0

        def test__x_and_y_coordinates_are_one__r_is_root_two(self, no_galaxies):
            coordinates = np.array([1.0, 1.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert source_plane.grids.coordinates_to_radius(coordinates) == pytest.approx(np.sqrt(2), 1e-5)

        def test__shift_x_coordinate_first__r_includes_shift(self, no_galaxies):
            coordinates = np.array([1.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(-1.0, 0.0))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert source_plane.grids.coordinates_to_radius(coordinates) == pytest.approx(2.0, 1e-5)

        def test__shift_x_and_y_coordinates_first__r_includes_shift(self, no_galaxies):
            coordinates = np.array([3.0, 3.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(2.0, 2.0))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert source_plane.grids.coordinates_to_radius(coordinates) == pytest.approx(np.sqrt(2.0), 1e-5)


class TestTraceImageAndSoure(object):

        def test__coordinates_only__no_galaxy__image_and_source_plane_setup(self, no_galaxies):

            coordinates = np.array([[1.0, 0.0]])

            ray_tracing_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            lensing = ray_tracing.TraceImageAndSource(lens_galaxies=no_galaxies, source_galaxies=no_galaxies,
                                                      image_plane_grids=ray_tracing_coordinates)

            assert lensing.image_plane.coordinates.image_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.deflection_angles.image_grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert lensing.source_plane.coordinates.image_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test__lens_galaxy_on__source_plane_is_deflected(self, lens_sis):

            coordinates = np.array([[1.0, 0.0]])

            ray_tracing_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            lensing = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_sis], source_galaxies=no_galaxies,
                                                      image_plane_grids=ray_tracing_coordinates)

            assert lensing.image_plane.coordinates.image_grid == pytest.approx(np.array([[1.0, 0.0]]), 1e-3)
            assert lensing.image_plane.deflection_angles.image_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.source_plane.coordinates.image_grid == pytest.approx(np.array([[0.0, 0.0]]), 1e-3)

        def test__two_lens_galaxies__source_plane_deflected_doubles(self, lens_sis):

            coordinates = np.array([[1.0, 0.0]])

            ray_tracing_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            lensing = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_sis, lens_sis], source_galaxies=no_galaxies,
                                                      image_plane_grids=ray_tracing_coordinates)

            assert lensing.image_plane.coordinates.image_grid == pytest.approx(np.array([[1.0, 0.0]]), 1e-3)
            assert lensing.image_plane.deflection_angles.image_grid[0] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
            assert lensing.source_plane.coordinates.image_grid == pytest.approx(np.array([[-1.0, 0.0]]), 1e-3)

        def test__all_coordinates(self, lens_sis):

            coordinates = np.array([[1.0, 0.0]])
            sub_coordinates = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
            sparse_coordinates = np.array([[-1.0, 0.0]])
            blurring_coordinates = np.array([[-1.0, -1.0]])

            image_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates,
                                                                   blurring_coordinates)

            lensing = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_sis], source_galaxies=no_galaxies,
                                                      image_plane_grids=image_plane_coordinates)

            assert lensing.image_plane.coordinates.image_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.coordinates.sub_grid[0,0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.coordinates.sub_grid[1,0] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
            assert lensing.image_plane.coordinates.sparse_grid[0] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            assert lensing.image_plane.coordinates.blurring_grid[0] == pytest.approx(np.array([-1.0, -1.0]), 1e-3)

            assert lensing.image_plane.deflection_angles.image_grid[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.deflection_angles.sub_grid[0,0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.deflection_angles.sub_grid[1,0] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
            assert lensing.image_plane.deflection_angles.sparse_grid[0] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            assert lensing.image_plane.deflection_angles.blurring_grid[0] == pytest.approx(np.array([-0.707, -0.707]), 1e-3)

            assert lensing.source_plane.coordinates.image_grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert lensing.source_plane.coordinates.sub_grid[0,0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert lensing.source_plane.coordinates.sub_grid[1,0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert lensing.source_plane.coordinates.sparse_grid[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert lensing.source_plane.coordinates.blurring_grid[0] == pytest.approx(np.array([-0.293, -0.293]), 1e-3)