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
    sis = mass_profiles.SphericalIsothermalMassProfile(einstein_radius=1.0)
    lens_sis = galaxy.Galaxy(mass_profiles=[sis])
    return lens_sis

class TestTraceImageAndSoure(object):

    class TestCoordinatesOnly(object):
    
        def test__no_galaxy__no_deflections__image_and_source_plane_setup(self, no_galaxies):

            coordinates = np.array([[1.0, 0.0]])

            ray_tracing_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            lensing = ray_tracing.TraceImageAndSource(lens_galaxies=no_galaxies, source_galaxies=no_galaxies,
                                                      image_plane_coordinates=ray_tracing_coordinates)

            assert lensing.image_plane.coordinates == pytest.approx(np.array([[1.0, 0.0]]), 1e-3)
            assert lensing.source_plane.coordinates == pytest.approx(np.array([[1.0, 0.0]]), 1e-3)

        def test__lens_galaxy_on__source_plane_is_deflected(self, lens_sis):

            coordinates = np.array([[1.0, 0.0]])

            ray_tracing_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            lensing = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_sis], source_galaxies=no_galaxies,
                                                      image_plane_coordinates=ray_tracing_coordinates)

            assert lensing.image_plane.coordinates == pytest.approx(np.array([[1.0, 0.0]]), 1e-3)
            assert lensing.source_plane.coordinates == pytest.approx(np.array([[0.0, 0.0]]), 1e-3)

        def test__two_lens_galaxies__source_plane_deflected_doubles(self, lens_sis):

            coordinates = np.array([[1.0, 0.0]])

            ray_tracing_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            lensing = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_sis, lens_sis], source_galaxies=no_galaxies,
                                                      image_plane_coordinates=ray_tracing_coordinates)

            assert lensing.image_plane.coordinates == pytest.approx(np.array([[1.0, 0.0]]), 1e-3)
            assert lensing.source_plane.coordinates == pytest.approx(np.array([[-1.0, 0.0]]), 1e-3)

        def test__sub_and_sparse_coordinates__source_plane_deflections(self, lens_sis):
            
            coordinates = np.array([[1.0, 0.0]])
            sub_coordinates = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
            sparse_coordinates = np.array([[-1.0, 0.0]])
            
            image_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates)

            lensing = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_sis], source_galaxies=no_galaxies,
                                                      image_plane_coordinates=image_plane_coordinates)

            assert lensing.image_plane.coordinates[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.sub_coordinates[0,0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.sub_coordinates[1,0] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
            assert lensing.image_plane.sparse_coordinates[0] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            
            assert lensing.image_plane.deflection_angles[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.sub_deflection_angles[0,0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert lensing.image_plane.sub_deflection_angles[1,0] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
            assert lensing.image_plane.sparse_deflection_angles[0] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            
            assert lensing.source_plane.coordinates[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert lensing.source_plane.sub_coordinates[0,0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert lensing.source_plane.sub_coordinates[1,0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert lensing.source_plane.sparse_coordinates[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)


class TestPlaneCoordinates(object):

    class TestBasicSetup(object):
        coordinates = np.array([[1.0, 1.0]])
        sub_coordinates = np.array([[[1.0, 1.0], [1.0, 2.0]]])
        sparse_coordinates = np.array([0.0, 0.0])

        ray_trace_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates)

        assert (ray_trace_coordinates.coordinates == np.array([[1.0, 1.0]])).all()
        assert (ray_trace_coordinates.sub_coordinates == np.array([[[1.0, 1.0], [1.0, 2.0]]])).all()
        assert (ray_trace_coordinates.sparse_coordinates == np.array([0.0, 0.0])).all()
        assert ray_trace_coordinates.blurring_coordinates == None


class TestImagePlane(object):
    
    class TestBasicSetup(object):

        def test__setup_correct_values(self, no_galaxies):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
            sparse_coordinates = np.array([[0.0, 0.0]])

            image_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates)

            image_plane = ray_tracing.ImagePlane(no_galaxies, image_plane_coordinates)

            assert image_plane.coordinates[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert image_plane.sub_coordinates[0,0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert image_plane.sub_coordinates[1,0] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
            assert image_plane.blurring_coordinates == None

    class TestDeflectionAngleCalc:

        def test__coordinates_only__simple_lens(self, lens_sis):

            coordinates = np.array([[1.0, 0.0]])

            image_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            image_plane = ray_tracing.ImagePlane(galaxies=[lens_sis], plane_coordinates=image_plane_coordinates)

            assert image_plane.deflection_angles[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test__coordinates_only__two_lens_galaxies(self, lens_sis):

            coordinates = np.array([[1.0, 0.0]])

            image_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            image_plane = ray_tracing.ImagePlane(galaxies=[lens_sis, lens_sis], plane_coordinates=image_plane_coordinates)

            assert image_plane.deflection_angles[0] == pytest.approx(np.array([2.0, 0.0]), 1e-3)

        def test__coordinates_only__complex_mas_model(self):

            coordinates = np.array([[1.0, 0.0]])

            power_law = mass_profiles.EllipticalPowerLawMassProfile(centre=(1.0, 4.0), axis_ratio=0.7, phi=30.0,
                                                                    einstein_radius=1.0, slope=2.2)
            nfw = mass_profiles.SphericalNFWMassProfile(kappa_s=0.1, scale_radius=5.0)

            defls = power_law.deflection_angles_at_coordinates(coordinates[0]) + \
                      nfw.deflection_angles_at_coordinates(coordinates[0])

            image_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            lens_galaxy = galaxy.Galaxy(mass_profiles=[power_law, nfw])

            image_plane = ray_tracing.ImagePlane(galaxies=[lens_galaxy], plane_coordinates=image_plane_coordinates)

            assert image_plane.deflection_angles[0] == pytest.approx(defls, 1e-3)

        def test__coordinates_sub_and_sparse__simple_lens(self, lens_sis):

            coordinates = np.array([[1.0, 0.0]])
            sub_coordinates = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
            sparse_coordinates = np.array([[-1.0, 0.0]])

            image_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates)

            image_plane = ray_tracing.ImagePlane(galaxies=[lens_sis], plane_coordinates=image_plane_coordinates)

            assert image_plane.deflection_angles[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert image_plane.sub_deflection_angles[0,0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert image_plane.sub_deflection_angles[1,0] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
            assert image_plane.sparse_deflection_angles[0] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)

        def test__coordinates_sub_and_sparse__two_lens_galaxies(self, lens_sis):

            coordinates = np.array([[1.0, 0.0]])
            sub_coordinates = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
            sparse_coordinates = np.array([[-1.0, 0.0]])

            image_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates)

            image_plane = ray_tracing.ImagePlane(galaxies=[lens_sis, lens_sis], plane_coordinates=image_plane_coordinates)

            assert image_plane.deflection_angles[0] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
            assert image_plane.sub_deflection_angles[0,0] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
            assert image_plane.sub_deflection_angles[1,0] == pytest.approx(np.array([0.0, 2.0]), 1e-3)
            assert image_plane.sparse_deflection_angles[0] == pytest.approx(np.array([-2.0, 0.0]), 1e-3)

        def test__coordinates_sub_and_sparse__complex_mass_model(self):

            coordinates = np.array([[1.0, 0.0]])
            sub_coordinates = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
            sparse_coordinates = np.array([[-1.0, 0.0]])

            power_law = mass_profiles.EllipticalPowerLawMassProfile(centre=(1.0, 4.0), axis_ratio=0.7, phi=30.0,
                                                                    einstein_radius=1.0, slope=2.2)
            nfw = mass_profiles.SphericalNFWMassProfile(kappa_s=0.1, scale_radius=5.0)

            defls = power_law.deflection_angles_at_coordinates(coordinates[0]) + \
                      nfw.deflection_angles_at_coordinates(coordinates[0])

            sub_defls_0 = power_law.deflection_angles_at_coordinates(sub_coordinates[0,0]) + \
                      nfw.deflection_angles_at_coordinates(sub_coordinates[0,0])

            sub_defls_1 = power_law.deflection_angles_at_coordinates(sub_coordinates[1,0]) + \
                      nfw.deflection_angles_at_coordinates(sub_coordinates[1,0])

            sparse_defls = power_law.deflection_angles_at_coordinates(sparse_coordinates[0]) + \
                      nfw.deflection_angles_at_coordinates(sparse_coordinates[0])

            image_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates)

            lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profiles=[power_law, nfw])

            image_plane = ray_tracing.ImagePlane(galaxies=[lens_galaxy], plane_coordinates=image_plane_coordinates)

            assert image_plane.deflection_angles[0] == pytest.approx(defls, 1e-3)
            assert image_plane.sub_deflection_angles[0,0] == pytest.approx(sub_defls_0, 1e-3)
            assert image_plane.sub_deflection_angles[1,0] == pytest.approx(sub_defls_1, 1e-3)
            assert image_plane.sparse_deflection_angles[0] == pytest.approx(sparse_defls, 1e-3)


class TestSourcePlane(object):

    class TestCoordinatesInit(object):

        def test__sets_correct_values(self, no_galaxies):

            coordinates = np.array([[1.0, 1.0]])
            sub_coordinates = np.array([[[1.0, 1.0], [1.0, 2.0]]])
            sparse_coordinates = np.array([0.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert (source_plane.coordinates == np.array([[1.0, 1.0]])).all()
            assert (source_plane.sub_coordinates == np.array([[[1.0, 1.0], [1.0, 2.0]]])).all()
            assert (source_plane.sparse_coordinates == np.array([0.0, 0.0])).all()
            assert source_plane.blurring_coordinates == None

        def test__four_coordinates__correct_source_plane(self, no_galaxies):

            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert (source_plane.coordinates == [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]).all()

        def test__four_coordinates_and_offset_centre__doesnt_change_coordinate_values(self, no_galaxies):

            # The centre is used by SourcePlaneGeomtry, but doesn't change the input coordinate values
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert (source_plane.coordinates == np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])).all()

    class TestCoordinatesToCentre(object):
        def test__source_plane_centre_zeros_by_default__no_shift(self, no_galaxies):
            coordinates = np.array([0.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            coordinates_shift = source_plane.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == 0.0

        def test__source_plane_centre_x_shift__x_shifts(self, no_galaxies):

            coordinates = np.array([0.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(0.5, 0.0))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            coordinates_shift = source_plane.coordinates_to_centre(source_plane.coordinates)

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == 0.0

        def test__source_plane_centre_y_shift__y_shifts(self, no_galaxies):
            coordinates = np.array([0.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(0.0, 0.5))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            coordinates_shift = source_plane.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == 0.0
            assert coordinates_shift[1] == -0.5

        def test__source_plane_centre_x_and_y_shift__x_and_y_both_shift(self, no_galaxies):
            coordinates = np.array([0.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(0.5, 0.5))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            coordinates_shift = source_plane.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == -0.5
            assert coordinates_shift[1] == -0.5

        def test__source_plane_centre_and_coordinates__correct_shifts(self, no_galaxies):
            coordinates = np.array([0.2, 0.4])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(1.0, 0.5))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            coordinates_shift = source_plane.coordinates_to_centre(coordinates)

            assert coordinates_shift[0] == -0.8
            assert coordinates_shift[1] == pytest.approx(-0.1, 1e-5)

    class TestCoordinatesToRadius(object):
        def test__coordinates_overlap_source_plane_analysis__r_is_zero(self, no_galaxies):
            coordinates = np.array([0.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert source_plane.coordinates_to_radius(coordinates) == 0.0

        def test__x_coordinates_is_one__r_is_one(self, no_galaxies):
            coordinates = np.array([1.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert source_plane.coordinates_to_radius(coordinates) == 1.0

        def test__x_and_y_coordinates_are_one__r_is_root_two(self, no_galaxies):
            coordinates = np.array([1.0, 1.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert source_plane.coordinates_to_radius(coordinates) == pytest.approx(np.sqrt(2), 1e-5)

        def test__shift_x_coordinate_first__r_includes_shift(self, no_galaxies):
            coordinates = np.array([1.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(-1.0, 0.0))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert source_plane.coordinates_to_radius(coordinates) == pytest.approx(2.0, 1e-5)

        def test__shift_x_and_y_coordinates_first__r_includes_shift(self, no_galaxies):
            coordinates = np.array([3.0, 3.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(2.0, 2.0))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            assert source_plane.coordinates_to_radius(coordinates) == pytest.approx(np.sqrt(2.0), 1e-5)

    class TestCoordinatesAngleFromX(object):
        def test__angle_is_zero__angles_follow_trig(self, no_galaxies):
            coordinates = np.array([1.0, 0.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 0.0

        def test__angle_is_forty_five__angles_follow_trig(self, no_galaxies):
            coordinates = np.array([1.0, 1.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 45.0

        def test__angle_is_sixty__angles_follow_trig(self, no_galaxies):
            coordinates = np.array([1.0, 1.7320])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == pytest.approx(60.0, 1e-3)

        def test__top_left_quandrant__angle_goes_above_90(self, no_galaxies):
            coordinates = np.array([-1.0, 1.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 135.0

        def test__bottom_left_quandrant__angle_continues_above_180(self, no_galaxies):
            coordinates = np.array([-1.0, -1.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 225.0

        def test__bottom_right_quandrant__angle_flips_back_to_above_90(self, no_galaxies):
            coordinates = np.array([1.0, -1.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 315.0

        def test__include_source_plane_centre__angle_takes_into_accounts(self, no_galaxies):
            coordinates = np.array([2.0, 2.0])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(1.0, 1.0))

            source_plane = ray_tracing.SourcePlane(no_galaxies, source_plane_coordinates)

            theta_from_x = source_plane.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 45.0


class TestSorucePlaneWithBorder(object):
    
    class TestCoordinatesThetasAndRadii(object):

        def test__four_coordinates_in_circle__correct_border(self, no_galaxies):
            
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            border_pixels = np.array([0, 1, 2, 3])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert source_plane.coordinates == \
                   pytest.approx(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]), 1e-3)
            assert source_plane.border_coordinates == \
                   pytest.approx(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]), 1e-3)
            assert source_plane.border_radii == [1.0, 1.0, 1.0, 1.0]
            assert source_plane.border_thetas == [0.0, 90.0, 180.0, 270.0]

        def test__test_other_thetas_radii(self, no_galaxies):
            coordinates = np.array([[2.0, 0.0], [2.0, 2.0], [-1.0, -1.0], [0.0, -3.0]])
            border_pixels = np.array([0, 1, 2, 3])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert source_plane.coordinates == \
                   pytest.approx(np.array([[2.0, 0.0], [2.0, 2.0], [-1.0, -1.0], [0.0, -3.0]]), 1e-3)
            assert source_plane.border_coordinates == \
                   pytest.approx(np.array([[2.0, 0.0], [2.0, 2.0], [-1.0, -1.0], [0.0, -3.0]]), 1e-3)
            assert source_plane.border_radii == [2.0, 2.0 * np.sqrt(2), np.sqrt(2.0), 3.0]
            assert source_plane.border_thetas == [0.0, 45.0, 225.0, 270.0]

        def test__source_plane_centre_offset__coordinates_same_r_and_theta_shifted(self, no_galaxies):
            coordinates = np.array([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
            border_pixels = np.array([0, 1, 2, 3])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(1.0, 1.0))

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert source_plane.coordinates == \
                   pytest.approx(np.array([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 0.0]]), 1e-3)
            assert source_plane.border_coordinates == \
                   pytest.approx(np.array([(2.0, 1.0), (1.0, 2.0), (0.0, 1.0), (1.0, 0.0)]), 1e-3)
            assert source_plane.border_radii == [1.0, 1.0, 1.0, 1.0]
            assert source_plane.border_thetas == [0.0, 90.0, 180.0, 270.0]

    class TestBorderPolynomial(object):

        def test__four_coordinates_in_circle__thetas_at_radius_are_each_coordinates_radius(self, no_galaxies):
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]])
            border_pixels = np.array([0, 1, 2, 3])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert source_plane.border_radius_at_theta(theta=0.0) == pytest.approx(1.0, 1e-3)
            assert source_plane.border_radius_at_theta(theta=90.0) == pytest.approx(1.0, 1e-3)
            assert source_plane.border_radius_at_theta(theta=180.0) == pytest.approx(1.0, 1e-3)
            assert source_plane.border_radius_at_theta(theta=270.0) == pytest.approx(1.0, 1e-3)

        def test__eight_coordinates_in_circle__thetas_at_each_coordinates_are_the_radius(self, no_galaxies):
            coordinates = np.array([[1.0, 0.0], [0.5 * np.sqrt(2), 0.5 * np.sqrt(2)],
                                    [0.0, 1.0], [-0.5 * np.sqrt(2), 0.5 * np.sqrt(2)],
                                    [-1.0, 0.0], [-0.5 * np.sqrt(2), -0.5 * np.sqrt(2)],
                                    [0.0, -1.0], [0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]])

            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert source_plane.border_radius_at_theta(theta=0.0) == pytest.approx(1.0, 1e-3)
            assert source_plane.border_radius_at_theta(theta=45.0) == pytest.approx(1.0, 1e-3)
            assert source_plane.border_radius_at_theta(theta=90.0) == pytest.approx(1.0, 1e-3)
            assert source_plane.border_radius_at_theta(theta=135.0) == pytest.approx(1.0, 1e-3)
            assert source_plane.border_radius_at_theta(theta=180.0) == pytest.approx(1.0, 1e-3)
            assert source_plane.border_radius_at_theta(theta=225.0) == pytest.approx(1.0, 1e-3)
            assert source_plane.border_radius_at_theta(theta=270.0) == pytest.approx(1.0, 1e-3)
            assert source_plane.border_radius_at_theta(theta=315.0) == pytest.approx(1.0, 1e-3)

    class TestMoveFactors(object):

        def test__inside_border__move_factor_is_1(self, no_galaxies):
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            border_pixels = np.array([0, 1, 2, 3])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert source_plane.move_factor(coordinate=(0.5, 0.0)) == 1.0
            assert source_plane.move_factor(coordinate=(-0.5, 0.0)) == 1.0
            assert source_plane.move_factor(coordinate=(0.25, 0.25)) == 1.0
            assert source_plane.move_factor(coordinate=(0.0, 0.0)) == 1.0

        def test__outside_border_double_its_radius__move_factor_is_05(self, no_galaxies):
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            border_pixels = np.array([0, 1, 2, 3])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert source_plane.move_factor(coordinate=(2.0, 0.0)) == pytest.approx(0.5, 1e-3)
            assert source_plane.move_factor(coordinate=(0.0, 2.0)) == pytest.approx(0.5, 1e-3)
            assert source_plane.move_factor(coordinate=(-2.0, 0.0)) == pytest.approx(0.5, 1e-3)
            assert source_plane.move_factor(coordinate=(0.0, -2.0)) == pytest.approx(0.5, 1e-3)

        def test__outside_border_double_its_radius_and_offset__move_factor_is_05(self, no_galaxies):
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            border_pixels = np.array([0, 1, 2, 3])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert source_plane.move_factor(coordinate=(2.0, 0.0)) == pytest.approx(0.5, 1e-3)
            assert source_plane.move_factor(coordinate=(0.0, 2.0)) == pytest.approx(0.5, 1e-3)
            assert source_plane.move_factor(coordinate=(0.0, 2.0)) == pytest.approx(0.5, 1e-3)
            assert source_plane.move_factor(coordinate=(2.0, 0.0)) == pytest.approx(0.5, 1e-3)

        def test__outside_border_as_above__but_shift_for_source_plane_centre(self, no_galaxies):
            coordinates = np.array([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
            border_pixels = np.array([0, 1, 2, 3])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, centre=(1.0, 1.0))

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert source_plane.move_factor(coordinate=(3.0, 1.0)) == pytest.approx(0.5, 1e-3)
            assert source_plane.move_factor(coordinate=(1.0, 3.0)) == pytest.approx(0.5, 1e-3)
            assert source_plane.move_factor(coordinate=(1.0, 3.0)) == pytest.approx(0.5, 1e-3)
            assert source_plane.move_factor(coordinate=(3.0, 1.0)) == pytest.approx(0.5, 1e-3)

    class TestRelocateCoordinates(object):

        def test__outside_border_simple_cases__relocates_to_source_border(self, no_galaxies):
            thetas = np.linspace(0.0, 2.0 * np.pi, 32)
            coordinates = np.asarray(list(map(lambda x: (np.cos(x), np.sin(x)), thetas)))
            border_pixels = np.arange(32)

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            relocated_coordinate = source_plane.relocated_coordinate(coordinate=np.array([2.5, 0.37]))
            assert source_plane.coordinates_to_radius(relocated_coordinate) == pytest.approx(1.0, 1e-3)

            relocated_coordinate = source_plane.relocated_coordinate(coordinate=np.array([25.3, -9.2]))
            assert source_plane.coordinates_to_radius(relocated_coordinate) == pytest.approx(1.0, 1e-3)

            relocated_coordinate = source_plane.relocated_coordinate(coordinate=np.array([13.5, 0.0]))
            assert source_plane.coordinates_to_radius(relocated_coordinate) == pytest.approx(1.0, 1e-3)

            relocated_coordinate = source_plane.relocated_coordinate(coordinate=np.array([-2.5, -0.37]))
            assert source_plane.coordinates_to_radius(relocated_coordinate) == pytest.approx(1.0, 1e-3)

        def test__outside_border_simple_cases_2__relocates_to_source_border(self, no_galaxies):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            coordinates = np.asarray(list(map(lambda x: (np.cos(x), np.sin(x)), thetas)))
            border_pixels = np.arange(16)

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            relocated_coordinate = source_plane.relocated_coordinate(coordinate=(2.0, 0.0))
            assert relocated_coordinate == pytest.approx((1.0, 0.0), 1e-3)

            relocated_coordinate = source_plane.relocated_coordinate(coordinate=(1.0, 1.0))
            assert relocated_coordinate == pytest.approx((0.5 * np.sqrt(2), 0.5 * np.sqrt(2)), 1e-3)

            relocated_coordinate = source_plane.relocated_coordinate(coordinate=(0.0, 2.0))
            assert relocated_coordinate == pytest.approx((0.0, 1.0), 1e-3)

            relocated_coordinate = source_plane.relocated_coordinate(coordinate=(-1.0, 1.0))
            assert relocated_coordinate == pytest.approx((-0.5 * np.sqrt(2), 0.5 * np.sqrt(2)), 1e-3)

            relocated_coordinate = source_plane.relocated_coordinate(coordinate=(-2.0, 0.0))
            assert relocated_coordinate == pytest.approx((-1.0, 0.0), 1e-3)

            relocated_coordinate = source_plane.relocated_coordinate(coordinate=(-1.0, -1.0))
            assert relocated_coordinate == pytest.approx((-0.5 * np.sqrt(2), -0.5 * np.sqrt(2)), 1e-3)

            relocated_coordinate = source_plane.relocated_coordinate(coordinate=(0.0, -1.0))
            assert relocated_coordinate == pytest.approx((0.0, -1.0), 1e-3)

            relocated_coordinate = source_plane.relocated_coordinate(coordinate=(1.0, -1.0))
            assert relocated_coordinate == pytest.approx((0.5 * np.sqrt(2), -0.5 * np.sqrt(2)), 1e-3)

    class TestSourcePlanePixelsOutsideBorderAreRelocated:

        def test__all_coordinates_inside_border__no_relocations(self, no_galaxies):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates = np.asarray(circle + [(2.0, 0.0), (1.0, 1.0), (0.0, 2.0), (-1.0, 1.0),
                                               (-2.0, 0.0), (-1.0, -1.0), (0.0, -2.0), (1.0, -1.0)])

            border_pixels = np.arange(16)

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert (source_plane.coordinates == coordinates).all()

        def test__all_coordinates_inside_border_again__no_relocations(self, no_galaxies):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates = np.asarray(circle + [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (-0.5, 0.5),
                                               (-0.5, 0.0), (-0.5, -0.5), (0.0, -0.5), (0.5, -0.5)])

            border_pixels = np.arange(16)

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert (source_plane.coordinates == coordinates).all()

        def test__6_coordinates_total__2_outside_border__relocate_to_source_border(self, no_galaxies):

            coordinates = np.array([[1.0, 0.0], [20., 20.], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
            border_pixels = np.array([0, 2, 3, 4])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert (source_plane.coordinates[0] == coordinates[0]).all()
            assert source_plane.coordinates[1] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)
            assert (source_plane.coordinates[2] == coordinates[2]).all()
            assert (source_plane.coordinates[3] == coordinates[3]).all()
            assert (source_plane.coordinates[4] == coordinates[4]).all()
            assert source_plane.coordinates[5] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)

        def test__24_coordinates_total__8_coordinates_outside_border__relocate_to_source_border(self, no_galaxies):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates = np.asarray(circle + [(2.0, 0.0), (1.0, 1.0), (0.0, 2.0), (-1.0, 1.0),
                                               (-2.0, 0.0), (-1.0, -1.0), (0.0, -2.0), (1.0, -1.0)])

            border_pixels = np.arange(16)

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert (source_plane.coordinates[0:16] == coordinates[0:16]).all()
            assert source_plane.coordinates[16] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert source_plane.coordinates[17] == pytest.approx(np.array([0.5 * np.sqrt(2), 0.5 * np.sqrt(2)]))
            assert source_plane.coordinates[18] == pytest.approx(np.array([0.0, 1.0]))
            assert source_plane.coordinates[19] == pytest.approx(np.array([-0.5 * np.sqrt(2), 0.5 * np.sqrt(2)]))
            assert source_plane.coordinates[20] == pytest.approx(np.array([-1.0, 0.0]))
            assert source_plane.coordinates[21] == pytest.approx(np.array([-0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]))
            assert source_plane.coordinates[22] == pytest.approx(np.array([0.0, -1.0]))
            assert source_plane.coordinates[23] == pytest.approx(np.array([0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]))

        def test__24_coordinates_total__4_coordinates_outside_border__relates_to_source_border(self, no_galaxies):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates = np.asarray(circle + [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (-0.5, 0.5),
                                               (-2.0, 0.0), (-1.0, -1.0), (0.0, -2.0), (1.0, -1.0)])

            border_pixels = np.arange(16)

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert (source_plane.coordinates[0:20] == coordinates[0:20]).all()
            assert source_plane.coordinates[20] == pytest.approx(np.array([-1.0, 0.0]))
            assert source_plane.coordinates[21] == pytest.approx(np.array([-0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]))
            assert source_plane.coordinates[22] == pytest.approx(np.array([0.0, -1.0]))
            assert source_plane.coordinates[23] == pytest.approx(np.array([0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]))

        def test__change_pixel_order_and_border_pixels__works_as_above(self, no_galaxies):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates = np.asarray([(-2.0, 0.0), (-1.0, -1.0), (0.0, -2.0), (1.0, -1.0)] + circle + \
                                     [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (-0.5, 0.5)])

            border_pixels = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert source_plane.coordinates[0] == pytest.approx(np.array([-1.0, 0.0]))
            assert source_plane.coordinates[1] == pytest.approx(np.array([-0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]))
            assert source_plane.coordinates[2] == pytest.approx(np.array([0.0, -1.0]))
            assert source_plane.coordinates[3] == pytest.approx(np.array([0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]))
            assert (source_plane.coordinates[4:24] == coordinates[4:24]).all()

        def test__sub_pixels_in_border__are_not_relocated(self, no_galaxies):

            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            sub_coordinates = np.array([[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
                                        [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
                                        [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
                                        [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]]])

            border_pixels = np.array([0, 1, 2, 3])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert (source_plane.coordinates == coordinates).all()
            assert (source_plane.sub_coordinates == sub_coordinates).all()

        def test__sub_pixels_outside_border__are_relocated(self, no_galaxies):

            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            sub_coordinates = np.array([[[2.0, 0.0], [0.2, 0.0], [2.0, 2.0], [0.4, 0.0]],
                                        [[0.0, 2.0], [-2.0, 2.0], [0.3, 0.0], [0.4, 0.0]],
                                        [[-2.0, 0.0], [0.2, 0.0], [0.3, 0.0], [2.0, -2.0]],
                                        [[0.0, -2.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]]])

            border_pixels = np.array([0, 1, 2, 3])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sub_coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert (source_plane.coordinates == coordinates).all()

            assert (source_plane.sub_coordinates[0,0] == pytest.approx(np.array([1.0, 0.0]), 1e-3))
            assert (source_plane.sub_coordinates[0,1] == sub_coordinates[0,1]).all()
            assert (source_plane.sub_coordinates[0,2] == pytest.approx(np.array([0.707, 0.707]), 1e-3))
            assert (source_plane.sub_coordinates[0,3] == sub_coordinates[0,3]).all()

            assert (source_plane.sub_coordinates[1,0] == pytest.approx(np.array([0.0, 1.0]), 1e-3))
            assert (source_plane.sub_coordinates[1,1] == pytest.approx(np.array([-0.707, 0.707]), 1e-3))
            assert (source_plane.sub_coordinates[1,2] == sub_coordinates[1,2]).all()
            assert (source_plane.sub_coordinates[1,3] == sub_coordinates[1,3]).all()

            assert (source_plane.sub_coordinates[2,0] == pytest.approx(np.array([-1.0, 0.0]), 1e-3))
            assert (source_plane.sub_coordinates[2,1] == sub_coordinates[2,1]).all()
            assert (source_plane.sub_coordinates[2,2] == sub_coordinates[2,2]).all()
            assert (source_plane.sub_coordinates[2,3] == pytest.approx(np.array([0.707, -0.707]), 1e-3))

            assert (source_plane.sub_coordinates[3,0] == pytest.approx(np.array([0.0, -1.0]), 1e-3))
            assert (source_plane.sub_coordinates[3,1] == sub_coordinates[3,1]).all()
            assert (source_plane.sub_coordinates[3,2] == sub_coordinates[3,2]).all()
            assert (source_plane.sub_coordinates[3,3] == sub_coordinates[3,3]).all()

        def test__sparse_coordinates_are_relocated__same_as_normal_coordinates(self, no_galaxies):

            coordinates = np.array([[1.0, 0.0], [20., 20.], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
            sparse_coordinates = np.array([[1.0, 0.0], [20., 20.], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
            border_pixels = np.array([0, 2, 3, 4])

            source_plane_coordinates = ray_tracing.PlaneCoordinates(coordinates, sparse_coordinates=sparse_coordinates)

            source_plane = ray_tracing.SourcePlaneWithBorder(no_galaxies, border_pixels, 3, source_plane_coordinates)

            assert (source_plane.coordinates[0] == coordinates[0]).all()
            assert source_plane.coordinates[1] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)
            assert (source_plane.coordinates[2] == coordinates[2]).all()
            assert (source_plane.coordinates[3] == coordinates[3]).all()
            assert (source_plane.coordinates[4] == coordinates[4]).all()
            assert source_plane.coordinates[5] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)

            assert (source_plane.sparse_coordinates[0] == sparse_coordinates[0]).all()
            assert source_plane.sparse_coordinates[1] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)
            assert (source_plane.sparse_coordinates[2] == sparse_coordinates[2]).all()
            assert (source_plane.sparse_coordinates[3] == sparse_coordinates[3]).all()
            assert (source_plane.sparse_coordinates[4] == sparse_coordinates[4]).all()
            assert source_plane.sparse_coordinates[5] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)