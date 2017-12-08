import pytest
import decorator
import profile


@pytest.fixture(name='circular')
def circular_sersic():
    return profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                      effective_radius=0.6, sersic_index=4.0)


@pytest.fixture(name='elliptical')
def elliptical_sersic():
    return profile.SersicLightProfile(axis_ratio=0.5, phi=0.0, flux=1.0,
                                      effective_radius=0.6, sersic_index=4.0)


@pytest.fixture(name='vertical')
def vertical_sersic():
    return profile.SersicLightProfile(axis_ratio=0.5, phi=90.0, flux=1.0,
                                      effective_radius=0.6, sersic_index=4.0)


class MockMask(object):
    def __init__(self, masked_coordinates):
        self.masked_coordinates = masked_coordinates

    def is_masked(self, coordinates):
        # It's probably a good idea to use a numpy array in the real class for efficiency
        return coordinates in self.masked_coordinates


class TestDecorators(object):
    def test_subgrid_2x2(self):
        @decorator.subgrid
        def return_coords(coords):
            return coords[0], coords[1]

        coordinates = return_coords((0, 0), pixel_scale=1.0, grid_size=1)
        assert coordinates == [(0, 0)]

        coordinates = return_coords((0.5, 0.5), pixel_scale=1.0, grid_size=2)
        assert coordinates == [(1. / 3., 1. / 3.), (1. / 3., 2. / 3.), (2. / 3., 1. / 3.), (2. / 3., 2. / 3.)]

    def test_subgrid_3x3(self):
        @decorator.subgrid
        def return_coords(coords):
            return coords[0], coords[1]

        coordinates = return_coords((0, 0), pixel_scale=1.0, grid_size=1)
        assert coordinates == [(0, 0)]

        coordinates = return_coords((0.5, 0.5), pixel_scale=1.0, grid_size=3)
        assert coordinates == [(0.25, 0.25), (0.25, 0.5), (0.25, 0.75),
                               (0.50, 0.25), (0.50, 0.5), (0.50, 0.75),
                               (0.75, 0.25), (0.75, 0.5), (0.75, 0.75)]

    def test_subgrid_3x3_triple_pixel_scale_and_coordinate(self):
        @decorator.subgrid
        def return_coords(coords):
            return coords[0], coords[1]

        coordinates = return_coords((0, 0), pixel_scale=1.0, grid_size=1)
        assert coordinates == [(0, 0)]

        coordinates = return_coords((1.5, 1.5), pixel_scale=3.0, grid_size=3)

        assert coordinates == [(0.75, 0.75), (0.75, 1.5), (0.75, 2.25),
                               (1.50, 0.75), (1.50, 1.5), (1.50, 2.25),
                               (2.25, 0.75), (2.25, 1.5), (2.25, 2.25)]

    def test_subgrid_4x4_new_coordinates(self):
        @decorator.subgrid
        def return_coords(coords):
            return coords[0], coords[1]

        coordinates = return_coords((0, 0), pixel_scale=1.0, grid_size=1)
        assert coordinates == [(0, 0)]

        coordinates = return_coords((-2.0, 3.0), pixel_scale=0.1, grid_size=4)

        coordinates = map(lambda coords: (pytest.approx(coords[0], 1e-2), pytest.approx(coords[1], 1e-2)), coordinates)

        assert coordinates == [(-2.03, 2.97), (-2.03, 2.99), (-2.03, 3.01), (-2.03, 3.03),
                               (-2.01, 2.97), (-2.01, 2.99), (-2.01, 3.01), (-2.01, 3.03),
                               (-1.99, 2.97), (-1.99, 2.99), (-1.99, 3.01), (-1.99, 3.03),
                               (-1.97, 2.97), (-1.97, 2.99), (-1.97, 3.01), (-1.97, 3.03)]

    def test_average(self):
        @decorator.avg
        def return_input(input_list):
            return input_list

        assert return_input([1, 2, 3]) == 2
        assert return_input([(1, 10), (2, 20), (3, 30)]) == (2, 20)

    def test_iterative_subgrid(self):
        # noinspection PyUnusedLocal
        @decorator.iterative_subgrid
        def one_over_grid(coordinates, pixel_scale, grid_size):
            return 1.0 / grid_size

        assert one_over_grid(None, None, 0.51) == pytest.approx(0.5)
        assert one_over_grid(None, None, 0.21) == pytest.approx(0.2)

    def test_mask(self):
        mask = MockMask([(x, 0) for x in range(-5, 6)])
        array = decorator.array_function(lambda coordinates: 1)(-5, -5, 5, 5, 1, mask=mask)

        assert array[5][5] is None
        assert array[5][6] is not None
        assert array[6][5] is None
        assert array[0][0] is not None
        assert array[0][5] is None


class TestAuxiliary(object):
    def test__side_length(self):
        assert decorator.side_length(-5, 5, 0.1) == 100

    def test__pixel_to_coordinate(self):
        assert decorator.pixel_to_coordinate(-5, 0.1, 0) == -5
        assert decorator.pixel_to_coordinate(-5, 0.1, 100) == 5
        assert decorator.pixel_to_coordinate(-5, 0.1, 50) == 0


class TestArray(object):
    def test__simple_assumptions(self, circular):
        array = decorator.array_function(circular.flux_at_coordinates)(x_min=0, x_max=101, y_min=0, y_max=101,
                                                                       pixel_scale=1)
        assert array.shape == (101, 101)
        assert array[51][51] > array[51][52]
        assert array[51][51] > array[52][51]
        assert all(map(lambda i: i > 0, array[0]))

        array = decorator.array_function(circular.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                       pixel_scale=0.5)
        assert array.shape == (200, 200)

    def test__ellipticity(self, circular, elliptical, vertical):
        array = decorator.array_function(circular.flux_at_coordinates)(x_min=0, x_max=101, y_min=0, y_max=101,
                                                                       pixel_scale=1)
        assert array[60][0] == array[0][60]

        array = decorator.array_function(elliptical.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                         pixel_scale=1)

        assert array[60][51] > array[51][60]

        array = decorator.array_function(vertical.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                       pixel_scale=1)
        assert array[60][51] < array[51][60]

    # noinspection PyTypeChecker
    def test__flat_array(self, circular):
        array = decorator.array_function(circular.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                       pixel_scale=1)
        flat_array = decorator.array_function(circular.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                            pixel_scale=1).flatten()

        assert all(array[0] == flat_array[:100])
        assert all(array[1] == flat_array[100:200])

    def test_combined_array(self, circular):
        combined = profile.CombinedLightProfile(circular, circular)

        assert all(map(lambda i: i == 2,
                       decorator.array_function(combined.flux_at_coordinates)().flatten() / decorator.array_function(
                           circular.flux_at_coordinates)().flatten()))

    def test_symmetric_profile(self, circular):
        circular.centre = (50, 50)
        array = decorator.array_function(circular.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                       pixel_scale=1.0)

        assert array[50][50] > array[50][51]
        assert array[50][50] > array[49][50]
        assert array[49][50] == array[50][51]
        assert array[50][51] == array[50][49]
        assert array[50][49] == array[51][50]

        array = decorator.array_function(circular.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                       pixel_scale=0.5)

        assert array[100][100] > array[100][101]
        assert array[100][100] > array[99][100]
        assert array[99][100] == array[100][101]
        assert array[100][101] == array[100][99]
        assert array[100][99] == array[101][100]

    def test_origin_symmetric_profile(self, circular):
        array = decorator.array_function(circular.flux_at_coordinates)()

        assert circular.flux_at_coordinates((-5, 0)) < circular.flux_at_coordinates((0, 0))
        assert circular.flux_at_coordinates((5, 0)) < circular.flux_at_coordinates((0, 0))
        assert circular.flux_at_coordinates((0, -5)) < circular.flux_at_coordinates((0, 0))
        assert circular.flux_at_coordinates((0, 5)) < circular.flux_at_coordinates((0, 0))
        assert circular.flux_at_coordinates((5, 5)) < circular.flux_at_coordinates((0, 0))
        assert circular.flux_at_coordinates((-5, -5)) < circular.flux_at_coordinates((0, 0))

        assert array.shape == (100, 100)

        assert array[50][50] > array[50][51]
        assert array[50][50] > array[49][50]
        assert array[49][50] == pytest.approx(array[50][51], 1e-10)
        assert array[50][51] == pytest.approx(array[50][49], 1e-10)
        assert array[50][49] == pytest.approx(array[51][50], 1e-10)

    def test__deflection_angle_array(self):
        mass_profile = profile.EllipticalIsothermalMassProfile(centre=(0, 0), axis_ratio=0.5, phi=45.0,
                                                               einstein_radius=2.0)
        # noinspection PyTypeChecker
        assert all(decorator.array_function(mass_profile.compute_deflection_angle)(-1, -1, -0.5, -0.5, 0.1)[0][
                       0] == mass_profile.compute_deflection_angle((-1, -1)))


class MockProfile(object):
    @profile.transform_coordinates
    def is_transformed(self, coordinates):
        return isinstance(coordinates, profile.TransformedCoordinates)

    # noinspection PyMethodMayBeStatic
    def coordinates_rotate_to_elliptical(self, coordinates):
        return profile.TransformedCoordinates((coordinates[0] + 1, coordinates[1] + 1))

    # noinspection PyMethodMayBeStatic
    def coordinates_back_to_cartesian(self, coordinates):
        return coordinates[0], coordinates[1]

    @profile.transform_coordinates
    def return_coordinates(self, coordinates):
        return coordinates


class TestTransform(object):
    def test_transform(self):
        mock_profile = MockProfile()
        assert mock_profile.is_transformed((0, 0))
        assert mock_profile.return_coordinates((0, 0)) == (1, 1)
        assert mock_profile.return_coordinates(
            profile.TransformedCoordinates((0, 0))) == profile.TransformedCoordinates((0, 0))

    def test_exceptions(self, elliptical):
        with pytest.raises(profile.CoordinatesException):
            elliptical.coordinates_rotate_to_elliptical(profile.TransformedCoordinates((0, 0)))
