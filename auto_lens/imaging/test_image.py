from auto_lens.imaging import image
import numpy as np
import pytest


@pytest.fixture(name="data_grid")
def make_data_grid():
    return image.DataGrid(np.zeros((5, 5)), pixel_scale=0.5)


class TestDataGrid(object):

    def test__constructor(self, data_grid):
            # Does the data grid class correctly instantiate as an instance of ndarray?
            assert data_grid.shape == (5, 5)
            assert data_grid.pixel_scale == 0.5
            assert isinstance(data_grid, np.ndarray)
            assert isinstance(data_grid, image.DataGrid)

    class TestCentralPixel:

        def test__3x3_grid__central_pixel_is_1_and_1(self):
            grid = image.DataGrid(np.zeros((3, 3)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (1, 1)

        def test__4x4_grid__central_pixel_is_1dot5_and_1dot5(self):
            grid = image.DataGrid(np.zeros((4, 4)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (1.5, 1.5)

        def test__5x3_grid__central_pixel_is_2_and_1(self):
            grid = image.DataGrid(np.zeros((5, 3)), pixel_scale=0.1)
            assert grid.central_pixel_coordinates == (2, 1)

        def test__central_pixel_coordinates_5x5(self, data_grid):
            assert data_grid.central_pixel_coordinates == (2, 2)

    class TestConversion:

        def test__pixels_to_arcseconds(self, data_grid):
            assert data_grid.pixels_to_arc_seconds(1) == 0.5

        def test__arcseconds_to_pixels(self, data_grid):
            assert data_grid.arc_seconds_to_pixels(1) == 2

        def test__pixel_coordinates_to_arc_second_coordinates(self, data_grid):
            # Does the central pixel have (0, 0) coordinates in arcseconds?
            assert data_grid.pixel_coordinates_to_arc_second_coordinates((2, 2)) == (0, 0)

        def test__arc_second_coordinates_to_pixel_coordinates(self, data_grid):
            # Does the (0, 0) coordinates correspond to the central pixel?
            assert data_grid.arc_second_coordinates_to_pixel_coordinates((0, 0)) == (2, 2)
