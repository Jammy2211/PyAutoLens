from __future__ import division, print_function
import pytest
import numpy as np
from auto_lens.imaging import imaging
import os

test_data_dir = "{}/../../data/test_data/".format(os.path.dirname(os.path.realpath(__file__)))


class TestDataGrid(object):


    class TestConstructors:

        def test__setup_all_attributes_correctly(self):

            grid = imaging.DataGrid(pixel_dimensions=(2, 2), pixel_scale=1.0)

            assert (grid.grid_coordinates() == np.array([[[-0.5, 0.5], [0.5, 0.5]],
                                                         [[-0.5, -0.5], [0.5, -0.5]]])).all()

            assert grid.pixel_scale == 1.0
            assert grid.central_pixels == (0.5, 0.5)
            assert grid.pixel_dimensions == (2, 2)
            assert grid.arc_second_dimensions == (2.0, 2.0)

        def test__from_arcsecond_dimensions__setup_all_attrbitues_correctly(self):

            grid = imaging.DataGrid.from_arcsecond_dimensions(arc_second_dimensions=(2.0, 2.0), pixel_scale=1.0)

            assert (grid.grid_coordinates() == np.array([[[-0.5, 0.5], [0.5, 0.5]],
                                                         [[-0.5, -0.5], [0.5, -0.5]]])).all()

            assert grid.pixel_scale == 1.0
            assert grid.central_pixels == (0.5, 0.5)
            assert grid.pixel_dimensions == (2, 2)
            assert grid.arc_second_dimensions == (2.0, 2.0)


    class TestCentralPixel:

        def test__3x3_grid__central_pixel_is_1_and_1(self):

            grid = imaging.DataGrid(pixel_dimensions=(3, 3), pixel_scale=0.1)
            assert grid.central_pixels == (1, 1)

        def test__4x4_grid__central_pixel_is_1dot5_and_1dot5(self):

            grid = imaging.DataGrid(pixel_dimensions=(4, 4), pixel_scale=0.1)
            assert grid.central_pixels == (1.5, 1.5)

        def test__5x3_grid__central_pixel_is_2_and_1(self):

            grid = imaging.DataGrid(pixel_dimensions=(5, 3), pixel_scale=0.1)
            assert grid.central_pixels == (2, 1)


    # TODO : Unit Conversions tests


    class TestSetupGrid:

        def test__array_1x1__sets_up_arcsecond_coordinates(self):
            grid = imaging.DataGrid(pixel_dimensions=(1, 1), pixel_scale=1.0)

            grid_coordinates = grid.grid_coordinates()

            assert (grid_coordinates == np.array([[[0.0, 0.0]]])).all()

            assert (grid_coordinates[0, 0] == np.array([[0.0, 0.0]])).all()

        def test__array_2x2__sets_up_arcsecond_coordinates(self):
            grid = imaging.DataGrid(pixel_dimensions=(2, 2), pixel_scale=1.0)

            grid_coordinates = grid.grid_coordinates()

            assert (grid_coordinates == np.array([[[-0.5, 0.5], [0.5, 0.5]],
                                                  [[-0.5, -0.5], [0.5, -0.5]]])).all()

            assert (grid_coordinates[0, 0] == np.array([[-0.5, 0.5]])).all()
            assert (grid_coordinates[0, 1] == np.array([[0.5, 0.5]])).all()
            assert (grid_coordinates[1, 0] == np.array([[-0.5, -0.5]])).all()
            assert (grid_coordinates[1, 1] == np.array([[0.5, -0.5]])).all()

        def test__array_3x3__sets_up_arcsecond_coordinates(self):
            grid = imaging.DataGrid(pixel_dimensions=(3, 3), pixel_scale=1.0)

            grid_coordinates = grid.grid_coordinates()

            assert (grid_coordinates == np.array([[[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
                                                  [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
                                                  [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]]])).all()

            assert (grid_coordinates[0, 0] == np.array([[-1.0, 1.0]])).all()
            assert (grid_coordinates[0, 1] == np.array([[0.0, 1.0]])).all()
            assert (grid_coordinates[0, 2] == np.array([[1.0, 1.0]])).all()
            assert (grid_coordinates[1, 0] == np.array([[-1.0, 0.0]])).all()
            assert (grid_coordinates[1, 1] == np.array([[0.0, 0.0]])).all()
            assert (grid_coordinates[1, 2] == np.array([[1.0, 0.0]])).all()
            assert (grid_coordinates[2, 0] == np.array([[-1.0, -1.0]])).all()
            assert (grid_coordinates[2, 1] == np.array([[0.0, -1.0]])).all()
            assert (grid_coordinates[2, 2] == np.array([[1.0, -1.0]])).all()

        def test__array_4x4__sets_up_arcsecond_coordinates(self):
            grid = imaging.DataGrid(pixel_dimensions=(4, 4), pixel_scale=0.5)

            grid_coordinates = grid.grid_coordinates()

            assert (grid_coordinates == np.array([[[-0.75, 0.75], [-0.25, 0.75], [0.25, 0.75], [0.75, 0.75]],
                                                  [[-0.75, 0.25], [-0.25, 0.25], [0.25, 0.25], [0.75, 0.25]],
                                                  [[-0.75, -0.25], [-0.25, -0.25], [0.25, -0.25], [0.75, -0.25]],
                                                  [[-0.75, -0.75], [-0.25, -0.75], [0.25, -0.75],
                                                   [0.75, -0.75]]])).all()

        def test__array_2x3__sets_up_arcsecond_coordinates(self):
            grid = imaging.DataGrid(pixel_dimensions=(2, 3), pixel_scale=1.0)

            grid_coordinates = grid.grid_coordinates()

            assert (grid_coordinates == np.array([[[-1.0, 0.5], [0.0, 0.5], [1.0, 0.5]],
                                                  [[-1.0, -0.5], [0.0, -0.5], [1.0, -0.5]]])).all()

            assert (grid_coordinates[0, 0] == np.array([[-1.0, 0.5]])).all()
            assert (grid_coordinates[0, 1] == np.array([[0.0, 0.5]])).all()
            assert (grid_coordinates[0, 2] == np.array([[1.0, 0.5]])).all()
            assert (grid_coordinates[1, 0] == np.array([[-1.0, -0.5]])).all()
            assert (grid_coordinates[1, 1] == np.array([[0.0, -0.5]])).all()
            assert (grid_coordinates[1, 2] == np.array([[1.0, -0.5]])).all()

        def test__array_3x2__sets_up_arcsecond_coordinates(self):
            grid = imaging.DataGrid(pixel_dimensions=(3, 2), pixel_scale=1.0)

            grid_coordinates = grid.grid_coordinates()

            assert (grid_coordinates == np.array([[[-0.5, 1.0], [0.5, 1.0]],
                                                  [[-0.5, 0.0], [0.5, 0.0]],
                                                  [[-0.5, -1.0], [0.5, -1.0]]])).all()

            assert (grid_coordinates[0, 0] == np.array([[-0.5, 1.0]])).all()
            assert (grid_coordinates[0, 1] == np.array([[0.5, 1.0]])).all()
            assert (grid_coordinates[1, 0] == np.array([[-0.5, 0.0]])).all()
            assert (grid_coordinates[1, 1] == np.array([[0.5, 0.0]])).all()
            assert (grid_coordinates[2, 0] == np.array([[-0.5, -1.0]])).all()
            assert (grid_coordinates[2, 1] == np.array([[0.5, -1.0]])).all()


class TestData(object):


    class TestConstructor:

        def test__sets_up_all_attributes_and_inherites_from_data_grid(self):
        
            data = imaging.Data(data=np.ones((3, 3)), pixel_scale=0.1)

            assert (data.data == np.ones((3, 3))).all()
            assert data.pixel_scale == 0.1
            assert data.pixel_dimensions == (3, 3)
            assert data.central_pixels == (1.0, 1.0)
            assert data.arc_second_dimensions == pytest.approx((0.3, 0.3))


    class TestPad:

        def test__from_3x3_to_5x5(self):

            data = np.ones((3, 3))
            data[1, 1] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)
            data.pad(new_dimensions=(5, 5))

            assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.0, 1.0, 1.0, 1.0, 0.0],
                                             [0.0, 1.0, 2.0, 1.0, 0.0],
                                             [0.0, 1.0, 1.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0]])).all()
            
            assert data.pixel_dimensions == (5, 5)
            assert data.arc_second_dimensions == (5.0, 5.0)

        def test__from_5x5_to_9x9(self):

            data = np.ones((5, 5))
            data[2, 2] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)
            data.pad(new_dimensions=(9, 9))

            assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert data.pixel_dimensions == (9, 9)
            assert data.arc_second_dimensions == (9.0, 9.0)

        def test__from_3x3_to_4x4__goes_to_5x5_to_keep_symmetry(self):
            
            data = np.ones((3, 3))
            data[1, 1] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)
            data.pad(new_dimensions=(4, 4))

            assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 1.0, 1.0, 1.0, 0.0],
                                           [0.0, 1.0, 2.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert data.pixel_dimensions == (5, 5)
            assert data.arc_second_dimensions == (5.0, 5.0)

        def test__from_5x5_to_8x8__goes_to_9x9_to_keep_symmetry(self):
            
            data = np.ones((5, 5))
            data[2, 2] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)
            data.pad(new_dimensions=(8, 8))

            assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert data.pixel_dimensions == (9, 9)
            assert data.arc_second_dimensions == (9.0, 9.0)

        def test__from_4x4_to_6x6(self):
            
            data = np.ones((4, 4))
            data[1:3, 1:3] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)
            data.pad(new_dimensions=(6, 6))

            assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                           [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                           [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert data.pixel_dimensions == (6, 6)
            assert data.arc_second_dimensions == (6.0, 6.0)

        def test__from_4x4_to_8x8(self):
            
            data = np.ones((4, 4))
            data[1:3, 1:3] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)
            data.pad(new_dimensions=(8, 8))

            assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert data.pixel_dimensions == (8, 8)
            assert data.arc_second_dimensions == (8.0, 8.0)

        def test__from_4x4_to_5x5__goes_to_6x6_to_keep_symmetry(self):
            
            data = np.ones((4, 4))
            data[1:3, 1:3] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)
            data.pad(new_dimensions=(5, 5))

            assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                           [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                           [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert data.pixel_dimensions == (6, 6)
            assert data.arc_second_dimensions == (6.0, 6.0)

        def test__from_4x4_to_7x7__goes_to_8x8_to_keep_symmetry(self):
            
            data = np.ones((4, 4))
            data[1:3, 1:3] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)
            data.pad(new_dimensions=(7, 7))

            assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert data.pixel_dimensions == (8, 8)
            assert data.arc_second_dimensions == (8.0, 8.0)

        def test__from_5x4_to_7x6(self):
            
            data = np.ones((5, 4))
            data[2, 1:3] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)
            data.pad(new_dimensions=(7, 6))

            assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                           [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert data.pixel_dimensions == (7, 6)
            assert data.arc_second_dimensions == (7.0, 6.0)

        def test__from_2x3_to_6x7(self):
            
            data = np.ones((2, 3))
            data[0:2, 1] = 2.0
            data[1, 2] = 9

            data = imaging.Data(data, pixel_scale=1.0)
            data.pad(new_dimensions=(6, 7))

            assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 2.0, 9.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert data.pixel_dimensions == (6, 7)
            assert data.arc_second_dimensions == (6.0, 7.0)

        def test__from_2x3_to_5x6__goes_to_6x7_to_keep_symmetry(self):
            
            data = np.ones((2, 3))
            data[0:2, 1] = 2.0
            data[1, 2] = 9

            data = imaging.Data(data, pixel_scale=1.0)
            data.pad(new_dimensions=(5, 6))

            assert (data.data == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 2.0, 9.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).all()

            assert data.pixel_dimensions == (6, 7)
            assert data.arc_second_dimensions == (6.0, 7.0)

        def test__x_size_smaller_than_data__raises_error(self):
            
            data = np.ones((5, 5))
            data[2, 2] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            with pytest.raises(ValueError):
                assert data.trim(new_dimensions=(3, 8))

        def test__y_size_smaller_than_data__raises_error(self):
            
            data = np.ones((5, 5))
            data[2, 2] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            with pytest.raises(ValueError):
                assert data.trim(new_dimensions=(8, 3))


    class TestTrim:

        def test__from_5x5_to_3x3(self):

            data = np.ones((5, 5))
            data[2, 2] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(3, 3))

            assert (data.data == np.array([[1.0, 1.0, 1.0],
                                           [1.0, 2.0, 1.0],
                                           [1.0, 1.0, 1.0]])).all()

            assert data.pixel_dimensions == (3, 3)
            assert data.arc_second_dimensions == (3.0, 3.0)

        def test__from_7x7_to_3x3(self):

            data = np.ones((7, 7))
            data[3, 3] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(3, 3))

            assert (data.data == np.array([[1.0, 1.0, 1.0],
                                           [1.0, 2.0, 1.0],
                                           [1.0, 1.0, 1.0]])).all()

            assert data.pixel_dimensions == (3, 3)
            assert data.arc_second_dimensions == (3.0, 3.0)

        def test__from_11x11_to_5x5(self):

            data = np.ones((11, 11))
            data[5, 5] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(5, 5))

            assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 2.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

            assert data.pixel_dimensions == (5, 5)
            assert data.arc_second_dimensions == (5.0, 5.0)

        def test__from_5x5_to_2x2__goes_to_3x3_to_keep_symmetry(self):

            data = np.ones((5, 5))
            data[2, 2] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(2, 2))

            assert (data.data == np.array([[1.0, 1.0, 1.0],
                                           [1.0, 2.0, 1.0],
                                           [1.0, 1.0, 1.0]])).all()

            assert data.pixel_dimensions == (3, 3)
            assert data.arc_second_dimensions == (3.0, 3.0)

        def test__from_5x5_to_4x4__goes_to_5x5_to_keep_symmetry(self):

            data = np.ones((5, 5))
            data[2, 2] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(4, 4))

            assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 2.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

            assert data.pixel_dimensions == (5, 5)
            assert data.arc_second_dimensions == (5.0, 5.0)

        def test__from_11x11_to_4x4__goes_to_5x5_to_keep_symmetry(self):

            data = np.ones((11, 11))
            data[5, 5] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(4, 4))

            assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 2.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0]])).all()

            assert data.pixel_dimensions == (5, 5)
            assert data.arc_second_dimensions == (5.0, 5.0)

        def test__from_4x4_to_2x2(self):

            data = np.ones((4, 4))
            data[1:3, 1:3] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(2, 2))

            assert (data.data == np.array([[2.0, 2.0],
                                           [2.0, 2.0]])).all()

            assert data.pixel_dimensions == (2, 2)
            assert data.arc_second_dimensions == (2.0, 2.0)

        def test__from_6x6_to_4x4(self):

            data = np.ones((6, 6))
            data[2:4, 2:4] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(4, 4))

            assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0],
                                           [1.0, 2.0, 2.0, 1.0],
                                           [1.0, 2.0, 2.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0]])).all()

            assert data.pixel_dimensions == (4, 4)
            assert data.arc_second_dimensions == (4.0, 4.0)

        def test__from_12x12_to_6x6(self):

            data = np.ones((12, 12))
            data[5:7, 5:7] = 2.0
            data[4, 4] = 9.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(6, 6))

            assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 9.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                           [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])).all()

            assert data.pixel_dimensions == (6, 6)
            assert data.arc_second_dimensions == (6.0, 6.0)

        def test__from_4x4_to_3x3__goes_to_4x4_to_keep_symmetry(self):

            data = np.ones((4, 4))
            data[1:3, 1:3] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(3, 3))

            assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0],
                                           [1.0, 2.0, 2.0, 1.0],
                                           [1.0, 2.0, 2.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0]])).all()

            assert data.pixel_dimensions == (4, 4)
            assert data.arc_second_dimensions == (4.0, 4.0)

        def test__from_6x6_to_3x3_goes_to_4x4_to_keep_symmetry(self):

            data = np.ones((6, 6))
            data[2:4, 2:4] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(3, 3))

            assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0],
                                           [1.0, 2.0, 2.0, 1.0],
                                           [1.0, 2.0, 2.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0]])).all()

            assert data.pixel_dimensions == (4, 4)
            assert data.arc_second_dimensions == (4.0, 4.0)

        def test__from_12x12_to_5x5__goes_to_6x6_to_keep_symmetry(self):

            data = np.ones((12, 12))
            data[5:7, 5:7] = 2.0
            data[4, 4] = 9.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(5, 5))

            assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 9.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                           [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])).all()

            assert data.pixel_dimensions == (6, 6)
            assert data.arc_second_dimensions == (6.0, 6.0)

        def test__from_5x4_to_3x2(self):

            data = np.ones((5, 4))
            data[2, 1:3] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(3, 2))

            assert (data.data == np.array([[1.0, 1.0],
                                           [2.0, 2.0],
                                           [1.0, 1.0]])).all()

            assert data.pixel_dimensions == (3, 2)
            assert data.arc_second_dimensions == (3.0, 2.0)

        def test__from_4x5_to_2x3(self):

            data = np.ones((4, 5))
            data[1:3, 2] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(2, 3))

            assert (data.data == np.array([[1.0, 2.0, 1.0],
                                           [1.0, 2.0, 1.0]])).all()

            assert data.pixel_dimensions == (2, 3)
            assert data.arc_second_dimensions == (2.0, 3.0)

        def test__from_5x4_to_4x3__goes_to_5x4_to_keep_symmetry(self):

            data = np.ones((5, 4))
            data[2, 1:3] = 2.0
            data[4, 3] = 9.0

            data = imaging.Data(data, pixel_scale=1.0)

            data.trim(new_dimensions=(4, 3))

            assert (data.data == np.array([[1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0],
                                           [1.0, 2.0, 2.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 9.0]])).all()

            assert data.pixel_dimensions == (5, 4)
            assert data.arc_second_dimensions == (5.0, 4.0)

        def test__x_size_bigger_than_data__raises_error(self):

            data = np.ones((5, 5))
            data[2, 2] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            with pytest.raises(ValueError):
                assert data.trim(new_dimensions=(8, 3))

        def test__y_size_bigger_than_data__raises_error(self):

            data = np.ones((5, 5))
            data[2, 2] = 2.0

            data = imaging.Data(data, pixel_scale=1.0)

            with pytest.raises(ValueError):
                assert data.trim(new_dimensions=(3, 8))


class TestImage(object):


    class TestConstructors(object):

        def test__input_image_3x3__all_attributes_correct_including_data_inheritance(self):

            image = imaging.Image(data=np.ones((3, 3)), pixel_scale=0.1)

            assert (image.data == np.ones((3, 3))).all()
            assert image.pixel_scale == 0.1
            assert image.pixel_dimensions == (3, 3)
            assert image.central_pixels == (1.0, 1.0)
            assert image.arc_second_dimensions == pytest.approx((0.3, 0.3))

        def test__init__input_image_4x3__all_attributes_correct_including_data_inheritance(self):
            image = imaging.Image(data=np.ones((4, 3)), pixel_scale=0.1)

            assert (image.data == np.ones((4, 3))).all()
            assert image.pixel_scale == 0.1
            assert image.pixel_dimensions == (4, 3)
            assert image.central_pixels == (1.5, 1.0)
            assert image.arc_second_dimensions == pytest.approx((0.4, 0.3))

        def test__from_fits__input_image_3x3__all_attributes_correct_including_data_inheritance(self):
            
            image = imaging.Image.from_fits(path=test_data_dir, filename='3x3_ones.fits', hdu=0, pixel_scale=0.1)

            assert (image.data == np.ones((3, 3))).all()
            assert image.pixel_scale == 0.1
            assert image.data.shape == (3, 3)
            assert image.central_pixels == (1.0, 1.0)
            assert image.arc_second_dimensions == pytest.approx((0.3, 0.3))

        def test__from_fits__input_image_4x3__all_attributes_correct_including_data_inheritance(self):

            image = imaging.Image.from_fits(path=test_data_dir, filename='4x3_ones.fits', hdu=0, pixel_scale=0.1)

            assert (image.data == np.ones((4, 3))).all()
            assert image.pixel_scale == 0.1
            assert image.data.shape == (4, 3)
            assert image.central_pixels == (1.5, 1.0)
            assert image.arc_second_dimensions == pytest.approx((0.4, 0.3))


    class TestEstimateBackgroundNoise(object):

        def test__via_edges__input_all_ones__sky_bg_level_1(self):

            image = imaging.Image(data=np.ones((3, 3)), pixel_scale=0.1)
            sky_noise = image.estimate_background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__3x3_image_simple_gaussian__answer_ignores_central_pixel(self):

            image_data = np.array([[1, 1, 1],
                                   [1, 100, 1],
                                   [1, 1, 1]])

            image = imaging.Image(data=image_data, pixel_scale=0.1)
            sky_noise = image.estimate_background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__4x3_image_simple_gaussian__ignores_central_pixels(self):

            image_data = np.array([[1, 1, 1],
                                   [1, 100, 1],
                                   [1, 100, 1],
                                   [1, 1, 1]])

            image = imaging.Image(data=image_data, pixel_scale=0.1)
            sky_noise = image.estimate_background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__4x4_image_simple_gaussian__ignores_central_pixels(self):
            image_data = np.array([[1, 1, 1, 1],
                                   [1, 100, 100, 1],
                                   [1, 100, 100, 1],
                                   [1, 1, 1, 1]])

            image = imaging.Image(data=image_data, pixel_scale=0.1)
            sky_noise = image.estimate_background_noise_from_edges(no_edges=1)

            assert sky_noise == 0.0

        def test__via_edges__5x5_image_simple_gaussian_two_edges__ignores_central_pixel(self):
            image_data = np.array([[1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 100, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1]])

            image = imaging.Image(data=image_data, pixel_scale=0.1)
            sky_noise = image.estimate_background_noise_from_edges(no_edges=2)

            assert sky_noise == 0.0

        def test__via_edges__6x5_image_two_edges__values(self):
            image_data = np.array([[0, 1, 2, 3, 4],
                                   [5, 6, 7, 8, 9],
                                   [10, 11, 100, 12, 13],
                                   [14, 15, 100, 16, 17],
                                   [18, 19, 20, 21, 22],
                                   [23, 24, 25, 26, 27]])

            image = imaging.Image(data=image_data, pixel_scale=0.1)
            sky_noise = image.estimate_background_noise_from_edges(no_edges=2)

            assert sky_noise == np.std(np.arange(28))

        def test__via_edges__7x7_image_three_edges__values(self):
            image_data = np.array([[0, 1, 2, 3, 4, 5, 6],
                                   [7, 8, 9, 10, 11, 12, 13],
                                   [14, 15, 16, 17, 18, 19, 20],
                                   [21, 22, 23, 100, 24, 25, 26],
                                   [27, 28, 29, 30, 31, 32, 33],
                                   [34, 35, 36, 37, 38, 39, 40],
                                   [41, 42, 43, 44, 45, 46, 47]])

            image = imaging.Image(data=image_data, pixel_scale=0.1)
            sky_noise = image.estimate_background_noise_from_edges(no_edges=3)

            assert sky_noise == np.std(np.arange(48))


    class TestImagingConstructors(object):

        def test__circular_mask(self):

            mask = imaging.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1.0, radius_mask=5)

            image = imaging.Image(data=np.ones((3, 3)), pixel_scale=1.0)

            image_mask = image.circle_mask(radius_mask=5.0)

            assert (mask.mask == image_mask.mask).all()

        def test__annulus_mask(self):

            mask = imaging.Mask.annular(arc_second_dimensions=(3, 3), pixel_scale=1, inner_radius_mask=0,
                                        outer_radius_mask=0.5)

            image = imaging.Image(data=np.ones((3, 3)), pixel_scale=1.0)

            image_mask = image.annulus_mask(inner_radius_mask=0.0, outer_radius_mask=0.5)

            assert (mask.mask == image_mask.mask).all()

        def test__unmasked(self):

            image = imaging.Image(data=np.ones((3, 3)), pixel_scale=1.0)

            image_mask = image.unmasked()

            assert (image_mask.mask == np.array([[False, False, False],
                                      [False, False, False],
                                      [False, False, False]])).all()


class TestNoise(object):


    class TestConstructors(object):

        def test__init__input_noise_3x3__all_attributes_correct_including_data_inheritance(self):
            
            noise = imaging.Noise(data=np.ones((3, 3)), pixel_scale=0.1)

            assert (noise.data == np.ones((3, 3))).all()
            assert noise.pixel_scale == 0.1
            assert noise.pixel_dimensions == (3, 3)
            assert noise.central_pixels == (1.0, 1.0)
            assert noise.arc_second_dimensions == pytest.approx((0.3, 0.3))

        def test__init__input_noise_4x3__all_attributes_correct_including_data_inheritance(self):
            
            noise = imaging.Noise(data=np.ones((4, 3)), pixel_scale=0.1)

            assert (noise.data == np.ones((4, 3))).all()
            assert noise.pixel_scale == 0.1
            assert noise.pixel_dimensions == (4, 3)
            assert noise.central_pixels == (1.5, 1.0)
            assert noise.arc_second_dimensions == pytest.approx((0.4, 0.3))

        def test__from_fits__input_noise_3x3__all_attributes_correct_including_data_inheritance(self):
            
            noise = imaging.Noise.from_fits(path=test_data_dir, filename='3x3_ones.fits', hdu=0, pixel_scale=1.0)

            assert (noise.data == np.ones((3, 3))).all()
            assert noise.pixel_scale == 1.0
            assert noise.pixel_dimensions == (3, 3)
            assert noise.central_pixels == (1.0, 1.0)
            assert noise.arc_second_dimensions == pytest.approx((3.0, 3.0))


        def test__from_fits__input_noise_4x3__all_attributes_correct_including_data_inheritance(self):

            noise = imaging.Noise.from_fits(path=test_data_dir, filename='4x3_ones.fits', hdu=0, pixel_scale=0.1)

            assert (noise.data == np.ones((4, 3))).all()
            assert noise.pixel_scale == 0.1
            assert noise.pixel_dimensions == (4, 3)
            assert noise.central_pixels == (1.5, 1.0)
            assert noise.arc_second_dimensions == pytest.approx((0.4, 0.3))


class TestNoiseBackground(object):
    
    class TestConstructors(object):

        def test__init__input_background_noise_single_value__all_attributes_correct_including_data_inheritance(self):

            background_noise = imaging.NoiseBackground.from_one_value(background_noise=5.0, pixel_dimensions=(3,3),
                                                                      pixel_scale=1.0)

            assert (background_noise.data == 5.0*np.ones((3,3))).all()
            assert background_noise.pixel_scale == 1.0
            assert background_noise.pixel_dimensions == (3, 3)
            assert background_noise.central_pixels == (1.0, 1.0)
            assert background_noise.arc_second_dimensions == pytest.approx((3.0, 3.0))

        def test__init__input_background_noise_3x3__all_attributes_correct_including_data_inheritance(self):
            
            background_noise = imaging.NoiseBackground(data=np.ones((3, 3)), pixel_scale=1.0)

            assert background_noise.pixel_scale == 1.0
            assert background_noise.pixel_dimensions == (3, 3)
            assert background_noise.central_pixels == (1.0, 1.0)
            assert background_noise.arc_second_dimensions == pytest.approx((3.0, 3.0))
            assert (background_noise.data == np.ones((3, 3))).all()

        def test__init__input_background_noise_4x3__all_attributes_correct_including_data_inheritance(self):
            
            background_noise = imaging.NoiseBackground(data=np.ones((4, 3)), pixel_scale=0.1)

            assert (background_noise.data == np.ones((4, 3))).all()
            assert background_noise.pixel_scale == 0.1
            assert background_noise.pixel_dimensions == (4, 3)
            assert background_noise.central_pixels == (1.5, 1.0)
            assert background_noise.arc_second_dimensions == pytest.approx((0.4, 0.3))

        def test__from_fits__input_background_noise_3x3__all_attributes_correct_including_data_inheritance(self):
            
            background_noise = imaging.NoiseBackground.from_fits(path=test_data_dir, filename='3x3_ones.fits', hdu=0,
                                                           pixel_scale=1.0)

            assert (background_noise.data == np.ones((3, 3))).all()
            assert background_noise.pixel_scale == 1.0
            assert background_noise.pixel_dimensions == (3, 3)
            assert background_noise.central_pixels == (1.0, 1.0)
            assert background_noise.arc_second_dimensions == pytest.approx((3.0, 3.0))

        def test__from_fits__input_background_noise_4x3__all_attributes_correct_including_data_inheritance(self):
            
            background_noise = imaging.NoiseBackground.from_fits(path=test_data_dir, filename='4x3_ones.fits', hdu=0,
                                                           pixel_scale=0.1)

            assert (background_noise.data == np.ones((4, 3))).all()
            assert background_noise.pixel_scale == 0.1
            assert background_noise.pixel_dimensions == (4, 3)
            assert background_noise.central_pixels == (1.5, 1.0)
            assert background_noise.arc_second_dimensions == pytest.approx((0.4, 0.3))


class TestPSF(object):


    class TestConstructors(object):

        def test__init__input_psf_3x3__all_attributes_correct_including_data_inheritance(self):
            
            psf = imaging.PSF(data=np.ones((3, 3)), pixel_scale=1.0, renormalize=False)

            assert psf.pixel_scale == 1.0
            assert psf.pixel_dimensions == (3, 3)
            assert psf.central_pixels == (1.0, 1.0)
            assert psf.arc_second_dimensions == pytest.approx((3.0, 3.0))
            assert (psf.data == np.ones((3, 3))).all()

        def test__init__input_psf_4x3__all_attributes_correct_including_data_inheritance(self):
            
            psf = imaging.PSF(data=np.ones((4, 3)), pixel_scale=0.1, renormalize=False)

            assert (psf.data == np.ones((4, 3))).all()
            assert psf.pixel_scale == 0.1
            assert psf.pixel_dimensions == (4, 3)
            assert psf.central_pixels == (1.5, 1.0)
            assert psf.arc_second_dimensions == pytest.approx((0.4, 0.3))

        def test__from_fits__input_psf_3x3__all_attributes_correct_including_data_inheritance(self):
            psf = imaging.PSF.from_fits(path=test_data_dir, filename='3x3_ones.fits', hdu=0, pixel_scale=1.0,
                                             renormalize=False)

            assert (psf.data == np.ones((3, 3))).all()
            assert psf.pixel_scale == 1.0
            assert psf.pixel_dimensions == (3, 3)
            assert psf.central_pixels == (1.0, 1.0)
            assert psf.arc_second_dimensions == pytest.approx((3.0, 3.0))

        def test__from_fits__input_psf_4x3__all_attributes_correct_including_data_inheritance(self):
            psf = imaging.PSF.from_fits(path=test_data_dir, filename='4x3_ones.fits', hdu=0, pixel_scale=0.1,
                                             renormalize=False)

            assert (psf.data == np.ones((4, 3))).all()
            assert psf.pixel_scale == 0.1
            assert psf.pixel_dimensions == (4, 3)
            assert psf.central_pixels == (1.5, 1.0)
            assert psf.arc_second_dimensions == pytest.approx((0.4, 0.3))


    class TestRenormalize(object):

        def test__input_is_already_normalized__no_change(self):

            psf_data = np.ones((3, 3)) / 9.0

            psf = imaging.PSF(data=psf_data, pixel_scale=1.0, renormalize=True)

            assert psf.data == pytest.approx(psf_data, 1e-3)

        def test__input_is_above_normalization_so_is_normalized(self):

            psf_data = np.ones((3, 3)) / 9.0

            psf = imaging.PSF(data=psf_data, pixel_scale=1.0, renormalize=True)

            assert psf.data == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

        def test__input_is_below_normalization_so_is_normalized(self):

            psf_data = np.ones((3, 3)) / 90.0

            psf = imaging.PSF(data=psf_data, pixel_scale=1.0, renormalize=True)

            assert psf.data == pytest.approx(np.ones((3, 3)) / 90.0, 1e-3)


    class TestConvolve(object):

        def test__kernel_is_not_odd_x_odd__raises_error(self):

            image = np.array([[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]])

            kernel = np.array([[0.0, 1.0],
                               [1.0, 2.0]])

            psf = imaging.PSF(data=kernel, pixel_scale=1.0)

            with pytest.raises(imaging.KernelException):
                psf.convolve_with_image(image)

        def test__image_is_3x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(self):
        
            image = np.array([[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]])
        
            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = imaging.PSF(data=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve_with_image(image)
        
            assert (blurred_image == kernel).all()

        def test__image_is_4x4_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(self):

            image = np.array([[0.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]])
        
            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = imaging.PSF(data=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve_with_image(image)
        
            assert (blurred_image == np.array([[0.0, 1.0, 0.0, 0.0],
                                               [1.0, 2.0, 1.0, 0.0],
                                               [0.0, 1.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 0.0]])).all()
        
        def test__image_is_4x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(self):

            image = np.array([[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]])
        
            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])
        
            psf = imaging.PSF(data=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve_with_image(image)

            assert (blurred_image == np.array([[0.0, 1.0, 0.0],
                                               [1.0, 2.0, 1.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 0.0, 0.0]])).all()
        
        def test__image_is_3x4_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(self):
            image = np.array([[0.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]])
        
            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])
        
            psf = imaging.PSF(data=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve_with_image(image)
        
            assert (blurred_image == np.array([[0.0, 1.0, 0.0, 0.0],
                                               [1.0, 2.0, 1.0, 0.0],
                                               [0.0, 1.0, 0.0, 0.0]])).all()
        
        def test__image_is_4x4_has_two_central_values__kernel_is_asymmetric__blurred_image_follows_convolution(self):

            image = np.array([[0.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]])
        
            kernel = np.array([[1.0, 1.0, 1.0],
                               [2.0, 2.0, 1.0],
                               [1.0, 3.0, 3.0]])
        
            psf = imaging.PSF(data=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve_with_image(image)
        
            assert (blurred_image == np.array([[1.0, 1.0, 1.0, 0.0],
                                               [2.0, 3.0, 2.0, 1.0],
                                               [1.0, 5.0, 5.0, 1.0],
                                               [0.0, 1.0, 3.0, 3.0]])).all()
        
        def test__image_is_4x4_values_are_on_edge__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(self):

            image = np.array([[0.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0],
                              [0.0, 0.0, 0.0, 0.0]])
        
            kernel = np.array([[1.0, 1.0, 1.0],
                               [2.0, 2.0, 1.0],
                               [1.0, 3.0, 3.0]])
        
            psf = imaging.PSF(data=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve_with_image(image)

            assert (blurred_image == np.array([[1.0, 1.0, 0.0, 0.0],
                                               [2.0, 1.0, 1.0, 1.0],
                                               [3.0, 3.0, 2.0, 2.0],
                                               [0.0, 0.0, 1.0, 3.0]])).all()
        
        def test__image_is_4x4_values_are_on_corner__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(self):

            image = np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
        
            kernel = np.array([[1.0, 1.0, 1.0],
                               [2.0, 2.0, 1.0],
                               [1.0, 3.0, 3.0]])

            psf = imaging.PSF(data=kernel, pixel_scale=1.0)

            blurred_image = psf.convolve_with_image(image)

            assert (blurred_image == np.array([[2.0, 1.0, 0.0, 0.0],
                                               [3.0, 3.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 1.0],
                                               [0.0, 0.0, 2.0, 2.0]])).all()


class TestExpsoureTime(object):


    class TestConstructors(object):

        def test__init__input_exposure_time_single_value__all_attributes_correct_including_data_inheritance(self):

            exposure_time = imaging.ExposureTime.from_one_value(exposure_time=5.0, pixel_dimensions=(3, 3),
                                                                pixel_scale=1.0)

            assert (exposure_time.data == 5.0*np.ones((3,3))).all()
            assert exposure_time.pixel_scale == 1.0
            assert exposure_time.pixel_dimensions == (3, 3)
            assert exposure_time.central_pixels == (1.0, 1.0)
            assert exposure_time.arc_second_dimensions == pytest.approx((3.0, 3.0))

        def test__init__input_exposure_time_3x3__all_attributes_correct_including_data_inheritance(self):
            
            exposure_time = imaging.ExposureTime(data=np.ones((3, 3)), pixel_scale=1.0)

            assert exposure_time.pixel_scale == 1.0
            assert exposure_time.pixel_dimensions == (3, 3)
            assert exposure_time.central_pixels == (1.0, 1.0)
            assert exposure_time.arc_second_dimensions == pytest.approx((3.0, 3.0))
            assert (exposure_time.data == np.ones((3, 3))).all()

        def test__init__input_exposure_time_4x3__all_attributes_correct_including_data_inheritance(self):
            
            exposure_time = imaging.ExposureTime(data=np.ones((4, 3)), pixel_scale=0.1)

            assert (exposure_time.data == np.ones((4, 3))).all()
            assert exposure_time.pixel_scale == 0.1
            assert exposure_time.pixel_dimensions == (4, 3)
            assert exposure_time.central_pixels == (1.5, 1.0)
            assert exposure_time.arc_second_dimensions == pytest.approx((0.4, 0.3))

        def test__from_fits__input_exposure_time_3x3__all_attributes_correct_including_data_inheritance(self):

            exposure_time = imaging.ExposureTime.from_fits(path=test_data_dir, filename='3x3_ones.fits', hdu=0,
                                                           pixel_scale=1.0)

            assert (exposure_time.data == np.ones((3, 3))).all()
            assert exposure_time.pixel_scale == 1.0
            assert exposure_time.pixel_dimensions == (3, 3)
            assert exposure_time.central_pixels == (1.0, 1.0)
            assert exposure_time.arc_second_dimensions == pytest.approx((3.0, 3.0))

        def test__from_fits__input_exposure_time_4x3__all_attributes_correct_including_data_inheritance(self):

            exposure_time = imaging.ExposureTime.from_fits(path=test_data_dir, filename='4x3_ones.fits', hdu=0,
                                                           pixel_scale=0.1)

            assert (exposure_time.data == np.ones((4, 3))).all()
            assert exposure_time.pixel_scale == 0.1
            assert exposure_time.pixel_dimensions == (4, 3)
            assert exposure_time.central_pixels == (1.5, 1.0)
            assert exposure_time.arc_second_dimensions == pytest.approx((0.4, 0.3))


class TestEstimateNoiseFromImage:

    def test__image_and_exposure_times_float_1__no_background__noise_is_all_1s(self):
        
        # Image (eps) = 1.0
        # Background (eps) = 0.0
        # Exposure times = 1.0 s
        # Image (counts) = 1.0
        # Background (counts) = 0.0

        # Noise (counts) = sqrt(1.0 + 0.0**2) = 1.0
        # Noise (eps) = 1.0 / 1.0

        image = np.ones((3, 3))

        exposure_time = 1.0

        noise_estimate = imaging.estimate_noise_from_image(image, exposure_time,
                                                                   background_noise=0.0)

        assert (noise_estimate == np.ones((3, 3))).all()

    def test__image_and_exposure_time_ndarray_all_1s__no_background__noise_is_all_1s(self):

        # Image (eps) = 1.0
        # Background (eps) = 0.0
        # Exposure times = 1.0 s
        # Image (counts) = 1.0
        # Background (counts) = 0.0

        # Noise (counts) = sqrt(1.0 + 0.0**2) = 1.0
        # Noise (eps) = 1.0 / 1.0

        image = np.ones((3, 3))

        exposure_time = np.ones((3, 3))

        noise_estimate = imaging.estimate_noise_from_image(image, exposure_time,
                                                                   background_noise=0.0)

        assert (noise_estimate == np.ones((3, 3))).all()

    def test__image_all_4s__exposure_time_all_1s__no_background__noise_is_all_2s(self):

        # Image (eps) = 4.0
        # Background (eps) = 0.0
        # Exposure times = 1.0 s
        # Image (counts) = 4.0
        # Background (counts) = 0.0

        # Noise (counts) = sqrt(4.0 + 0.0**2) = 2.0
        # Noise (eps) = 2.0 / 1.0

        image = 4.0 * np.ones((4, 2))

        exposure_time = np.ones((4, 2))

        noise_estimate = imaging.estimate_noise_from_image(image, exposure_time, background_noise=0.0)

        assert (noise_estimate == 2.0 * np.ones((4, 2))).all()

    def test__image_all_1s__exposure_time_all_4s__no_background__noise_is_all_2_divided_4_so_halves(self):

        # Image (eps) = 1.0
        # Background (eps) = 0.0
        # Exposure times = 4.0 s
        # Image (counts) = 4.0
        # Background (counts) = 0.0

        # Noise (counts) = sqrt(4.0 + 0.0**2) = 2.0
        # Noise (eps) = 2.0 / 4.0 = 0.5

        image = np.ones((1, 5))

        exposure_time = 4.0 * np.ones((1, 5))

        noise_estimate = imaging.estimate_noise_from_image(image, exposure_time, background_noise=0.0)

        assert (noise_estimate == 0.5 * np.ones((1, 5))).all()

    def test__image_and_exposure_times_range_of_values__no_background__noises_estimates_correct(self):

        # Noise (eps) = sqrt( image (counts) + 0.0 ) / exposure_time

        image = np.array([[5.0, 3.0],
                          [10.0, 20.0]])

        exposure_time = np.array([[1.0, 2.0],
                                 [3.0, 4.0]])

        noise_estimate = imaging.estimate_noise_from_image(image, exposure_time, background_noise=0.0)

        assert (noise_estimate == np.array([[np.sqrt(5.0),     np.sqrt(6.0)/2.0],
                                                   [np.sqrt(30.0)/3.0, np.sqrt(80.0)/4.0]])).all()

    def test__image_and_exposure_times_all_1s__background_is_float_sqrt_3__noise_is_all_2s(self):

        # Image (eps) = 1.0
        # Background (eps) = sqrt(3.0)
        # Exposure times = 1.0 s
        # Image (counts) = 1.0
        # Background (counts) = sqrt(3.0)

        # Noise (counts) = sqrt(1.0 + sqrt(3.0)**2) = sqrt(1.0 + 3.0) = 2.0
        # Noise (eps) = 2.0 / 1.0 = 2.0

        image = np.ones((3, 3))

        exposure_time = np.ones((3, 3))

        noise_estimate = imaging.estimate_noise_from_image(image, exposure_time,
                                                                   background_noise=3.0 ** 0.5)

        assert noise_estimate == pytest.approx(2.0 * np.ones((3, 3)), 1e-2)

    def test__image_and_exposure_times_all_1s__background_is_float_5__noise_all_correct(self):

        # Image (eps) = 1.0
        # Background (eps) = 5.0
        # Exposure times = 1.0 s
        # Image (counts) = 1.0
        # Background (counts) = 5.0

        # Noise (counts) = sqrt(1.0 + 5**2)
        # Noise (eps) = sqrt(1.0 + 5**2) / 1.0

        image = np.ones((2, 3))

        exposure_time = np.ones((2, 3))

        noise_estimate = imaging.estimate_noise_from_image(image, exposure_time,
                                                                   background_noise=5.0)

        assert noise_estimate == \
               pytest.approx(np.array([[np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0)],
                                       [np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0)]]), 1e-2)

    def test__image_all_1s__exposure_times_all_2s__background_is_float_5__noise_all_correct(self):

        # Image (eps) = 1.0
        # Background (eps) = 5.0
        # Exposure times = 2.0 s
        # Image (counts) = 2.0
        # Background (counts) = 10.0

        # Noise (counts) = sqrt(2.0 + 10**2) = sqrt(2.0 + 100.0)
        # Noise (eps) = sqrt(2.0 + 100.0) / 2.0

        image = np.ones((2, 3))

        exposure_time = 2.0*np.ones((2, 3))

        noise_estimate = imaging.estimate_noise_from_image(image, exposure_time,
                                                                   background_noise=5.0)

        assert noise_estimate == \
               pytest.approx(np.array([[np.sqrt(2.0 + 100.0)/2.0, np.sqrt(2.0 + 100.0)/2.0, np.sqrt(2.0 + 100.0)/2.0],
                                       [np.sqrt(2.0 + 100.0)/2.0, np.sqrt(2.0 + 100.0)/2.0, np.sqrt(2.0 + 100.0)/2.0]]),
                             1e-2)

    def test__same_as_above_but_different_image_values_in_each_pixel_and_new_background_values(self):

        # Can use pattern from previous test for values

        image = np.array([[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0]])

        exposure_time = np.ones((3, 2))

        noise_estimate = imaging.estimate_noise_from_image(image, exposure_time,
                                                                   background_noise=12.0)

        assert noise_estimate == pytest.approx(np.array([[np.sqrt(1.0 + 144.0), np.sqrt(2.0 + 144.0)],
                                                         [np.sqrt(3.0 + 144.0), np.sqrt(4.0 + 144.0)],
                                                         [np.sqrt(5.0 + 144.0), np.sqrt(6.0 + 144.0)]]), 1e-2)

    def test__image_and_exposure_times_range_of_values__background_has_value_9___noise_estimates_correct(self):

        # Use same pattern as above, noting that here our background values are now being converts to counts using
        # different exposure time and then being squared.

        image = np.array([[5.0, 3.0],
                         [10.0, 20.0]])

        exposure_time = np.array([[1.0, 2.0],
                                  [3.0, 4.0]])

        noise_estimate = imaging.estimate_noise_from_image(image, exposure_time, background_noise=9.0)

        assert noise_estimate == pytest.approx(np.array([[np.sqrt(5.0 + 81.0),     np.sqrt(6.0 + 18.0**2.0)/2.0],
                                                         [np.sqrt(30.0 + 27.0**2.0)/3.0, np.sqrt(80.0 + 36.0**2.0)/4.0]]),
                                                                                                                1e-2)

    def test__image_and_exposure_times_and_background_are_all_ranges_of_values__noise_estimates_correct(self):

        # Use same pattern as above, noting that we are now also using a variable background signal_to_noise_ratio map.

        image = np.array([[5.0, 3.0],
                         [10.0, 20.0]])

        exposure_time = np.array([[1.0, 2.0],
                                  [3.0, 4.0]])
        
        background_noise = np.array([[5.0, 6.0],
                                     [7.0, 8.0]])

        noise_estimate = imaging.estimate_noise_from_image(image, exposure_time, background_noise)

        assert noise_estimate == pytest.approx(np.array([[np.sqrt(5.0 + 5.0**2.0), np.sqrt(6.0 + 12.0**2.0)/2.0],
                                                         [np.sqrt(30.0 + 21.0**2.0)/3.0, np.sqrt(80.0 + 32.0**2.0)/4.0]]),
                                                                                                                 1e-2)


class TestMask(object):


    class TestConstructor(object):

        def test__simple_array_in(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=1)

            assert (mask.mask == np.array([[True, True, True],
                                      [True, False, True],
                                      [True, True, True]])).all()
            assert mask.pixel_scale == 1.0
            assert mask.central_pixels == (1.0, 1.0)
            assert mask.pixel_dimensions == (3, 3)
            assert mask.arc_second_dimensions == (3.0, 3.0)

        def test__rectangular_array_in(self):

            mask = np.array([[True, True, True, True],
                                  [True, False, False, True],
                                  [True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=1)

            assert (mask.mask == np.array([[True, True, True, True],
                                      [True, False, False, True],
                                      [True, True, True, True]])).all()
            assert mask.pixel_scale == 1.0
            assert mask.central_pixels == (1.0, 1.5)
            assert mask.pixel_dimensions == (3, 4)
            assert mask.arc_second_dimensions == (3.0, 4.0)


    class TestCircular(object):
        
        def test__input_big_mask__mask(self):

            mask = imaging.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1.0, radius_mask=5)

            assert mask.mask.shape == (3, 3)
            assert (mask.mask == np.array([[False, False, False],
                                           [False, False, False],
                                           [False, False, False]])).all()

        def test__odd_x_odd_mask_input_radius_small__mask(self):
            
            mask = imaging.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius_mask=0.5)
            assert (mask.mask == np.array([[True, True, True],
                                      [True, False, True],
                                      [True, True, True]])).all()

        def test__odd_x_odd_mask_input_radius_medium__mask(self):
            mask = imaging.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius_mask=1)

            assert (mask.mask == np.array([[True, False, True],
                                      [False, False,False],
                                      [True, False, True]])).all()

        def test__odd_x_odd_mask_input_radius_large__mask(self):
            mask = imaging.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius_mask=3)

            assert (mask.mask == np.array([[False, False, False],
                                      [False, False, False],
                                      [False, False, False]])).all()

        def test__even_x_odd_mask_input_radius_small__mask(self):
            mask = imaging.Mask.circular(arc_second_dimensions=(4, 3), pixel_scale=1, radius_mask=0.5)

            assert (mask.mask == np.array([[True, True, True],
                                      [True, False, True],
                                      [True, False, True],
                                      [True, True, True]])).all()

        def test__even_x_odd_mask_input_radius_medium__mask(self):
            mask = imaging.Mask.circular(arc_second_dimensions=(4, 3), pixel_scale=1, radius_mask=1.50001)

            assert (mask.mask == np.array([[True, False, True],
                                      [False, False, False],
                                      [False, False, False],
                                      [True, False, True]])).all()

        def test__even_x_odd_mask_input_radius_large__mask(self):
            mask = imaging.Mask.circular(arc_second_dimensions=(4, 3), pixel_scale=1, radius_mask=3)

            assert (mask.mask == np.array([[False, False, False],
                                      [False, False, False],
                                      [False, False, False],
                                      [False, False, False]])).all()

        def test__even_x_even_mask_input_radius_small__mask(self):
            mask = imaging.Mask.circular(arc_second_dimensions=(4, 4), pixel_scale=1, radius_mask=0.72)

            assert (mask.mask == np.array([[True, True, True, True],
                                      [True, False, False, True],
                                      [True, False, False, True],
                                      [True, True, True, True]])).all()

        def test__even_x_even_mask_input_radius_medium__mask(self):
            mask = imaging.Mask.circular(arc_second_dimensions=(4, 4), pixel_scale=1, radius_mask=1.7)

            assert (mask.mask == np.array([[True, False, False, True],
                                      [False, False, False, False],
                                      [False, False, False, False],
                                      [True, False, False, True]])).all()

        def test__even_x_even_mask_input_radius_large__mask(self):
            mask = imaging.Mask.circular(arc_second_dimensions=(4, 4), pixel_scale=1, radius_mask=3)

            assert (mask.mask == np.array([[False, False, False, False],
                                      [False, False, False, False],
                                      [False, False, False, False],
                                      [False, False, False, False]])).all()

        def test__centre_shift__simple_shift_back(self):

            mask = imaging.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius_mask=0.5, centre=(-1, 0))

            assert mask.pixel_dimensions == (3, 3)
            assert (mask.mask == np.array([[True, True, True],
                                           [True, True, True],
                                           [True, False, True]])).all()

        def test__centre_shift__simple_shift_forward(self):
            mask = imaging.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius_mask=0.5, centre=(0, 1))

            assert mask.pixel_dimensions == (3, 3)
            assert (mask.mask == np.array([[True, True, True],
                                           [True, True, False],
                                           [True, True, True]])).all()

        def test__centre_shift__diagonal_shift(self):
            mask = imaging.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius_mask=0.5, centre=(1, 1))

            assert (mask.mask == np.array([[True, True, False],
                                           [True, True, True],
                                           [True, True, True]])).all()


    class TestAnnular(object):

        def test__odd_x_odd_mask_inner_radius_zero_outer_radius_small__mask(self):
            
            mask = imaging.Mask.annular(arc_second_dimensions=(3, 3), pixel_scale=1, inner_radius_mask=0,
                                        outer_radius_mask=0.5)

            assert (mask.mask == np.array([[True, True, True],
                                               [True, False, True],
                                               [True, True, True]])).all()

        def test__odd_x_odd_mask_inner_radius_small_outer_radius_large__mask(self):
            mask = imaging.Mask.annular(arc_second_dimensions=(3, 3), pixel_scale=1, inner_radius_mask=0.5,
                                        outer_radius_mask=3)

            assert (mask.mask == np.array([[False, False, False],
                                                [False, True, False],
                                                [False, False, False]])).all()

        def test__even_x_odd_mask_inner_radius_small_outer_radius_medium__mask(self):
            mask = imaging.Mask.annular(arc_second_dimensions=(4, 3), pixel_scale=1, inner_radius_mask=0.51,
                                        outer_radius_mask=1.51)

            assert (mask.mask == np.array([[True, False, True],
                                                [False, True, False],
                                                [False, True, False],
                                                [True, False, True]])).all()

        def test__even_x_odd_mask_inner_radius_medium_outer_radius_large__mask(self):
            mask = imaging.Mask.annular(arc_second_dimensions=(4, 3), pixel_scale=1, inner_radius_mask=1.51,
                                        outer_radius_mask=3)

            assert (mask.mask == np.array([[False, True, False],
                                                  [True, True, True],
                                                  [True, True, True],
                                                [False, True, False]])).all()

        def test__even_x_even_mask_inner_radius_small_outer_radius_medium__mask(self):
            mask = imaging.Mask.annular(arc_second_dimensions=(4, 4), pixel_scale=1, inner_radius_mask=0.81,
                                        outer_radius_mask=2)

            assert (mask.mask == np.array([[True, False, False, True],
                                                [False, True, True, False],
                                                [False, True, True, False],
                                                [True, False, False, True]])).all()

        def test__even_x_even_mask_inner_radius_medium_outer_radius_large__mask(self):
            mask = imaging.Mask.annular(arc_second_dimensions=(4, 4), pixel_scale=1, inner_radius_mask=1.71,
                                        outer_radius_mask=3)

            assert (mask.mask == np.array([[False, True, True, False],
                                                [True, True, True, True],
                                                [True, True, True, True],
                                                [False, True, True, False]])).all()

        def test__centre_shift__simple_shift_back(self):

            mask = imaging.Mask.annular(arc_second_dimensions=(3, 3), pixel_scale=1, inner_radius_mask=0.5,
                                        outer_radius_mask=3, centre=(-1.0, 0.0))

            assert mask.pixel_dimensions == (3, 3)
            assert (mask.mask == np.array([[False, False, False],
                                           [False, False, False],
                                           [False, True, False]])).all()

        def test__centre_shift__simple_shift_forward(self):

            mask = imaging.Mask.annular(arc_second_dimensions=(3, 3), pixel_scale=1, inner_radius_mask=0.5,
                                        outer_radius_mask=3, centre=(0.0, 1.0))

            assert mask.pixel_dimensions == (3, 3)
            assert (mask.mask == np.array([[False, False, False],
                                           [False, False, True],
                                           [False, False, False]])).all()

        def test__centre_shift__diagonal_shift(self):

            mask = imaging.Mask.annular(arc_second_dimensions=(3, 3), pixel_scale=1, inner_radius_mask=0.5,
                                        outer_radius_mask=3, centre=(1.0, 1.0))

            assert mask.pixel_dimensions == (3, 3)
            assert (mask.mask == np.array([[False, False, True],
                                           [False, False, False],
                                           [False, False, False]])).all()


    class TestUnmasked(object):

        def test__3x3__input__all_are_false(self):

            mask = imaging.Mask.unmasked(arc_second_dimensions=(3, 3), pixel_scale=1)

            assert mask.pixel_dimensions == (3, 3)
            assert (mask.mask == np.array([[False, False, False],
                                      [False, False, False],
                                      [False, False, False]])).all()

        def test__3x2__input__all_are_false(self):

            mask = imaging.Mask.unmasked(arc_second_dimensions=(1.5, 1.0), pixel_scale=0.5)

            assert mask.pixel_dimensions == (3, 2)
            assert (mask.mask == np.array([[False, False],
                                      [False, False],
                                      [False, False]])).all()

        def test__5x5__input__all_are_false(self):

            mask = imaging.Mask.unmasked(arc_second_dimensions=(5, 5), pixel_scale=1)

            assert mask.pixel_dimensions == (5, 5)
            assert (mask.mask == np.array([[False, False, False, False, False],
                                      [False, False, False, False, False],
                                      [False, False, False, False, False],
                                      [False, False, False, False, False],
                                      [False, False, False, False, False]])).all()


    class TestComputeGridCoordsImage(object):

        def test__setup_3x3_image_one_coordinate(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_grid = mask.compute_grid_coords_image()

            assert (image_grid[0] == np.array([0.0, 0.0])).all()

        def test__setup_3x3_image__five_coordinates(self):

            mask = np.array([[True, False, True],
                             [False, False, False],
                             [True, False, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_grid = mask.compute_grid_coords_image()
            
            assert (image_grid[0] == np.array([0.0, 3.0])).all()
            assert (image_grid[1] == np.array([-3.0, 0.0])).all()
            assert (image_grid[2] == np.array([0.0, 0.0])).all()
            assert (image_grid[3] == np.array([3.0, 0.0])).all()
            assert (image_grid[4] == np.array([0.0, -3.0])).all()

        def test__setup_4x4_image__ten_coordinates__new_pixel_scale(self):
            mask = np.array([[True, False, False, True],
                             [False, False, False, True],
                             [True, False, False, True],
                             [False, False, False, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=1.0)

            image_grid = mask.compute_grid_coords_image()

            assert (image_grid[0] == np.array([-0.5, 1.5])).all()
            assert (image_grid[1] == np.array([0.5, 1.5])).all()
            assert (image_grid[2] == np.array([-1.5, 0.5])).all()
            assert (image_grid[3] == np.array([-0.5, 0.5])).all()
            assert (image_grid[4] == np.array([0.5, 0.5])).all()
            assert (image_grid[5] == np.array([-0.5, -0.5])).all()
            assert (image_grid[6] == np.array([0.5, -0.5])).all()
            assert (image_grid[7] == np.array([-1.5, -1.5])).all()
            assert (image_grid[8] == np.array([-0.5, -1.5])).all()
            assert (image_grid[9] == np.array([0.5, -1.5])).all()

        def test__setup_3x4_image__six_coordinates(self):
            mask = np.array([[True, False, True, True],
                             [False, False, False, True],
                             [True, False, True, False]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_grid = mask.compute_grid_coords_image()

            assert (image_grid[0] == np.array([-1.5, 3.0])).all()
            assert (image_grid[1] == np.array([-4.5, 0.0])).all()
            assert (image_grid[2] == np.array([-1.5, 0.0])).all()
            assert (image_grid[3] == np.array([1.5, 0.0])).all()
            assert (image_grid[4] == np.array([-1.5, -3.0])).all()
            assert (image_grid[5] == np.array([4.5, -3.0])).all()


    class TestComputeGridCoordsImageSub(object):

        def test__3x3_mask_with_one_pixel__2x2_sub_grid__coordinates(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_sub_grid = mask.compute_grid_coords_image_sub(sub_grid_size=2)

            assert (image_sub_grid == np.array
            ([[[-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]])).all()

            assert (image_sub_grid[0, 0] == np.array([-0.5, 0.5])).all()
            assert (image_sub_grid[0, 1] == np.array([0.5, 0.5])).all()
            assert (image_sub_grid[0, 2] == np.array([-0.5, -0.5])).all()
            assert (image_sub_grid[0, 3] == np.array([0.5, -0.5])).all()

        def test__3x3_mask_with_row_of_pixels__2x2_sub_grid__coordinates(self):
            mask = np.array([[True, True, True],
                             [False, False, False],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_sub_grid = mask.compute_grid_coords_image_sub(sub_grid_size=2)

            assert (image_sub_grid == np.array([[[-3.5, 0.5], [-2.5, 0.5], [-3.5, -0.5], [-2.5, -0.5]],
                                                    [[-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]],
                                                    [[2.5, 0.5], [3.5, 0.5], [2.5, -0.5], [3.5, -0.5]]])).all()

            assert (image_sub_grid[0, 0] == np.array([-3.5, 0.5])).all()
            assert (image_sub_grid[0, 1] == np.array([-2.5, 0.5])).all()
            assert (image_sub_grid[0, 2] == np.array([-3.5, -0.5])).all()
            assert (image_sub_grid[0, 3] == np.array([-2.5, -0.5])).all()

            assert (image_sub_grid[1, 0] == np.array([-0.5, 0.5])).all()
            assert (image_sub_grid[1, 1] == np.array([0.5, 0.5])).all()
            assert (image_sub_grid[1, 2] == np.array([-0.5, -0.5])).all()
            assert (image_sub_grid[1, 3] == np.array([0.5, -0.5])).all()

            assert (image_sub_grid[2, 0] == np.array([2.5, 0.5])).all()
            assert (image_sub_grid[2, 1] == np.array([3.5, 0.5])).all()
            assert (image_sub_grid[2, 2] == np.array([2.5, -0.5])).all()
            assert (image_sub_grid[2, 3] == np.array([3.5, -0.5])).all()

        def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__coordinates(self):
            mask = np.array([[True, True, False],
                             [False, False, False],
                             [True, True, False]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_sub_grid = mask.compute_grid_coords_image_sub(sub_grid_size=2)

            assert (image_sub_grid == np.array([[[2.5, 3.5], [3.5, 3.5], [2.5, 2.5], [3.5, 2.5]],
                                                    [[-3.5, 0.5], [-2.5, 0.5], [-3.5, -0.5], [-2.5, -0.5]],
                                                    [[-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]],
                                                    [[2.5, 0.5], [3.5, 0.5], [2.5, -0.5], [3.5, -0.5]],
                                                    [[2.5, -2.5], [3.5, -2.5], [2.5, -3.5], [3.5, -3.5]]])).all()

            assert (image_sub_grid[0, 0] == np.array([2.5, 3.5])).all()
            assert (image_sub_grid[0, 1] == np.array([3.5, 3.5])).all()
            assert (image_sub_grid[0, 2] == np.array([2.5, 2.5])).all()
            assert (image_sub_grid[0, 3] == np.array([3.5, 2.5])).all()

            assert (image_sub_grid[1, 0] == np.array([-3.5, 0.5])).all()
            assert (image_sub_grid[1, 1] == np.array([-2.5, 0.5])).all()
            assert (image_sub_grid[1, 2] == np.array([-3.5, -0.5])).all()
            assert (image_sub_grid[1, 3] == np.array([-2.5, -0.5])).all()

            assert (image_sub_grid[2, 0] == np.array([-0.5, 0.5])).all()
            assert (image_sub_grid[2, 1] == np.array([0.5, 0.5])).all()
            assert (image_sub_grid[2, 2] == np.array([-0.5, -0.5])).all()
            assert (image_sub_grid[2, 3] == np.array([0.5, -0.5])).all()

            assert (image_sub_grid[3, 0] == np.array([2.5, 0.5])).all()
            assert (image_sub_grid[3, 1] == np.array([3.5, 0.5])).all()
            assert (image_sub_grid[3, 2] == np.array([2.5, -0.5])).all()
            assert (image_sub_grid[3, 3] == np.array([3.5, -0.5])).all()

            assert (image_sub_grid[4, 0] == np.array([2.5, -2.5])).all()
            assert (image_sub_grid[4, 1] == np.array([3.5, -2.5])).all()
            assert (image_sub_grid[4, 2] == np.array([2.5, -3.5])).all()
            assert (image_sub_grid[4, 3] == np.array([3.5, -3.5])).all()

        def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__different_pixel_scale(self):
            mask = np.array([[True, True, False],
                             [False, False, False],
                             [True, True, False]])

            mask = imaging.Mask(mask=mask, pixel_scale=0.3)

            image_sub_grid = mask.compute_grid_coords_image_sub(sub_grid_size=2)

            image_sub_grid = np.round(image_sub_grid, decimals=2)

            assert (image_sub_grid == np.array([[[0.25, 0.35], [0.35, 0.35], [0.25, 0.25], [0.35, 0.25]],
                                                    [[-0.35, 0.05], [-0.25, 0.05], [-0.35, -0.05], [-0.25, -0.05]],
                                                    [[-0.05, 0.05], [0.05, 0.05], [-0.05, -0.05], [0.05, -0.05]],
                                                    [[0.25, 0.05], [0.35, 0.05], [0.25, -0.05], [0.35, -0.05]],
                                                    [[0.25, -0.25], [0.35, -0.25], [0.25, -0.35],
                                                     [0.35, -0.35]]])).all()

            assert (image_sub_grid[0, 0] == np.array([0.25, 0.35])).all()
            assert (image_sub_grid[0, 1] == np.array([0.35, 0.35])).all()
            assert (image_sub_grid[0, 2] == np.array([0.25, 0.25])).all()
            assert (image_sub_grid[0, 3] == np.array([0.35, 0.25])).all()

            assert (image_sub_grid[1, 0] == np.array([-0.35, 0.05])).all()
            assert (image_sub_grid[1, 1] == np.array([-0.25, 0.05])).all()
            assert (image_sub_grid[1, 2] == np.array([-0.35, -0.05])).all()
            assert (image_sub_grid[1, 3] == np.array([-0.25, -0.05])).all()

            assert (image_sub_grid[2, 0] == np.array([-0.05, 0.05])).all()
            assert (image_sub_grid[2, 1] == np.array([0.05, 0.05])).all()
            assert (image_sub_grid[2, 2] == np.array([-0.05, -0.05])).all()
            assert (image_sub_grid[2, 3] == np.array([0.05, -0.05])).all()

            assert (image_sub_grid[3, 0] == np.array([0.25, 0.05])).all()
            assert (image_sub_grid[3, 1] == np.array([0.35, 0.05])).all()
            assert (image_sub_grid[3, 2] == np.array([0.25, -0.05])).all()
            assert (image_sub_grid[3, 3] == np.array([0.35, -0.05])).all()

            assert (image_sub_grid[4, 0] == np.array([0.25, -0.25])).all()
            assert (image_sub_grid[4, 1] == np.array([0.35, -0.25])).all()
            assert (image_sub_grid[4, 2] == np.array([0.25, -0.35])).all()
            assert (image_sub_grid[4, 3] == np.array([0.35, -0.35])).all()

        def test__3x3_mask_with_one_pixel__3x3_sub_grid__coordinates(self):
            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_sub_grid = mask.compute_grid_coords_image_sub(sub_grid_size=3)

            assert (image_sub_grid == np.array([[[-0.75, 0.75], [0.0, 0.75], [0.75, 0.75],
                                                     [-0.75, 0.0], [0.0, 0.0], [0.75, 0.0],
                                                     [-0.75, -0.75], [0.0, -0.75], [0.75, -0.75]]])).all()

            assert (image_sub_grid[0, 0] == np.array([-0.75, 0.75])).all()
            assert (image_sub_grid[0, 1] == np.array([0.0, 0.75])).all()
            assert (image_sub_grid[0, 2] == np.array([0.75, 0.75])).all()
            assert (image_sub_grid[0, 3] == np.array([-0.75, 0.0])).all()
            assert (image_sub_grid[0, 4] == np.array([0.0, 0.0])).all()
            assert (image_sub_grid[0, 5] == np.array([0.75, 0.0])).all()
            assert (image_sub_grid[0, 6] == np.array([-0.75, -0.75])).all()
            assert (image_sub_grid[0, 7] == np.array([0.0, -0.75])).all()
            assert (image_sub_grid[0, 8] == np.array([0.75, -0.75])).all()

        def test__3x3_mask_with_one_row__3x3_sub_grid__coordinates(self):
            mask = np.array([[True, True, False],
                             [True, False, True],
                             [True, True, False]])

            mask = imaging.Mask(mask=mask, pixel_scale=2.0)

            image_sub_grid = mask.compute_grid_coords_image_sub(sub_grid_size=3)

            assert (image_sub_grid[0, 0] == np.array([1.5, 2.5])).all()
            assert (image_sub_grid[0, 1] == np.array([2.0, 2.5])).all()
            assert (image_sub_grid[0, 2] == np.array([2.5, 2.5])).all()
            assert (image_sub_grid[0, 3] == np.array([1.5, 2.0])).all()
            assert (image_sub_grid[0, 4] == np.array([2.0, 2.0])).all()
            assert (image_sub_grid[0, 5] == np.array([2.5, 2.0])).all()
            assert (image_sub_grid[0, 6] == np.array([1.5, 1.5])).all()
            assert (image_sub_grid[0, 7] == np.array([2.0, 1.5])).all()
            assert (image_sub_grid[0, 8] == np.array([2.5, 1.5])).all()

            assert (image_sub_grid[1, 0] == np.array([-0.5, 0.5])).all()
            assert (image_sub_grid[1, 1] == np.array([0.0, 0.5])).all()
            assert (image_sub_grid[1, 2] == np.array([0.5, 0.5])).all()
            assert (image_sub_grid[1, 3] == np.array([-0.5, 0.0])).all()
            assert (image_sub_grid[1, 4] == np.array([0.0, 0.0])).all()
            assert (image_sub_grid[1, 5] == np.array([0.5, 0.0])).all()
            assert (image_sub_grid[1, 6] == np.array([-0.5, -0.5])).all()
            assert (image_sub_grid[1, 7] == np.array([0.0, -0.5])).all()
            assert (image_sub_grid[1, 8] == np.array([0.5, -0.5])).all()

            assert (image_sub_grid[2, 0] == np.array([1.5, -1.5])).all()
            assert (image_sub_grid[2, 1] == np.array([2.0, -1.5])).all()
            assert (image_sub_grid[2, 2] == np.array([2.5, -1.5])).all()
            assert (image_sub_grid[2, 3] == np.array([1.5, -2.0])).all()
            assert (image_sub_grid[2, 4] == np.array([2.0, -2.0])).all()
            assert (image_sub_grid[2, 5] == np.array([2.5, -2.0])).all()
            assert (image_sub_grid[2, 6] == np.array([1.5, -2.5])).all()
            assert (image_sub_grid[2, 7] == np.array([2.0, -2.5])).all()
            assert (image_sub_grid[2, 8] == np.array([2.5, -2.5])).all()

        def test__4x4_mask_with_one_pixel__4x4_sub_grid__coordinates(self):
            mask = np.array([[True, True, True, True],
                             [True, False, False, True],
                             [True, False, False, True],
                             [True, True, True, False]])

            mask = imaging.Mask(mask=mask, pixel_scale=2.0)

            image_sub_grid = mask.compute_grid_coords_image_sub(sub_grid_size=4)

            image_sub_grid = np.round(image_sub_grid, decimals=1)

            assert (image_sub_grid[0, 0] == np.array([-1.6, 1.6])).all()
            assert (image_sub_grid[0, 1] == np.array([-1.2, 1.6])).all()
            assert (image_sub_grid[0, 2] == np.array([-0.8, 1.6])).all()
            assert (image_sub_grid[0, 3] == np.array([-0.4, 1.6])).all()
            assert (image_sub_grid[0, 4] == np.array([-1.6, 1.2])).all()
            assert (image_sub_grid[0, 5] == np.array([-1.2, 1.2])).all()
            assert (image_sub_grid[0, 6] == np.array([-0.8, 1.2])).all()
            assert (image_sub_grid[0, 7] == np.array([-0.4, 1.2])).all()
            assert (image_sub_grid[0, 8] == np.array([-1.6, 0.8])).all()
            assert (image_sub_grid[0, 9] == np.array([-1.2, 0.8])).all()
            assert (image_sub_grid[0, 10] == np.array([-0.8, 0.8])).all()
            assert (image_sub_grid[0, 11] == np.array([-0.4, 0.8])).all()
            assert (image_sub_grid[0, 12] == np.array([-1.6, 0.4])).all()
            assert (image_sub_grid[0, 13] == np.array([-1.2, 0.4])).all()
            assert (image_sub_grid[0, 14] == np.array([-0.8, 0.4])).all()
            assert (image_sub_grid[0, 15] == np.array([-0.4, 0.4])).all()

            assert (image_sub_grid[1, 0] == np.array([0.4, 1.6])).all()
            assert (image_sub_grid[1, 1] == np.array([0.8, 1.6])).all()
            assert (image_sub_grid[1, 2] == np.array([1.2, 1.6])).all()
            assert (image_sub_grid[1, 3] == np.array([1.6, 1.6])).all()
            assert (image_sub_grid[1, 4] == np.array([0.4, 1.2])).all()
            assert (image_sub_grid[1, 5] == np.array([0.8, 1.2])).all()
            assert (image_sub_grid[1, 6] == np.array([1.2, 1.2])).all()
            assert (image_sub_grid[1, 7] == np.array([1.6, 1.2])).all()
            assert (image_sub_grid[1, 8] == np.array([0.4, 0.8])).all()
            assert (image_sub_grid[1, 9] == np.array([0.8, 0.8])).all()
            assert (image_sub_grid[1, 10] == np.array([1.2, 0.8])).all()
            assert (image_sub_grid[1, 11] == np.array([1.6, 0.8])).all()
            assert (image_sub_grid[1, 12] == np.array([0.4, 0.4])).all()
            assert (image_sub_grid[1, 13] == np.array([0.8, 0.4])).all()
            assert (image_sub_grid[1, 14] == np.array([1.2, 0.4])).all()
            assert (image_sub_grid[1, 15] == np.array([1.6, 0.4])).all()

            assert (image_sub_grid[2, 0] == np.array([-1.6, -0.4])).all()
            assert (image_sub_grid[2, 1] == np.array([-1.2, -0.4])).all()
            assert (image_sub_grid[2, 2] == np.array([-0.8, -0.4])).all()
            assert (image_sub_grid[2, 3] == np.array([-0.4, -0.4])).all()
            assert (image_sub_grid[2, 4] == np.array([-1.6, -0.8])).all()
            assert (image_sub_grid[2, 5] == np.array([-1.2, -0.8])).all()
            assert (image_sub_grid[2, 6] == np.array([-0.8, -0.8])).all()
            assert (image_sub_grid[2, 7] == np.array([-0.4, -0.8])).all()
            assert (image_sub_grid[2, 8] == np.array([-1.6, -1.2])).all()
            assert (image_sub_grid[2, 9] == np.array([-1.2, -1.2])).all()
            assert (image_sub_grid[2, 10] == np.array([-0.8, -1.2])).all()
            assert (image_sub_grid[2, 11] == np.array([-0.4, -1.2])).all()
            assert (image_sub_grid[2, 12] == np.array([-1.6, -1.6])).all()
            assert (image_sub_grid[2, 13] == np.array([-1.2, -1.6])).all()
            assert (image_sub_grid[2, 14] == np.array([-0.8, -1.6])).all()
            assert (image_sub_grid[2, 15] == np.array([-0.4, -1.6])).all()

            assert (image_sub_grid[3, 0] == np.array([0.4, -0.4])).all()
            assert (image_sub_grid[3, 1] == np.array([0.8, -0.4])).all()
            assert (image_sub_grid[3, 2] == np.array([1.2, -0.4])).all()
            assert (image_sub_grid[3, 3] == np.array([1.6, -0.4])).all()
            assert (image_sub_grid[3, 4] == np.array([0.4, -0.8])).all()
            assert (image_sub_grid[3, 5] == np.array([0.8, -0.8])).all()
            assert (image_sub_grid[3, 6] == np.array([1.2, -0.8])).all()
            assert (image_sub_grid[3, 7] == np.array([1.6, -0.8])).all()
            assert (image_sub_grid[3, 8] == np.array([0.4, -1.2])).all()
            assert (image_sub_grid[3, 9] == np.array([0.8, -1.2])).all()
            assert (image_sub_grid[3, 10] == np.array([1.2, -1.2])).all()
            assert (image_sub_grid[3, 11] == np.array([1.6, -1.2])).all()
            assert (image_sub_grid[3, 12] == np.array([0.4, -1.6])).all()
            assert (image_sub_grid[3, 13] == np.array([0.8, -1.6])).all()
            assert (image_sub_grid[3, 14] == np.array([1.2, -1.6])).all()
            assert (image_sub_grid[3, 15] == np.array([1.6, -1.6])).all()

            assert (image_sub_grid[4, 0] == np.array([2.4, -2.4])).all()
            assert (image_sub_grid[4, 1] == np.array([2.8, -2.4])).all()
            assert (image_sub_grid[4, 2] == np.array([3.2, -2.4])).all()
            assert (image_sub_grid[4, 3] == np.array([3.6, -2.4])).all()
            assert (image_sub_grid[4, 4] == np.array([2.4, -2.8])).all()
            assert (image_sub_grid[4, 5] == np.array([2.8, -2.8])).all()
            assert (image_sub_grid[4, 6] == np.array([3.2, -2.8])).all()
            assert (image_sub_grid[4, 7] == np.array([3.6, -2.8])).all()
            assert (image_sub_grid[4, 8] == np.array([2.4, -3.2])).all()
            assert (image_sub_grid[4, 9] == np.array([2.8, -3.2])).all()
            assert (image_sub_grid[4, 10] == np.array([3.2, -3.2])).all()
            assert (image_sub_grid[4, 11] == np.array([3.6, -3.2])).all()
            assert (image_sub_grid[4, 12] == np.array([2.4, -3.6])).all()
            assert (image_sub_grid[4, 13] == np.array([2.8, -3.6])).all()
            assert (image_sub_grid[4, 14] == np.array([3.2, -3.6])).all()
            assert (image_sub_grid[4, 15] == np.array([3.6, -3.6])).all()

        def test__4x3_mask_with_one_pixel__2x2_sub_grid__coordinates(self):
            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, False, False],
                             [False, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_sub_grid = mask.compute_grid_coords_image_sub(sub_grid_size=2)

            assert (image_sub_grid[0, 0] == np.array([-0.5, 2.0])).all()
            assert (image_sub_grid[0, 1] == np.array([0.5, 2.0])).all()
            assert (image_sub_grid[0, 2] == np.array([-0.5, 1.0])).all()
            assert (image_sub_grid[0, 3] == np.array([0.5, 1.0])).all()

            assert (image_sub_grid[1, 0] == np.array([-0.5, -1.0])).all()
            assert (image_sub_grid[1, 1] == np.array([0.5, -1.0])).all()
            assert (image_sub_grid[1, 2] == np.array([-0.5, -2.0])).all()
            assert (image_sub_grid[1, 3] == np.array([0.5, -2.0])).all()

            assert (image_sub_grid[2, 0] == np.array([2.5, -1.0])).all()
            assert (image_sub_grid[2, 1] == np.array([3.5, -1.0])).all()
            assert (image_sub_grid[2, 2] == np.array([2.5, -2.0])).all()
            assert (image_sub_grid[2, 3] == np.array([3.5, -2.0])).all()

            assert (image_sub_grid[3, 0] == np.array([-3.5, -4.0])).all()
            assert (image_sub_grid[3, 1] == np.array([-2.5, -4.0])).all()
            assert (image_sub_grid[3, 2] == np.array([-3.5, -5.0])).all()
            assert (image_sub_grid[3, 3] == np.array([-2.5, -5.0])).all()

        def test__3x4_mask_with_one_pixel__2x2_sub_grid__coordinates(self):
            
            mask = np.array([[True, True, True, False],
                             [True, False, False, True],
                             [False, True, False, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_sub_grid = mask.compute_grid_coords_image_sub(sub_grid_size=2)

            assert (image_sub_grid[0, 0] == np.array([4.0, 3.5])).all()
            assert (image_sub_grid[0, 1] == np.array([5.0, 3.5])).all()
            assert (image_sub_grid[0, 2] == np.array([4.0, 2.5])).all()
            assert (image_sub_grid[0, 3] == np.array([5.0, 2.5])).all()

            assert (image_sub_grid[1, 0] == np.array([-2.0, 0.5])).all()
            assert (image_sub_grid[1, 1] == np.array([-1.0, 0.5])).all()
            assert (image_sub_grid[1, 2] == np.array([-2.0, -0.5])).all()
            assert (image_sub_grid[1, 3] == np.array([-1.0, -0.5])).all()

            assert (image_sub_grid[2, 0] == np.array([1.0, 0.5])).all()
            assert (image_sub_grid[2, 1] == np.array([2.0, 0.5])).all()
            assert (image_sub_grid[2, 2] == np.array([1.0, -0.5])).all()
            assert (image_sub_grid[2, 3] == np.array([2.0, -0.5])).all()

            assert (image_sub_grid[3, 0] == np.array([-5.0, -2.5])).all()
            assert (image_sub_grid[3, 1] == np.array([-4.0, -2.5])).all()
            assert (image_sub_grid[3, 2] == np.array([-5.0, -3.5])).all()
            assert (image_sub_grid[3, 3] == np.array([-4.0, -3.5])).all()

            assert (image_sub_grid[4, 0] == np.array([1.0, -2.5])).all()
            assert (image_sub_grid[4, 1] == np.array([2.0, -2.5])).all()
            assert (image_sub_grid[4, 2] == np.array([1.0, -3.5])).all()
            assert (image_sub_grid[4, 3] == np.array([2.0, -3.5])).all()


    class TestComputeGridCoordsBlurring(object):
        
        def test__3x3_blurring_mask_correct_coordinates(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_grid = mask.compute_grid_coords_blurring(psf_size=(3, 3))

            assert (blurring_grid[0] == np.array([-3.0, 3.0])).all()
            assert (blurring_grid[1] == np.array([0.0, 3.0])).all()
            assert (blurring_grid[2] == np.array([3.0, 3.0])).all()
            assert (blurring_grid[3] == np.array([-3.0, 0.0])).all()
            assert (blurring_grid[4] == np.array([3.0, 0.0])).all()
            assert (blurring_grid[5] == np.array([-3.0, -3.0])).all()
            assert (blurring_grid[6] == np.array([0.0, -3.0])).all()
            assert (blurring_grid[7] == np.array([3.0, -3.0])).all()

        def test__3x5_blurring_mask_correct_coordinates(self):

            mask = np.array([[True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, False, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            # Blurring mask

            # [[True, True, True, True, True, True, True],
            # [True, True, False, False, False, True, True],
            # [True, True, False, False, False, True, True],
            # [True, True, False, True, False, True, True],
            # [True, True, False, False, False, True, True],
            # [True, True, False, False, False, True, True],
            # [True, True, True, True, True, True, True]])

            blurring_grid = mask.compute_grid_coords_blurring(psf_size=(3, 5))

            assert (blurring_grid[0] == np.array([-3.0, 6.0])).all()
            assert (blurring_grid[1] == np.array([0.0, 6.0])).all()
            assert (blurring_grid[2] == np.array([3.0, 6.0])).all()
            assert (blurring_grid[3] == np.array([-3.0, 3.0])).all()
            assert (blurring_grid[4] == np.array([0.0, 3.0])).all()
            assert (blurring_grid[5] == np.array([3.0, 3.0])).all()
            assert (blurring_grid[6] == np.array([-3.0, 0.0])).all()
            assert (blurring_grid[7] == np.array([3.0, 0.0])).all()
            assert (blurring_grid[8] == np.array([-3.0, -3.0])).all()
            assert (blurring_grid[9] == np.array([0.0, -3.0])).all()
            assert (blurring_grid[10] == np.array([3.0, -3.0])).all()
            assert (blurring_grid[11] == np.array([-3.0, -6.0])).all()
            assert (blurring_grid[12] == np.array([0.0, -6.0])).all()
            assert (blurring_grid[13] == np.array([3.0, -6.0])).all()

        def test__5x3_blurring_mask_correct_coordinates(self):

            mask = np.array([[True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, False, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            # Blurring mask

            # [[True, True, True, True, True, True, True],
            #  [True, True, True, True, True, True, True],
            #  [True, False, False, False, False, False, True],
            #  [True, False, False, True, False, False, True],
            #  [True, False, False, False, False, False, True],
            #  [True, True, True, True, True, True, True],
            #  [True, True, True, True, True, True, True]]

            blurring_grid = mask.compute_grid_coords_blurring(psf_size=(5, 3))

            assert (blurring_grid[0] == np.array([-6.0, 3.0])).all()
            assert (blurring_grid[1] == np.array([-3.0, 3.0])).all()
            assert (blurring_grid[2] == np.array([0.0, 3.0])).all()
            assert (blurring_grid[3] == np.array([3.0, 3.0])).all()
            assert (blurring_grid[4] == np.array([6.0, 3.0])).all()
            assert (blurring_grid[5] == np.array([-6.0, 0.0])).all()
            assert (blurring_grid[6] == np.array([-3.0, 0.0])).all()
            assert (blurring_grid[7] == np.array([3.0, 0.0])).all()
            assert (blurring_grid[8] == np.array([6.0, 0.0])).all()
            assert (blurring_grid[9] == np.array([-6.0, -3.0])).all()
            assert (blurring_grid[10] == np.array([-3.0, -3.0])).all()
            assert (blurring_grid[11] == np.array([0.0, -3.0])).all()
            assert (blurring_grid[12] == np.array([3.0, -3.0])).all()
            assert (blurring_grid[13] == np.array([6.0, -3.0])).all()


    class TestComputeGridData(object):

        def test__setup_3x3_data(self):

            data = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            grid_data = mask.compute_grid_data(data)

            assert (grid_data[0] == np.array([5])).all()

        def test__setup_3x3_data__five_now_in_mask(self):

            data = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

            mask = np.array([[True,  False, True],
                             [False, False, False],
                             [True,  False, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            grid_data = mask.compute_grid_data(data)

            assert (grid_data[0] == np.array([2])).all()
            assert (grid_data[1] == np.array([4])).all()
            assert (grid_data[2] == np.array([5])).all()
            assert (grid_data[3] == np.array([6])).all()
            assert (grid_data[4] == np.array([8])).all()

        def test__setup_3x4_data(self):

            data = np.array([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12]])

            mask = np.array([[True, False, True, True],
                             [False, False, False, True],
                             [True, False, True, False]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            grid_data = mask.compute_grid_data(data)

            assert (grid_data[0] == np.array([2])).all()
            assert (grid_data[1] == np.array([5])).all()
            assert (grid_data[2] == np.array([6])).all()
            assert (grid_data[3] == np.array([7])).all()
            assert (grid_data[4] == np.array([10])).all()

        def test__setup_4x3_data__five_now_in_mask(self):

            data = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9],
                             [10, 11, 12]])

            mask = np.array([[True,  False, True],
                             [False, False, False],
                             [True,  False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            grid_data = mask.compute_grid_data(data)

            assert (grid_data[0] == np.array([2])).all()
            assert (grid_data[1] == np.array([4])).all()
            assert (grid_data[2] == np.array([5])).all()
            assert (grid_data[3] == np.array([6])).all()
            assert (grid_data[4] == np.array([8])).all()


    class TestComputeBlurringMask(object):

        def test__size__3x3_small_mask(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(3, 3))

            assert (blurring_mask.mask == np.array([[False, False, False],
                                                     [False, True, False],
                                                     [False, False, False]])).all()

        def test__size__3x3__large_mask(self):

            mask = np.array([[True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, False, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(3, 3))

            assert (blurring_mask.mask == np.array([[True, True, True, True, True, True, True],
                                                   [True, True, True, True, True, True, True],
                                                   [True, True, False, False, False, True, True],
                                                   [True, True, False, True, False, True, True],
                                                   [True, True, False, False, False, True, True],
                                                   [True, True, True, True, True, True, True],
                                                   [True, True, True, True, True, True, True]])).all()

        def test__size__5x5__large_mask(self):

            mask = np.array([[True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, False, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(5, 5))

            assert (blurring_mask.mask == np.array([[True, True, True, True, True, True, True],
                                               [True, False, False, False, False, False, True],
                                               [True, False, False, False, False, False, True],
                                               [True, False, False, True, False, False, True],
                                               [True, False, False, False, False, False, True],
                                               [True, False, False, False, False, False, True],
                                               [True, True, True, True, True, True, True]])).all()

        def test__size__5x3__large_mask(self):

            mask = np.array([[True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, False, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(5, 3))

            assert (blurring_mask.mask == np.array([[True, True, True, True, True, True, True],
                                               [True, True, True, True, True, True, True],
                                               [True, False, False, False, False, False, True],
                                               [True, False, False, True, False, False, True],
                                               [True, False, False, False, False, False, True],
                                               [True, True, True, True, True, True, True],
                                               [True, True, True, True, True, True, True]])).all()

        def test__size__3x5__large_mask(self):
            
            mask = np.array([[True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, False, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(3, 5))

            assert (blurring_mask.mask == np.array([[True, True, True, True, True, True, True],
                                               [True, True, False, False, False, True, True],
                                               [True, True, False, False, False, True, True],
                                               [True, True, False, True, False, True, True],
                                               [True, True, False, False, False, True, True],
                                               [True, True, False, False, False, True, True],
                                               [True, True, True, True, True, True, True]])).all()

        def test__size__3x3__multiple_points(self):
            mask = np.array([[True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True],
                             [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(3, 3))

            assert (blurring_mask.mask == np.array([[False, False, False, True, False, False, False],
                                               [False, True, False, True, False, True, False],
                                               [False, False, False, True, False, False, False],
                                               [True, True, True, True, True, True, True],
                                               [False, False, False, True, False, False, False],
                                               [False, True, False, True, False, True, False],
                                               [False, False, False, True, False, False, False]])).all()

        def test__size__5x5__multiple_points(self):
            mask = np.array([[True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(5, 5))

            assert (blurring_mask.mask == np.array([[False, False, False, False, False, False, False, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, True, False, False, False, True, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, True, False, False, False, True, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, False, False, False, False, False, False, False]])).all()

        def test__size__5x3__multiple_points(self):
            mask = np.array([[True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(5, 3))

            assert (blurring_mask.mask == np.array([[True, True, True, True, True, True, True, True, True],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, True, False, False, False, True, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [True, True, True, True, True, True, True, True, True],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, True, False, False, False, True, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [True, True, True, True, True, True, True, True, True]])).all()

        def test__size__3x5__multiple_points(self):
            mask = np.array([[True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(3, 5))

            assert (blurring_mask.mask == np.array([[True, False, False, False, True, False, False, False, True],
                                               [True, False, False, False, True, False, False, False, True],
                                               [True, False, True, False, True, False, True, False, True],
                                               [True, False, False, False, True, False, False, False, True],
                                               [True, False, False, False, True, False, False, False, True],
                                               [True, False, False, False, True, False, False, False, True],
                                               [True, False, True, False, True, False, True, False, True],
                                               [True, False, False, False, True, False, False, False, True],
                                               [True, False, False, False, True, False, False, False, True]])).all()

        def test__size__3x3__even_sized_image(self):
            
            mask = np.array([[True, True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(3, 3))

            assert (blurring_mask.mask == np.array([[False, False, False, True, False, False, False, True],
                                               [False, True, False, True, False, True, False, True],
                                               [False, False, False, True, False, False, False, True],
                                               [True, True, True, True, True, True, True, True],
                                               [False, False, False, True, False, False, False, True],
                                               [False, True, False, True, False, True, False, True],
                                               [False, False, False, True, False, False, False, True],
                                               [True, True, True, True, True, True, True, True]])).all()

        def test__size__5x5__even_sized_image(self):
            
            mask = np.array([[True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(5, 5))

            assert (blurring_mask.mask == np.array([[True, True, True, True, True, True, True, True],
                                               [True, True, True, True, True, True, True, True],
                                               [True, True, True, True, True, True, True, True],
                                               [True, True, True, False, False, False, False, False],
                                               [True, True, True, False, False, False, False, False],
                                               [True, True, True, False, False, True, False, False],
                                               [True, True, True, False, False, False, False, False],
                                               [True, True, True, False, False, False, False, False]])).all()

        def test__size__3x3__rectangular_8x9_image(self):
            
            mask = np.array([[True, True, True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(3, 3))

            assert (blurring_mask.mask == np.array([[False, False, False, True, False, False, False, True, True],
                                               [False, True, False, True, False, True, False, True, True],
                                               [False, False, False, True, False, False, False, True, True],
                                               [True, True, True, True, True, True, True, True, True],
                                               [False, False, False, True, False, False, False, True, True],
                                               [False, True, False, True, False, True, False, True, True],
                                               [False, False, False, True, False, False, False, True, True],
                                               [True, True, True, True, True, True, True, True, True]])).all()

        def test__size__3x3__rectangular_9x8_image(self):
            
            mask = np.array([[True, True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_mask = mask.compute_blurring_mask(psf_size=(3, 3))

            assert (blurring_mask.mask == np.array([[False, False, False, True, False, False, False, True],
                                               [False, True, False, True, False, True, False, True],
                                               [False, False, False, True, False, False, False, True],
                                               [True, True, True, True, True, True, True, True],
                                               [False, False, False, True, False, False, False, True],
                                               [False, True, False, True, False, True, False, True],
                                               [False, False, False, True, False, False, False, True],
                                               [True, True, True, True, True, True, True, True],
                                               [True, True, True, True, True, True, True, True]])).all()

        def test__size__5x5__multiple_points__mask_extends_beyond_border_so_raises_mask_exception(self):
            
            mask = np.array([[True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True],
                             [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            with pytest.raises(imaging.MaskException):
                blurring_mask = mask.compute_blurring_mask(psf_size=(5, 5))
                
                
    class TestComputeBorderPixels(object):

        def test__7x7_mask_one_central_pixel__is_entire_border(self):
            
            mask = np.array([[True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, False, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            border_pixels = mask.compute_grid_border()

            assert (border_pixels == np.array([0])).all()

        def test__7x7_mask_nine_central_pixels__is_border(self):
            mask = np.array([[True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            border_pixels = mask.compute_grid_border()

            assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

        def test__7x7_mask_rectangle_of_fifteen_central_pixels__is_border(self):
            mask = np.array([[True, True, True, True, True, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            border_pixels = mask.compute_grid_border()

            assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14])).all()

        def test__8x7_mask_add_edge_pixels__also_in_border(self):
            mask = np.array([[True, True, True, True, True, True, True],
                                   [True, True, True, False, True, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, False, False, False, False, False, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            border_pixels = mask.compute_grid_border()

            assert (border_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17])).all()

        def test__8x7_mask_big_square(self):
            mask = np.array([[True, True, True, True, True, True, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, True],
                                   [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            border_pixels = mask.compute_grid_border()

            assert (border_pixels == np.array
            ([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 26, 27, 28, 29])).all()

        def test__7x8_mask_add_edge_pixels__also_in_border(self):
            mask = np.array([[True, True, True, True, True, True, True, True],
                                   [True, True, True, False, True, True, True, True],
                                   [True, True, False, False, False, True, True, True],
                                   [True, True, False, False, False, True, True, True],
                                   [True, False, False, False, False, False, True, True],
                                   [True, True, False, False, False, True, True, True],
                                   [True, True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            border_pixels = mask.compute_grid_border()

            assert (border_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14])).all()

        def test__7x8_mask_big_square(self):
            mask = np.array([[True, True, True, True, True, True, True, True],
                                   [True, False, False, False, False, False, True, True],
                                   [True, False, False, False, False, False, True, True],
                                   [True, False, False, False, False, False, True, True],
                                   [True, False, False, False, False, False, True, True],
                                   [True, False, False, False, False, False, True, True],
                                   [True, True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            border_pixels = mask.compute_grid_border()

            assert (border_pixels == np.array
            ([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24])).all()


    class TestSparsePixels(object):

        # TODO : These tests are over crowded, should break up into more self contained things.

        def test__7x7_circle_mask__five_central_pixels__sparse_grid_size_1(self):
            
            mask = np.array([[True, True, True, True, True, True, True],
                               [True, True, True, True, True, True, True],
                               [True, True, True, True, True, True, True],
                               [True, False, False, False, False, False, True],
                               [True, True, True, True, True, True, True],
                               [True, True, True, True, True, True, True],
                               [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=1)

            assert (sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__7x7_circle_mask__sparse_grid_size_1(self):

            mask = np.array([[True, True, True, True, True, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=1)

            assert (sparse_to_image == np.arange(21)).all()
            assert (image_to_sparse == np.arange(21)).all()

        def test__7x7_rectangle_mask__sparse_grid_size_1(self):

            mask = np.array([[False, False, False, False, False, False, False],
                                   [False, False, False, False, False, False, False],
                                   [False, False, False, False, False, False, False],
                                   [False, False, False, False, False, False, False],
                                   [False, False, False, False, False, False, False],
                                   [False, False, False, False, False, False, False],
                                   [False, False, False, False, False, False, False]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=1)

            assert (sparse_to_image == np.arange(49)).all()
            assert (image_to_sparse == np.arange(49)).all()

        def test__7x7_circle_mask__sparse_grid_size_2(self):

            mask = np.array([[True, True, True, True, True, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=2)

            assert (sparse_to_image == np.array([4, 6, 14, 16])).all()
            assert (image_to_sparse == np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
                                                 1, 2, 2, 2, 3, 3, 2, 2, 3])).all()

        def test__8x8_sporadic_mask__sparse_grid_size_2(self):

            mask = np.array([[True, True, True, True, True, True, False, False],
                                   [True, True, False, False, False, True, False, False],
                                   [True, False, False, False, False, False, False, False],
                                   [True, False, False, False, False, False, False, False],
                                   [True, False, False, False, False, False, False, False],
                                   [True, True, False, False, False, True, False, False],
                                   [True, True, True, True, True, True, False, False],
                                   [True, True, False, False, False, True, False, False]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=2)

            assert (sparse_to_image == np.array([0, 8, 10, 12, 22, 24, 26, 33])).all()
            assert (image_to_sparse == np.array([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 2, 2, 3, 3,
                                                 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 4, 4, 5, 6, 6,
                                                 7, 7, 4, 4, 7, 7, 7])).all()

        def test__7x7_circle_mask_trues_on_even_values__sparse_grid_size_2(self):

            mask = np.array([[False, True, False, True, False, True, False],
                                   [True, True, True, True, True, True, True],
                                   [False, True, False, True, False, True, False],
                                   [True, True, True, True, True, True, True],
                                   [False, True, False, True, False, True, False],
                                   [True, True, True, True, True, True, True],
                                   [False, True, False, True, False, True, False]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=2)

            assert (sparse_to_image == np.arange(16)).all()
            assert (image_to_sparse == np.arange(16)).all()

        def test__7x7_circle_mask__sparse_grid_size_3(self):

            mask = np.array([[True, True, True, True, True, True, True],
                                   [True, True, False, False, False, True, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=3)

            assert (sparse_to_image == np.array([10])).all()
            assert (image_to_sparse == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all()

        def test__7x7_circle_mask_more_points_added__sparse_grid_size_3(self):

            mask = np.array([[False, True, True, False, True, False, False],
                                   [True, True, False, False, False, True, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, False],
                                   [True, False, False, False, False, False, True],
                                   [True, True, False, False, False, True, True],
                                   [True, True, True, True, True, True, False]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=3)

            assert (sparse_to_image == np.array([0, 1, 3, 14, 17, 26])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 2, 1, 1, 1, 0, 3, 3, 3, 4, 3, 3, 3, 3, 4, 4, 3, 3, 3,
                                                 3, 4, 3, 3, 3, 5])).all()

        def test__7x7_mask_trues_on_values_which_divide_by_3__sparse_grid_size_3(self):

            mask = np.array([[False, True, True, False, True, True, False],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [False, True, True, False, True, True, False],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [False, True, True, False, True, True, False]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=3)

            assert (sparse_to_image == np.arange(9)).all()
            assert (image_to_sparse == np.arange(9)).all()

        def test__8x8_mask_trues_on_values_which_divide_by_3_and_other_values__sparse_grid_size_3(self):

            mask = np.array([[False, True, False, False, True, True, False],
                                   [True, True, True, True, True, True, True],
                                   [True, True, False, False, False, True, True],
                                   [False, True, True, False, True, True, False],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [False, False, False, False, False, False, False]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=3)

            assert (sparse_to_image == np.array([0, 2, 3, 7, 8, 9, 10, 13, 16])).all()
            assert (image_to_sparse == np.array([0, 1, 1, 2, 4, 4, 4, 3, 4, 5, 6, 6, 7, 7, 7, 8, 8])).all()

        def test__8x7__five_central_pixels__sparse_grid_size_1(self):

            mask = np.array([[True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, False, False, False, False, False, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=1)

            assert (sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__8x7__five_central_pixels_2__sparse_grid_size_1(self):

            mask = np.array([[True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, False, False, False, False, False, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=1)

            assert (sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__8x7__five_central_pixels__sparse_grid_size_2(self):

            mask = np.array([[True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, False, False, False, False, False, True],
                                   [True, False, False, False, False, False, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=2)

            assert (sparse_to_image == np.array([1, 3])).all()
            assert (image_to_sparse == np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])).all()

        def test__7x8__five_central_pixels__sparse_grid_size_1(self):

            mask = np.array([[True, True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True, True],
                                   [True, False, False, False, False, False, True, True],
                                   [True, True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=1)

            assert (sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__7x8__five_central_pixels__sparse_grid_size_2(self):

            mask = np.array([[True, True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True, True],
                                   [True, False, False, False, False, False, True, True],
                                   [True, False, False, False, False, False, True, True],
                                   [True, True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=2)

            assert (sparse_to_image == np.array([1, 3])).all()
            assert (image_to_sparse == np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])).all()

        def test__7x8__more_central_pixels__sparse_grid_size_2(self):

            mask = np.array([[True, True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True, True],
                                   [True, False, False, False, False, False, True, True],
                                   [True, False, False, False, False, False, True, True],
                                   [True, False, False, False, False, False, True, True],
                                   [True, True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size=2)

            assert (sparse_to_image == np.array([1, 3, 11, 13])).all()
            assert (image_to_sparse == np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3])).all()