from __future__ import division, print_function

import numpy as np
import pytest
from LensData.ArrayTools import *

class TestArrayTools:

    def test_get_dimensions__input_arrays__correct_dimensions(self):

        xdim, ydim = get_dimensions(np.zeros((2, 2)))

        assert xdim == 2
        assert ydim == 2

        xdim, ydim = get_dimensions(np.zeros((2, 4)))

        assert xdim == 2
        assert ydim == 4

        xdim, ydim = get_dimensions(np.zeros((3, 1)))

        assert xdim == 3
        assert ydim == 1

    def test_get_dimensions__not_2d_array__raise_index_error(self):

        with pytest.raises(IndexError):
            get_dimensions(np.zeros((2)))

        with pytest.raises(IndexError):
            get_dimensions(np.zeros((2, 2, 2)))

    def test_get_dimensions__input_not_numpy_array__raises_type_error(self):

        with pytest.raises(TypeError):
            get_dimensions(array='')

    def test_extract_edges__array_all_zeros__correct_list(self):

        assert (extract_edges(array=np.zeros((3,3))) == [0, 0, 0, 0, 0, 0, 0, 0]).all()

    def test_extract_edges__array_sequential__correct_list(self):

        array=np.zeros((3,3))

        array[0, 0] = 1
        array[0, 1] = 2
        array[0, 2] = 3
        array[1, 0] = 4
        array[1, 1] = 5
        array[1, 2] = 6
        array[2, 0] = 7
        array[2, 1] = 8
        array[2, 2] = 9

        assert (extract_edges(array=array) == [1, 2, 3, 6, 7, 8, 9, 4]).all()

    def test_extract_edges__array_sequential_bigger_x__correct_list(self):

        array = np.zeros((4, 3))

        array[0, 0] = 1
        array[0, 1] = 2
        array[0, 2] = 3
        array[1, 0] = 4
        array[1, 1] = 5
        array[1, 2] = 6
        array[2, 0] = 7
        array[2, 1] = 8
        array[2, 2] = 9
        array[3, 0] = 10
        array[3, 1] = 11
        array[3, 2] = 12

        assert (extract_edges(array=array) == [1, 2, 3, 6, 9, 10, 11, 12, 4, 7]).all()

    def test_extract_edges__array_sequential_bigger_y__correct_list(self):

        array = np.zeros((3, 4))

        array[0, 0] = 1
        array[0, 1] = 2
        array[0, 2] = 3
        array[0, 3] = 4
        array[1, 0] = 5
        array[1, 1] = 6
        array[1, 2] = 7
        array[1, 3] = 8
        array[2, 0] = 9
        array[2, 1] = 10
        array[2, 2] = 11
        array[2, 3] = 12

        assert (extract_edges(array=array) == [1, 2, 3, 4, 8, 9, 10, 11, 12, 5]).all()

        #   def test_median_of_edges__array_all_ones__correct_value(self):

    #    assert get_median_of_edges(array=np.ones((3,3))) == 1

 #   def test_median_of_edges__array_all_zeros__correct_value(self):

  #      assert get_median_of_edges(array=np.zeros((3,3))) == 0

#    def test_median_of_edges__array_all_zeros__correct_value(self):
#
#         array = np.zeros((3,3))
#
#         array[0][0] = 0
#         array[0][1] = 0
#         array[0][2] = 0
#         array[1][0] = 4
#         array[1][1] = 5
#         array[1][2] = 6
#         array[2][0] = 7
#         array[2][1] = 8
#         array[2][2] = 9
#
#         assert get_median_of_edges(array=array) == 0