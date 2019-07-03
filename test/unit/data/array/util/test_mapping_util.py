import os

import numpy as np

from autolens.data.array.util import mapping_util, mask_util
from autolens.data.array import mask

test_data_dir = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestMap1dIndexesTo2dIndex(object):

    def test__9_1d_indexes_from_0_to_8__map_to_shape_3x3(self):

        indexes_1d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

        indexes_2d = mapping_util.map_1d_indexes_to_2d_indexes_for_shape(indexes_1d=indexes_1d, shape=(3, 3))

        assert (indexes_2d == np.array([[0,0], [0,1], [0,2],
                                        [1,0], [1,1], [1,2],
                                        [2,0], [2,1], [2,2]])).all()

    def test__6_1d_indexes_from_0_to_5__map_to_shape_2x3(self):

        indexes_1d = np.array([0, 1, 2, 3, 4, 5])

        indexes_2d = mapping_util.map_1d_indexes_to_2d_indexes_for_shape(indexes_1d=indexes_1d, shape=(2, 3))

        assert (indexes_2d == np.array([[0,0], [0,1], [0,2],
                                        [1,0], [1,1], [1,2]])).all()

    def test__6_1d_indexes_from_0_to_5__map_to_shape_3x2(self):

        indexes_1d = np.array([0, 1, 2, 3, 4, 5])

        indexes_2d = mapping_util.map_1d_indexes_to_2d_indexes_for_shape(indexes_1d=indexes_1d, shape=(3, 2))

        assert (indexes_2d == np.array([[0,0], [0,1],
                                        [1,0], [1,1],
                                        [2,0], [2,1]])).all()

    def test__9_1d_indexes_from_0_to_8_different_order__map_to_shape_3x3(self):

        indexes_1d = np.array([1, 4, 7, 8, 0, 2, 3, 5, 6])

        indexes_2d = mapping_util.map_1d_indexes_to_2d_indexes_for_shape(indexes_1d=indexes_1d, shape=(3, 3))

        assert (indexes_2d == np.array([[0,1], [1,1], [2,1],
                                        [2,2], [0,0], [0,2],
                                        [1,0], [1,2], [2,0]])).all()


class TestMap2dIndexesTo1dIndex(object):

    def test__9_2d_indexes_from_0_0_to_2_2__map_to_shape_3x3(self):

        indexes_2d = np.array([[0,0], [0,1], [0,2],
                               [1,0], [1,1], [1,2],
                               [2,0], [2,1], [2,2]])

        indexes_1d = mapping_util.map_2d_indexes_to_1d_indexes_for_shape(indexes_2d=indexes_2d, shape=(3, 3))

        assert (indexes_1d == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__6_1d_indexes_from_0_0_to_1_2__map_to_shape_2x3(self):

        indexes_2d =np.array([[0,0], [0,1], [0,2],
                              [1,0], [1,1], [1,2]])

        indexes_1d = mapping_util.map_2d_indexes_to_1d_indexes_for_shape(indexes_2d=indexes_2d, shape=(2, 3))

        assert (indexes_1d == np.array([0, 1, 2, 3, 4, 5])).all()

    def test__6_1d_indexes_from_0_0_to_2_1__map_to_shape_3x2(self):

        indexes_2d =np.array([[0,0], [0,1],
                              [1,0], [1,1],
                              [2,0], [2,1]])


        indexes_1d = mapping_util.map_2d_indexes_to_1d_indexes_for_shape(indexes_2d=indexes_2d, shape=(3, 2))

        assert (indexes_1d == np.array([0, 1, 2, 3, 4, 5])).all()

    def test__9_1d_indexes_from_0_0_to_2_2_different_order__map_to_shape_3x3(self):

        indexes_2d = np.array([[0,1], [1,1], [2,1],
                               [2,2], [0,0], [0,2],
                               [1,0], [1,2], [2,0]])


        indexes_1d = mapping_util.map_2d_indexes_to_1d_indexes_for_shape(indexes_2d=indexes_2d, shape=(3, 3))

        assert (indexes_1d == np.array([1, 4, 7, 8, 0, 2, 3, 5, 6])).all()


class TestSubToImage(object):

    def test__3x3_mask_with_1_pixel__2x2_sub_grid__correct_sub_to_image(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        sub_to_image = mapping_util.sub_to_regular_from_mask(mask, sub_grid_size=2)

        assert (sub_to_image == np.array([0, 0, 0, 0])).all()

    def test__3x3_mask_with_row_of_pixels_pixel__2x2_sub_grid__correct_sub_to_image(self):
        mask = np.array([[True, True, True],
                         [False, False, False],
                         [True, True, True]])

        sub_to_image = mapping_util.sub_to_regular_from_mask(mask, sub_grid_size=2)

        assert (sub_to_image == np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])).all()

    def test__3x3_mask_with_row_of_pixels_pixel__3x3_sub_grid__correct_sub_to_image(self):
        mask = np.array([[True, True, True],
                         [False, False, False],
                         [True, True, True]])

        sub_to_image = mapping_util.sub_to_regular_from_mask(mask, sub_grid_size=3)

        assert (sub_to_image == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1,
                                          2, 2, 2, 2, 2, 2, 2, 2, 2])).all()


class TestMap2DArrayTo1d(object):

    def test__setup_3x3_data(self):
        array_2d = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        array_1d = mapping_util.map_array_2d_to_masked_array_1d_from_array_2d_and_mask(mask, array_2d)

        assert (array_1d == np.array([5])).all()

    def test__setup_3x3_array__five_now_in_mask(self):
        array_2d = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        array_1d = mapping_util.map_array_2d_to_masked_array_1d_from_array_2d_and_mask(mask, array_2d)

        assert (array_1d == np.array([2, 4, 5, 6, 8])).all()

    def test__setup_3x4_array(self):
        array_2d = np.array([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12]])

        mask = np.array([[True, False, True, True],
                         [False, False, False, True],
                         [True, False, True, False]])

        array_1d = mapping_util.map_array_2d_to_masked_array_1d_from_array_2d_and_mask(mask, array_2d)

        assert (array_1d == np.array([2, 5, 6, 7, 10, 12])).all()

    def test__setup_4x3_array__five_now_in_mask(self):
        array_2d = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9],
                             [10, 11, 12]])

        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True],
                         [True, True, True]])

        array_1d = mapping_util.map_array_2d_to_masked_array_1d_from_array_2d_and_mask(mask, array_2d)

        assert (array_1d == np.array([2, 4, 5, 6, 8])).all()


class TestMap2DGridTo1d(object):

    def test__setup_3x3_data(self):
        
        grid_2d = np.array([[[1, 1], [2, 2], [3, 3]],
                             [[4, 4], [5, 5], [6, 6]],
                             [[7, 7], [8, 8], [9, 9]]])

        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        grid_1d = mapping_util.map_grid_2d_to_masked_grid_1d_from_grid_2d_and_mask(mask=mask, grid_2d=grid_2d)

        assert (grid_1d == np.array([[5, 5]])).all()

    def test__setup_3x3_grid__five_now_in_mask(self):

        grid_2d = np.array([[[1, 1], [2, 2], [3, 3]],
                            [[4, 4], [5, 5], [6, 6]],
                            [[7, 7], [8, 8], [9, 9]]])

        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        grid_1d = mapping_util.map_grid_2d_to_masked_grid_1d_from_grid_2d_and_mask(mask=mask, grid_2d=grid_2d)

        assert (grid_1d == np.array([[2,2], [4,4], [5,5], [6,6], [8,8]])).all()

    def test__setup_3x4_grid(self):

        grid_2d = np.array([[[1,1], [2,2], [3,3], [4,4]],
                             [[5,5], [6,6], [7,7], [8,8]],
                             [[9,9], [10,10], [11,11], [12,12]]])

        mask = np.array([[True, False, True, True],
                         [False, False, False, True],
                         [True, False, True, False]])

        grid_1d = mapping_util.map_grid_2d_to_masked_grid_1d_from_grid_2d_and_mask(mask=mask, grid_2d=grid_2d)

        assert (grid_1d == np.array([[2,2], [5,5], [6,6], [7,7], [10,10], [12,12]])).all()

    def test__setup_4x3_grid__five_now_in_mask(self):

        grid_2d = np.array([[[1, 1], [2, 2], [3, 3]],
                            [[4, 4], [5, 5], [6, 6]],
                            [[7, 7], [8, 8], [9, 9]],
                            [[10, 10], [11,11], [12,12]]])

        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True],
                         [True, True, True]])

        grid_1d = mapping_util.map_grid_2d_to_masked_grid_1d_from_grid_2d_and_mask(mask=mask, grid_2d=grid_2d)

        assert (grid_1d == np.array([[2,2], [4,4], [5,5], [6,6], [8,8]])).all()


class TestMap2DSubArrayTo1D(object):
    
    def test__setup_3x3_data__sub_grid_size_1(self):

        sub_array_2d = np.array([[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]])

        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        sub_array_1d = mapping_util.map_sub_array_2d_to_masked_sub_array_1d_from_sub_array_2d_mask_and_sub_grid_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_grid_size=1)

        assert (sub_array_1d == np.array([5])).all()

    def test__setup_3x3_data__sub_grid_size_2(self):

        sub_array_2d = np.array([[1,  2,  3,   4,  5,  6],
                                 [7,  8,  9,  10, 11, 12],
                                 [13, 14, 15, 16, 17, 18],
                                 [1,   2,  3,  4,  5,  6],
                                 [7,   8,  9, 10, 11, 12],
                                 [13, 14, 15, 16, 17, 18]])

        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        sub_array_1d = mapping_util.map_sub_array_2d_to_masked_sub_array_1d_from_sub_array_2d_mask_and_sub_grid_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_grid_size=2)

        assert (sub_array_1d == np.array([15, 16, 3, 4])).all()

        mask = np.array([[True, False, True],
                         [True, False, True],
                         [True, True, False]])

        sub_array_1d = mapping_util.map_sub_array_2d_to_masked_sub_array_1d_from_sub_array_2d_mask_and_sub_grid_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_grid_size=2)

        assert (sub_array_1d == np.array([3, 4, 9, 10, 15, 16, 3, 4, 11, 12, 17, 18])).all()


    def test__setup_3x4_sub_array(self):

        sub_array_2d = np.array([[1,  2,  3,   4,  5,  6, 7, 7, 7],
                                 [7,  8,  9,  10, 11, 12, 7, 7, 7],
                                 [13, 14, 15, 16, 17, 18, 7, 7, 7],
                                 [1,   2,  3,  4,  5,  6, 7, 7, 7],
                                 [7,   8,  9, 10, 11, 12, 7, 7, 7],
                                 [13, 14, 15, 16, 17, 18, 7, 7, 7]])

        mask = np.array([[True, False, True, True],
                         [False, False, False, True],
                         [True, False, True, False]])

        sub_array_1d = mapping_util.map_sub_array_2d_to_masked_sub_array_1d_from_sub_array_2d_mask_and_sub_grid_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_grid_size=2)

        assert (sub_array_1d == np.array([3, 4, 9, 10, 13, 14, 1, 2, 15, 16, 3, 4, 17, 18, 5, 6, 9, 10, 15, 16, 7, 7, 7, 7])).all()

    def test__setup_4x3_sub_array__five_now_in_mask(self):

        sub_array_2d = np.array([[1,  2,  3,   4,  5,  6],
                                 [7,  8,  9,  10, 11, 12],
                                 [13, 14, 15, 16, 17, 18],
                                 [1,   2,  3,  4,  5,  6],
                                 [7,   8,  9, 10, 11, 12],
                                 [13, 14, 15, 16, 17, 18],
                                 [7,   7,  7,  7,  7,  7],
                                 [7,   7,  7,  7,  7,  7]])

        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True],
                         [True, True, True]])

        sub_array_1d = mapping_util.map_sub_array_2d_to_masked_sub_array_1d_from_sub_array_2d_mask_and_sub_grid_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_grid_size=2)

        assert (sub_array_1d == np.array([3, 4, 9, 10, 13, 14, 1, 2, 15, 16, 3, 4, 17, 18, 5, 6, 9, 10, 15, 16])).all()


    def test__setup_2x2_data__sub_grid_size_3(self):

        sub_array_2d = np.array([[1,  2,  3,   4,  5,  6],
                                 [7,  8,  9,  10, 11, 12],
                                 [13, 14, 15, 16, 17, 18],
                                 [1,   2,  3,  4,  5,  6],
                                 [7,   8,  9, 10, 11, 12],
                                 [13, 14, 15, 16, 17, 18]])

        mask = np.array([[False, True],
                         [True, False]])

        sub_array_1d = mapping_util.map_sub_array_2d_to_masked_sub_array_1d_from_sub_array_2d_mask_and_sub_grid_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_grid_size=3)

        assert (sub_array_1d == np.array([1, 2, 3, 7, 8, 9, 13, 14, 15, 4, 5, 6, 10, 11, 12, 16, 17, 18])).all()


class TestMapMasked1DArrayTo2d(object):

    def test__2d_array_is_2x2__is_not_masked__maps_correctly(self):
        array_1d = np.array([1.0, 2.0, 3.0, 4.0])

        one_to_two = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        shape = (2, 2)

        array_2d = mapping_util.map_masked_array_1d_to_array_2d_from_array_1d_shape_and_one_to_two(array_1d, shape, one_to_two)

        assert (array_2d == np.array([[1.0, 2.0],
                                      [3.0, 4.0]])).all()

    def test__2d_array_is_2x2__is_masked__maps_correctly(self):
        array_1d = np.array([1.0, 2.0, 3.0])

        one_to_two = np.array([[0, 0], [0, 1], [1, 0]])
        shape = (2, 2)

        array_2d = mapping_util.map_masked_array_1d_to_array_2d_from_array_1d_shape_and_one_to_two(array_1d, shape, one_to_two)

        assert (array_2d == np.array([[1.0, 2.0],
                                      [3.0, 0.0]])).all()

    def test__different_shape_and_mappings(self):
        array_1d = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])

        one_to_two = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [2, 3]])
        shape = (3, 4)

        array_2d = mapping_util.map_masked_array_1d_to_array_2d_from_array_1d_shape_and_one_to_two(array_1d, shape, one_to_two)

        assert (array_2d == np.array([[1.0, 2.0, 0.0, 0.0],
                                      [3.0, 0.0, 0.0, 0.0],
                                      [-1.0, -2.0, 0.0, -3.0]])).all()


class TestMapMasked1DGridTo2d(object):

    def test__2d_grid_is_2x2__is_not_masked__maps_correctly(self):
        
        grid_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        one_to_two = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        shape = (2, 2)

        grid_2d = mapping_util.map_masked_1d_grid_to_2d_grid_from_grid_1d_shape_and_one_to_two(grid_1d, shape, one_to_two)

        assert (grid_2d == np.array([[[1.0, 1.0], [2.0, 2.0]],
                                     [[3.0, 3.0], [4.0, 4.0]]])).all()

    def test__2d_grid_is_2x2__is_masked__maps_correctly(self):

        grid_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 4.0]])

        one_to_two = np.array([[0, 0], [0, 1], [1, 0]])
        shape = (2, 2)

        grid_2d = mapping_util.map_masked_1d_grid_to_2d_grid_from_grid_1d_shape_and_one_to_two(grid_1d, shape, one_to_two)

        assert (grid_2d == np.array([[[1.0, 1.0], [2.0, 2.0]],
                                      [[3.0, 4.0], [0.0, 0.0]]])).all()

    def test__different_shape_and_mappings(self):

        grid_1d = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]])

        one_to_two = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [2, 3]])
        shape = (3, 4)

        grid_2d = mapping_util.map_masked_1d_grid_to_2d_grid_from_grid_1d_shape_and_one_to_two(grid_1d, shape, one_to_two)

        assert (grid_2d == np.array([[[1.0, -1.0], [2.0, -2.0], [0.0, 0.0], [0.0, 0.0]],
                                      [[3.0, -3.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                                      [[-1.0, 1.0], [-2.0, 2.0], [0.0, 0.0], [-3.0, 3.0]]])).all()


class TestMapUnmasked1dArrayTo2d(object):

    def test__1d_array_in__maps_it_to_4x4_2d_array(self):
        array_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        array_2d = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d, shape=(4, 4))

        assert (array_2d == np.array([[1.0, 2.0, 3.0, 4.0],
                                      [5.0, 6.0, 7.0, 8.0],
                                      [9.0, 10.0, 11.0, 12.0],
                                      [13.0, 14.0, 15.0, 16.0]])).all()

    def test__1d_array_in__can_map_it_to_2x3_2d_array(self):
        array_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        array_2d = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d, shape=(2, 3))

        assert (array_2d == np.array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]])).all()

    def test__1d_array_in__can_map_it_to_3x2_2d_array(self):
        array_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        array_2d = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d, shape=(3, 2))

        assert (array_2d == np.array([[1.0, 2.0],
                                      [3.0, 4.0],
                                      [5.0, 6.0]])).all()
        
        
class TestSparseToUnmaskedSparse:

    def test__mask_full_false__image_mask_and_pixel_centres_fully_overlap__each_sparse_maps_to_unmaked_sparse(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        unmasked_sparse_grid_pixel_centres = np.array \
            ([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [2 ,0], [2 ,1], [2 ,2]])

        total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                      unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres)

        sparse_to_unmasked_sparse = mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_masked_pixels, mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres)

        assert (sparse_to_unmasked_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__same_as_above__but_remove_some_centre_pixels_and_change_order__order_does_not_change_mapping(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        unmasked_sparse_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [2 ,2], [1 ,1], [0 ,2], [2 ,0], [0 ,2]])

        total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                      unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres)

        sparse_to_unmasked_sparse = mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_masked_pixels, mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres)

        assert (sparse_to_unmasked_sparse == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    def test__mask_is_cross__some_pix_pixels_are_masked__omitted_from_mapping(self):

        ma = mask.Mask(array=np.array([[True, False, True],
                                       [False, False, False],
                                       [True, False, True]]), pixel_scale=1.0)

        unmasked_sparse_grid_pixel_centres = np.array \
            ([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [2 ,0], [2 ,1], [2 ,2]])

        total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                      unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres)

        sparse_to_unmasked_sparse = mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_masked_pixels, mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres)

        assert (sparse_to_unmasked_sparse == np.array([1, 3, 4, 5, 7])).all()

    def test__same_as_above__different_mask_and_centres(self):

        ma = mask.Mask(array=np.array([[False, False, True],
                                       [False, False, False],
                                       [True, False, False]]), pixel_scale=1.0)

        unmasked_sparse_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1]])

        total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                      unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres)

        sparse_to_unmasked_sparse = mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_masked_pixels, mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres)

        assert (sparse_to_unmasked_sparse == np.array([0, 1, 5])).all()

    def test__same_as_above__but_3x4_mask(self):

        ma = mask.Mask(array=np.array([[True, True, False, True],
                                       [False, False, False, False],
                                       [True,  True,  False, True]]), pixel_scale=1.0)

        unmasked_sparse_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1], [2 ,3], [2, 2]])

        total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                      unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres)

        sparse_to_unmasked_sparse = mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_masked_pixels, mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres)

        assert (sparse_to_unmasked_sparse == np.array([2, 3, 4, 5, 7])).all()

    def test__same_as_above__but_4x3_mask(self):

        ma = mask.Mask(array=np.array([[True, False, True],
                                       [True,  False, True],
                                       [False, False, False],
                                       [True,  False, True]]), pixel_scale=1.0)

        unmasked_sparse_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1], [2 ,2], [3, 1]])

        total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                      unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres)

        sparse_to_unmasked_sparse = mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_masked_pixels, mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres)

        assert (sparse_to_unmasked_sparse == np.array([1, 5, 6, 7])).all()


class TestUnmaskedSparseToSparse:

    def test__mask_full_false__image_mask_and_pixel_centres_fully_overlap__each_pix_maps_to_unmaked_pix(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        unmasked_sparse_grid_pixel_centres = np.array \
            ([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [2 ,0], [2 ,1], [2 ,2]])

        unmasked_sparse_to_sparse = mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(mask=ma,
                                         unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
                                          total_sparse_pixels=9)

        assert (unmasked_sparse_to_sparse  == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__same_as_above__but_remove_some_centre_pixels_and_change_order__order_does_not_change_mapping(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        unmasked_sparse_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [2 ,2], [1 ,1], [0 ,2], [2 ,0], [0 ,2]])

        unmasked_sparse_to_sparse = mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(mask=ma,
                                    unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
                                    total_sparse_pixels=9)

        assert (unmasked_sparse_to_sparse  == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    def test__mask_is_cross__some_pix_pixels_are_masked__omitted_from_mapping(self):

        ma = mask.Mask(array=np.array([[True, False, True],
                                       [False, False, False],
                                       [True, False, True]]), pixel_scale=1.0)

        unmasked_sparse_grid_pixel_centres = np.array \
            ([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [2 ,0], [2 ,1], [2 ,2]])

        unmasked_sparse_to_sparse = mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(mask=ma,
                                    unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
                                    total_sparse_pixels=5)

        assert (unmasked_sparse_to_sparse  == np.array([0, 0, 1, 1, 2, 3, 4, 4, 4])).all()

    def test__same_as_above__different_mask_and_centres(self):

        ma = mask.Mask(array=np.array([[False, False, True],
                                       [False, False, False],
                                       [True, False, False]]), pixel_scale=1.0)

        unmasked_sparse_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1]])

        unmasked_sparse_to_sparse = mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(mask=ma,
                                            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
                                                                                           total_sparse_pixels=4)

        assert (unmasked_sparse_to_sparse  == np.array([0, 1, 2, 2, 2, 2])).all()

    def test__same_as_above__but_3x4_mask(self):

        ma = mask.Mask(array=np.array([[True, True, False, True],
                                       [False, False, False, False],
                                       [True,  True,  False, True]]), pixel_scale=1.0)

        unmasked_sparse_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1], [2 ,3], [0, 2]])

        unmasked_sparse_to_sparse = mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(mask=ma,
                   unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres, total_sparse_pixels=5)

        assert (unmasked_sparse_to_sparse  == np.array([0, 0, 0, 1, 2, 3, 4, 4])).all()

    def test__same_as_above__but_4x3_mask(self):

        ma = mask.Mask(array=np.array([[True, False, True],
                                       [True,  False, True],
                                       [False, False, False],
                                       [True,  False, True]]), pixel_scale=1.0)

        unmasked_sparse_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1], [2 ,2], [3, 1]])

        unmasked_sparse_to_sparse = mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(mask=ma,
                                         unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
                                                                                         total_sparse_pixels=6)

        assert (unmasked_sparse_to_sparse  == np.array([0, 0, 1, 1, 1, 1, 2, 3])).all()


class TestRegularToSparse:

    def test__simple_cases_for_regular_to_unmasked_sparse_and__unmasked_sparse_to_sparse(self):

        regular_to_unmasked_sparse = np.array([0, 1, 2, 3, 4])
        unmasked_sparse_to_sparse = np.array([0, 1, 2, 3, 4])
        regular_to_sparse = mapping_util.regular_to_sparse_from_sparse_mappings(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            unmasked_sparse_to_sparse=unmasked_sparse_to_sparse)

        assert (regular_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        regular_to_unmasked_sparse = np.array([0, 1, 2, 3, 4])
        unmasked_sparse_to_sparse = np.array([0, 1, 5, 7, 18])
        regular_to_sparse = mapping_util.regular_to_sparse_from_sparse_mappings(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            unmasked_sparse_to_sparse=unmasked_sparse_to_sparse)

        assert (regular_to_sparse == np.array([0, 1, 5, 7, 18])).all()

        regular_to_unmasked_sparse = np.array([1, 1, 1, 1, 2])
        unmasked_sparse_to_sparse = np.array([0, 10, 15, 3, 4])
        regular_to_sparse = mapping_util.regular_to_sparse_from_sparse_mappings(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            unmasked_sparse_to_sparse=unmasked_sparse_to_sparse)

        assert (regular_to_sparse == np.array([10, 10, 10, 10, 15])).all()

        regular_to_unmasked_sparse = np.array([5, 6, 7, 8, 9])
        unmasked_sparse_to_sparse = np.array([0, 1, 2, 3, 4, 19, 18, 17, 16, 15])
        regular_to_sparse = mapping_util.regular_to_sparse_from_sparse_mappings(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            unmasked_sparse_to_sparse=unmasked_sparse_to_sparse)

        assert (regular_to_sparse == np.array([19, 18, 17, 16, 15])).all()


class TestSparseGridFromUnmaskedSparseGrid:

    def test__simple_unmasked_sparse_grid__full_grid_pix_grid_same_size__straightforward_mappings(self):

        unmasked_sparse_grid = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        sparse_to_unmasked_sparse = np.array([0, 1, 2, 3])
        pix_grid = mapping_util.sparse_grid_from_unmasked_sparse_grid(unmasked_sparse_grid=unmasked_sparse_grid,
                                                                      sparse_to_unmasked_sparse=sparse_to_unmasked_sparse)

        assert (pix_grid == np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])).all()

        unmasked_sparse_grid = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])
        sparse_to_unmasked_sparse = np.array([0, 1, 2, 3])
        pix_grid = mapping_util.sparse_grid_from_unmasked_sparse_grid(unmasked_sparse_grid=unmasked_sparse_grid,
                                                                      sparse_to_unmasked_sparse=sparse_to_unmasked_sparse)

        assert (pix_grid == np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])).all()

        unmasked_sparse_grid = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])
        sparse_to_unmasked_sparse = np.array([1, 0, 3, 2])
        pix_grid = mapping_util.sparse_grid_from_unmasked_sparse_grid(unmasked_sparse_grid=unmasked_sparse_grid,
                                                                      sparse_to_unmasked_sparse=sparse_to_unmasked_sparse)

        assert (pix_grid == np.array([[4.0, 5.0], [0.0, 0.0], [8.0, 7.0], [2.0, 2.0]])).all()

    def test__simple_unmasked_sparse_grid__full_grid_pix_bigger_than_pix__straightforward_mappings(self):

        unmasked_sparse_grid = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        sparse_to_unmasked_sparse = np.array([1, 2])
        pix_grid = mapping_util.sparse_grid_from_unmasked_sparse_grid(unmasked_sparse_grid=unmasked_sparse_grid,
                                                                      sparse_to_unmasked_sparse=sparse_to_unmasked_sparse)

        assert (pix_grid == np.array([[1.0, 1.0], [2.0, 2.0]])).all()

        unmasked_sparse_grid = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])
        sparse_to_unmasked_sparse = np.array([2, 2, 3])
        pix_grid = mapping_util.sparse_grid_from_unmasked_sparse_grid(unmasked_sparse_grid=unmasked_sparse_grid,
                                                                      sparse_to_unmasked_sparse=sparse_to_unmasked_sparse)

        assert (pix_grid == np.array([[2.0, 2.0], [2.0, 2.0], [8.0, 7.0]])).all()

        unmasked_sparse_grid = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0], [11.0, 11.0], [-20.0, -15.0]])
        sparse_to_unmasked_sparse = np.array([1, 0, 5, 2])
        pix_grid = mapping_util.sparse_grid_from_unmasked_sparse_grid(unmasked_sparse_grid=unmasked_sparse_grid,
                                                                      sparse_to_unmasked_sparse=sparse_to_unmasked_sparse)

        assert (pix_grid == np.array([[4.0, 5.0], [0.0, 0.0], [-20.0, -15.0], [2.0, 2.0]])).all()