from __future__ import division, print_function
import pytest
import numpy as np
import os
from data import image, analysis_data

test_data_dir = "{}/../data/test_data/".format(os.path.dirname(os.path.realpath(__file__)))

class TestDataConversion(object):

    def test__setup_3x3___one_data_in_mask(self):

        data =  np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])

        mask_array = np.array([[True, True, True],
                               [True, False, True],
                               [True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        data_1d = analysis_data.setup_data(mask, data)

        assert (data_1d[0] == 5.0)

    def test__setup_3x3_image__five_coordinates(self):

        data =  np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])

        mask_array = np.array([[True, False, True],
                             [False, False, False],
                             [True, False, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        data_1d = analysis_data.setup_data(mask, data)

        assert (data_1d[0] == 2.0)
        assert (data_1d[1] == 4.0)
        assert (data_1d[2] == 5.0)
        assert (data_1d[3] == 6.0)
        assert (data_1d[4] == 8.0)

    def test__setup_4x4_image__ten_coordinates__new_pixel_scale(self):

        data =  np.array([[1.0, 2.0, 3.0, 4.0],
                          [8.0, 7.0, 6.0, 5.0],
                          [9.0, 10.0, 11.0, 12.0],
                          [16.0, 15.0, 14.0, 13.0]])

        mask_array = np.array([[True, False, False, True],
                              [False, False, False, True],
                              [True, False, False, True],
                              [False, False, False, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        data_1d = analysis_data.setup_data(mask, data)

        assert (data_1d[0] == 2.0)
        assert (data_1d[1] == 3.0)
        assert (data_1d[2] == 8.0)
        assert (data_1d[3] == 7.0)
        assert (data_1d[4] == 6.0)
        assert (data_1d[5] == 10.0)
        assert (data_1d[6] == 11.0)
        assert (data_1d[7] == 16.0)
        assert (data_1d[8] == 15.0)
        assert (data_1d[9] == 14.0)

    def test__setup_3x4_image__six_coordinates(self):

        mask_array = np.array([[True, False, True, True],
                         [False, False, False, True],
                         [True, False, True, False]])

        data =  np.array([[1.0, 2.0, 3.0, 4.0],
                          [8.0, 7.0, 6.0, 5.0],
                          [9.0, 10.0, 11.0, 12.0]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        data_1d = analysis_data.setup_data(mask, data)

        assert (data_1d[0] == 2.0)
        assert (data_1d[1] == 8.0)
        assert (data_1d[2] == 7.0)
        assert (data_1d[3] == 6.0)
        assert (data_1d[4] == 10.0)
        assert (data_1d[5] == 12.0)


class TestMapper2d(object):

    def test__setup_3x3___one_data_in_mask(self):

        mask_array = np.array([[True, True, True],
                               [True, False, True],
                               [True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        mapper_2d = analysis_data.setup_mapper_2d(mask)

        assert (mapper_2d[0] == np.array([1, 1])).all()

    def test__setup_3x3_image__five_coordinates(self):

        mask_array = np.array([[True, False, True],
                               [False, False, False],
                               [True, False, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        mapper_2d = analysis_data.setup_mapper_2d(mask)

        assert (mapper_2d[0] == np.array([0, 1])).all()
        assert (mapper_2d[1] == np.array([1, 0])).all()
        assert (mapper_2d[2] == np.array([1, 1])).all()
        assert (mapper_2d[3] == np.array([1, 2])).all()
        assert (mapper_2d[4] == np.array([2, 1])).all()

    def test__setup_4x4_image__ten_coordinates__new_pixel_scale(self):

        mask_array = np.array([[True, False, False, True],
                              [False, False, False, True],
                              [True, False, False, True],
                              [False, False, False, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        mapper_2d = analysis_data.setup_mapper_2d(mask)

        assert (mapper_2d[0] == np.array([0, 1])).all()
        assert (mapper_2d[1] == np.array([0, 2])).all()
        assert (mapper_2d[2] == np.array([1, 0])).all()
        assert (mapper_2d[3] == np.array([1, 1])).all()
        assert (mapper_2d[4] == np.array([1, 2])).all()
        assert (mapper_2d[5] == np.array([2, 1])).all()
        assert (mapper_2d[6] == np.array([2, 2])).all()
        assert (mapper_2d[7] == np.array([3, 0])).all()
        assert (mapper_2d[8] == np.array([3, 1])).all()
        assert (mapper_2d[9] == np.array([3, 2])).all()

    def test__setup_3x4_image__six_coordinates(self):

        mask_array = np.array([[True, False, True, True],
                             [False, False, False, True],
                              [True, False, True, False]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        mapper_2d = analysis_data.setup_mapper_2d(mask)

        assert (mapper_2d[0] == np.array([0, 1])).all()
        assert (mapper_2d[1] == np.array([1, 0])).all()
        assert (mapper_2d[2] == np.array([1, 1])).all()
        assert (mapper_2d[3] == np.array([1, 2])).all()
        assert (mapper_2d[4] == np.array([2, 1])).all()
        assert (mapper_2d[5] == np.array([2, 3])).all()


class TestImageCoordinates(object):

    def test__setup_3x3_image_one_coordinate(self):

        mask_array = np.array([[True, True, True],
                               [True, False, True],
                               [True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        image_coordinates = analysis_data.setup_image_coordinates(mask)

        assert (image_coordinates[0] == np.array([0.0, 0.0])).all()

    def test__setup_3x3_image__five_coordinates(self):

        mask_array = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        image_coordinates = analysis_data.setup_image_coordinates(mask)

        assert (image_coordinates[0] == np.array([ 0.0, 3.0])).all()
        assert (image_coordinates[1] == np.array([-3.0, 0.0])).all()
        assert (image_coordinates[2] == np.array([ 0.0, 0.0])).all()
        assert (image_coordinates[3] == np.array([ 3.0, 0.0])).all()
        assert (image_coordinates[4] == np.array([ 0.0,-3.0])).all()

    def test__setup_4x4_image__ten_coordinates__new_pixel_scale(self):

        mask_array = np.array([[True, False, False, True],
                         [False, False, False, True],
                         [True, False, False, True],
                         [False, False, False, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=1.0)

        image_coordinates = analysis_data.setup_image_coordinates(mask)

        assert (image_coordinates[0] == np.array([-0.5, 1.5])).all()
        assert (image_coordinates[1] == np.array([ 0.5, 1.5])).all()
        assert (image_coordinates[2] == np.array([-1.5, 0.5])).all()
        assert (image_coordinates[3] == np.array([-0.5, 0.5])).all()
        assert (image_coordinates[4] == np.array([ 0.5, 0.5])).all()
        assert (image_coordinates[5] == np.array([-0.5,-0.5])).all()
        assert (image_coordinates[6] == np.array([ 0.5,-0.5])).all()
        assert (image_coordinates[7] == np.array([-1.5,-1.5])).all()
        assert (image_coordinates[8] == np.array([-0.5,-1.5])).all()
        assert (image_coordinates[9] == np.array([ 0.5,-1.5])).all()

    def test__setup_3x4_image__six_coordinates(self):

        mask_array = np.array([[True, False, True, True],
                         [False, False, False, True],
                         [True, False, True, False]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        image_coordinates = analysis_data.setup_image_coordinates(mask)

        assert (image_coordinates[0] == np.array([-1.5, 3.0])).all()
        assert (image_coordinates[1] == np.array([-4.5, 0.0])).all()
        assert (image_coordinates[2] == np.array([-1.5, 0.0])).all()
        assert (image_coordinates[3] == np.array([ 1.5, 0.0])).all()
        assert (image_coordinates[4] == np.array([-1.5,-3.0])).all()
        assert (image_coordinates[5] == np.array([ 4.5,-3.0])).all()


class TestImageSubCoordinates(object):

    def test__3x3_mask_with_one_pixel__2x2_sub_grid__coordinates(self):

        mask_array = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        image_sub_coordinates = analysis_data.setup_sub_coordinates(mask=mask, sub_grid_size=2)

        assert (image_sub_coordinates == np.array
            ([[[-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]])).all()

        assert (image_sub_coordinates[0 ,0] == np.array([-0.5,  0.5])).all()
        assert (image_sub_coordinates[0 ,1] == np.array([ 0.5,  0.5])).all()
        assert (image_sub_coordinates[0 ,2] == np.array([-0.5, -0.5])).all()
        assert (image_sub_coordinates[0 ,3] == np.array([ 0.5, -0.5])).all()

    def test__3x3_mask_with_row_of_pixels__2x2_sub_grid__coordinates(self):

        mask_array = np.array([[True, True, True],
                         [False, False, False],
                         [True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        image_sub_coordinates = analysis_data.setup_sub_coordinates(mask=mask, sub_grid_size=2)

        assert (image_sub_coordinates == np.array([[[-3.5, 0.5], [-2.5, 0.5], [-3.5, -0.5], [-2.5, -0.5]],
                                                        [[-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]],
                                                        [[2.5, 0.5], [3.5, 0.5], [2.5, -0.5], [3.5, -0.5]]])).all()

        assert (image_sub_coordinates[0 ,0] == np.array([-3.5,  0.5])).all()
        assert (image_sub_coordinates[0 ,1] == np.array([-2.5,  0.5])).all()
        assert (image_sub_coordinates[0 ,2] == np.array([-3.5, -0.5])).all()
        assert (image_sub_coordinates[0 ,3] == np.array([-2.5, -0.5])).all()

        assert (image_sub_coordinates[1 ,0] == np.array([-0.5,  0.5])).all()
        assert (image_sub_coordinates[1 ,1] == np.array([ 0.5,  0.5])).all()
        assert (image_sub_coordinates[1 ,2] == np.array([-0.5, -0.5])).all()
        assert (image_sub_coordinates[1 ,3] == np.array([ 0.5, -0.5])).all()

        assert (image_sub_coordinates[2 ,0] == np.array([2.5,  0.5])).all()
        assert (image_sub_coordinates[2 ,1] == np.array([3.5,  0.5])).all()
        assert (image_sub_coordinates[2 ,2] == np.array([2.5, -0.5])).all()
        assert (image_sub_coordinates[2 ,3] == np.array([3.5, -0.5])).all()

    def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__coordinates(self):

        mask_array = np.array([[True, True, False],
                         [False, False, False],
                         [True, True, False]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        image_sub_coordinates = analysis_data.setup_sub_coordinates(mask=mask, sub_grid_size=2)

        assert (image_sub_coordinates == np.array([[[2.5, 3.5],  [3.5, 3.5],  [2.5, 2.5], [3.5, 2.5]],
                                                   [[-3.5, 0.5], [-2.5, 0.5], [-3.5, -0.5], [-2.5, -0.5]],
                                                   [[-0.5, 0.5], [0.5, 0.5],  [-0.5, -0.5], [0.5, -0.5]],
                                                   [[2.5, 0.5],  [3.5, 0.5],  [2.5, -0.5], [3.5, -0.5]],
                                                   [[2.5, -2.5], [3.5, -2.5], [2.5, -3.5], [3.5, -3.5]]])).all()

        assert (image_sub_coordinates[0 ,0] == np.array([2.5,  3.5])).all()
        assert (image_sub_coordinates[0 ,1] == np.array([3.5,  3.5])).all()
        assert (image_sub_coordinates[0 ,2] == np.array([2.5,  2.5])).all()
        assert (image_sub_coordinates[0 ,3] == np.array([3.5,  2.5])).all()

        assert (image_sub_coordinates[1 ,0] == np.array([-3.5,  0.5])).all()
        assert (image_sub_coordinates[1 ,1] == np.array([-2.5,  0.5])).all()
        assert (image_sub_coordinates[1 ,2] == np.array([-3.5, -0.5])).all()
        assert (image_sub_coordinates[1 ,3] == np.array([-2.5, -0.5])).all()

        assert (image_sub_coordinates[2 ,0] == np.array([-0.5,  0.5])).all()
        assert (image_sub_coordinates[2 ,1] == np.array([ 0.5,  0.5])).all()
        assert (image_sub_coordinates[2 ,2] == np.array([-0.5, -0.5])).all()
        assert (image_sub_coordinates[2 ,3] == np.array([ 0.5, -0.5])).all()

        assert (image_sub_coordinates[3 ,0] == np.array([2.5,  0.5])).all()
        assert (image_sub_coordinates[3 ,1] == np.array([3.5,  0.5])).all()
        assert (image_sub_coordinates[3 ,2] == np.array([2.5, -0.5])).all()
        assert (image_sub_coordinates[3 ,3] == np.array([3.5, -0.5])).all()

        assert (image_sub_coordinates[4 ,0] == np.array([2.5, -2.5])).all()
        assert (image_sub_coordinates[4 ,1] == np.array([3.5, -2.5])).all()
        assert (image_sub_coordinates[4 ,2] == np.array([2.5, -3.5])).all()
        assert (image_sub_coordinates[4 ,3] == np.array([3.5, -3.5])).all()

    def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__different_pixel_scale(self):

        mask_array = np.array([[True, True, False],
                         [False, False, False],
                         [True, True, False]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=0.3)

        image_sub_coordinates = analysis_data.setup_sub_coordinates(mask=mask, sub_grid_size=2)

        image_sub_coordinates = np.round(image_sub_coordinates, decimals=2)

        assert (image_sub_coordinates == np.array([[[0.25, 0.35], [0.35, 0.35], [0.25, 0.25], [0.35, 0.25]],
                                                        [[-0.35, 0.05], [-0.25, 0.05], [-0.35, -0.05], [-0.25, -0.05]],
                                                        [[-0.05, 0.05], [0.05, 0.05], [-0.05, -0.05], [0.05, -0.05]],
                                                        [[0.25, 0.05], [0.35, 0.05], [0.25, -0.05], [0.35, -0.05]],
                                                        [[0.25, -0.25], [0.35, -0.25], [0.25, -0.35], [0.35, -0.35]]])).all()

        assert (image_sub_coordinates[0 ,0] == np.array([0.25,  0.35])).all()
        assert (image_sub_coordinates[0 ,1] == np.array([0.35,  0.35])).all()
        assert (image_sub_coordinates[0 ,2] == np.array([0.25,  0.25])).all()
        assert (image_sub_coordinates[0 ,3] == np.array([0.35,  0.25])).all()

        assert (image_sub_coordinates[1 ,0] == np.array([-0.35,  0.05])).all()
        assert (image_sub_coordinates[1 ,1] == np.array([-0.25,  0.05])).all()
        assert (image_sub_coordinates[1 ,2] == np.array([-0.35, -0.05])).all()
        assert (image_sub_coordinates[1 ,3] == np.array([-0.25, -0.05])).all()

        assert (image_sub_coordinates[2 ,0] == np.array([-0.05,  0.05])).all()
        assert (image_sub_coordinates[2 ,1] == np.array([ 0.05,  0.05])).all()
        assert (image_sub_coordinates[2 ,2] == np.array([-0.05, -0.05])).all()
        assert (image_sub_coordinates[2 ,3] == np.array([ 0.05, -0.05])).all()

        assert (image_sub_coordinates[3 ,0] == np.array([0.25,  0.05])).all()
        assert (image_sub_coordinates[3 ,1] == np.array([0.35,  0.05])).all()
        assert (image_sub_coordinates[3 ,2] == np.array([0.25, -0.05])).all()
        assert (image_sub_coordinates[3 ,3] == np.array([0.35, -0.05])).all()

        assert (image_sub_coordinates[4 ,0] == np.array([0.25, -0.25])).all()
        assert (image_sub_coordinates[4 ,1] == np.array([0.35, -0.25])).all()
        assert (image_sub_coordinates[4 ,2] == np.array([0.25, -0.35])).all()
        assert (image_sub_coordinates[4 ,3] == np.array([0.35, -0.35])).all()

    def test__3x3_mask_with_one_pixel__3x3_sub_grid__coordinates(self):

        mask_array = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        image_sub_coordinates = analysis_data.setup_sub_coordinates(mask=mask, sub_grid_size=3)

        assert (image_sub_coordinates == np.array([[[-0.75, 0.75],  [0.0, 0.75],  [0.75, 0.75],
                                                         [-0.75, 0.0],   [0.0, 0.0],   [0.75, 0.0],
                                                         [-0.75, -0.75], [0.0, -0.75], [0.75, -0.75]]])).all()

        assert (image_sub_coordinates[0 ,0] == np.array([-0.75, 0.75])).all()
        assert (image_sub_coordinates[0 ,1] == np.array([0.0,   0.75])).all()
        assert (image_sub_coordinates[0 ,2] == np.array([0.75,  0.75])).all()
        assert (image_sub_coordinates[0 ,3] == np.array([-0.75, 0.0])).all()
        assert (image_sub_coordinates[0 ,4] == np.array([0.0,   0.0])).all()
        assert (image_sub_coordinates[0 ,5] == np.array([0.75,  0.0])).all()
        assert (image_sub_coordinates[0 ,6] == np.array([-0.75 ,-0.75])).all()
        assert (image_sub_coordinates[0 ,7] == np.array([0.0,  -0.75])).all()
        assert (image_sub_coordinates[0 ,8] == np.array([0.75, -0.75])).all()

    def test__3x3_mask_with_one_row__3x3_sub_grid__coordinates(self):

        mask_array = np.array([[True, True, False],
                         [True, False, True],
                         [True, True, False]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=2.0)

        image_sub_coordinates = analysis_data.setup_sub_coordinates(mask=mask, sub_grid_size=3)

        assert (image_sub_coordinates[0 ,0] == np.array([1.5, 2.5])).all()
        assert (image_sub_coordinates[0 ,1] == np.array([2.0, 2.5])).all()
        assert (image_sub_coordinates[0 ,2] == np.array([2.5, 2.5])).all()
        assert (image_sub_coordinates[0 ,3] == np.array([1.5, 2.0])).all()
        assert (image_sub_coordinates[0 ,4] == np.array([2.0, 2.0])).all()
        assert (image_sub_coordinates[0 ,5] == np.array([2.5, 2.0])).all()
        assert (image_sub_coordinates[0 ,6] == np.array([1.5, 1.5])).all()
        assert (image_sub_coordinates[0 ,7] == np.array([2.0, 1.5])).all()
        assert (image_sub_coordinates[0 ,8] == np.array([2.5, 1.5])).all()

        assert (image_sub_coordinates[1 ,0] == np.array([-0.5, 0.5])).all()
        assert (image_sub_coordinates[1 ,1] == np.array([0.0,   0.5])).all()
        assert (image_sub_coordinates[1 ,2] == np.array([0.5,  0.5])).all()
        assert (image_sub_coordinates[1 ,3] == np.array([-0.5, 0.0])).all()
        assert (image_sub_coordinates[1 ,4] == np.array([0.0,   0.0])).all()
        assert (image_sub_coordinates[1 ,5] == np.array([0.5,  0.0])).all()
        assert (image_sub_coordinates[1 ,6] == np.array([-0.5 ,-0.5])).all()
        assert (image_sub_coordinates[1 ,7] == np.array([0.0,  -0.5])).all()
        assert (image_sub_coordinates[1 ,8] == np.array([0.5, -0.5])).all()

        assert (image_sub_coordinates[2 ,0] == np.array([1.5, -1.5])).all()
        assert (image_sub_coordinates[2 ,1] == np.array([2.0, -1.5])).all()
        assert (image_sub_coordinates[2 ,2] == np.array([2.5, -1.5])).all()
        assert (image_sub_coordinates[2 ,3] == np.array([1.5, -2.0])).all()
        assert (image_sub_coordinates[2 ,4] == np.array([2.0, -2.0])).all()
        assert (image_sub_coordinates[2 ,5] == np.array([2.5, -2.0])).all()
        assert (image_sub_coordinates[2 ,6] == np.array([1.5, -2.5])).all()
        assert (image_sub_coordinates[2 ,7] == np.array([2.0, -2.5])).all()
        assert (image_sub_coordinates[2 ,8] == np.array([2.5, -2.5])).all()

    def test__4x4_mask_with_one_pixel__4x4_sub_grid__coordinates(self):

        mask_array = np.array([[True, True, True, True],
                         [True, False, False, True],
                         [True, False, False, True],
                         [True, True, True, False]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=2.0)

        image_sub_coordinates = analysis_data.setup_sub_coordinates(mask=mask, sub_grid_size=4)

        image_sub_coordinates = np.round(image_sub_coordinates, decimals=1)

        assert (image_sub_coordinates[0 ,0] == np.array([-1.6, 1.6])).all()
        assert (image_sub_coordinates[0 ,1] == np.array([-1.2, 1.6])).all()
        assert (image_sub_coordinates[0 ,2] == np.array([-0.8, 1.6])).all()
        assert (image_sub_coordinates[0 ,3] == np.array([-0.4, 1.6])).all()
        assert (image_sub_coordinates[0 ,4] == np.array([-1.6, 1.2])).all()
        assert (image_sub_coordinates[0 ,5] == np.array([-1.2, 1.2])).all()
        assert (image_sub_coordinates[0 ,6] == np.array([-0.8, 1.2])).all()
        assert (image_sub_coordinates[0 ,7] == np.array([-0.4, 1.2])).all()
        assert (image_sub_coordinates[0 ,8] == np.array([-1.6, 0.8])).all()
        assert (image_sub_coordinates[0 ,9] == np.array([-1.2, 0.8])).all()
        assert (image_sub_coordinates[0 ,10] == np.array([-0.8, 0.8])).all()
        assert (image_sub_coordinates[0 ,11] == np.array([-0.4, 0.8])).all()
        assert (image_sub_coordinates[0 ,12] == np.array([-1.6, 0.4])).all()
        assert (image_sub_coordinates[0 ,13] == np.array([-1.2, 0.4])).all()
        assert (image_sub_coordinates[0 ,14] == np.array([-0.8, 0.4])).all()
        assert (image_sub_coordinates[0 ,15] == np.array([-0.4, 0.4])).all()

        assert (image_sub_coordinates[1 ,0] == np.array([0.4, 1.6])).all()
        assert (image_sub_coordinates[1 ,1] == np.array([0.8, 1.6])).all()
        assert (image_sub_coordinates[1 ,2] == np.array([1.2, 1.6])).all()
        assert (image_sub_coordinates[1 ,3] == np.array([1.6, 1.6])).all()
        assert (image_sub_coordinates[1 ,4] == np.array([0.4, 1.2])).all()
        assert (image_sub_coordinates[1 ,5] == np.array([0.8, 1.2])).all()
        assert (image_sub_coordinates[1 ,6] == np.array([1.2, 1.2])).all()
        assert (image_sub_coordinates[1 ,7] == np.array([1.6, 1.2])).all()
        assert (image_sub_coordinates[1 ,8] == np.array([0.4, 0.8])).all()
        assert (image_sub_coordinates[1 ,9] == np.array([0.8, 0.8])).all()
        assert (image_sub_coordinates[1 ,10] == np.array([1.2, 0.8])).all()
        assert (image_sub_coordinates[1 ,11] == np.array([1.6, 0.8])).all()
        assert (image_sub_coordinates[1 ,12] == np.array([0.4, 0.4])).all()
        assert (image_sub_coordinates[1 ,13] == np.array([0.8, 0.4])).all()
        assert (image_sub_coordinates[1 ,14] == np.array([1.2, 0.4])).all()
        assert (image_sub_coordinates[1 ,15] == np.array([1.6, 0.4])).all()

        assert (image_sub_coordinates[2, 0] == np.array([-1.6, -0.4])).all()
        assert (image_sub_coordinates[2, 1] == np.array([-1.2, -0.4])).all()
        assert (image_sub_coordinates[2, 2] == np.array([-0.8, -0.4])).all()
        assert (image_sub_coordinates[2, 3] == np.array([-0.4, -0.4])).all()
        assert (image_sub_coordinates[2, 4] == np.array([-1.6, -0.8])).all()
        assert (image_sub_coordinates[2, 5] == np.array([-1.2, -0.8])).all()
        assert (image_sub_coordinates[2, 6] == np.array([-0.8, -0.8])).all()
        assert (image_sub_coordinates[2, 7] == np.array([-0.4, -0.8])).all()
        assert (image_sub_coordinates[2, 8] == np.array([-1.6, -1.2])).all()
        assert (image_sub_coordinates[2, 9] == np.array([-1.2, -1.2])).all()
        assert (image_sub_coordinates[2, 10] == np.array([-0.8, -1.2])).all()
        assert (image_sub_coordinates[2, 11] == np.array([-0.4, -1.2])).all()
        assert (image_sub_coordinates[2, 12] == np.array([-1.6, -1.6])).all()
        assert (image_sub_coordinates[2, 13] == np.array([-1.2, -1.6])).all()
        assert (image_sub_coordinates[2, 14] == np.array([-0.8, -1.6])).all()
        assert (image_sub_coordinates[2, 15] == np.array([-0.4, -1.6])).all()

        assert (image_sub_coordinates[3, 0] == np.array([0.4, -0.4])).all()
        assert (image_sub_coordinates[3, 1] == np.array([0.8, -0.4])).all()
        assert (image_sub_coordinates[3, 2] == np.array([1.2, -0.4])).all()
        assert (image_sub_coordinates[3, 3] == np.array([1.6, -0.4])).all()
        assert (image_sub_coordinates[3, 4] == np.array([0.4, -0.8])).all()
        assert (image_sub_coordinates[3, 5] == np.array([0.8, -0.8])).all()
        assert (image_sub_coordinates[3, 6] == np.array([1.2, -0.8])).all()
        assert (image_sub_coordinates[3, 7] == np.array([1.6, -0.8])).all()
        assert (image_sub_coordinates[3, 8] == np.array([0.4, -1.2])).all()
        assert (image_sub_coordinates[3, 9] == np.array([0.8, -1.2])).all()
        assert (image_sub_coordinates[3, 10] == np.array([1.2, -1.2])).all()
        assert (image_sub_coordinates[3, 11] == np.array([1.6, -1.2])).all()
        assert (image_sub_coordinates[3, 12] == np.array([0.4, -1.6])).all()
        assert (image_sub_coordinates[3, 13] == np.array([0.8, -1.6])).all()
        assert (image_sub_coordinates[3, 14] == np.array([1.2, -1.6])).all()
        assert (image_sub_coordinates[3, 15] == np.array([1.6, -1.6])).all()

        assert (image_sub_coordinates[4, 0] == np.array([2.4, -2.4])).all()
        assert (image_sub_coordinates[4, 1] == np.array([2.8, -2.4])).all()
        assert (image_sub_coordinates[4, 2] == np.array([3.2, -2.4])).all()
        assert (image_sub_coordinates[4, 3] == np.array([3.6, -2.4])).all()
        assert (image_sub_coordinates[4, 4] == np.array([2.4, -2.8])).all()
        assert (image_sub_coordinates[4, 5] == np.array([2.8, -2.8])).all()
        assert (image_sub_coordinates[4, 6] == np.array([3.2, -2.8])).all()
        assert (image_sub_coordinates[4, 7] == np.array([3.6, -2.8])).all()
        assert (image_sub_coordinates[4, 8] == np.array([2.4, -3.2])).all()
        assert (image_sub_coordinates[4, 9] == np.array([2.8, -3.2])).all()
        assert (image_sub_coordinates[4, 10] == np.array([3.2, -3.2])).all()
        assert (image_sub_coordinates[4, 11] == np.array([3.6, -3.2])).all()
        assert (image_sub_coordinates[4, 12] == np.array([2.4, -3.6])).all()
        assert (image_sub_coordinates[4, 13] == np.array([2.8, -3.6])).all()
        assert (image_sub_coordinates[4, 14] == np.array([3.2, -3.6])).all()
        assert (image_sub_coordinates[4, 15] == np.array([3.6, -3.6])).all()

    def test__4x3_mask_with_one_pixel__2x2_sub_grid__coordinates(self):

        mask_array = np.array([[True, True, True],
                         [True, False, True],
                         [True, False, False],
                         [False, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        image_sub_coordinates = analysis_data.setup_sub_coordinates(mask=mask, sub_grid_size=2)

        assert (image_sub_coordinates[0,0] == np.array([-0.5, 2.0])).all()
        assert (image_sub_coordinates[0,1] == np.array([ 0.5, 2.0])).all()
        assert (image_sub_coordinates[0,2] == np.array([-0.5, 1.0])).all()
        assert (image_sub_coordinates[0,3] == np.array([ 0.5, 1.0])).all()

        assert (image_sub_coordinates[1,0] == np.array([-0.5, -1.0])).all()
        assert (image_sub_coordinates[1,1] == np.array([ 0.5, -1.0])).all()
        assert (image_sub_coordinates[1,2] == np.array([-0.5, -2.0])).all()
        assert (image_sub_coordinates[1,3] == np.array([ 0.5, -2.0])).all()

        assert (image_sub_coordinates[2,0] == np.array([2.5, -1.0])).all()
        assert (image_sub_coordinates[2,1] == np.array([3.5, -1.0])).all()
        assert (image_sub_coordinates[2,2] == np.array([2.5, -2.0])).all()
        assert (image_sub_coordinates[2,3] == np.array([3.5, -2.0])).all()

        assert (image_sub_coordinates[3,0] == np.array([-3.5, -4.0])).all()
        assert (image_sub_coordinates[3,1] == np.array([-2.5, -4.0])).all()
        assert (image_sub_coordinates[3,2] == np.array([-3.5, -5.0])).all()
        assert (image_sub_coordinates[3,3] == np.array([-2.5, -5.0])).all()

    def test__3x4_mask_with_one_pixel__2x2_sub_grid__coordinates(self):

        mask_array = np.array([[True, True, True, False],
                         [True, False, False, True],
                         [False, True, False, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        image_sub_coordinates = analysis_data.setup_sub_coordinates(mask=mask, sub_grid_size=2)

        assert (image_sub_coordinates[0,0] == np.array([4.0, 3.5])).all()
        assert (image_sub_coordinates[0,1] == np.array([5.0, 3.5])).all()
        assert (image_sub_coordinates[0,2] == np.array([4.0, 2.5])).all()
        assert (image_sub_coordinates[0,3] == np.array([5.0, 2.5])).all()

        assert (image_sub_coordinates[1,0] == np.array([-2.0, 0.5])).all()
        assert (image_sub_coordinates[1,1] == np.array([-1.0, 0.5])).all()
        assert (image_sub_coordinates[1,2] == np.array([-2.0, -0.5])).all()
        assert (image_sub_coordinates[1,3] == np.array([-1.0, -0.5])).all()

        assert (image_sub_coordinates[2,0] == np.array([1.0, 0.5])).all()
        assert (image_sub_coordinates[2,1] == np.array([2.0, 0.5])).all()
        assert (image_sub_coordinates[2,2] == np.array([1.0, -0.5])).all()
        assert (image_sub_coordinates[2,3] == np.array([2.0, -0.5])).all()

        assert (image_sub_coordinates[3,0] == np.array([-5.0, -2.5])).all()
        assert (image_sub_coordinates[3,1] == np.array([-4.0, -2.5])).all()
        assert (image_sub_coordinates[3,2] == np.array([-5.0, -3.5])).all()
        assert (image_sub_coordinates[3,3] == np.array([-4.0, -3.5])).all()

        assert (image_sub_coordinates[4,0] == np.array([1.0, -2.5])).all()
        assert (image_sub_coordinates[4,1] == np.array([2.0, -2.5])).all()
        assert (image_sub_coordinates[4,2] == np.array([1.0, -3.5])).all()
        assert (image_sub_coordinates[4,3] == np.array([2.0, -3.5])).all()


class TestBlurringRegion(object):

    class TestBlurringMask:

        def test__size__3x3_small_mask(self):
    
            mask_array = np.array([[True, True, True],
                                  [True, False, True],
                                  [True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(3, 3))
    
            assert (blurring_mask == np.array([[False, False, False],
                                               [False, True, False],
                                               [False, False, False]])).all()
    
        def test__size__3x3__large_mask(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, False, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(3, 3))
    
            assert (blurring_mask == np.array([[True, True, True, True, True, True, True],
                                                 [True, True, True, True, True, True, True],
                                                 [True, True, False, False, False, True, True],
                                                 [True, True, False, True, False, True, True],
                                                 [True, True, False, False, False, True, True],
                                                 [True, True, True, True, True, True, True],
                                                 [True, True, True, True, True, True, True]])).all()
    
        def test__size__5x5__large_mask(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True],
                                     [True, True, True, False, True, True, True],
                                     [True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(5, 5))

            assert (blurring_mask == np.array([[True, True, True, True, True, True, True],
                                               [True, False, False, False, False, False, True],
                                               [True, False, False, False, False, False, True],
                                               [True, False, False, True, False, False,  True],
                                               [True, False, False, False, False, False,  True],
                                               [True, False, False, False, False, False,  True],
                                               [True, True, True, True, True, True, True]])).all()

        def test__size__5x3__large_mask(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, False, True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(5, 3))

            assert (blurring_mask == np.array([[True, True, True, True, True, True, True],
                                                 [True, True, True, True, True, True, True],
                                                 [True, False, False, False, False, False, True],
                                                 [True, False, False, True, False, False, True],
                                                 [True, False, False, False, False, False, True],
                                                 [True, True, True, True, True, True, True],
                                                 [True, True, True, True, True, True, True]])).all()
    
        def test__size__3x5__large_mask(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, False, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(3, 5))

            assert (blurring_mask == np.array([[True, True, True, True, True, True, True],
                                                 [True, True, False, False, False, True, True],
                                                 [True, True, False, False, False, True, True],
                                                 [True, True, False, True, False, True, True],
                                                 [True, True, False, False, False, True, True],
                                                 [True, True, False, False, False, True, True],
                                                 [True, True, True, True, True, True, True]])).all()
    
        def test__size__3x3__multiple_points(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True,  True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True],
                             [True, True, True, True, True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(3, 3))

            assert (blurring_mask == np.array([[False, False, False, True, False, False, False],
                                               [False, True,  False, True, False, True,  False],
                                               [False, False, False, True, False, False, False],
                                               [True,  True,  True,  True, True,  True,  True],
                                               [False, False, False, True, False, False, False],
                                               [False, True,  False, True, False, True,  False],
                                               [False, False, False, True, False, False, False]])).all()

        def test__size__5x5__multiple_points(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True,  True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(5, 5))

            assert (blurring_mask == np.array([[False, False, False, False, False, False, False, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, True, False, False, False, True, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, True, False, False, False, True, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, False, False, False, False, False, False, False]])).all()
    
        def test__size__5x3__multiple_points(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True,  True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(5, 3))

            assert (blurring_mask == np.array([[True,   True,  True,  True,  True,  True,  True,  True,  True],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, True, False, False, False, True, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [True,   True,  True,  True,  True,  True,  True,  True,  True],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, True, False, False, False, True, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [True,   True,  True,  True,  True,  True,  True,  True,  True]])).all()
    
        def test__size__3x5__multiple_points(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True,  True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(3, 5))

            assert (blurring_mask == np.array([[True, False, False,  False, True,  False,  False,  False, True],
                                               [True, False, False,  False, True,  False,  False,  False, True],
                                               [True, False,  True,  False, True,  False,   True,  False, True],
                                               [True, False, False,  False, True,  False,  False,  False, True],
                                               [True, False, False,  False, True,  False,  False,  False, True],
                                               [True, False, False,  False, True,  False,  False,  False, True],
                                               [True, False,  True,  False, True,  False,   True,  False, True],
                                               [True, False, False,  False, True,  False,  False,  False, True],
                                               [True, False, False,  False, True,  False,  False,  False, True]])).all()
    
        def test__size__3x3__even_sized_image(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True,  True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(3, 3))
            
            assert (blurring_mask == np.array([[False, False, False, True, False, False, False, True],
                                               [False,  True, False, True, False, True,  False, True],
                                               [False, False, False, True, False, False, False, True],
                                               [True,   True,  True, True,  True,  True,  True, True],
                                               [False, False, False, True, False, False, False, True],
                                               [False,  True, False, True, False,  True, False, True],
                                               [False, False, False, True, False, False, False, True],
                                               [ True,  True,  True, True,  True,  True,  True, True]])).all()

        def test__size__5x5__even_sized_image(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True, True],
                                  [True, True, True, True, True, True, True, True],
                                  [True, True, True, True, True, True, True, True],
                                  [True, True, True, True, True, True, True, True],
                                  [True, True, True, True, True, True, True, True],
                                  [True, True, True, True, True, False, True, True],
                                  [True, True, True, True, True, True, True, True],
                                  [True, True, True, True, True, True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(5, 5))

            assert (blurring_mask == np.array([[True, True, True,  True, True, True, True, True],
                                                 [True, True, True, True, True, True, True, True],
                                                 [True, True, True, True, True, True, True, True],
                                                 [True, True, True, False, False, False, False, False],
                                                 [True, True, True, False, False, False, False, False],
                                                 [True, True, True, False, False, True, False, False],
                                                 [True, True, True, False, False, False, False, False],
                                                 [True, True, True, False, False, False, False, False]])).all()
    
        def test__size__3x3__rectangular_8x9_image(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True,  True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True, True, True],
                             [True, True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(3, 3))
    
            assert (blurring_mask == np.array([[False, False, False, True,  False,  False,  False, True, True],
                                               [False,  True,  False, True,  False, True,  False,  True, True],
                                                [False, False, False, True,  False, False, False,  True, True],
                                                [True,   True,  True, True, True,   True,  True,  True, True],
                                                [False, False, False, True,  False, False, False,  True, True],
                                                [False,  True,  False, True, False,  True,  False,  True, True],
                                                [False, False, False, True,  False, False, False,  True, True],
                                                [True,  True,  True,  True,  True,  True,   True,  True, True]])).all()
    
        def test__size__3x3__rectangular_9x8_image(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True,  True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, False, True, True, True, False, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(3, 3))

            assert (blurring_mask == np.array([[False, False, False, True, False, False, False, True],
                                               [False, True,  False, True, False,  True, False, True],
                                               [False, False, False, True, False, False, False, True],
                                               [True,   True,  True, True,  True,  True,  True, True],
                                               [False, False, False, True, False, False, False, True],
                                               [False,  True, False, True, False,  True, False, True],
                                               [False, False, False, True, False, False, False, True],
                                               [True,   True,  True, True,  True,  True,  True, True],
                                               [True,   True,  True, True,  True,  True,  True, True]])).all()

        def test__size__5x5__multiple_points__mask_extends_beyond_border_so_raises_mask_exception(self):
    
            mask_array = np.array([[True, True, True, True, True, True, True],
                                 [True, False, True, True, True, False, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, True,  True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, False, True, True, True, False, True],
                                 [True, True, True, True, True, True, True]])
    
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)
    
            with pytest.raises(image.MaskException):
                blurring_mask = analysis_data.setup_blurring_mask(mask, psf_size=(5, 5))

    class TestBlurringCoordinates:
        
        def test__3x3_blurring_mask_correct_coordinates(self):

            mask_array = np.array([[True, True, True],
                                   [True, False, True],
                                   [True, True, True]])

            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            blurring_coordinates = analysis_data.setup_blurring_coordinates(mask, psf_size=(3, 3))

            assert (blurring_coordinates[0] == np.array([-3.0, 3.0])).all()
            assert (blurring_coordinates[1] == np.array([0.0, 3.0])).all()
            assert (blurring_coordinates[2] == np.array([3.0, 3.0])).all()
            assert (blurring_coordinates[3] == np.array([-3.0, 0.0])).all()
            assert (blurring_coordinates[4] == np.array([3.0, 0.0])).all()
            assert (blurring_coordinates[5] == np.array([-3.0, -3.0])).all()
            assert (blurring_coordinates[6] == np.array([0.0, -3.0])).all()
            assert (blurring_coordinates[7] == np.array([3.0, -3.0])).all()

        def test__3x5_blurring_mask_correct_coordinates(self):

            mask_array = np.array([[True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, False, True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True]])

            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            # Blurring mask

            #[[True, True, True, True, True, True, True],
            # [True, True, False, False, False, True, True],
            # [True, True, False, False, False, True, True],
            # [True, True, False, True, False, True, True],
            # [True, True, False, False, False, True, True],
            # [True, True, False, False, False, True, True],
            # [True, True, True, True, True, True, True]])

            blurring_coordinates = analysis_data.setup_blurring_coordinates(mask, psf_size=(3, 5))

            assert (blurring_coordinates[0] ==  np.array([-3.0,  6.0])).all()
            assert (blurring_coordinates[1] ==  np.array([ 0.0,  6.0])).all()
            assert (blurring_coordinates[2] ==  np.array([ 3.0,  6.0])).all()
            assert (blurring_coordinates[3] ==  np.array([-3.0,  3.0])).all()
            assert (blurring_coordinates[4] ==  np.array([ 0.0,  3.0])).all()
            assert (blurring_coordinates[5] ==  np.array([ 3.0,  3.0])).all()
            assert (blurring_coordinates[6] ==  np.array([-3.0,  0.0])).all()
            assert (blurring_coordinates[7] ==  np.array([ 3.0,  0.0])).all()
            assert (blurring_coordinates[8] ==  np.array([-3.0, -3.0])).all()
            assert (blurring_coordinates[9] ==  np.array([ 0.0, -3.0])).all()
            assert (blurring_coordinates[10] == np.array([ 3.0, -3.0])).all()
            assert (blurring_coordinates[11] == np.array([-3.0, -6.0])).all()
            assert (blurring_coordinates[12] == np.array([ 0.0, -6.0])).all()
            assert (blurring_coordinates[13] == np.array([ 3.0, -6.0])).all()

        def test__5x3_blurring_mask_correct_coordinates(self):

            mask_array = np.array([[True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, False, True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True]])

            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            # Blurring mask

            # [[True, True, True, True, True, True, True],
            #  [True, True, True, True, True, True, True],
            #  [True, False, False, False, False, False, True],
            #  [True, False, False, True, False, False, True],
            #  [True, False, False, False, False, False, True],
            #  [True, True, True, True, True, True, True],
            #  [True, True, True, True, True, True, True]]

            blurring_coordinates = analysis_data.setup_blurring_coordinates(mask, psf_size=(5, 3))

            assert (blurring_coordinates[0] ==  np.array([-6.0,  3.0])).all()
            assert (blurring_coordinates[1] ==  np.array([-3.0,  3.0])).all()
            assert (blurring_coordinates[2] ==  np.array([ 0.0,  3.0])).all()
            assert (blurring_coordinates[3] ==  np.array([ 3.0,  3.0])).all()
            assert (blurring_coordinates[4] ==  np.array([ 6.0,  3.0])).all()
            assert (blurring_coordinates[5] ==  np.array([-6.0,  0.0])).all()
            assert (blurring_coordinates[6] ==  np.array([-3.0,  0.0])).all()
            assert (blurring_coordinates[7] ==  np.array([ 3.0,  0.0])).all()
            assert (blurring_coordinates[8] ==  np.array([ 6.0,  0.0])).all()
            assert (blurring_coordinates[9] ==  np.array([-6.0, -3.0])).all()
            assert (blurring_coordinates[10] == np.array([-3.0, -3.0])).all()
            assert (blurring_coordinates[11] == np.array([ 0.0, -3.0])).all()
            assert (blurring_coordinates[12] == np.array([ 3.0, -3.0])).all()
            assert (blurring_coordinates[13] == np.array([ 6.0, -3.0])).all()


class TestBorderPixels(object):

    def test__7x7_mask_one_central_pixel__is_entire_border(self):

        mask_array = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, False,  True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        border_pixels = analysis_data.setup_border_pixels(mask)

        assert (border_pixels == np.array([0])).all()

    def test__7x7_mask_nine_central_pixels__is_border(self):

        mask_array = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        border_pixels = analysis_data.setup_border_pixels(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

    def test__7x7_mask_rectangle_of_fifteen_central_pixels__is_border(self):

        mask_array = np.array([[True, True, True, True, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        border_pixels = analysis_data.setup_border_pixels(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14])).all()

    def test__8x7_mask_add_edge_pixels__also_in_border(self):

        mask_array = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, False, False, False, False, False, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        border_pixels = analysis_data.setup_border_pixels(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17])).all()

    def test__8x7_mask_big_square(self):
        mask_array = np.array([[True, True, True, True, True, True, True],
                         [True, False,False, False, False,False, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, True, True, True, True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        border_pixels = analysis_data.setup_border_pixels(mask)

        assert (border_pixels == np.array
            ([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 26, 27, 28, 29])).all()

    def test__7x8_mask_add_edge_pixels__also_in_border(self):

        mask_array = np.array([[True, True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True, True],
                         [True, True, False, False, False, True, True, True],
                         [True, True, False, False, False, True, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, True, False, False, False, True, True, True],
                         [True, True, True, True, True, True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        border_pixels = analysis_data.setup_border_pixels(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14])).all()

    def test__7x8_mask_big_square(self):

        mask_array = np.array([[True, True, True, True, True, True, True, True],
                         [True, False,False, False, False,False, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, True, True, True, True, True, True, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        border_pixels = analysis_data.setup_border_pixels(mask)

        assert (border_pixels == np.array
            ([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24])).all()


class TestSparsePixels(object):

    class TestSetupAll:

        def test__7x7_circle_mask__five_central_pixels__sparse_grid_size_1(self):

            mask_array = np.array([[True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, False,False,False,False,False, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=1)

            assert (sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__7x7_circle_mask__sparse_grid_size_1(self):

            mask_array = np.array([[True, True, True, True, True, True, True],
                             [True, True, False, False, False, True, True],
                             [True,False, False, False,False, False, True],
                             [True, False, False, False,False, False, True],
                             [True, False, False, False,False, False, True],
                             [True, True, False, False, False, True, True],
                             [True, True, True, True, True, True, True]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=1)

            assert (sparse_to_image == np.arange(21)).all()
            assert (image_to_sparse == np.arange(21)).all()

        def test__7x7_rectangle_mask__sparse_grid_size_1(self):

            mask_array = np.array([[False, False, False, False, False, False, False],
                             [False, False, False,  False,  False, False, False],
                             [False, False, False,  False,  False, False, False],
                             [False, False, False,  False,  False, False, False],
                             [False, False, False,  False,  False, False, False],
                             [False, False, False,  False, False, False, False],
                             [False, False, False, False, False, False, False]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=1)

            assert (sparse_to_image == np.arange(49)).all()
            assert (image_to_sparse == np.arange(49)).all()

        def test__7x7_circle_mask__sparse_grid_size_2(self):

            mask_array = np.array([[True, True, True, True, True, True, True],
                             [True, True, False, False, False, True, True],
                             [True, False, False, False,False, False, True],
                             [True, False, False, False,False, False, True],
                             [True, False, False, False,False, False, True],
                             [True, True, False, False, False, True, True],
                             [True, True, True, True, True, True, True]])

            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=2)

            assert (sparse_to_image == np.array([4, 6, 14, 16])).all()
            assert (image_to_sparse == np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
                                                 1, 2, 2, 2, 3, 3, 2, 2, 3])).all()

        def test__8x8_sporadic_mask__sparse_grid_size_2(self):

            mask_array = np.array([[True, True, True, True, True, True, False, False],
                             [True, True, False, False, False, True, False, False],
                             [True, False, False, False,False, False, False,False],
                             [True, False, False, False, False, False, False, False],
                             [True, False, False, False, False, False, False, False],
                             [True, True, False, False, False, True, False, False],
                             [True, True, True, True, True, True, False, False],
                             [True, True, False, False, False, True, False, False]])

            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=2)

            assert (sparse_to_image == np.array([0, 8, 10, 12, 22, 24, 26, 33])).all()
            assert (image_to_sparse == np.array([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 2, 2, 3, 3,
                                                 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 4, 4, 5, 6, 6,
                                                 7, 7, 4, 4, 7, 7, 7])).all()

        def test__7x7_circle_mask_trues_on_even_values__sparse_grid_size_2(self):

            mask_array = np.array([[False, True, False, True, False, True, False],
                             [True,  True, True,  True, True,  True, True],
                             [False, True, False, True, False, True, False],
                             [True,  True, True,  True, True,  True, True],
                             [False, True, False, True, False, True, False],
                             [True,  True, True,  True, True,  True, True],
                             [False, True, False, True, False, True, False]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=2)

            assert (sparse_to_image == np.arange(16)).all()
            assert (image_to_sparse == np.arange(16)).all()

        def test__7x7_circle_mask__sparse_grid_size_3(self):

            mask_array = np.array([[True, True, True, True, True, True, True],
                             [True, True, False, False, False, True, True],
                             [True, False, False, False, False, False, True],
                             [True, False, False, False, False, False, True],
                             [True, False, False, False, False, False, True],
                             [True, True, False, False, False, True, True],
                             [True, True, True, True, True, True, True]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=3)

            assert (sparse_to_image == np.array([10])).all()
            assert (image_to_sparse == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all()

        def test__7x7_circle_mask_more_points_added__sparse_grid_size_3(self):

            mask_array = np.array([[False, True, True, False, True, False, False],
                             [True, True, False, False, False, True, True],
                             [True, False, False, False, False, False, True],
                             [True, False, False, False, False, False, False],
                             [True, False, False, False, False, False, True],
                             [True, True, False, False, False, True, True],
                             [True, True, True, True, True, True, False]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=3)

            assert (sparse_to_image == np.array([0, 1, 3, 14, 17, 26])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 2, 1, 1, 1, 0, 3, 3, 3, 4, 3, 3, 3, 3, 4, 4, 3, 3, 3,
                                                 3, 4, 3, 3, 3, 5])).all()

        def test__7x7_mask_trues_on_values_which_divide_by_3__sparse_grid_size_3(self):

            mask_array = np.array([[False,  True, True, False, True, True, False],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [False, True, True,  False, True, True,  False],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [False,  True, True, False, True, True, False]])

            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=3)

            assert (sparse_to_image == np.arange(9)).all()
            assert (image_to_sparse == np.arange(9)).all()

        def test__8x8_mask_trues_on_values_which_divide_by_3_and_other_values__sparse_grid_size_3(self):

            mask_array = np.array([[False,  True, False, False, True, True, False],
                             [True, True, True, True, True, True, True],
                             [True, True, False, False, False, True, True],
                             [False, True, True,  False, True, True,  False],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [False, False, False, False, False, False, False]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=3)

            assert (sparse_to_image == np.array([0, 2, 3, 7, 8, 9, 10, 13, 16])).all()
            assert (image_to_sparse == np.array([0, 1, 1, 2, 4, 4, 4, 3, 4, 5, 6, 6, 7, 7, 7, 8, 8])).all()

        def test__8x7__five_central_pixels__sparse_grid_size_1(self):

            mask_array = np.array([[True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, False, False, False, False, False, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=1)

            assert (sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__8x7__five_central_pixels_2__sparse_grid_size_1(self):

            mask_array = np.array([[True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, False, False, False, False, False, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=1)

            assert (sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__8x7__five_central_pixels__sparse_grid_size_2(self):

            mask_array = np.array([[True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, False, False, False, False, False, True],
                             [True, False, False, False, False, False, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=2)

            assert (sparse_to_image == np.array([1, 3])).all()
            assert (image_to_sparse == np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])).all()

        def test__7x8__five_central_pixels__sparse_grid_size_1(self):

            mask_array = np.array([[True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, False, False, False, False, False, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=1)

            assert (sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__7x8__five_central_pixels__sparse_grid_size_2(self):

            mask_array = np.array([[True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, False, False, False, False, False, True, True],
                             [True, False, False, False, False, False, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True],
                             [True, True, True, True, True, True, True, True]])
                        
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=2)

            assert (sparse_to_image == np.array([1, 3])).all()
            assert (image_to_sparse == np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])).all()

        def test__7x8__more_central_pixels__sparse_grid_size_2(self):

            mask_array = np.array([[True, True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True, True],
                                 [True, False, False, False, False, False, True, True],
                                 [True, False, False, False, False, False, True, True],
                                 [True, False, False, False, False, False, True, True],
                                 [True, True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True, True]])
            
            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = analysis_data.setup_sparse_pixels(mask, sparse_grid_size=2)

            assert (sparse_to_image == np.array([1, 3, 11, 13])).all()
            assert (image_to_sparse == np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3])).all()


class TestAnalysisArray:

    def test__sets_up_array_fully_unmasked__maps_back_to_2d(self):

        array = np.ones((2,2))
        mask = image.Mask.from_array(mask_array=np.zeros((2, 2)), pixel_scale=1.0)

        analysis_array = analysis_data.AnalysisArray(mask, array)

        assert (analysis_array == np.array([1, 1, 1, 1])).all()
        assert (analysis_array.map_to_2d() == np.ones((2,2))).all()

    def test__set_up_array_with_mask__maps_back_to_2d(self):

        array = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])

        mask_array = np.array([[True,  False, True],
                               [False, False, False],
                                [True, False, True]])

        mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=1.0)

        analysis_array = analysis_data.AnalysisArray(mask, array)

        assert (analysis_array == np.array([2.0, 4.0, 5.0, 6.0, 8.0])).all()

        assert (analysis_array.map_to_2d() == np.array([[0.0, 2.0, 0.0],
                                                        [4.0, 5.0, 6.0],
                                                        [0.0, 8.0, 0.0]])).all()


class TestAnalysisImage:

    class TestConstructor:

        def test__sets_up_image_fully_unmasked__maps_back_to_2d(self):

            image_data = np.ones((2, 2))
            mask = image.Mask.from_array(mask_array=np.zeros((2, 2)), pixel_scale=1.0)

            analysis_image = analysis_data.AnalysisImage(mask, image_data)

            assert (analysis_image == np.array([1, 1, 1, 1])).all()
            assert (analysis_image.map_to_2d() == np.ones((2, 2))).all()

        def test__set_up_image_with_mask__maps_back_to_2d(self):

            image_data = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])

            mask_array = np.array([[True, False, True],
                                   [False, False, False],
                                   [True, False, True]])

            mask = image.Mask.from_array(mask_array=mask_array, pixel_scale=1.0)

            analysis_image = analysis_data.AnalysisArray(mask, image_data)

            assert (analysis_image == np.array([2.0, 4.0, 5.0, 6.0, 8.0])).all()

            assert (analysis_image.map_to_2d() == np.array([[0.0, 2.0, 0.0],
                                                            [4.0, 5.0, 6.0],
                                                            [0.0, 8.0, 0.0]])).all()


class TestAnalyisData(object):

    class TestConstructor:

        def test__4x4_input_all_image_properties__4_central_pixels_unmasked(self):

            test_image = image.Image(array=np.ones((4, 4)), pixel_scale=1.5)
            test_noise = image.Noise.from_array(array=np.ones((4, 4)))
            test_psf = image.PSF.from_array(array=np.ones((3, 3)), renormalize=False)
            test_mask = image.Mask.from_array(mask_array=np.array([[True, True, True, True],
                                                                   [True, False, False, True],
                                                                   [True, False, False, True],
                                                                   [True, True, True, True]]), pixel_scale=1.5)

            adata = analysis_data.AnalysisData(test_mask, test_image, test_noise, test_psf, sub_grid_size=2)

            assert (adata.image == np.array([1, 1, 1, 1])).all()
            assert (adata.image.map_to_2d() == np.array([[0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 1.0, 1.0, 0.0],
                                                         [0.0, 1.0, 1.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0]])).all()

            assert (adata.noise == np.array([1, 1, 1, 1])).all()
            assert (adata.psf == np.ones((3, 3))).all()
            assert (adata.coordinates == np.array([[-0.75, 0.75], [0.75, 0.75], [-0.75, -0.75], [0.75, -0.75]])).all()
            assert (adata.sub_coordinates == analysis_data.setup_sub_coordinates(test_mask, sub_grid_size=2)).all()
            assert (adata.blurring_coordinates == analysis_data.setup_blurring_coordinates(test_mask, psf_size=(3, 3))).all()
            assert (adata.border_pixels == analysis_data.setup_border_pixels(test_mask)).all()