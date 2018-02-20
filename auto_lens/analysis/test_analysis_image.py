from __future__ import division, print_function
import pytest
import numpy as np
import os
from analysis import analysis_image
import image

test_data_dir = "{}/../data/test_data/".format(os.path.dirname(os.path.realpath(__file__)))

class TestAnalysisRegions(object):

    # TODO : Add Rectagular image tests to all masked region tests

    class TestImageSubGrid(object):

        def test__3x3_mask_with_one_pixel__2x2_sub_grid__coordinates(self):

            mask = np.array([[False, False, False],
                             [False, True, False],
                             [False, False, False]])

            image_sub_grid_coordinates = analysis_image.setup_image_sub_grid_coordinates(mask=mask, 
                                                                                         pixel_scale=3.0, sub_grid_size=2)

            assert (image_sub_grid_coordinates == np.array
                ([[[-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]])).all()

            assert (image_sub_grid_coordinates[0 ,0] == np.array([-0.5,  0.5])).all()
            assert (image_sub_grid_coordinates[0 ,1] == np.array([ 0.5,  0.5])).all()
            assert (image_sub_grid_coordinates[0 ,2] == np.array([-0.5, -0.5])).all()
            assert (image_sub_grid_coordinates[0 ,3] == np.array([ 0.5, -0.5])).all()

        def test__3x3_mask_with_row_of_pixels__2x2_sub_grid__coordinates(self):

            mask = np.array([[False, False, False],
                             [True, True, True],
                             [False, False, False]])

            image_sub_grid_coordinates = analysis_image.setup_image_sub_grid_coordinates(mask=mask, pixel_scale=3.0,
                                                                                         sub_grid_size=2)

            assert (image_sub_grid_coordinates == np.array([[[-3.5, 0.5], [-2.5, 0.5], [-3.5, -0.5], [-2.5, -0.5]],
                                                            [[-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]],
                                                            [[2.5, 0.5], [3.5, 0.5], [2.5, -0.5], [3.5, -0.5]]])).all()

            assert (image_sub_grid_coordinates[0 ,0] == np.array([-3.5,  0.5])).all()
            assert (image_sub_grid_coordinates[0 ,1] == np.array([-2.5,  0.5])).all()
            assert (image_sub_grid_coordinates[0 ,2] == np.array([-3.5, -0.5])).all()
            assert (image_sub_grid_coordinates[0 ,3] == np.array([-2.5, -0.5])).all()

            assert (image_sub_grid_coordinates[1 ,0] == np.array([-0.5,  0.5])).all()
            assert (image_sub_grid_coordinates[1 ,1] == np.array([ 0.5,  0.5])).all()
            assert (image_sub_grid_coordinates[1 ,2] == np.array([-0.5, -0.5])).all()
            assert (image_sub_grid_coordinates[1 ,3] == np.array([ 0.5, -0.5])).all()

            assert (image_sub_grid_coordinates[2 ,0] == np.array([2.5,  0.5])).all()
            assert (image_sub_grid_coordinates[2 ,1] == np.array([3.5,  0.5])).all()
            assert (image_sub_grid_coordinates[2 ,2] == np.array([2.5, -0.5])).all()
            assert (image_sub_grid_coordinates[2 ,3] == np.array([3.5, -0.5])).all()

        def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__coordinates(self):

            mask = np.array([[False, False, True],
                             [True,  True,  True],
                             [False, False, True]])

            image_sub_grid_coordinates = analysis_image.setup_image_sub_grid_coordinates(mask=mask, pixel_scale=3.0,
                                                                                         sub_grid_size=2)

            assert (image_sub_grid_coordinates == np.array([[[2.5, 3.5], [3.5, 3.5], [2.5, 2.5], [3.5, 2.5]],
                                                            [[-3.5, 0.5], [-2.5, 0.5], [-3.5, -0.5], [-2.5, -0.5]],
                                                            [[-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]],
                                                            [[2.5, 0.5], [3.5, 0.5], [2.5, -0.5], [3.5, -0.5]],
                                                            [[2.5, -2.5], [3.5, -2.5], [2.5, -3.5], [3.5, -3.5]]])).all()

            assert (image_sub_grid_coordinates[0 ,0] == np.array([2.5,  3.5])).all()
            assert (image_sub_grid_coordinates[0 ,1] == np.array([3.5,  3.5])).all()
            assert (image_sub_grid_coordinates[0 ,2] == np.array([2.5,  2.5])).all()
            assert (image_sub_grid_coordinates[0 ,3] == np.array([3.5,  2.5])).all()

            assert (image_sub_grid_coordinates[1 ,0] == np.array([-3.5,  0.5])).all()
            assert (image_sub_grid_coordinates[1 ,1] == np.array([-2.5,  0.5])).all()
            assert (image_sub_grid_coordinates[1 ,2] == np.array([-3.5, -0.5])).all()
            assert (image_sub_grid_coordinates[1 ,3] == np.array([-2.5, -0.5])).all()

            assert (image_sub_grid_coordinates[2 ,0] == np.array([-0.5,  0.5])).all()
            assert (image_sub_grid_coordinates[2 ,1] == np.array([ 0.5,  0.5])).all()
            assert (image_sub_grid_coordinates[2 ,2] == np.array([-0.5, -0.5])).all()
            assert (image_sub_grid_coordinates[2 ,3] == np.array([ 0.5, -0.5])).all()

            assert (image_sub_grid_coordinates[3 ,0] == np.array([2.5,  0.5])).all()
            assert (image_sub_grid_coordinates[3 ,1] == np.array([3.5,  0.5])).all()
            assert (image_sub_grid_coordinates[3 ,2] == np.array([2.5, -0.5])).all()
            assert (image_sub_grid_coordinates[3 ,3] == np.array([3.5, -0.5])).all()

            assert (image_sub_grid_coordinates[4 ,0] == np.array([2.5, -2.5])).all()
            assert (image_sub_grid_coordinates[4 ,1] == np.array([3.5, -2.5])).all()
            assert (image_sub_grid_coordinates[4 ,2] == np.array([2.5, -3.5])).all()
            assert (image_sub_grid_coordinates[4 ,3] == np.array([3.5, -3.5])).all()

        def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__different_pixel_scale(self):

            mask = np.array([[False, False, True],
                             [True,  True,  True],
                             [False, False, True]])

            image_sub_grid_coordinates = analysis_image.setup_image_sub_grid_coordinates(mask=mask, pixel_scale=0.3,
                                                                                         sub_grid_size=2)

            image_sub_grid_coordinates = np.round(image_sub_grid_coordinates, decimals=2)

            assert (image_sub_grid_coordinates == np.array([[[0.25, 0.35], [0.35, 0.35], [0.25, 0.25], [0.35, 0.25]],
                                                            [[-0.35, 0.05], [-0.25, 0.05], [-0.35, -0.05], [-0.25, -0.05]],
                                                            [[-0.05, 0.05], [0.05, 0.05], [-0.05, -0.05], [0.05, -0.05]],
                                                            [[0.25, 0.05], [0.35, 0.05], [0.25, -0.05], [0.35, -0.05]],
                                                            [[0.25, -0.25], [0.35, -0.25], [0.25, -0.35], [0.35, -0.35]]])).all()

            assert (image_sub_grid_coordinates[0 ,0] == np.array([0.25,  0.35])).all()
            assert (image_sub_grid_coordinates[0 ,1] == np.array([0.35,  0.35])).all()
            assert (image_sub_grid_coordinates[0 ,2] == np.array([0.25,  0.25])).all()
            assert (image_sub_grid_coordinates[0 ,3] == np.array([0.35,  0.25])).all()

            assert (image_sub_grid_coordinates[1 ,0] == np.array([-0.35,  0.05])).all()
            assert (image_sub_grid_coordinates[1 ,1] == np.array([-0.25,  0.05])).all()
            assert (image_sub_grid_coordinates[1 ,2] == np.array([-0.35, -0.05])).all()
            assert (image_sub_grid_coordinates[1 ,3] == np.array([-0.25, -0.05])).all()

            assert (image_sub_grid_coordinates[2 ,0] == np.array([-0.05,  0.05])).all()
            assert (image_sub_grid_coordinates[2 ,1] == np.array([ 0.05,  0.05])).all()
            assert (image_sub_grid_coordinates[2 ,2] == np.array([-0.05, -0.05])).all()
            assert (image_sub_grid_coordinates[2 ,3] == np.array([ 0.05, -0.05])).all()

            assert (image_sub_grid_coordinates[3 ,0] == np.array([0.25,  0.05])).all()
            assert (image_sub_grid_coordinates[3 ,1] == np.array([0.35,  0.05])).all()
            assert (image_sub_grid_coordinates[3 ,2] == np.array([0.25, -0.05])).all()
            assert (image_sub_grid_coordinates[3 ,3] == np.array([0.35, -0.05])).all()

            assert (image_sub_grid_coordinates[4 ,0] == np.array([0.25, -0.25])).all()
            assert (image_sub_grid_coordinates[4 ,1] == np.array([0.35, -0.25])).all()
            assert (image_sub_grid_coordinates[4 ,2] == np.array([0.25, -0.35])).all()
            assert (image_sub_grid_coordinates[4 ,3] == np.array([0.35, -0.35])).all()

        def test__3x3_mask_with_one_pixel__3x3_sub_grid__coordinates(self):

            mask = np.array([[False, False, False],
                             [False, True, False],
                             [False, False, False]])

            image_sub_grid_coordinates = analysis_image.setup_image_sub_grid_coordinates(mask=mask, pixel_scale=3.0,
                                                                                         sub_grid_size=3)

            assert (image_sub_grid_coordinates == np.array([[[-0.75, 0.75],  [0.0, 0.75],  [0.75, 0.75],
                                                             [-0.75, 0.0],   [0.0, 0.0],   [0.75, 0.0],
                                                             [-0.75, -0.75], [0.0, -0.75], [0.75, -0.75]]])).all()

            assert (image_sub_grid_coordinates[0 ,0] == np.array([-0.75, 0.75])).all()
            assert (image_sub_grid_coordinates[0 ,1] == np.array([0.0,   0.75])).all()
            assert (image_sub_grid_coordinates[0 ,2] == np.array([0.75,  0.75])).all()
            assert (image_sub_grid_coordinates[0 ,3] == np.array([-0.75, 0.0])).all()
            assert (image_sub_grid_coordinates[0 ,4] == np.array([0.0,   0.0])).all()
            assert (image_sub_grid_coordinates[0 ,5] == np.array([0.75,  0.0])).all()
            assert (image_sub_grid_coordinates[0 ,6] == np.array([-0.75 ,-0.75])).all()
            assert (image_sub_grid_coordinates[0 ,7] == np.array([0.0,  -0.75])).all()
            assert (image_sub_grid_coordinates[0 ,8] == np.array([0.75, -0.75])).all()

        def test__3x3_mask_with_one_row__3x3_sub_grid__coordinates(self):

            mask = np.array([[False, False, True],
                             [False, True, False],
                             [False, False, True]])

            image_sub_grid_coordinates = analysis_image.setup_image_sub_grid_coordinates(mask=mask, pixel_scale=2.0,
                                                                                         sub_grid_size=3)

            assert (image_sub_grid_coordinates[0 ,0] == np.array([1.5, 2.5])).all()
            assert (image_sub_grid_coordinates[0 ,1] == np.array([2.0, 2.5])).all()
            assert (image_sub_grid_coordinates[0 ,2] == np.array([2.5, 2.5])).all()
            assert (image_sub_grid_coordinates[0 ,3] == np.array([1.5, 2.0])).all()
            assert (image_sub_grid_coordinates[0 ,4] == np.array([2.0, 2.0])).all()
            assert (image_sub_grid_coordinates[0 ,5] == np.array([2.5, 2.0])).all()
            assert (image_sub_grid_coordinates[0 ,6] == np.array([1.5, 1.5])).all()
            assert (image_sub_grid_coordinates[0 ,7] == np.array([2.0, 1.5])).all()
            assert (image_sub_grid_coordinates[0 ,8] == np.array([2.5, 1.5])).all()

            assert (image_sub_grid_coordinates[1 ,0] == np.array([-0.5, 0.5])).all()
            assert (image_sub_grid_coordinates[1 ,1] == np.array([0.0,   0.5])).all()
            assert (image_sub_grid_coordinates[1 ,2] == np.array([0.5,  0.5])).all()
            assert (image_sub_grid_coordinates[1 ,3] == np.array([-0.5, 0.0])).all()
            assert (image_sub_grid_coordinates[1 ,4] == np.array([0.0,   0.0])).all()
            assert (image_sub_grid_coordinates[1 ,5] == np.array([0.5,  0.0])).all()
            assert (image_sub_grid_coordinates[1 ,6] == np.array([-0.5 ,-0.5])).all()
            assert (image_sub_grid_coordinates[1 ,7] == np.array([0.0,  -0.5])).all()
            assert (image_sub_grid_coordinates[1 ,8] == np.array([0.5, -0.5])).all()

            assert (image_sub_grid_coordinates[2 ,0] == np.array([1.5, -1.5])).all()
            assert (image_sub_grid_coordinates[2 ,1] == np.array([2.0, -1.5])).all()
            assert (image_sub_grid_coordinates[2 ,2] == np.array([2.5, -1.5])).all()
            assert (image_sub_grid_coordinates[2 ,3] == np.array([1.5, -2.0])).all()
            assert (image_sub_grid_coordinates[2 ,4] == np.array([2.0, -2.0])).all()
            assert (image_sub_grid_coordinates[2 ,5] == np.array([2.5, -2.0])).all()
            assert (image_sub_grid_coordinates[2 ,6] == np.array([1.5, -2.5])).all()
            assert (image_sub_grid_coordinates[2 ,7] == np.array([2.0, -2.5])).all()
            assert (image_sub_grid_coordinates[2 ,8] == np.array([2.5, -2.5])).all()

        def test__4x4_mask_with_one_pixel__4x4_sub_grid__coordinates(self):

            mask = np.array([[False, False, False, False],
                             [False, True, True, False],
                             [False, True, True, False],
                             [False, False, False, True]])

            image_sub_grid_coordinates = analysis_image.setup_image_sub_grid_coordinates(mask=mask, pixel_scale=2.0,
                                                                                         sub_grid_size=4)

            image_sub_grid_coordinates = np.round(image_sub_grid_coordinates, decimals=1)

            assert (image_sub_grid_coordinates[0 ,0] == np.array([-1.6, 1.6])).all()
            assert (image_sub_grid_coordinates[0 ,1] == np.array([-1.2, 1.6])).all()
            assert (image_sub_grid_coordinates[0 ,2] == np.array([-0.8, 1.6])).all()
            assert (image_sub_grid_coordinates[0 ,3] == np.array([-0.4, 1.6])).all()
            assert (image_sub_grid_coordinates[0 ,4] == np.array([-1.6, 1.2])).all()
            assert (image_sub_grid_coordinates[0 ,5] == np.array([-1.2, 1.2])).all()
            assert (image_sub_grid_coordinates[0 ,6] == np.array([-0.8, 1.2])).all()
            assert (image_sub_grid_coordinates[0 ,7] == np.array([-0.4, 1.2])).all()
            assert (image_sub_grid_coordinates[0 ,8] == np.array([-1.6, 0.8])).all()
            assert (image_sub_grid_coordinates[0 ,9] == np.array([-1.2, 0.8])).all()
            assert (image_sub_grid_coordinates[0 ,10] == np.array([-0.8, 0.8])).all()
            assert (image_sub_grid_coordinates[0 ,11] == np.array([-0.4, 0.8])).all()
            assert (image_sub_grid_coordinates[0 ,12] == np.array([-1.6, 0.4])).all()
            assert (image_sub_grid_coordinates[0 ,13] == np.array([-1.2, 0.4])).all()
            assert (image_sub_grid_coordinates[0 ,14] == np.array([-0.8, 0.4])).all()
            assert (image_sub_grid_coordinates[0 ,15] == np.array([-0.4, 0.4])).all()

            assert (image_sub_grid_coordinates[1 ,0] == np.array([0.4, 1.6])).all()
            assert (image_sub_grid_coordinates[1 ,1] == np.array([0.8, 1.6])).all()
            assert (image_sub_grid_coordinates[1 ,2] == np.array([1.2, 1.6])).all()
            assert (image_sub_grid_coordinates[1 ,3] == np.array([1.6, 1.6])).all()
            assert (image_sub_grid_coordinates[1 ,4] == np.array([0.4, 1.2])).all()
            assert (image_sub_grid_coordinates[1 ,5] == np.array([0.8, 1.2])).all()
            assert (image_sub_grid_coordinates[1 ,6] == np.array([1.2, 1.2])).all()
            assert (image_sub_grid_coordinates[1 ,7] == np.array([1.6, 1.2])).all()
            assert (image_sub_grid_coordinates[1 ,8] == np.array([0.4, 0.8])).all()
            assert (image_sub_grid_coordinates[1 ,9] == np.array([0.8, 0.8])).all()
            assert (image_sub_grid_coordinates[1 ,10] == np.array([1.2, 0.8])).all()
            assert (image_sub_grid_coordinates[1 ,11] == np.array([1.6, 0.8])).all()
            assert (image_sub_grid_coordinates[1 ,12] == np.array([0.4, 0.4])).all()
            assert (image_sub_grid_coordinates[1 ,13] == np.array([0.8, 0.4])).all()
            assert (image_sub_grid_coordinates[1 ,14] == np.array([1.2, 0.4])).all()
            assert (image_sub_grid_coordinates[1 ,15] == np.array([1.6, 0.4])).all()

            assert (image_sub_grid_coordinates[2, 0] == np.array([-1.6, -0.4])).all()
            assert (image_sub_grid_coordinates[2, 1] == np.array([-1.2, -0.4])).all()
            assert (image_sub_grid_coordinates[2, 2] == np.array([-0.8, -0.4])).all()
            assert (image_sub_grid_coordinates[2, 3] == np.array([-0.4, -0.4])).all()
            assert (image_sub_grid_coordinates[2, 4] == np.array([-1.6, -0.8])).all()
            assert (image_sub_grid_coordinates[2, 5] == np.array([-1.2, -0.8])).all()
            assert (image_sub_grid_coordinates[2, 6] == np.array([-0.8, -0.8])).all()
            assert (image_sub_grid_coordinates[2, 7] == np.array([-0.4, -0.8])).all()
            assert (image_sub_grid_coordinates[2, 8] == np.array([-1.6, -1.2])).all()
            assert (image_sub_grid_coordinates[2, 9] == np.array([-1.2, -1.2])).all()
            assert (image_sub_grid_coordinates[2, 10] == np.array([-0.8, -1.2])).all()
            assert (image_sub_grid_coordinates[2, 11] == np.array([-0.4, -1.2])).all()
            assert (image_sub_grid_coordinates[2, 12] == np.array([-1.6, -1.6])).all()
            assert (image_sub_grid_coordinates[2, 13] == np.array([-1.2, -1.6])).all()
            assert (image_sub_grid_coordinates[2, 14] == np.array([-0.8, -1.6])).all()
            assert (image_sub_grid_coordinates[2, 15] == np.array([-0.4, -1.6])).all()

            assert (image_sub_grid_coordinates[3, 0] == np.array([0.4, -0.4])).all()
            assert (image_sub_grid_coordinates[3, 1] == np.array([0.8, -0.4])).all()
            assert (image_sub_grid_coordinates[3, 2] == np.array([1.2, -0.4])).all()
            assert (image_sub_grid_coordinates[3, 3] == np.array([1.6, -0.4])).all()
            assert (image_sub_grid_coordinates[3, 4] == np.array([0.4, -0.8])).all()
            assert (image_sub_grid_coordinates[3, 5] == np.array([0.8, -0.8])).all()
            assert (image_sub_grid_coordinates[3, 6] == np.array([1.2, -0.8])).all()
            assert (image_sub_grid_coordinates[3, 7] == np.array([1.6, -0.8])).all()
            assert (image_sub_grid_coordinates[3, 8] == np.array([0.4, -1.2])).all()
            assert (image_sub_grid_coordinates[3, 9] == np.array([0.8, -1.2])).all()
            assert (image_sub_grid_coordinates[3, 10] == np.array([1.2, -1.2])).all()
            assert (image_sub_grid_coordinates[3, 11] == np.array([1.6, -1.2])).all()
            assert (image_sub_grid_coordinates[3, 12] == np.array([0.4, -1.6])).all()
            assert (image_sub_grid_coordinates[3, 13] == np.array([0.8, -1.6])).all()
            assert (image_sub_grid_coordinates[3, 14] == np.array([1.2, -1.6])).all()
            assert (image_sub_grid_coordinates[3, 15] == np.array([1.6, -1.6])).all()

            assert (image_sub_grid_coordinates[4, 0] == np.array([2.4, -2.4])).all()
            assert (image_sub_grid_coordinates[4, 1] == np.array([2.8, -2.4])).all()
            assert (image_sub_grid_coordinates[4, 2] == np.array([3.2, -2.4])).all()
            assert (image_sub_grid_coordinates[4, 3] == np.array([3.6, -2.4])).all()
            assert (image_sub_grid_coordinates[4, 4] == np.array([2.4, -2.8])).all()
            assert (image_sub_grid_coordinates[4, 5] == np.array([2.8, -2.8])).all()
            assert (image_sub_grid_coordinates[4, 6] == np.array([3.2, -2.8])).all()
            assert (image_sub_grid_coordinates[4, 7] == np.array([3.6, -2.8])).all()
            assert (image_sub_grid_coordinates[4, 8] == np.array([2.4, -3.2])).all()
            assert (image_sub_grid_coordinates[4, 9] == np.array([2.8, -3.2])).all()
            assert (image_sub_grid_coordinates[4, 10] == np.array([3.2, -3.2])).all()
            assert (image_sub_grid_coordinates[4, 11] == np.array([3.6, -3.2])).all()
            assert (image_sub_grid_coordinates[4, 12] == np.array([2.4, -3.6])).all()
            assert (image_sub_grid_coordinates[4, 13] == np.array([2.8, -3.6])).all()
            assert (image_sub_grid_coordinates[4, 14] == np.array([3.2, -3.6])).all()
            assert (image_sub_grid_coordinates[4, 15] == np.array([3.6, -3.6])).all()

        def test__4x3_mask_with_one_pixel__2x2_sub_grid__coordinates(self):

            mask = np.array([[False, False, False],
                             [False, True, False],
                             [False, True, True],
                             [True, False, False]])

            image_sub_grid_coordinates = analysis_image.setup_image_sub_grid_coordinates(mask=mask, pixel_scale=3.0,
                                                                                         sub_grid_size=2)

            assert (image_sub_grid_coordinates[0,0] == np.array([-0.5, 2.0])).all()
            assert (image_sub_grid_coordinates[0,1] == np.array([ 0.5, 2.0])).all()
            assert (image_sub_grid_coordinates[0,2] == np.array([-0.5, 1.0])).all()
            assert (image_sub_grid_coordinates[0,3] == np.array([ 0.5, 1.0])).all()

            assert (image_sub_grid_coordinates[1,0] == np.array([-0.5, -1.0])).all()
            assert (image_sub_grid_coordinates[1,1] == np.array([ 0.5, -1.0])).all()
            assert (image_sub_grid_coordinates[1,2] == np.array([-0.5, -2.0])).all()
            assert (image_sub_grid_coordinates[1,3] == np.array([ 0.5, -2.0])).all()

            assert (image_sub_grid_coordinates[2,0] == np.array([2.5, -1.0])).all()
            assert (image_sub_grid_coordinates[2,1] == np.array([3.5, -1.0])).all()
            assert (image_sub_grid_coordinates[2,2] == np.array([2.5, -2.0])).all()
            assert (image_sub_grid_coordinates[2,3] == np.array([3.5, -2.0])).all()

            assert (image_sub_grid_coordinates[3,0] == np.array([-3.5, -4.0])).all()
            assert (image_sub_grid_coordinates[3,1] == np.array([-2.5, -4.0])).all()
            assert (image_sub_grid_coordinates[3,2] == np.array([-3.5, -5.0])).all()
            assert (image_sub_grid_coordinates[3,3] == np.array([-2.5, -5.0])).all()

        def test__3x4_mask_with_one_pixel__2x2_sub_grid__coordinates(self):

            mask = np.array([[False, False, False, True],
                             [False, True, True, False],
                             [True, False, True, False]])

            image_sub_grid_coordinates = analysis_image.setup_image_sub_grid_coordinates(mask=mask, pixel_scale=3.0,
                                                                                         sub_grid_size=2)

            assert (image_sub_grid_coordinates[0,0] == np.array([4.0, 3.5])).all()
            assert (image_sub_grid_coordinates[0,1] == np.array([5.0, 3.5])).all()
            assert (image_sub_grid_coordinates[0,2] == np.array([4.0, 2.5])).all()
            assert (image_sub_grid_coordinates[0,3] == np.array([5.0, 2.5])).all()

            assert (image_sub_grid_coordinates[1,0] == np.array([-2.0, 0.5])).all()
            assert (image_sub_grid_coordinates[1,1] == np.array([-1.0, 0.5])).all()
            assert (image_sub_grid_coordinates[1,2] == np.array([-2.0, -0.5])).all()
            assert (image_sub_grid_coordinates[1,3] == np.array([-1.0, -0.5])).all()

            assert (image_sub_grid_coordinates[2,0] == np.array([1.0, 0.5])).all()
            assert (image_sub_grid_coordinates[2,1] == np.array([2.0, 0.5])).all()
            assert (image_sub_grid_coordinates[2,2] == np.array([1.0, -0.5])).all()
            assert (image_sub_grid_coordinates[2,3] == np.array([2.0, -0.5])).all()

            assert (image_sub_grid_coordinates[3,0] == np.array([-5.0, -2.5])).all()
            assert (image_sub_grid_coordinates[3,1] == np.array([-4.0, -2.5])).all()
            assert (image_sub_grid_coordinates[3,2] == np.array([-5.0, -3.5])).all()
            assert (image_sub_grid_coordinates[3,3] == np.array([-4.0, -3.5])).all()

            assert (image_sub_grid_coordinates[4,0] == np.array([1.0, -2.5])).all()
            assert (image_sub_grid_coordinates[4,1] == np.array([2.0, -2.5])).all()
            assert (image_sub_grid_coordinates[4,2] == np.array([1.0, -3.5])).all()
            assert (image_sub_grid_coordinates[4,3] == np.array([2.0, -3.5])).all()

    class TestBlurringRegion(object):

        def test__size__3x3_small_mask(self):

            mask = np.array([[False, False, False],
                             [False, True, False],
                             [False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(3, 3))

            assert (blurring_region == np.array([[True, True, True],
                                                 [True, False, True],
                                                 [True, True, True]])).all()

        def test__size__3x3__large_mask(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, True,  False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(3, 3))

            assert (blurring_region == np.array([[False, False, False, False, False, False, False],
                                                 [False, False, False, False, False, False, False],
                                                 [False, False, True, True, True, False, False],
                                                 [False, False, True, False, True, False, False],
                                                 [False, False, True, True, True, False, False],
                                                 [False, False, False, False, False, False, False],
                                                 [False, False, False, False, False, False, False]])).all()

        def test__size__5x5__large_mask(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, True,  False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(5, 5))

            assert (blurring_region == np.array([[False, False, False, False, False, False, False],
                                                 [False, True, True, True, True, True, False],
                                                 [False, True, True, True, True, True, False],
                                                 [False, True, True, False, True, True, False],
                                                 [False, True, True, True, True, True, False],
                                                 [False, True, True, True, True, True, False],
                                                 [False, False, False, False, False, False, False]])).all()

        def test__size__5x3__large_mask(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, True, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(5, 3))

            assert (blurring_region == np.array([[False, False, False, False, False, False, False],
                                                 [False, False, False, False, False, False, False],
                                                 [False, True, True, True, True, True, False],
                                                 [False, True, True, False, True, True, False],
                                                 [False, True, True, True, True, True, False],
                                                 [False, False, False, False, False, False, False],
                                                 [False, False, False, False, False, False, False]])).all()

        def test__size__3x5__large_mask(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, True, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(3, 5))

            assert (blurring_region == np.array([[False, False, False, False, False, False, False],
                                                 [False, False, True, True, True, False, False],
                                                 [False, False, True, True, True, False, False],
                                                 [False, False, True, False, True, False, False],
                                                 [False, False, True, True, True, False, False],
                                                 [False, False, True, True, True, False, False],
                                                 [False, False, False, False, False, False, False]])).all()

        def test__size__3x3__multiple_points(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False,  False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False],
                             [False, False, False, False, False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(3, 3))

            assert (blurring_region == np.array([[True,   True,  True, False,  True,  True,  True],
                                                 [True,  False,  True, False,  True, False,  True],
                                                 [True,   True,  True, False,  True,  True,  True],
                                                 [False, False, False, False, False, False, False],
                                                 [True,   True,  True, False,  True,  True,  True],
                                                 [True,  False,  True, False,  True, False,  True],
                                                 [True,   True,  True, False,  True,  True,  True]])).all()

        def test__size__5x5__multiple_points(self):

            mask = np.array([[False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False,  False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(5, 5))

            assert (blurring_region == np.array([[True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,  False,  True, True,  True, False,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,  False,  True, True,  True, False,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True]])).all()

        def test__size__5x3__multiple_points(self):

            mask = np.array([[False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False,  False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(5, 3))

            assert (blurring_region == np.array([[False, False, False, False, False, False, False, False, False],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,  False,  True, True,  True, False,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [False, False, False, False, False, False, False, False, False],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [True, True,  False,  True, True,  True, False,  True, True],
                                                 [True, True,   True,  True, True,  True,  True,  True, True],
                                                 [False, False, False, False, False, False, False, False, False]])).all()

        def test__size__3x5__multiple_points(self):

            mask = np.array([[False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False,  False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(3, 5))

            assert (blurring_region == np.array([[False, True,   True,  True, False,  True,  True,  True, False],
                                                 [False, True,   True,  True, False,  True,  True,  True, False],
                                                 [False, True,  False,  True, False,  True, False,  True, False],
                                                 [False, True,   True,  True, False,  True,  True,  True, False],
                                                 [False, True,   True,  True, False,  True,  True,  True, False],
                                                 [False, True,   True,  True, False,  True,  True,  True, False],
                                                 [False, True,  False,  True, False,  True, False,  True, False],
                                                 [False, True,   True,  True, False,  True,  True,  True, False],
                                                 [False, True,   True,  True, False,  True,  True,  True, False]])).all()

        def test__size__3x3__even_sized_image(self):

            mask = np.array([[False, False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False,  False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(3, 3))

            assert (blurring_region == np.array([[True,   True,  True, False,  True,  True,  True, False],
                                                 [True,  False,  True, False,  True, False,  True, False],
                                                 [True,   True,  True, False,  True,  True,  True, False],
                                                 [False, False, False, False, False, False, False, False],
                                                 [True,   True,  True, False,  True,  True,  True, False],
                                                 [True,  False,  True, False,  True, False,  True, False],
                                                 [True,   True,  True, False,  True,  True,  True, False],
                                                 [False, False, False, False, False, False, False, False]])).all()

        def test__size__5x5__even_sized_image(self):

            mask = np.array([[False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(5, 5))

            assert (blurring_region == np.array([[False, False, False,  False, False, False, False, False],
                                                 [False, False, False, False, False, False, False, False],
                                                 [False, False, False, False, False, False, False, False],
                                                 [False, False, False, True,  True,  True,  True, True],
                                                 [False, False, False, True,  True,  True,  True, True],
                                                 [False, False, False, True,  True, False,  True, True],
                                                 [False, False, False, True,  True,  True,  True, True],
                                                 [False, False, False,  True,  True,  True,  True, True]])).all()

        def test__size__3x3__rectangular_8x9_image(self):

            mask = np.array([[False, False, False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False,  False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False, False, False],
                             [False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(3, 3))

            assert (blurring_region == np.array([[True,   True,  True, False,  True,  True,  True, False, False],
                                                 [True,  False,  True, False,  True, False,  True, False, False],
                                                 [True,   True,  True, False,  True,  True,  True, False, False],
                                                 [False, False, False, False, False, False, False, False, False],
                                                 [True,   True,  True, False,  True,  True,  True, False, False],
                                                 [True,  False,  True, False,  True, False,  True, False, False],
                                                 [True,   True,  True, False,  True,  True,  True, False, False],
                                                 [False, False, False, False, False, False, False, False, False]])).all()

        def test__size__3x3__rectangular_9x8_image(self):

            mask = np.array([[False, False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False,  False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False]])

            blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(3, 3))

            assert (blurring_region == np.array([[True,   True,  True, False,  True,  True,  True, False],
                                                 [True,  False,  True, False,  True, False,  True, False],
                                                 [True,   True,  True, False,  True,  True,  True, False],
                                                 [False, False, False, False, False, False, False, False],
                                                 [True,   True,  True, False,  True,  True,  True, False],
                                                 [True,  False,  True, False,  True, False,  True, False],
                                                 [True,   True,  True, False,  True,  True,  True, False],
                                                 [False, False, False, False, False, False, False, False],
                                                 [False, False, False, False, False, False, False, False]])).all()


        def test__size__5x5__multiple_points__mask_extends_beyond_border_so_raises_mask_exception(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False,  False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, True, False, False, False, True, False],
                             [False, False, False, False, False, False, False]])

            with pytest.raises(analysis_image.MaskException):
                blurring_region = analysis_image.setup_blurring_region(mask, blurring_region_size=(5, 5))

    class TestBorderPixels(object):

        def test__7x7_mask_one_central_pixel__is_entire_border(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, True,  False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            border_pixels = analysis_image.setup_border_pixels(mask)

            assert (border_pixels == np.array([0])).all()

        def test__7x7_mask_nine_central_pixels__is_border(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            border_pixels = analysis_image.setup_border_pixels(mask)

            assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

        def test__7x7_mask_rectangle_of_fifteen_central_pixels__is_border(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True, True, False, False],
                             [False, False, False, False, False, False, False]])

            border_pixels = analysis_image.setup_border_pixels(mask)

            assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14])).all()

        def test__8x7_mask_add_edge_pixels__also_in_border(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, True, False, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, True,  True,  True,  True, True, False],
                             [False, False, True,  True,  True, False, False],
                             [False, False, True,  True, True, False, False],
                             [False, False, False, False, False, False, False]])

            border_pixels = analysis_image.setup_border_pixels(mask)

            assert (border_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17])).all()

        def test__8x7_mask_big_square(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True, True, True, False],
                             [False, True, True,  True, True, True, False],
                             [False, True, True,  True, True, True, False],
                             [False, True, True,  True, True, True, False],
                             [False, True, True,  True, True, True, False],
                             [False, False, False, False, False, False, False]])

            border_pixels = analysis_image.setup_border_pixels(mask)

            assert (border_pixels == np.array
                ([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 26, 27, 28, 29])).all()

        def test__7x8_mask_add_edge_pixels__also_in_border(self):

            mask = np.array([[False, False, False, False, False, False, False, False],
                             [False, False, False, True, False, False, False, False],
                             [False, False, True,  True,  True, False, False, False],
                             [False, False, True,  True,  True, False, False, False],
                             [False, True,  True,  True,  True, True, False, False],
                             [False, False, True,  True,  True, False, False, False],
                             [False, False, False, False, False, False, False, False]])

            border_pixels = analysis_image.setup_border_pixels(mask)

            assert (border_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14])).all()

        def test__7x8_mask_big_square(self):

            mask = np.array([[False, False, False, False, False, False, False, False],
                             [False, True, True,  True,  True, True, False, False],
                             [False, True, True,  True, True, True, False, False],
                             [False, True, True,  True, True, True, False, False],
                             [False, True, True,  True, True, True, False, False],
                             [False, True, True,  True, True, True, False, False],
                             [False, False, False, False, False, False, False, False]])

            border_pixels = analysis_image.setup_border_pixels(mask)

            assert (border_pixels == np.array
                ([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24])).all()

    class TestSparseClusteringPixels(object):

        def test__7x7_circle_mask__five_central_pixels__sparse_grid_size_1(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=1)

            assert (sparse_list == np.array([0, 1, 2, 3, 4])).all()

        def test__7x7_circle_mask__sparse_grid_size_1(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, False, True,  True, True, False, False],
                             [False, False, False, False, False, False, False]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=1)

            assert (sparse_list == np.arange(21)).all()

        def test__7x7_rectangle_mask__sparse_grid_size_1(self):

            mask = np.array([[True, True, True, True, True, True, True],
                             [True, True, True,  True,  True, True, True],
                             [True, True, True,  True,  True, True, True],
                             [True, True, True,  True,  True, True, True],
                             [True, True, True,  True,  True, True, True],
                             [True, True, True,  True, True, True, True],
                             [True, True, True, True, True, True, True]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=1)

            assert (sparse_list == np.arange(49)).all()

        def test__7x7_circle_mask__sparse_grid_size_2(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, False, True,  True, True, False, False],
                             [False, False, False, False, False, False, False]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=2)

            assert (sparse_list == np.array([4, 6, 14, 16])).all()

        def test__8x8_sporadic_mask__sparse_grid_size_2(self):

            mask = np.array([[False, False, False, False, False, False, True, True],
                             [False, False, True,  True,  True, False, True, True],
                             [False, True, True,  True,  True, True, True, True],
                             [False, True, True,  True,  True, True, True, True],
                             [False, True, True,  True,  True, True, True, True],
                             [False, False, True,  True, True, False, True, True],
                             [False, False, False, False, False, False, True, True],
                             [False, False, True, True, True, False, True, True]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=2)

            assert (sparse_list == np.array([0, 8, 10, 12, 22, 24, 26, 33])).all()

        def test__7x7_circle_mask_trues_on_even_values__sparse_grid_size_2(self):

            mask = np.array([[True,  False, True, False, True, False, True],
                             [False, False, False, False, False, False, False],
                             [True,  False, True, False, True, False, True],
                             [False, False, False, False, False, False, False],
                             [True,  False, True, False, True, False, True],
                             [False, False, False, False, False, False, False],
                             [True,  False, True, False, True, False, True]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=2)

            assert (sparse_list == np.arange(16)).all()

        def test__7x7_circle_mask__sparse_grid_size_3(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, True,  True,  True, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, False, True,  True, True, False, False],
                             [False, False, False, False, False, False, False]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=3)

            assert (sparse_list == np.array([10])).all()

        def test__7x7_circle_mask_more_points_added__sparse_grid_size_3(self):

            mask = np.array([[True, False, False, True, False, True, True],
                             [False, False, True,  True,  True, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, True],
                             [False, True, True,  True,  True, True, False],
                             [False, False, True,  True, True, False, False],
                             [False, False, False, False, False, False, True]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=3)

            assert (sparse_list == np.array([0, 1, 3, 14, 17, 26])).all()

        def test__7x7_mask_trues_on_values_which_divide_by_3__sparse_grid_size_3(self):

            mask = np.array([[True,  False, False, True, False, False, True],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [True, False, False,  True, False, False,  True],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [True,  False, False, True, False, False, True]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=3)

            assert (sparse_list == np.arange(9)).all()

        def test__8x8_mask_trues_on_values_which_divide_by_3_and_other_values__sparse_grid_size_3(self):

            mask = np.array([[True,  False, True, True, False, False, True],
                             [False, False, False, False, False, False, False],
                             [False, False, True, True, True, False, False],
                             [True, False, False,  True, False, False,  True],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [True,  True, True, True, True, True, True]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=3)

            assert (sparse_list == np.array([0, 2, 3, 7, 8, 9, 10, 13, 16])).all()

        def test__8x7__five_central_pixels__sparse_grid_size_1(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=1)

            assert (sparse_list == np.array([0, 1, 2, 3, 4])).all()

        def test__8x7__five_central_pixels_2__sparse_grid_size_1(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=1)

            assert (sparse_list == np.array([0, 1, 2, 3, 4])).all()

        def test__8x7__five_central_pixels__sparse_grid_size_2(self):

            mask = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, True, True,  True,  True, True, False],
                             [False, True, True,  True,  True, True, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=2)

            assert (sparse_list == np.array([1, 3])).all()

        def test__7x8__five_central_pixels__sparse_grid_size_1(self):

            mask = np.array([[False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, True, True,  True,  True, True, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=1)

            assert (sparse_list == np.array([0, 1, 2, 3, 4])).all()

        def test__7x8__five_central_pixels__sparse_grid_size_2(self):

            mask = np.array([[False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, True, True,  True,  True, True, False, False],
                             [False, True, True,  True,  True, True, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=2)

            assert (sparse_list == np.array([1, 3])).all()

        def test__7x8__more_central_pixels__sparse_grid_size_2(self):

            mask = np.array([[False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, True, True,  True,  True, True, False, False],
                             [False, True, True,  True,  True, True, False, False],
                             [False, True, True,  True,  True, True, False, False],
                             [False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False]])

            sparse_list = analysis_image.setup_sparse_clustering_pixels(mask, sparse_grid_size=2)

            assert (sparse_list == np.array([1, 3, 11, 13])).all()


class TestMask(object):
    class TestPixelScale(object):
        def test__central_pixel(self):
            assert image.central_pixel((3, 3)) == (1.0, 1.0)

        def test__shape(self):
            assert analysis_image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=5).shape == (3, 3)
            assert analysis_image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=0.5, radius=5).shape == (6, 6)
            assert analysis_image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=0.2, radius=5).shape == (15, 15)

        def test__odd_x_odd_mask_input_radius_small__mask(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=0.5, radius=0.5)
            assert (mask == np.array([[False, False, False, False, False, False],
                                      [False, False, False, False, False, False],
                                      [False, False, True, True, False, False],
                                      [False, False, True, True, False, False],
                                      [False, False, False, False, False, False],
                                      [False, False, False, False, False, False]])).all()

    class TestCentre(object):
        def test__simple_shift_back(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=0.5, centre=(-1, 0))
            assert mask.shape == (3, 3)
            assert (mask == np.array([[False, True, False],
                                      [False, False, False],
                                      [False, False, False]])).all()

        def test__simple_shift_forward(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=0.5, centre=(0, 1))
            assert mask.shape == (3, 3)
            assert (mask == np.array([[False, False, False],
                                      [False, False, True],
                                      [False, False, False]])).all()

        def test__diagonal_shift(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=0.5, centre=(1, 1))
            assert (mask == np.array([[False, False, False],
                                      [False, False, False],
                                      [False, False, True]])).all()

    class TestCircular(object):
        def test__input_big_mask__mask(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=5)
            assert mask.shape == (3, 3)
            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [True, True, True]])).all()

        def test__odd_x_odd_mask_input_radius_small__mask(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=0.5)
            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__odd_x_odd_mask_input_radius_medium__mask(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=1)

            assert (mask == np.array([[False, True, False],
                                      [True, True, True],
                                      [False, True, False]])).all()

        def test__odd_x_odd_mask_input_radius_large__mask(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(3, 3), pixel_scale=1, radius=3)

            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [True, True, True]])).all()

        def test__even_x_odd_mask_input_radius_small__mask(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(4, 3), pixel_scale=1, radius=0.5)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__even_x_odd_mask_input_radius_medium__mask(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(4, 3), pixel_scale=1, radius=1.50001)

            assert (mask == np.array([[False, True, False],
                                      [True, True, True],
                                      [True, True, True],
                                      [False, True, False]])).all()

        def test__even_x_odd_mask_input_radius_large__mask(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(4, 3), pixel_scale=1, radius=3)

            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [True, True, True],
                                      [True, True, True]])).all()

        def test__even_x_even_mask_input_radius_small__mask(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(4, 4), pixel_scale=1, radius=0.72)

            assert (mask == np.array([[False, False, False, False],
                                      [False, True, True, False],
                                      [False, True, True, False],
                                      [False, False, False, False]])).all()

        def test__even_x_even_mask_input_radius_medium__mask(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(4, 4), pixel_scale=1, radius=1.7)

            assert (mask == np.array([[False, True, True, False],
                                      [True, True, True, True],
                                      [True, True, True, True],
                                      [False, True, True, False]])).all()

        def test__even_x_even_mask_input_radius_large__mask(self):
            mask = analysis_image.Mask.circular(arc_second_dimensions=(4, 4), pixel_scale=1, radius=3)

            assert (mask == np.array([[True, True, True, True],
                                      [True, True, True, True],
                                      [True, True, True, True],
                                      [True, True, True, True]])).all()

    class TestAnnular(object):
        def test__odd_x_odd_mask_inner_radius_zero_outer_radius_small__mask(self):
            mask = analysis_image.Mask.annular(arc_second_dimensions=(3, 3), pixel_scale=1, inner_radius=0, outer_radius=0.5)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__odd_x_odd_mask_inner_radius_small_outer_radius_large__mask(self):
            mask = analysis_image.Mask.annular(arc_second_dimensions=(3, 3), pixel_scale=1, inner_radius=0.5, outer_radius=3)

            assert (mask == np.array([[True, True, True],
                                      [True, False, True],
                                      [True, True, True]])).all()

        def test__even_x_odd_mask_inner_radius_small_outer_radius_medium__mask(self):
            mask = analysis_image.Mask.annular(arc_second_dimensions=(4, 3), pixel_scale=1, inner_radius=0.51, outer_radius=1.51)

            assert (mask == np.array([[False, True, False],
                                      [True, False, True],
                                      [True, False, True],
                                      [False, True, False]])).all()

        def test__even_x_odd_mask_inner_radius_medium_outer_radius_large__mask(self):
            mask = analysis_image.Mask.annular(arc_second_dimensions=(4, 3), pixel_scale=1, inner_radius=1.51, outer_radius=3)

            assert (mask == np.array([[True, False, True],
                                      [False, False, False],
                                      [False, False, False],
                                      [True, False, True]])).all()

        def test__even_x_even_mask_inner_radius_small_outer_radius_medium__mask(self):
            mask = analysis_image.Mask.annular(arc_second_dimensions=(4, 4), pixel_scale=1, inner_radius=0.81, outer_radius=2)

            assert (mask == np.array([[False, True, True, False],
                                      [True, False, False, True],
                                      [True, False, False, True],
                                      [False, True, True, False]])).all()

        def test__even_x_even_mask_inner_radius_medium_outer_radius_large__mask(self):
            mask = analysis_image.Mask.annular(arc_second_dimensions=(4, 4), pixel_scale=1, inner_radius=1.71, outer_radius=3)

            assert (mask == np.array([[True, False, False, True],
                                      [False, False, False, False],
                                      [False, False, False, False],
                                      [True, False, False, True]])).all()