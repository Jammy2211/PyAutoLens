from __future__ import division, print_function
import pytest
import numpy as np
import os
from auto_lens.imaging import grids
from auto_lens.imaging import imaging

test_data_dir = "{}/../data/test_data/".format(os.path.dirname(os.path.realpath(__file__)))


class TestDataConversion(object):

    def test__setup_3x3___one_data_in_mask(self):
        data = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]])

        mask_array = np.array([[True, True, True],
                               [True, False, True],
                               [True, True, True]])

        mask = imaging.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        data_1d = grids.setup_data(mask, data)

        assert (data_1d[0] == 5.0)

    def test__setup_3x3_image__five_coordinates(self):
        data = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]])

        mask_array = np.array([[True, False, True],
                               [False, False, False],
                               [True, False, True]])

        mask = imaging.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        data_1d = grids.setup_data(mask, data)

        assert (data_1d[0] == 2.0)
        assert (data_1d[1] == 4.0)
        assert (data_1d[2] == 5.0)
        assert (data_1d[3] == 6.0)
        assert (data_1d[4] == 8.0)

    def test__setup_4x4_image__ten_coordinates__new_pixel_scale(self):
        data = np.array([[1.0, 2.0, 3.0, 4.0],
                         [8.0, 7.0, 6.0, 5.0],
                         [9.0, 10.0, 11.0, 12.0],
                         [16.0, 15.0, 14.0, 13.0]])

        mask_array = np.array([[True, False, False, True],
                               [False, False, False, True],
                               [True, False, False, True],
                               [False, False, False, True]])

        mask = imaging.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        data_1d = grids.setup_data(mask, data)

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

        data = np.array([[1.0, 2.0, 3.0, 4.0],
                         [8.0, 7.0, 6.0, 5.0],
                         [9.0, 10.0, 11.0, 12.0]])

        mask = imaging.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        data_1d = grids.setup_data(mask, data)

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

        mask = imaging.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        mapper_2d = grids.setup_mapper_2d(mask)

        assert (mapper_2d[0] == np.array([1, 1])).all()

    def test__setup_3x3_image__five_coordinates(self):
        mask_array = np.array([[True, False, True],
                               [False, False, False],
                               [True, False, True]])

        mask = imaging.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        mapper_2d = grids.setup_mapper_2d(mask)

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

        mask = imaging.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        mapper_2d = grids.setup_mapper_2d(mask)

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

        mask = imaging.Mask.from_array(mask_array=mask_array, pixel_scale=3.0)

        mapper_2d = grids.setup_mapper_2d(mask)

        assert (mapper_2d[0] == np.array([0, 1])).all()
        assert (mapper_2d[1] == np.array([1, 0])).all()
        assert (mapper_2d[2] == np.array([1, 1])).all()
        assert (mapper_2d[3] == np.array([1, 2])).all()
        assert (mapper_2d[4] == np.array([2, 1])).all()
        assert (mapper_2d[5] == np.array([2, 3])).all()


class TestAnalysisArray:

    def test__sets_up_array_fully_unmasked__maps_back_to_2d(self):
        array = np.ones((2, 2))
        mask = imaging.Mask.from_array(mask_array=np.zeros((2, 2)), pixel_scale=1.0)

        analysis_array = grids.AnalysisArray(mask, array)

        assert (analysis_array == np.array([1, 1, 1, 1])).all()
        assert (analysis_array.map_to_2d() == np.ones((2, 2))).all()

    def test__set_up_array_with_mask__maps_back_to_2d(self):
        array = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])

        mask_array = np.array([[True, False, True],
                               [False, False, False],
                               [True, False, True]])

        mask = imaging.Mask.from_array(mask_array=mask_array, pixel_scale=1.0)

        analysis_array = grids.AnalysisArray(mask, array)

        assert (analysis_array == np.array([2.0, 4.0, 5.0, 6.0, 8.0])).all()

        assert (analysis_array.map_to_2d() == np.array([[0.0, 2.0, 0.0],
                                                        [4.0, 5.0, 6.0],
                                                        [0.0, 8.0, 0.0]])).all()


class TestAnalysisImage:
    class TestConstructor:

        def test__sets_up_image_fully_unmasked__maps_back_to_2d(self):
            image_data = np.ones((2, 2))
            mask = imaging.Mask.from_array(mask_array=np.zeros((2, 2)), pixel_scale=1.0)

            analysis_image = grids.AnalysisImage(mask, image_data)

            assert (analysis_image == np.array([1, 1, 1, 1])).all()
            assert (analysis_image.map_to_2d() == np.ones((2, 2))).all()

        def test__set_up_image_with_mask__maps_back_to_2d(self):
            image_data = np.array([[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0],
                                   [7.0, 8.0, 9.0]])

            mask_array = np.array([[True, False, True],
                                   [False, False, False],
                                   [True, False, True]])

            mask = imaging.Mask.from_array(mask_array=mask_array, pixel_scale=1.0)

            analysis_image = grids.AnalysisArray(mask, image_data)

            assert (analysis_image == np.array([2.0, 4.0, 5.0, 6.0, 8.0])).all()

            assert (analysis_image.map_to_2d() == np.array([[0.0, 2.0, 0.0],
                                                            [4.0, 5.0, 6.0],
                                                            [0.0, 8.0, 0.0]])).all()


class TestAnalyisData(object):
    class TestConstructor:

        def test__4x4_input_all_image_properties__4_central_pixels_unmasked(self):
            test_image = imaging.Image(data=np.ones((4, 4)), pixel_scale=1.5)
            test_noise = imaging.Noise.from_array(array=np.ones((4, 4)))
            test_psf = imaging.PSF.from_array(array=np.ones((3, 3)), renormalize=False)
            test_mask = imaging.Mask.from_array(mask_array=np.array([[True, True, True, True],
                                                                     [True, False, False, True],
                                                                     [True, False, False, True],
                                                                     [True, True, True, True]]), pixel_scale=1.5)

            adata = grids.AnalysisData(test_mask, test_image, test_noise, test_psf, sub_grid_size=2)

            assert (adata.image == np.array([1, 1, 1, 1])).all()
            assert (adata.image.map_to_2d() == np.array([[0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 1.0, 1.0, 0.0],
                                                         [0.0, 1.0, 1.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0]])).all()

            assert (adata.noise == np.array([1, 1, 1, 1])).all()
            assert (adata.psf == np.ones((3, 3))).all()
            assert (adata.coordinates == np.array([[-0.75, 0.75], [0.75, 0.75], [-0.75, -0.75], [0.75, -0.75]])).all()
            assert (adata.sub_coordinates == grids.setup_sub_coordinates(test_mask, sub_grid_size=2)).all()
            assert (adata.blurring_coordinates == grids.setup_blurring_coordinates(test_mask,
                                                                                   psf_size=(3, 3))).all()
            assert (adata.border_pixels == grids.setup_border_pixels(test_mask)).all()