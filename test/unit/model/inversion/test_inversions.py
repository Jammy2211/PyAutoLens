import numpy as np
import pytest

from autolens import exc
from autolens.data.array import grids, mask
from autolens.model.inversion.util import inversion_util
from autolens.model.inversion import inversions
from test.unit.mock.model import mock_inversion


class TestRegularizationTerm:

    def test__solution_all_1s__regularization_matrix_simple(self):

        matrix_shape = (3,3)

        inv = inversions.Inversion(
            image_1d=np.ones(9), noise_map_1d=np.ones(9), convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape=matrix_shape),
            regularization=mock_inversion.MockRegularization(matrix_shape))

        inv.pixelization_values = np.array([1.0, 1.0, 1.0])

        inv.regularization_matrix = np.array([[1.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0],
                                              [0.0, 0.0, 1.0]])

        # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

        # G_l = s_T * H * s

        # Matrix multiplication:

        # s_T * H = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [(1.0*1.0) + (1.0*0.0) + (1.0*0.0)] = [1.0, 1.0, 1.0]
        #                             [1.0, 1.0, 1.0]   [(1.0*0.0) + (1.0*1.0) + (1.0*0.0)]
        #                             [1.0, 1.0, 1.0]   [(1.0*0.0) + (1.0*0.0) + (1.0*1.0)]

        # (s_T * H) * s = [1.0, 1.0, 1.0] * [1.0] = 3.0
        #                                   [1.0]
        #                                   [1.0]

        assert inv.regularization_term == 3.0

    def test__solution_and_regularization_matrix_range_of_values(self):

        matrix_shape = (3,3)

        inv = inversions.Inversion(
            image_1d=np.ones(9), noise_map_1d=np.ones(9), convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape), regularization=mock_inversion.MockRegularization(matrix_shape))

        # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

        # G_l = s_T * H * s

        # Matrix multiplication:

        # s_T * H = [2.0, 3.0, 5.0] * [2.0,  -1.0,  0.0] = [(2.0* 2.0) + (3.0*-1.0) + (5.0 *0.0)] = [1.0, -1.0, 7.0]
        #                             [-1.0,  2.0, -1.0]   [(2.0*-1.0) + (3.0* 2.0) + (5.0*-1.0)]
        #                             [ 0.0, -1.0,  2.0]   [(2.0* 0.0) + (3.0*-1.0) + (5.0 *2.0)]

        # (s_T * H) * s = [1.0, -1.0, 7.0] * [2.0] = 34.0
        #                                    [3.0]
        #                                    [5.0]

        inv.pixelization_values = np.array([2.0, 3.0, 5.0])

        inv.regularization_matrix = np.array([[2.0, -1.0, 0.0],
                                          [-1.0, 2.0, -1.0],
                                          [0.0, -1.0, 2.0]])

        assert inv.regularization_term == 34.0


class TestLogDetMatrix:

    def test__determinant_of_positive_definite_matrix_via_cholesky(self):

        matrix_shape = (3,3)

        inv = inversions.Inversion(
            image_1d=np.ones(9), noise_map_1d=np.ones(9), convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape), regularization=mock_inversion.MockRegularization(matrix_shape))

        matrix = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])

        log_determinant = np.log(np.linalg.det(matrix))

        assert log_determinant == pytest.approx(inv.log_determinant_of_matrix_cholesky(matrix), 1e-4)

    def test__determinant_of_positive_definite_matrix_2_via_cholesky(self):

        matrix_shape = (3,3)

        inv = inversions.Inversion(
            image_1d=np.ones(9), noise_map_1d=np.ones(9), convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape), regularization=mock_inversion.MockRegularization(matrix_shape))

        matrix = np.array([[2.0, -1.0, 0.0],
                           [-1.0, 2.0, -1.0],
                           [0.0, -1.0, 2.0]])

        log_determinant = np.log(np.linalg.det(matrix))

        assert log_determinant == pytest.approx(inv.log_determinant_of_matrix_cholesky(matrix), 1e-4)

    def test__matrix_not_positive_definite__raises_reconstruction_exception(self):

        matrix_shape = (3,3)

        inv = inversions.Inversion(
            image_1d=np.ones(9), noise_map_1d=np.ones(9), convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape), regularization=mock_inversion.MockRegularization(matrix_shape))

        matrix = np.array([[2.0, 0.0, 0.0],
                           [-1.0, 2.0, -1.0],
                           [0.0, -1.0, 0.0]])

        with pytest.raises(exc.InversionException):
            assert pytest.approx(inv.log_determinant_of_matrix_cholesky(matrix), 1e-4)


class TestReconstructedDataVectorAndImage:

    def test__solution_all_1s__simple_blurred_mapping_matrix__correct_reconstructed_image(self):

        matrix_shape = (3,3)

        msk = mask.Mask(array=np.array([[True, True, True],
                                        [False, False, False],
                                        [True, True, True]]), pixel_scale=1.0)

        grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=msk, sub_grid_size=1, psf_shape=(1,1))

        inv = inversions.Inversion(
            image_1d=np.ones(9), noise_map_1d=np.ones(9), convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape, grid_stack),
            regularization=mock_inversion.MockRegularization(matrix_shape))

        inv.pixelization_values = np.array([1.0, 1.0, 1.0, 1.0])

        inv.blurred_mapping_matrix = np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 0.0, 1.0, 1.0],
                                               [1.0, 0.0, 0.0, 0.0]])
        # CCD pixel 0 maps to 4 pixs pixxels -> value is 4.0
        # CCD pixel 1 maps to 3 pixs pixxels -> value is 3.0
        # CCD pixel 2 maps to 1 pixs pixxels -> value is 1.0

        assert (inv.reconstructed_data_1d == np.array([4.0, 3.0, 1.0])).all()
        assert (inv.reconstructed_data_2d == np.array(
            [[0.0, 0.0, 0.0],
             [4.0, 3.0, 1.0],
             [0.0, 0.0, 0.0]]))

        assert inv.pixelization_errors_with_covariance == pytest.approx(np.array(
            [[ 0.7, -0.3, -0.3],
             [-0.3,  0.7, -0.3],
             [-0.3, -0.3,  0.7]]), 1.0e-4)
        assert inv.pixelization_errors == pytest.approx(np.array([0.7, 0.7, 0.7]), 1.0e-4)

    def test__solution_different_values__simple_blurred_mapping_matrix__correct_reconstructed_image(self):

        matrix_shape = (3,3)

        msk = mask.Mask(array=np.array([[True, True, True],
                                        [False, False, False],
                                        [True, True, True]]), pixel_scale=1.0)

        grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=msk, sub_grid_size=1, psf_shape=(1,1))

        inv = inversions.Inversion(
            image_1d=np.ones(9), noise_map_1d=np.ones(9), convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape, grid_stack),
            regularization=mock_inversion.MockRegularization(matrix_shape))

        inv.pixelization_values = np.array([1.0, 2.0, 3.0, 4.0])

        inv.blurred_mapping_matrix = np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 0.0, 1.0, 1.0],
                                               [1.0, 0.0, 0.0, 0.0]])

        # # CCD pixel 0 maps to 4 pixs pixxels -> value is 1.0 + 2.0 + 3.0 + 4.0 = 10.0
        # # CCD pixel 1 maps to 3 pixs pixxels -> value is 1.0 + 3.0 + 4.0
        # # CCD pixel 2 maps to 1 pixs pixxels -> value is 1.0

        assert (inv.reconstructed_data_1d == np.array([10.0, 8.0, 1.0])).all()
        assert (inv.reconstructed_data_2d == np.array([[0.0, 0.0, 0.0],
                                                       [10.0, 8.0, 1.0],
                                                       [0.0, 0.0, 0.0]]))

        assert inv.pixelization_errors_with_covariance == pytest.approx(np.array(
            [[ 0.7, -0.3, -0.3],
             [-0.3,  0.7, -0.3],
             [-0.3, -0.3,  0.7]]), 1.0e-4)
        assert inv.pixelization_errors == pytest.approx(np.array([0.7, 0.7, 0.7]), 1.0e-4)

#
# class TestPixelizationQuantities:
#
#     def test__compare_to_inversion_utils(self):
#
#         matrix_shape = (3,3)
#
#         msk = mask.Mask(array=np.array([[True, True, True],
#                                         [False, False, False],
#                                         [True, True, True]]), pixel_scale=1.0)
#
#         grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
#             mask=msk, sub_grid_size=1, psf_shape=(1,1))
#
#         inv = inversions.Inversion(
#             image_1d=np.ones(9), noise_map_1d=np.ones(9), convolver=mock_inversion.MockConvolver(matrix_shape),
#             mapper=mock_inversion.MockMapper(matrix_shape, grid_stack),
#             regularization=mock_inversion.MockRegularization(matrix_shape))
#
#         inv.pixelization_values = np.array([1.0, 1.0, 1.0, 1.0])
#
#         inv.blurred_mapping_matrix = np.array([[1.0, 1.0, 1.0, 1.0],
#                                                [1.0, 0.0, 1.0, 1.0],
#                                                [1.0, 0.0, 0.0, 0.0]])
#
#         pixelization_residuals_util = \
#             inversion_util.pixelization_residuals_from_pixelization_values_reconstructed_data_1d_and_mapping_quantities(
#                 pixelization_values=inv.pixelization_values, reconstructed_data_1d=inv.reconstructed_data_1d,
#                 sub_to_regular=inv.mapper.sub_to_regular, pixelization_to_sub_all=inv.mapper.pixelization_to_sub_all)