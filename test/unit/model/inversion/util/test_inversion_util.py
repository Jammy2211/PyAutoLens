import numpy as np
import pytest

from autolens import exc
from autolens.data.array import grids, mask
from autolens.model.inversion.util import inversion_util

class TestDataVectorFromData(object):

    def test__simple_blurred_mapping_matrix__correct_data_vector(self):

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        data_vector = inversion_util.data_vector_from_blurred_mapping_matrix_and_data(
            blurred_mapping_matrix=blurred_mapping_matrix, image_1d=image, noise_map_1d=noise_map)

        assert (data_vector == np.array([2.0, 3.0, 1.0])).all()

    def test__simple_blurred_mapping_matrix__change_image_values__correct_data_vector(self):

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        image = np.array([3.0, 1.0, 1.0, 10.0, 1.0, 1.0])
        noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        data_vector = inversion_util.data_vector_from_blurred_mapping_matrix_and_data(
            blurred_mapping_matrix=blurred_mapping_matrix, image_1d=image, noise_map_1d=noise_map)

        assert (data_vector == np.array([4.0, 14.0, 10.0])).all()

    def test__simple_blurred_mapping_matrix__change_noise_values__correct_data_vector(self):

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        image = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
        noise_map = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

        data_vector = inversion_util.data_vector_from_blurred_mapping_matrix_and_data(
            blurred_mapping_matrix=blurred_mapping_matrix, image_1d=image, noise_map_1d=noise_map)

        assert (data_vector == np.array([2.0, 3.0, 1.0])).all()
        

class TestCurvatureMatrixFromBlurred(object):

    def test__simple_blurred_mapping_matrix(self):

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        curvature_matrix = inversion_util.curvature_matrix_from_blurred_mapping_matrix(
            blurred_mapping_matrix=blurred_mapping_matrix, noise_map_1d=noise_map)

        assert (curvature_matrix == np.array([[2.0, 1.0, 0.0],
                                              [1.0, 3.0, 1.0],
                                              [0.0, 1.0, 1.0]])).all()

    def test__simple_blurred_mapping_matrix__change_noise_values(self):

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        noise_map = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        curvature_matrix = inversion_util.curvature_matrix_from_blurred_mapping_matrix(
            blurred_mapping_matrix=blurred_mapping_matrix, noise_map_1d=noise_map)

        assert (curvature_matrix == np.array([[1.25, 0.25, 0.0],
                                              [0.25, 2.25, 1.0],
                                              [0.0, 1.0, 1.0]])).all()


class TestPixelizationResiduals(object):

    def test__pixelization_perfectly_reconstructed_data__quantities_like_residuals_all_zeros(self):

        pixelization_values = np.ones(3)
        reconstructed_data_1d = np.ones(9)
        sub_to_regular = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        pixelization_to_sub_all = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

        pixelization_residuals = \
            inversion_util.pixelization_residual_map_from_pixelization_values_and_reconstructed_data_1d(
                pixelization_values=pixelization_values, reconstructed_data_1d=reconstructed_data_1d,
                sub_to_regular=sub_to_regular, pixelization_to_sub_all=pixelization_to_sub_all)

        assert (pixelization_residuals == np.zeros(3)).all()

    def test__pixelization_not_perfect_fit__quantities_like_residuals_non_zero(self):

        pixelization_values = np.ones(3)
        reconstructed_data_1d = 2.0*np.ones(9)
        sub_to_regular = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        pixelization_to_sub_all = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        pixelization_residuals = \
            inversion_util.pixelization_residual_map_from_pixelization_values_and_reconstructed_data_1d(
                pixelization_values=pixelization_values, reconstructed_data_1d=reconstructed_data_1d,
                sub_to_regular=sub_to_regular, pixelization_to_sub_all=pixelization_to_sub_all)

        assert (pixelization_residuals == 3.0*np.ones(3)).all()

        pixelization_values = np.ones(3)
        reconstructed_data_1d = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        sub_to_regular = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        pixelization_to_sub_all = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        pixelization_residuals = \
            inversion_util.pixelization_residual_map_from_pixelization_values_and_reconstructed_data_1d(
                pixelization_values=pixelization_values, reconstructed_data_1d=reconstructed_data_1d,
                sub_to_regular=sub_to_regular, pixelization_to_sub_all=pixelization_to_sub_all)

        assert (pixelization_residuals == np.array([0.0, 3.0, 6.0])).all()

class TestPixelizationNormalizedResiduals(object):

    def test__pixelization_perfectly_reconstructed_data__quantities_like_residuals_all_zeros(self):

        pixelization_values = np.ones(3)
        reconstructed_data_1d = np.ones(9)
        noise_map_1d = np.ones(9)
        sub_to_regular = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        pixelization_to_sub_all = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

        pixelization_normalized_residuals = \
            inversion_util.pixelization_normalized_residual_map_from_pixelization_values_and_reconstructed_data_1d(
                pixelization_values=pixelization_values, reconstructed_data_1d=reconstructed_data_1d,
                noise_map_1d=noise_map_1d, sub_to_regular=sub_to_regular,
                pixelization_to_sub_all=pixelization_to_sub_all)

        assert (pixelization_normalized_residuals == np.zeros(3)).all()

    def test__pixelization_not_perfect_fit__quantities_like_residuals_non_zero(self):

        pixelization_values = np.ones(3)
        reconstructed_data_1d = 2.0*np.ones(9)
        noise_map_1d = 2.0*np.ones(9)
        sub_to_regular = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        pixelization_to_sub_all = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        pixelization_normalized_residuals = \
            inversion_util.pixelization_normalized_residual_map_from_pixelization_values_and_reconstructed_data_1d(
                pixelization_values=pixelization_values, reconstructed_data_1d=reconstructed_data_1d,
                noise_map_1d=noise_map_1d, sub_to_regular=sub_to_regular,
                pixelization_to_sub_all=pixelization_to_sub_all)

        assert (pixelization_normalized_residuals == 1.5*np.ones(3)).all()

        pixelization_values = np.ones(3)
        reconstructed_data_1d = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        noise_map_1d = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        sub_to_regular = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        pixelization_to_sub_all = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        pixelization_normalized_residuals = \
            inversion_util.pixelization_normalized_residual_map_from_pixelization_values_and_reconstructed_data_1d(
                pixelization_values=pixelization_values, reconstructed_data_1d=reconstructed_data_1d,
                noise_map_1d=noise_map_1d,  sub_to_regular=sub_to_regular,
                pixelization_to_sub_all=pixelization_to_sub_all)

        assert (pixelization_normalized_residuals == np.array([0.0, 3.0, 3.0])).all()
        

class TestPixelizationChiSquareds(object):

    def test__pixelization_perfectly_reconstructed_data__quantities_like_residuals_all_zeros(self):

        pixelization_values = np.ones(3)
        reconstructed_data_1d = np.ones(9)
        noise_map_1d = np.ones(9)
        sub_to_regular = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        pixelization_to_sub_all = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

        pixelization_chi_squareds = \
            inversion_util.pixelization_chi_squared_map_from_pixelization_values_and_reconstructed_data_1d(
                pixelization_values=pixelization_values, reconstructed_data_1d=reconstructed_data_1d,
                noise_map_1d=noise_map_1d, sub_to_regular=sub_to_regular,
                pixelization_to_sub_all=pixelization_to_sub_all)

        assert (pixelization_chi_squareds == np.zeros(3)).all()

    def test__pixelization_not_perfect_fit__quantities_like_residuals_non_zero(self):

        pixelization_values = np.ones(3)
        reconstructed_data_1d = 2.0*np.ones(9)
        noise_map_1d = 2.0*np.ones(9)
        sub_to_regular = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        pixelization_to_sub_all = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        pixelization_chi_squareds = \
            inversion_util.pixelization_chi_squared_map_from_pixelization_values_and_reconstructed_data_1d(
                pixelization_values=pixelization_values, reconstructed_data_1d=reconstructed_data_1d,
                noise_map_1d=noise_map_1d, sub_to_regular=sub_to_regular,
                pixelization_to_sub_all=pixelization_to_sub_all)

        assert (pixelization_chi_squareds == 0.75*np.ones(3)).all()

        pixelization_values = np.ones(3)
        reconstructed_data_1d = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        noise_map_1d = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 4.0, 4.0, 4.0])
        sub_to_regular = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        pixelization_to_sub_all = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        pixelization_chi_squareds = \
            inversion_util.pixelization_chi_squared_map_from_pixelization_values_and_reconstructed_data_1d(
                pixelization_values=pixelization_values, reconstructed_data_1d=reconstructed_data_1d,
                noise_map_1d=noise_map_1d,  sub_to_regular=sub_to_regular,
                pixelization_to_sub_all=pixelization_to_sub_all)

        assert (pixelization_chi_squareds == np.array([0.0, 12.0, 0.75])).all()