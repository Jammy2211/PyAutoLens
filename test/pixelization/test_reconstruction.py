from autolens.pixelization import reconstruction

import numpy as np
import pytest


class TestCovarianceMatrixFromBlurred(object):

    def test__simple_blurred_mapping_matrix__correct_covariance_matrix(self):

        recon = reconstruction.Reconstructor(mapping=np.ones((6,3)), regularization=np.ones((1,1)),
                                             image_to_pix=np.ones((1,1)), sub_to_pix=np.ones((1,1)))

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        noise_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        cov = recon.covariance_matrix_from_blurred_mapping_jit(blurred_mapping_matrix, noise_vector)

        assert (cov == np.array([[2.0, 1.0, 0.0],
                                 [1.0, 3.0, 1.0],
                                 [0.0, 1.0, 1.0]])).all()

    def test__simple_blurred_mapping_matrix__change_noise_values__correct_covariance_matrix(self):

        recon = reconstruction.Reconstructor(mapping=np.ones((6,3)), regularization=np.ones((1,1)),
                                             image_to_pix=np.ones((1,1)), sub_to_pix=np.ones((1,1)))

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        noise_vector = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        cov = recon.covariance_matrix_from_blurred_mapping_jit(blurred_mapping_matrix, noise_vector)

        assert (cov == np.array([[1.25, 0.25, 0.0],
                                 [0.25, 2.25, 1.0],
                                 [0.0, 1.0, 1.0]])).all()

    def test__jitted_and_normal(self):

        recon = reconstruction.Reconstructor(mapping=np.ones((6,3)), regularization=np.ones((1,1)),
                                             image_to_pix=np.ones((1,1)), sub_to_pix=np.ones((1,1)))

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [2.0, 6.0, 1.0],
                                           [0.0, 1.0, 8.0],
                                           [3.0, 1.0, 1.0],
                                           [4.0, 3.0, 0.0],
                                           [0.0, 0.0, 3.0]])

        noise_vector = np.array([3.0, 1.0, 4.0, 1.0, 10.0, 1.0])

        cov = recon.covariance_matrix_from_blurred_mapping_jit(blurred_mapping_matrix, noise_vector)
        cov_jitted = recon.covariance_matrix_from_blurred_mapping_jit(blurred_mapping_matrix, noise_vector)

        assert (cov == cov_jitted).all()


class TestDataVectorFromData(object):

    def test__simple_blurred_mapping_matrix__correct_d_matrix(self):

        recon = reconstruction.Reconstructor(mapping=np.ones((6,3)), regularization=np.ones((1,1)),
                                             image_to_pix=np.ones((1,1)), sub_to_pix=np.ones((1,1)))

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        image_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        noise_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        d = recon.data_vector_from_blurred_mapping_and_data_jit(blurred_mapping_matrix, image_vector, noise_vector)

        assert (d == np.array([2.0, 3.0, 1.0])).all()

    def test__simple_blurred_mapping_matrix__change_image_values__correct_d_matrix(self):

        recon = reconstruction.Reconstructor(mapping=np.ones((6,3)), regularization=np.ones((1,1)),
                                             image_to_pix=np.ones((1,1)), sub_to_pix=np.ones((1,1)))

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        image_vector = np.array([3.0, 1.0, 1.0, 10.0, 1.0, 1.0])
        noise_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        d = recon.data_vector_from_blurred_mapping_and_data_jit(blurred_mapping_matrix, image_vector, noise_vector)

        assert (d == np.array([4.0, 14.0, 10.0])).all()

    def test__simple_blurred_mapping_matrix__change_noise_values__correct_d_matrix(self):

        recon = reconstruction.Reconstructor(mapping=np.ones((6,3)), regularization=np.ones((1,1)),
                                             image_to_pix=np.ones((1,1)), sub_to_pix=np.ones((1,1)))

        blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])

        image_vector = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
        noise_vector = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

        d = recon.data_vector_from_blurred_mapping_and_data_jit(blurred_mapping_matrix, image_vector, noise_vector)

        assert (d == np.array([2.0, 3.0, 1.0])).all()


class TestComputeRegularizationTerm:

    def test__solution_all_1s__regularization_matrix_simple(self):
        solution = np.array([1.0, 1.0, 1.0])

        regularization_matrix = np.array([[1.0, 0.0, 0.0],
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

        pix_fit = reconstruction.Reconstruction(data_vector=None, blurred_mapping=None,
                                                regularization=regularization_matrix,
                                                covariance=None, covariance_regularization=None,
                                                reconstruction=solution)

        assert pix_fit.regularization_term_from_reconstruction() == 3.0

    def test__solution_and_regularization_matrix_range_of_values(self):
        solution = np.array([2.0, 3.0, 5.0])

        regularization_matrix = np.array([[2.0, -1.0, 0.0],
                                          [-1.0, 2.0, -1.0],
                                          [0.0, -1.0, 2.0]])

        # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

        # G_l = s_T * H * s

        # Matrix multiplication:

        # s_T * H = [2.0, 3.0, 5.0] * [2.0,  -1.0,  0.0] = [(2.0* 2.0) + (3.0*-1.0) + (5.0 *0.0)] = [1.0, -1.0, 7.0]
        #                             [-1.0,  2.0, -1.0]   [(2.0*-1.0) + (3.0* 2.0) + (5.0*-1.0)]
        #                             [ 0.0, -1.0,  2.0]   [(2.0* 0.0) + (3.0*-1.0) + (5.0 *2.0)]

        # (s_T * H) * s = [1.0, -1.0, 7.0] * [2.0] = 34.0
        #                                    [3.0]
        #                                    [5.0]

        pix_fit = reconstruction.Reconstruction(data_vector=None, blurred_mapping=None,
                                                regularization=regularization_matrix,
                                                covariance=None, covariance_regularization=None,
                                                reconstruction=solution)

        assert pix_fit.regularization_term_from_reconstruction() == 34.0


class TestLogDetMatrix:

    def test__determinant_of_positive_definite_matrix_via_cholesky(self):
        matrix = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])

        log_determinant = np.log(np.linalg.det(matrix))

        pix_fit = reconstruction.Reconstruction(data_vector=None, blurred_mapping=None, regularization=None,
                                                covariance=None, covariance_regularization=None, reconstruction=None)

        assert log_determinant == pytest.approx(pix_fit.log_determinant_of_matrix_cholesky(matrix), 1e-4)

    def test__determinant_of_positive_definite_matrix_2_via_cholesky(self):
        matrix = np.array([[2.0, -1.0, 0.0],
                           [-1.0, 2.0, -1.0],
                           [0.0, -1.0, 2.0]])

        log_determinant = np.log(np.linalg.det(matrix))

        pix_fit = reconstruction.Reconstruction(data_vector=None, blurred_mapping=None, regularization=None,
                                                covariance=None, covariance_regularization=None, reconstruction=None)

        assert log_determinant == pytest.approx(pix_fit.log_determinant_of_matrix_cholesky(matrix), 1e-4)


class TestModelImageFromSolution:

    def test__solution_all_1s__simple_blurred_mapping__correct_model_image(self):
        solution = np.array([1.0, 1.0, 1.0, 1.0])

        blurred_mapping = np.array([[1.0, 1.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0, 0.0]])

        pix_fit = reconstruction.Reconstruction(data_vector=None, blurred_mapping=blurred_mapping,
                                                regularization=None,
                                                covariance=None, covariance_regularization=None,
                                                reconstruction=solution)

        model_image = pix_fit.model_image_from_reconstruction_jit()

        # Image pixel 0 maps to 4 pixs pixxels -> value is 4.0
        # Image pixel 1 maps to 3 pixs pixxels -> value is 3.0
        # Image pixel 2 maps to 1 pixs pixxels -> value is 1.0

        assert (model_image == np.array([4.0, 3.0, 1.0])).all()

    def test__solution_different_values__simple_blurred_mapping__correct_model_image(self):
        solution = np.array([1.0, 2.0, 3.0, 4.0])

        blurred_mapping = np.array([[1.0, 1.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0, 0.0]])

        pix_fit = reconstruction.Reconstruction(data_vector=None, blurred_mapping=blurred_mapping,
                                                regularization=None,
                                                covariance=None, covariance_regularization=None,
                                                reconstruction=solution)

        model_image = pix_fit.model_image_from_reconstruction_jit()

        # Image pixel 0 maps to 4 pixs pixxels -> value is 1.0 + 2.0 + 3.0 + 4.0 = 10.0
        # Image pixel 1 maps to 3 pixs pixxels -> value is 1.0 + 3.0 + 4.0
        # Image pixel 2 maps to 1 pixs pixxels -> value is 1.0

        assert (model_image == np.array([10.0, 8.0, 1.0])).all()
