import numpy as np
import pytest
from autolens.pixelization import reconstruction as recon


class TestReconstructor:
    class TestCurvatureMatrixFromBlurred(object):

        def test__simple_blurred_mapping_matrix__correct_covariance_matrix(self):
            re = recon.Reconstructor(mapping_matrix=np.ones((6, 3)), regularization_matrix=np.ones((1, 1)))

            blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                               [1.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 1.0, 1.0],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]])

            noise_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

            curvature_matrix = re.curvature_matrix_from_blurred_mapping_matrix_jit(blurred_mapping_matrix,
                                                                                      noise_vector)

            assert (curvature_matrix == np.array([[2.0, 1.0, 0.0],
                                                  [1.0, 3.0, 1.0],
                                                  [0.0, 1.0, 1.0]])).all()

        def test__simple_blurred_mapping_matrix__change_noise_values__correct_covariance_matrix(self):
            re = recon.Reconstructor(mapping_matrix=np.ones((6, 3)), regularization_matrix=np.ones((1, 1)))

            blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                               [1.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 1.0, 1.0],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]])

            noise_vector = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

            curvature_matrix = re.curvature_matrix_from_blurred_mapping_matrix_jit(blurred_mapping_matrix,
                                                                                      noise_vector)

            assert (curvature_matrix == np.array([[1.25, 0.25, 0.0],
                                                  [0.25, 2.25, 1.0],
                                                  [0.0, 1.0, 1.0]])).all()

    class TestDataVectorFromData(object):

        def test__simple_blurred_mapping_matrix__correct_data_vector(self):

            re = recon.Reconstructor(mapping_matrix=np.ones((6, 3)), regularization_matrix=np.ones((1, 1)))

            blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                               [1.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 1.0, 1.0],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]])

            image_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            noise_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

            data_vector = re.data_vector_from_blurred_mapping_matrix_and_data_jit(blurred_mapping_matrix,
                                                                                     image_vector, noise_vector)

            assert (data_vector == np.array([2.0, 3.0, 1.0])).all()

        def test__simple_blurred_mapping_matrix__change_image_values__correct_data_vector(self):
            re = recon.Reconstructor(mapping_matrix=np.ones((6, 3)), regularization_matrix=np.ones((1, 1)))

            blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                               [1.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 1.0, 1.0],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]])

            image_vector = np.array([3.0, 1.0, 1.0, 10.0, 1.0, 1.0])
            noise_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

            data_vector = re.data_vector_from_blurred_mapping_matrix_and_data_jit(blurred_mapping_matrix,
                                                                                     image_vector, noise_vector)

            assert (data_vector == np.array([4.0, 14.0, 10.0])).all()

        def test__simple_blurred_mapping_matrix__change_noise_values__correct_data_vector(self):
            re = recon.Reconstructor(mapping_matrix=np.ones((6, 3)), regularization_matrix=np.ones((1, 1)))

            blurred_mapping_matrix = np.array([[1.0, 1.0, 0.0],
                                               [1.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 1.0, 1.0],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]])

            image_vector = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
            noise_vector = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

            data_vector = re.data_vector_from_blurred_mapping_matrix_and_data_jit(blurred_mapping_matrix,
                                                                                     image_vector, noise_vector)

            assert (data_vector == np.array([2.0, 3.0, 1.0])).all()


class TestReconstruction:

    class TestRegularizationTerm:

        def test__solution_all_1s__regularization_matrix_simple(self):

            solution_vector = np.array([1.0, 1.0, 1.0])

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

            re = recon.Reconstruction(blurred_mapping_matrix=np.zeros((1, 1)),
                                      regularization_matrix=regularization_matrix, curvature_matrix=np.zeros((1, 1)),
                                      curvature_reg_matrix=np.zeros((1, 1)), solution_vector=solution_vector)

            assert re.regularization_term == 3.0

        def test__solution_and_regularization_matrix_range_of_values(self):
            solution_vector = np.array([2.0, 3.0, 5.0])

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

            re = recon.Reconstruction(blurred_mapping_matrix=np.ones((1, 1)),
                                      regularization_matrix=regularization_matrix, curvature_matrix=np.ones((1, 1)),
                                      curvature_reg_matrix=np.ones((1, 1)), solution_vector=solution_vector)

            assert re.regularization_term == 34.0

    class TestLogDetMatrix:

        def test__determinant_of_positive_definite_matrix_via_cholesky(self):
            matrix = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])

            log_determinant = np.log(np.linalg.det(matrix))

            re = recon.Reconstruction(blurred_mapping_matrix=np.zeros((1, 1)), regularization_matrix=np.zeros((1, 1)),
                                      curvature_matrix=np.zeros((1, 1)), curvature_reg_matrix=np.zeros((1, 1)),
                                      solution_vector=np.zeros((1)))

            assert log_determinant == pytest.approx(re.log_determinant_of_matrix_cholesky(matrix), 1e-4)

        def test__determinant_of_positive_definite_matrix_2_via_cholesky(self):
            matrix = np.array([[2.0, -1.0, 0.0],
                               [-1.0, 2.0, -1.0],
                               [0.0, -1.0, 2.0]])

            log_determinant = np.log(np.linalg.det(matrix))

            re = recon.Reconstruction(blurred_mapping_matrix=np.zeros((1, 1)), regularization_matrix=np.zeros((1, 1)),
                                      curvature_matrix=np.zeros((1, 1)), curvature_reg_matrix=np.zeros((1, 1)),
                                      solution_vector=np.zeros((1)))

            assert log_determinant == pytest.approx(re.log_determinant_of_matrix_cholesky(matrix), 1e-4)

    class TestReconstructedImageFromSolution:

        def test__solution_all_1s__simple_blurred_mapping_matrix__correct_reconstructed_image(self):

            solution_vector = np.array([1.0, 1.0, 1.0, 1.0])

            blurred_mapping_matrix = np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 0.0, 1.0, 1.0],
                                               [1.0, 0.0, 0.0, 0.0]])

            re = recon.Reconstruction(blurred_mapping_matrix=blurred_mapping_matrix,
                                      regularization_matrix=np.ones((1, 1)), curvature_matrix=np.ones((1, 1)),
                                      curvature_reg_matrix=np.ones((1, 1)), solution_vector=solution_vector)

            # Image pixel 0 maps to 4 pixs pixxels -> value is 4.0
            # Image pixel 1 maps to 3 pixs pixxels -> value is 3.0
            # Image pixel 2 maps to 1 pixs pixxels -> value is 1.0

            assert (re.reconstructed_image == np.array([4.0, 3.0, 1.0])).all()

        def test__solution_different_values__simple_blurred_mapping_matrix__correct_reconstructed_image(self):
            solution_vector = np.array([1.0, 2.0, 3.0, 4.0])

            blurred_mapping_matrix = np.array([[1.0, 1.0, 1.0, 1.0],
                                               [1.0, 0.0, 1.0, 1.0],
                                               [1.0, 0.0, 0.0, 0.0]])

            re = recon.Reconstruction(blurred_mapping_matrix=blurred_mapping_matrix,
                                      regularization_matrix=np.ones((1, 1)), curvature_matrix=np.ones((1, 1)),
                                      curvature_reg_matrix=np.ones((1, 1)), solution_vector=solution_vector)

            # Image pixel 0 maps to 4 pixs pixxels -> value is 1.0 + 2.0 + 3.0 + 4.0 = 10.0
            # Image pixel 1 maps to 3 pixs pixxels -> value is 1.0 + 3.0 + 4.0
            # Image pixel 2 maps to 1 pixs pixxels -> value is 1.0

            assert (re.reconstructed_image == np.array([10.0, 8.0, 1.0])).all()
