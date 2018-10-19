import numpy as np

from autolens.inversion import regularization


class TestRegularizationConstant:
    class TestRegularizationMatrixFromNeighbors:

        def test__1_b_matrix_size_3x3__weights_all_1s__makes_correct_regularization_matrix(self):
            # Here, we define the pixel_neighbors first here and make the B matrices based on them.

            # You'll notice that actually, the B Matrix doesn't have to have the -1's going down the diagonal and we
            # don't have to have as many B matrices as we do the pix pixel with the most  vertices. We can combine
            # the rows of each B matrix wherever we like ;0.

            pixel_neighbors = np.array([[1, 2], [0], [0]])

            test_b_matrix = np.array([[-1, 1, 0],  # Pair 1
                                      [-1, 0, 1],  # Pair 2
                                      [0, 0, 0, ]])  # Pair 1 flip

            test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix) + 1e-8 * np.identity(3)

            reg = regularization.Constant(coefficients=(1.0,))
            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pixel_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()
            assert (abs(np.linalg.det(regularization_matrix)) > 1e-8)

        def test__1_b_matrix_size_4x4__weights_all_1s__makes_correct_regularization_matrix(self):
            test_b_matrix = np.array([[-1, 1, 0, 0],
                                      [0, -1, 1, 0],
                                      [0, 0, -1, 1],
                                      [1, 0, 0, -1]])

            test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix) + 1e-8 * np.identity(4)

            pixel_neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])

            reg = regularization.Constant(coefficients=(1.0,))
            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pixel_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()
            assert (abs(np.linalg.det(regularization_matrix)) > 1e-8)

        def test__1_b_matrix_size_4x4__coefficient_2__makes_correct_regularization_matrix(self):
            pixel_neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])

            test_b_matrix = 2.0 * np.array([[-1, 1, 0, 0],
                                            [0, -1, 1, 0],
                                            [0, 0, -1, 1],
                                            [1, 0, 0, -1]])

            test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix) + 1e-8 * np.identity(4)

            reg = regularization.Constant(coefficients=(2.0,))
            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pixel_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()
            assert (abs(np.linalg.det(regularization_matrix)) > 1e-8)

        def test__1_b_matrix_size_9x9__coefficient_2__makes_correct_regularization_matrix(self):
            pixel_neighbors = [[1, 3], [4, 2, 0], [1, 5], [4, 6, 0], [7, 1, 5, 3], [4, 2, 8], [7, 3], [4, 8, 6],
                               [7, 5]]

            test_b_matrix_0 = np.array([[-1, 1, 0, 0, 0, 0, 0, 0, 0],
                                        [-1, 0, 0, 1, 0, 0, 0, 0, 0],
                                        [0, -1, 1, 0, 0, 0, 0, 0, 0],
                                        [0, -1, 0, 0, 1, 0, 0, 0, 0],
                                        [0, 0, -1, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, -1, 1, 0, 0, 0, 0],
                                        [0, 0, 0, -1, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, -1, 1, 0, 0, 0],
                                        [0, 0, 0, 0, -1, 0, 0, 1, 0]])

            test_b_matrix_1 = np.array([[0, 0, 0, 0, 0, -1, 0, 0, 1],
                                        [0, 0, 0, 0, 0, 0, -1, 1, 0],
                                        [0, 0, 0, 0, 0, 0, 0, -1, 1],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0]])

            test_regularization_matrix_0 = np.matmul(test_b_matrix_0.T, test_b_matrix_0)
            test_regularization_matrix_1 = np.matmul(test_b_matrix_1.T, test_b_matrix_1)

            test_regularization_matrix = test_regularization_matrix_0 + \
                                         test_regularization_matrix_1 + 1e-8 * np.identity(9)

            reg = regularization.Constant(coefficients=(1.0,))
            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pixel_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()
            assert (abs(np.linalg.det(regularization_matrix)) > 1e-8)


class TestRegularizationWeighted:
    class TestPixelSignals:

        def test__x3_image_pixels_signals_1s__pixel_scale_1__pixel_signals_all_1s(self):
            reg = regularization.Weighted(signal_scale=1.0)

            image_to_pixelization = np.array([0, 1, 2])
            galaxy_image = np.array([1.0, 1.0, 1.0])

            pixel_signals = reg.pixel_signals_from_images(pixels=3, image_to_pixelization=image_to_pixelization,
                                                          galaxy_image=galaxy_image)

            assert (pixel_signals == np.array([1.0, 1.0, 1.0])).all()

        def test__x4_image_pixels_signals_1s__pixel_signals_still_all_1s(self):
            reg = regularization.Weighted(signal_scale=1.0)

            image_to_pixelization = np.array([0, 1, 2, 0])
            galaxy_image = np.array([1.0, 1.0, 1.0, 1.0])

            pixel_signals = reg.pixel_signals_from_images(pixels=3, image_to_pixelization=image_to_pixelization,
                                                          galaxy_image=galaxy_image)

            assert (pixel_signals == np.array([1.0, 1.0, 1.0])).all()

        def test__galaxy_flux_in_a_pixel_pixel_is_double_the_others__pixel_signal_is_1_others_a_half(self):
            reg = regularization.Weighted(signal_scale=1.0)

            image_to_pixelization = np.array([0, 1, 2])
            galaxy_image = np.array([2.0, 1.0, 1.0])

            pixel_signals = reg.pixel_signals_from_images(pixels=3, image_to_pixelization=image_to_pixelization,
                                                          galaxy_image=galaxy_image)

            assert (pixel_signals == np.array([1.0, 0.5, 0.5])).all()

        def test__same_as_above_but_pixel_scale_2__scales_pixel_signals(self):
            reg = regularization.Weighted(signal_scale=2.0)

            image_to_pixelization = np.array([0, 1, 2])
            galaxy_image = np.array([2.0, 1.0, 1.0])

            pixel_signals = reg.pixel_signals_from_images(pixels=3, image_to_pixelization=image_to_pixelization,
                                                          galaxy_image=galaxy_image)

            assert (pixel_signals == np.array([1.0, 0.25, 0.25])).all()

    class TestRegularizationWeights(object):

        def test__pixel_signals_all_1s__coefficients_all_1s__weights_all_1s(self):
            reg = regularization.Weighted(coefficients=(1.0, 1.0))

            pixel_signals = np.array([1.0, 1.0, 1.0])

            weights = reg.regularization_weights_from_pixel_signals(pixel_signals)

            assert (weights == np.array([1.0, 1.0, 1.0])).all()

        def test__pixel_signals_vary__coefficents_all_1s__weights_still_all_1s(self):
            reg = regularization.Weighted(coefficients=(1.0, 1.0))

            pixel_signals = np.array([0.25, 0.5, 0.75])

            weights = reg.regularization_weights_from_pixel_signals(pixel_signals)

            assert (weights == np.array([1.0, 1.0, 1.0])).all()

        def test__pixel_signals_vary__coefficents_1_and_0__weights_are_pixel_signals_squared(self):
            reg = regularization.Weighted(coefficients=(1.0, 0.0))

            pixel_signals = np.array([0.25, 0.5, 0.75])

            weights = reg.regularization_weights_from_pixel_signals(pixel_signals)

            assert (weights == np.array([0.25 ** 2.0, 0.5 ** 2.0, 0.75 ** 2.0])).all()

        def test__pixel_signals_vary__coefficents_0_and_1__weights_are_1_minus_pixel_signals_squared(self):
            reg = regularization.Weighted(coefficients=(0.0, 1.0))

            pixel_signals = np.array([0.25, 0.5, 0.75])

            weights = reg.regularization_weights_from_pixel_signals(pixel_signals)

            assert (weights == np.array([0.75 ** 2.0, 0.5 ** 2.0, 0.25 ** 2.0])).all()

    class TestRegularizationMatrix:

        def test__1_b_matrix_size_4x4__weights_all_1s__makes_correct_regularization_matrix(self):
            reg = regularization.Weighted()

            pixel_neighbors = np.array([[2], [3], [0], [1]])

            test_b_matrix = np.array([[-1, 0, 1, 0],
                                      [0, -1, 0, 1],
                                      [1, 0, -1, 0],
                                      [0, 1, 0, -1]])

            test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix)

            regularization_weights = np.ones((4,))

            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(regularization_weights,
                                                                                   pixel_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()

        def test__2_b_matrices_size_3x3__weights_all_1s__makes_correct_regularization_matrix(self):
            # Here, we define the pixel_neighbors first here and make the B matrices based on them.

            # You'll notice that actually, the B Matrix doesn't have to have the -1's going down the diagonal and we
            # don't have to have as many B matrices as we do the pix pixel with the most  vertices. We can combine
            # the rows of each B matrix wherever we like ;0.

            reg = regularization.Weighted()

            pixel_neighbors = np.array([[1, 2], [0], [0]])

            test_b_matrix_1 = np.array([[-1, 1, 0],  # Pair 1
                                        [-1, 0, 1],  # Pair 2
                                        [1, -1, 0]])  # Pair 1 flip

            test_regularization_matrix_1 = np.matmul(test_b_matrix_1.T, test_b_matrix_1)

            test_b_matrix_2 = np.array([[1, 0, -1],  # Pair 2 flip
                                        [0, 0, 0],
                                        [0, 0, 0]])

            test_regularization_matrix_2 = np.matmul(test_b_matrix_2.T, test_b_matrix_2)

            test_regularization_matrix = test_regularization_matrix_1 + test_regularization_matrix_2

            regularization_weights = np.ones((3))

            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(regularization_weights,
                                                                                   pixel_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()

        def test__2_b_matrices_size_4x4__weights_all_1s__makes_correct_regularization_matrix(self):
            reg = regularization.Weighted()

            test_b_matrix_1 = np.array([[-1, 1, 0, 0],
                                        [0, -1, 1, 0],
                                        [0, 0, -1, 1],
                                        [1, 0, 0, -1]])

            test_regularization_matrix_1 = np.matmul(test_b_matrix_1.T, test_b_matrix_1)

            test_b_matrix_2 = np.array([[-1, 0, 0, 1],
                                        [1, -1, 0, 0],
                                        [0, 1, -1, 0],
                                        [0, 0, 1, -1]])

            test_regularization_matrix_2 = np.matmul(test_b_matrix_2.T, test_b_matrix_2)

            test_regularization_matrix = test_regularization_matrix_1 + test_regularization_matrix_2

            pixel_neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])
            regularization_weights = np.ones((4,))

            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(regularization_weights,
                                                                                   pixel_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()

        def test__4_b_matrices_size_6x6__weights_all_1s__makes_correct_regularization_matrix(self):
            # Again, lets exploit the freedom we have when setting up our B matrices to make matching it to pairs a
            # lot less Stressful.

            reg = regularization.Weighted()

            pixel_neighbors = [[2, 3, 4], [2, 5], [0, 1, 3, 5], [0, 2], [5, 0], [4, 1, 2]]

            test_b_matrix_1 = np.array([[-1, 0, 1, 0, 0, 0],  # Pair 1
                                        [0, -1, 1, 0, 0, 0],  # Pair 2
                                        [-1, 0, 0, 1, 0, 0],  # Pair 3
                                        [0, 0, 0, 0, -1, 1],  # Pair 4
                                        [0, -1, 0, 0, 0, 1],  # Pair 5
                                        [-1, 0, 0, 0, 1, 0]])  # Pair 6

            test_regularization_matrix_1 = np.matmul(test_b_matrix_1.T, test_b_matrix_1)

            test_b_matrix_2 = np.array([[0, 0, -1, 1, 0, 0],  # Pair 7
                                        [0, 0, -1, 0, 0, 1],  # Pair 8
                                        [0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0]])

            test_regularization_matrix_2 = np.matmul(test_b_matrix_2.T, test_b_matrix_2)

            test_b_matrix_3 = np.array([[1, 0, -1, 0, 0, 0],  # Pair 1 flip
                                        [0, 1, -1, 0, 0, 0],  # Pair 2 flip
                                        [1, 0, 0, -1, 0, 0],  # Pair 3 flip
                                        [0, 0, 0, 0, 1, -1],  # Pair 4 flip
                                        [0, 1, 0, 0, 0, -1],  # Pair 5 flip
                                        [1, 0, 0, 0, -1, 0]])  # Pair 6 flip

            test_regularization_matrix_3 = np.matmul(test_b_matrix_3.T, test_b_matrix_3)

            test_b_matrix_4 = np.array([[0, 0, 1, -1, 0, 0],  # Pair 7 flip
                                        [0, 0, 1, 0, 0, -1],  # Pair 8 flip
                                        [0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0]])

            test_regularization_matrix_4 = np.matmul(test_b_matrix_4.T, test_b_matrix_4)

            test_regularization_matrix = test_regularization_matrix_1 + test_regularization_matrix_2 + \
                                         test_regularization_matrix_3 + + test_regularization_matrix_4

            regularization_weights = np.ones((6))

            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(regularization_weights,
                                                                                   pixel_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()

        def test__1_b_matrix_size_3x3_variables_regularization_weights__makes_correct_regularization_matrix(self):
            # Simple case, where we have just one regularization direction, regularizing pixel 0 -> 1 and 1 -> 2.

            # This means our B matrix is:

            # [-1, 1, 0]
            # [0, -1, 1]
            # [0, 0, -1]

            # Regularization Matrix, H = B * B.T.I can

            reg = regularization.Weighted()

            regularization_weights = np.array([2.0, 4.0, 1.0])

            test_b_matrix = np.array([[-1, 1, 0],  # [[-2, 2, 0], (Matrix)
                                      [1, -1, 0],  # [4, -4, 0], (after)
                                      [0, 0, 0]])  # [0, 0,  0]]) (weights)

            test_b_matrix = (test_b_matrix.T * regularization_weights).T

            test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix)

            pixel_neighbors = [[1], [0], []]

            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(regularization_weights,
                                                                                   pixel_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()

        def test__2_b_matrices_size_4x4_variables_regularization_weights__makes_correct_regularization_matrix(self):
            # Simple case, where we have just one regularization direction, regularizing pixel 0 -> 1 and 1 -> 2.

            # This means our B matrix is:

            # [-1, 1, 0]
            # [0, -1, 1]
            # [0, 0, -1]

            # Regularization Matrix, H = B * B.T.I can

            reg = regularization.Weighted()

            regularization_weights = np.array([2.0, 4.0, 1.0, 8.0])

            test_b_matrix_1 = np.array([[-2, 2, 0, 0],
                                        [-2, 0, 2, 0],
                                        [0, -4, 4, 0],
                                        [0, -4, 0, 4]])

            test_b_matrix_2 = np.array([[4, -4, 0, 0],
                                        [1, 0, -1, 0],
                                        [0, 1, -1, 0],
                                        [0, 8, 0, -8]])

            test_regularization_matrix_1 = np.matmul(test_b_matrix_1.T, test_b_matrix_1)
            test_regularization_matrix_2 = np.matmul(test_b_matrix_2.T, test_b_matrix_2)

            test_regularization_matrix = test_regularization_matrix_1 + test_regularization_matrix_2

            pixel_neighbors = [[1, 2], [0, 2, 3], [0, 1], [1]]

            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(regularization_weights,
                                                                                   pixel_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()

        def test__4_b_matrices_size_6x6_with_regularization_weights__makes_correct_regularization_matrix(self):
            reg = regularization.Weighted()

            pixel_neighbors = [[1, 4], [2, 4, 0], [3, 4, 5, 1], [5, 2], [5, 0, 1, 2], [2, 3, 4]]
            regularization_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

            # I'm inputting the regularization weights directly thiss time, as it'd be a pain to multiply with a
            # loop.

            test_b_matrix_1 = np.array([[-1, 1, 0, 0, 0, 0],  # Pair 1
                                        [-1, 0, 0, 0, 1, 0],  # Pair 2
                                        [0, -2, 2, 0, 0, 0],  # Pair 3
                                        [0, -2, 0, 0, 2, 0],  # Pair 4
                                        [0, 0, -3, 3, 0, 0],  # Pair 5
                                        [0, 0, -3, 0, 3, 0]])  # Pair 6

            test_b_matrix_2 = np.array([[0, 0, -3, 0, 0, 3],  # Pair 7
                                        [0, 0, 0, -4, 0, 4],  # Pair 8
                                        [0, 0, 0, 0, -5, 5],  # Pair 9
                                        [0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0]])

            # Now do the same pairs but with the regularization direction and weights swapped.

            test_b_matrix_3 = np.array([[2, -2, 0, 0, 0, 0],  # Pair 1
                                        [5, 0, 0, 0, -5, 0],  # Pair 2
                                        [0, 3, -3, 0, 0, 0],  # Pair 3
                                        [0, 5, 0, 0, -5, 0],  # Pair 4
                                        [0, 0, 4, -4, 0, 0],  # Pair 5
                                        [0, 0, 5, 0, -5, 0]])  # Pair 6

            test_b_matrix_4 = np.array([[0, 0, 6, 0, 0, -6],  # Pair 7
                                        [0, 0, 0, 6, 0, -6],  # Pair 8
                                        [0, 0, 0, 0, 6, -6],  # Pair 9
                                        [0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0]])

            test_regularization_matrix_1 = np.matmul(test_b_matrix_1.T, test_b_matrix_1)
            test_regularization_matrix_2 = np.matmul(test_b_matrix_2.T, test_b_matrix_2)
            test_regularization_matrix_3 = np.matmul(test_b_matrix_3.T, test_b_matrix_3)
            test_regularization_matrix_4 = np.matmul(test_b_matrix_4.T, test_b_matrix_4)

            test_regularization_matrix = test_regularization_matrix_1 + test_regularization_matrix_2 + \
                                         test_regularization_matrix_3 + test_regularization_matrix_4

            regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(regularization_weights,
                                                                                   pixel_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()
