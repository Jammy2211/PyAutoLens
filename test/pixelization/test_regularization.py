from autolens.pixelization import pixelization

import pytest
import numpy as np


# The regularization matrix, H, is calculated by defining a set of B matrices which describe how pix-plane
# pixels map to one another. For example, if we had a 3x3 square grid:

# ______
# |0|1|2|
# |3|4|5|
# |6|7|8|
# ^^^^^^^

# Lets say we want to regularize this grid so that each square pixel is regularized with a pixel to its right
# and below it.

# So, 0 is regularized with pixels 1 and 3, pixel 1 with 2 and 4, but pixel 2 with only pixel 5, etc. So,
#
# We make two 9 x 9 B matrices, which describe regularization in each direction. So for regularization to the
# right of each pixel:

# B_x = [-1,  1,  0,  0,  0,  0,  0,  0,  0] # [0->1] This, row 0, correspomds to pixel 0 (signified by the -1). The 1's in columns 1 is saying we want to regularize pixel 0 with pixel 1.
#       [ 0, -1,  1,  0,  0,  0,  0,  0,  0] # [1->2] Row 1 for pixel 1 (again, the -1 tells us this), regularized with pixels 2.
#       [ 0,  0, -1,  0,  0,  0,  0,  0,  0] # [] NOTE - pixel 2 is NOT regularized with pixel 3 (check the square grid)!
#       [ 0,  0,  0, -1,  1,  0,  0,  0,  0] # [3->4]
#       [ 0,  0,  0,  0, -1,  1,  0,  0,  0] # [4->5]
#       [ 0,  0,  0,  0,  0, -1,  0,  0,  0] # [] NOTE - pixel 5 not regularized with pixel 6!
#       [ 0,  0,  0,  0,  0,  0, -1,  1,  0] # [6->7]
#       [ 0,  0,  0,  0,  0,  0,  0, -1,  1] # [7->8]
#       [ 0,  0,  0,  0,  0,  0,  0,  0, -1] # [] NOTE - Not regularized with anything

# We now make another B matrix for the regularization beneath each pixel:

# B_y = [-1,  0,  0,  1,  0,  0,  0,  0,  0] # [0->3] This, row 0, correspodns to pixel 0 (signified by the -1). The 1's in columns 3 is saying we want to regularize pixel 0 with pixel 3.
#       [ 0, -1,  0,  0,  1,  0,  0,  0,  0] # [1->4] Row 1 for pixel 1 (again, the -1 tells us this), regularized with pixel 4
#       [ 0,  0, -1,  0,  0,  1,  0,  0,  0] # [2->5]
#       [ 0,  0,  0, -1,  0,  0,  1,  0,  0] # [3->6]
#       [ 0,  0,  0,  0, -1,  0,  0,  1,  0] # [4->7]
#       [ 0,  0,  0,  0,  0, -1,  0,  0,  1] # [5->8]
#       [ 0,  0,  0,  0,  0,  0, -1,  0,  0] # [] No regularized performed in these last 3 rows / pixels
#       [ 0,  0,  0,  0,  0,  0,  0, -1,  0] # []
#       [ 0,  0,  0,  0,  0,  0,  0,  0, -1] # []

# So, we basically just make B matrices representing regularization in each direction. For each, we
# can then compute their corresponding regularization matrix, H, as, H = B * B.T (matrix multiplication)

# So, H_x = B_x.T, * B_x H_y = B_y.T * B_y
# And our overall regularization matrix, H = H_x + H_y

# For an adaptive Voronoi grid, we do this, however we make a B matrix for every shared Voronoi vertex
# of each pix-pixel cluster. This means the number of B matrices we compute is equal to the the number of
# Voronoi vertices in the pix-pixel with the most Voronoi vertices (i.e. the most neighbours a pix-pixel has).

### COMBINING B MATRICES ###

# Whereas the paper_plots above had each -1 going down the diagonal, this is not necessary. It valid to put each pairing
# anywhere. So, if we had a 4x4 B matrix, where pixel 0 regularizes 1, 2 -> 3 and 3 -> 0, we can set this up
# as one matrix even though the pixel 0 comes up twice!

# B = [-1, 1, 0 ,0] # [0->1]
#     [0, 0, 0 ,0] # We can skip rows by making them all zeros.
#     [0, 0, -1 ,1] # [2->3]
#     [1, 0, 0 ,-1] # [3->0] This is valid!

# So, we don't have to make the same number of B matrices as Voronoi vertices, as we can combine them into a few B
# matrices like this

# SKIPPING THE B MATRIX CALCULATION #

# The create_regularization_matrix routines in pixelization don't use the B matrices to compute H!.

# This is because, if you know all the pairs between pix pixels (which the Voronoi gridding can tell you), you
# can bypass the B matrix multiplicaion entirely and enter the values directly into the H matrix (making the
# calculation significantly faster).

# POSITIVE DEFINITE MATRIX #

# The regularization matrix must be positive-definite, to ensure that its determinant is 0. To ensure this
# criteria is met, two forms of regularization schemes are applied in slightly different ways.

# CONSTANT REGULARIZATION #

# For the constant regularization scheme, there is only 1 regularization coefficient that is applied to all
# pix-pair regularizations equally. This means that we when write B, we only need to regularize pixels
# in one direction (e.g. pix pixel 0 regularizes pix pixel 1, but NOT visa versa). For example:

# B = [-1, 1]  [0->1]
#     [0, -1]  1 does not regularization with 0

# WEIGHTED REGULARIZATION #

# For the weighted regularization scheme, each pixis given an 'effective regularization weight', instead
# of applying one constant regularization coefficient to each pix-pixel pair. scheme overall. This is
# because different regions of a pix-plane want different levels of regularization
# (see Nightingale, Dye and Massey) 2018.

# Unlike the constant regularization scheme, regularization pairs are now defined such that all pixels are
# regularized with one another (e.g. if pix pixel 0 regularizes pix pixel 1, pix pixel 1 also
# regularizes pix pixel 0). For example :
#
# B = [-1, 1]  [0->1]
#     [-1, -1]  1 now also regularizes 0
#
# For a constant regularization coefficient this would NOT produce a positive-definite matrix. However, for
# the weighted scheme, it does!

# The regularize weights change the B matrix as shown below, we simply multiply each pix-pixel's effective
# regularization weight by each row of B it has a -1 in, so:

# regularization_weights = [1, 2, 3, 4]

# B = [-1, 1, 0 ,0] # [0->1]
#     [0, -2, 2 ,0] # [1->2]
#     [0, 0, -3 ,3] # [2->3]
#     [4, 0, 0 ,-4] # [3->0] This is valid!

# If our -1's werent down the diagonal this would look like:

# B = [4, 0, 0 ,-4] # [3->0]
#     [0, -2, 2 ,0] # [1->2]
#     [-1, 1, 0 ,0] # [0->1]
#     [0, 0, -3 ,3] # [2->3] This is valid!

class TestRegularizationConstant:

    class TestRegularizationMatrixFromNeighbors:

        def test__1_b_matrix_size_3x3__weights_all_1s__makes_correct_regularization_matrix(self):
            # Here, we define the pix_neighbors first here and make the B matrices based on them.

            # You'll notice that actually, the B Matrix doesn't have to have the -1's going down the diagonal and we
            # don't have to have as many B matrices as we do the pix pixel with the most  vertices. We can combine
            # the rows of each B matrix wherever we like ;0.

            pix_neighbors = np.array([[1, 2], [0], [0]])

            test_b_matrix = np.array([[-1, 1, 0],  # Pair 1
                                      [-1, 0, 1],  # Pair 2
                                      [0, 0, 0, ]])  # Pair 1 flip

            test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix) + 1e-8 * np.identity(3)

            pix = pixelization.ClusterRegConst(pixels=3, regularization_coefficients=(1.0,))
            regularization_matrix = pix.regularization_matrix_from_pix_neighbors(pix_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()
            assert (abs(np.linalg.det(regularization_matrix)) > 1e-8)

        def test__1_b_matrix_size_4x4__weights_all_1s__makes_correct_regularization_matrix(self):
            test_b_matrix = np.array([[-1, 1, 0, 0],
                                      [0, -1, 1, 0],
                                      [0, 0, -1, 1],
                                      [1, 0, 0, -1]])

            test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix) + 1e-8 * np.identity(4)

            pix_neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])

            pix = pixelization.ClusterRegConst(pixels=4, regularization_coefficients=(1.0,))
            regularization_matrix = pix.regularization_matrix_from_pix_neighbors(pix_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()
            assert (abs(np.linalg.det(regularization_matrix)) > 1e-8)

        def test__1_b_matrix_size_4x4__coefficient_2__makes_correct_regularization_matrix(self):
            pix_neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])

            test_b_matrix = 2.0 * np.array([[-1, 1, 0, 0],
                                            [0, -1, 1, 0],
                                            [0, 0, -1, 1],
                                            [1, 0, 0, -1]])

            test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix) + 1e-8 * np.identity(4)

            pix = pixelization.ClusterRegConst(pixels=4, regularization_coefficients=(2.0,))
            regularization_matrix = pix.regularization_matrix_from_pix_neighbors(pix_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()
            assert (abs(np.linalg.det(regularization_matrix)) > 1e-8)

        def test__1_b_matrix_size_9x9__coefficient_2__makes_correct_regularization_matrix(self):
            pix_neighbors = [[1, 3], [4, 2, 0], [1, 5], [4, 6, 0], [7, 1, 5, 3], [4, 2, 8], [7, 3], [4, 8, 6],
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

            test_regularization_matrix = test_regularization_matrix_0 + test_regularization_matrix_1 + 1e-8 * np.identity(
                9)

            pix = pixelization.ClusterRegConst(pixels=9, regularization_coefficients=(1.0,))
            regularization_matrix = pix.regularization_matrix_from_pix_neighbors(pix_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()
            assert (abs(np.linalg.det(regularization_matrix)) > 1e-8)


class TestRegularizationWeighted:

    class TestComputeSourceSignals:

        def test__x3_image_pixels_signals_1s__pix_scale_1__pix_signals_all_1s(self):

            pix = pixelization.ClusterRegWeighted(pixels=3, pix_signal_scale=1.0)

            image_to_pix = np.array([0, 1, 2])
            galaxy_image = np.array([1.0, 1.0, 1.0])

            pix_signals = pix.pix_signals_from_images(image_to_pix, galaxy_image)

            assert (pix_signals == np.array([1.0, 1.0, 1.0])).all()

        def test__x4_image_pixels_signals_1s__pix_signals_still_all_1s(self):
            pix = pixelization.ClusterRegWeighted(pixels=3, pix_signal_scale=1.0)

            image_to_pix = np.array([0, 1, 2, 0])
            galaxy_image = np.array([1.0, 1.0, 1.0, 1.0])

            pix_signals = pix.pix_signals_from_images(image_to_pix, galaxy_image)

            assert (pix_signals == np.array([1.0, 1.0, 1.0])).all()

        def test__galaxy_flux_in_a_pix_pixel_is_double_the_others__pix_signal_is_1_others_a_half(self):
            pix = pixelization.ClusterRegWeighted(pixels=3, pix_signal_scale=1.0)

            image_to_pix = np.array([0, 1, 2])
            galaxy_image = np.array([2.0, 1.0, 1.0])

            pix_signals = pix.pix_signals_from_images(image_to_pix, galaxy_image)

            assert (pix_signals == np.array([1.0, 0.5, 0.5])).all()

        def test__same_as_above_but_pix_scale_2__scales_pix_signals(self):
            pix = pixelization.ClusterRegWeighted(pixels=3, pix_signal_scale=2.0)

            image_to_pix = np.array([0, 1, 2])
            galaxy_image = np.array([2.0, 1.0, 1.0])

            pix_signals = pix.pix_signals_from_images(image_to_pix, galaxy_image)

            assert (pix_signals == np.array([1.0, 0.25, 0.25])).all()

    class TestComputeRegularizationWeights(object):

        def test__pix_signals_all_1s__coefficients_all_1s__weights_all_1s(self):
            pix = pixelization.ClusterRegWeighted(pixels=3, regularization_coefficients=(1.0, 1.0))

            pix_signals = np.array([1.0, 1.0, 1.0])

            weights = pix.regularization_weights_from_pix_signals(pix_signals)

            assert (weights == np.array([1.0, 1.0, 1.0])).all()

        def test__pix_signals_vary__coefficents_all_1s__weights_still_all_1s(self):
            pix = pixelization.ClusterRegWeighted(pixels=3, regularization_coefficients=(1.0, 1.0))

            pix_signals = np.array([0.25, 0.5, 0.75])

            weights = pix.regularization_weights_from_pix_signals(pix_signals)

            assert (weights == np.array([1.0, 1.0, 1.0])).all()

        def test__pix_signals_vary__coefficents_1_and_0__weights_are_pix_signals_squared(self):
            pix = pixelization.ClusterRegWeighted(pixels=3, regularization_coefficients=(1.0, 0.0))

            pix_signals = np.array([0.25, 0.5, 0.75])

            weights = pix.regularization_weights_from_pix_signals(pix_signals)

            assert (weights == np.array([0.25 ** 2.0, 0.5 ** 2.0, 0.75 ** 2.0])).all()

        def test__pix_signals_vary__coefficents_0_and_1__weights_are_1_minus_pix_signals_squared(self):
            pix = pixelization.ClusterRegWeighted(pixels=3, regularization_coefficients=(0.0, 1.0))

            pix_signals = np.array([0.25, 0.5, 0.75])

            weights = pix.regularization_weights_from_pix_signals(pix_signals)

            assert (weights == np.array([0.75 ** 2.0, 0.5 ** 2.0, 0.25 ** 2.0])).all()

    class TestRegularizationMatrixFromNeighbors:

        def test__1_b_matrix_size_4x4__weights_all_1s__makes_correct_regularization_matrix(self):
            pix_neighbors = np.array([[2], [3], [0], [1]])

            test_b_matrix = np.array([[-1, 0, 1, 0],
                                      [0, -1, 0, 1],
                                      [1, 0, -1, 0],
                                      [0, 1, 0, -1]])

            test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix)

            regularization_weights = np.ones((4,))

            pix = pixelization.ClusterRegWeighted(pixels=4)
            regularization_matrix = pix.regularization_matrix_from_pix_neighbors(regularization_weights,
                                                                                          pix_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()

        def test__2_b_matrices_size_3x3__weights_all_1s__makes_correct_regularization_matrix(self):
            # Here, we define the pix_neighbors first here and make the B matrices based on them.

            # You'll notice that actually, the B Matrix doesn't have to have the -1's going down the diagonal and we
            # don't have to have as many B matrices as we do the pix pixel with the most  vertices. We can combine
            # the rows of each B matrix wherever we like ;0.

            pix_neighbors = np.array([[1, 2], [0], [0]])

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

            pix = pixelization.ClusterRegWeighted(pixels=3)
            regularization_matrix = pix.regularization_matrix_from_pix_neighbors(regularization_weights,
                                                                                          pix_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()

        def test__2_b_matrices_size_4x4__weights_all_1s__makes_correct_regularization_matrix(self):
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

            pix_neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])
            regularization_weights = np.ones((4,))

            pix = pixelization.ClusterRegWeighted(pixels=4)
            regularization_matrix = pix.regularization_matrix_from_pix_neighbors(regularization_weights,
                                                                                          pix_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()

        def test__4_b_matrices_size_6x6__weights_all_1s__makes_correct_regularization_matrix(self):
            # Again, lets exploit the freedom we have when setting up our B matrices to make matching it to pairs a
            # lot less Stressful.

            pix_neighbors = [[2, 3, 4], [2, 5], [0, 1, 3, 5], [0, 2], [5, 0], [4, 1, 2]]

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

            pix = pixelization.ClusterRegWeighted(pixels=6)
            regularization_matrix = pix.regularization_matrix_from_pix_neighbors(regularization_weights,
                                                                                          pix_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()

        def test__1_b_matrix_size_3x3_variables_regularization_weights__makes_correct_regularization_matrix(self):
            # Simple case, where we have just one regularization direction, regularizing pixel 0 -> 1 and 1 -> 2.

            # This means our B matrix is:

            # [-1, 1, 0]
            # [0, -1, 1]
            # [0, 0, -1]

            # Regularization Matrix, H = B * B.T.I can

            regularization_weights = np.array([2.0, 4.0, 1.0])

            test_b_matrix = np.array([[-1, 1, 0],  # [[-2, 2, 0], (Matrix)
                                      [1, -1, 0],  # [4, -4, 0], (after)
                                      [0, 0, 0]])  # [0, 0,  0]]) (weights)

            test_b_matrix = (test_b_matrix.T * regularization_weights).T

            test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix)

            pix_neighbors = [[1], [0], []]

            pix = pixelization.ClusterRegWeighted(pixels=3)
            regularization_matrix = pix.regularization_matrix_from_pix_neighbors(regularization_weights,
                                                                                          pix_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()

        def test__2_b_matrices_size_4x4_variables_regularization_weights__makes_correct_regularization_matrix(self):
            # Simple case, where we have just one regularization direction, regularizing pixel 0 -> 1 and 1 -> 2.

            # This means our B matrix is:

            # [-1, 1, 0]
            # [0, -1, 1]
            # [0, 0, -1]

            # Regularization Matrix, H = B * B.T.I can

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

            pix_neighbors = [[1, 2], [0, 2, 3], [0, 1], [1]]

            pix = pixelization.ClusterRegWeighted(pixels=4)
            regularization_matrix = pix.regularization_matrix_from_pix_neighbors(regularization_weights,
                                                                                          pix_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()

        def test__4_b_matrices_size_6x6_with_regularization_weights__makes_correct_regularization_matrix(self):
            pix_neighbors = [[1, 4], [2, 4, 0], [3, 4, 5, 1], [5, 2], [5, 0, 1, 2], [2, 3, 4]]
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

            test_regularization_matrix = test_regularization_matrix_1 + test_regularization_matrix_2 + test_regularization_matrix_3 + test_regularization_matrix_4

            pix = pixelization.ClusterRegWeighted(pixels=6)
            regularization_matrix = pix.regularization_matrix_from_pix_neighbors(regularization_weights,
                                                                                          pix_neighbors)

            assert (regularization_matrix == test_regularization_matrix).all()
