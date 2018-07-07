from src.pixelization import pixelization

import pytest
import numpy as np
import math


def coordinates_to_source_pixels_via_nearest_neighbour(coordinates, source_centers):
    """ Match a set of image_grid image_grid-pixel image_grid to their closest source-data_to_image, using the source-pixel centers (x,y).

        This method uses a nearest neighbour search between every image_image-pixel coordinate and set of source-pixel \
        centers, thus it is slow when the number of image_grid image_grid-pixel image_grid or source-data_to_image is large. However, it
        is probably the fastest routine for low numbers of image_grid image_grid-data_to_image and source-data_to_image.

        Parameters
        ----------
        coordinates : [(float, float)]
            The x and y image_grid image_grid-pixel image_grid to be matched to the source-pixel centers.
        image_total : int
            The total number of image_grid data_to_image in the image_grid.
        image_total : int
            The total number of image_grid data_to_image in the image_grid image_grid-grid_coords.
        source_centers: [(float, float)
            The source-data_to_image centers the image_grid image_grid-pixel image_grid are matched with.

        Returns
        ----------
        image_image_to_source : [int, int]
            The index in source_pixel_centers each image_grid and image_grid-image_coordinate is matched with. (e.g. if the fifth
            image_coordinate of the third image_grid pixel is closest to the 3rd source-pixel in source_pixel_centers,
            image_image_to_source[2,4] = 2).

     """

    def compute_squared_separation(coordinate1, coordinate2):
        """Computes the squared separation of two image_grid (no square root for efficiency)"""
        return (coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2

    image_pixels = coordinates.shape[0]

    image_to_source = np.zeros((image_pixels))

    for image_index, image_coordinate in enumerate(coordinates):
        distances = list(map(lambda centers: compute_squared_separation(image_coordinate, centers),
                             source_centers))

        image_to_source[image_index] = (np.argmin(distances))

    return image_to_source


class MockMapperCluster(object):

    def __init__(self, cluster_to_image, image_to_cluster):
        self.cluster_to_image = cluster_to_image
        self.image_to_cluster = image_to_cluster


class MockGeometry(object):

    def __init__(self, y_min, y_max, x_min, x_max, y_pixel_scale, x_pixel_scale):

        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_pixel_scale = y_pixel_scale
        self.x_pixel_scale = x_pixel_scale


class TestPixelization:

    class TestMappingMatrix:

        def test__3_image_pixels__6_source_pixels__sub_grid_1x1(self):

            sub_to_source = np.array([0, 1, 2])
            sub_to_image = np.array([0, 1, 2])

            pix = pixelization.Pixelization(pixels=6)
            mapping_matrix = pix.create_mapping_matrix(sub_to_source, sub_to_image, image_pixels=3, sub_grid_size=1)

            assert (mapping_matrix == np.array([[1, 0, 0, 0, 0, 0],  # Image pixel 0 maps to source pixel 0.
                                                [0, 1, 0, 0, 0, 0],  # Image pixel 1 maps to source pixel 1.
                                                [0, 0, 1, 0, 0, 0]])).all()  # Image pixel 2 maps to source pixel 2

        def test__5_image_pixels__8_source_pixels__sub_grid_1x1(self):

            sub_to_source = np.array([0, 1, 2, 7, 6])
            sub_to_image = np.array([0, 1, 2, 3, 4])

            pix = pixelization.Pixelization(pixels=8)
            mapping_matrix = pix.create_mapping_matrix(sub_to_source, sub_to_image, image_pixels=5, sub_grid_size=1)

            assert (mapping_matrix == np.array(
                [[1, 0, 0, 0, 0, 0, 0, 0],  # Image image_to_pixel 0 and 3 map to source pixel 0.
                 [0, 1, 0, 0, 0, 0, 0, 0],  # Image image_to_pixel 1 and 4 map to source pixel 1.
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0]])).all()  # Image image_to_pixel 2 and 5 map to source pixel 2

        def test__5_image_pixels__8_source_pixels__sub_grid_2x2__no_overlapping_pixels(self):

            sub_to_source = np.array([0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2])
            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

            pix = pixelization.Pixelization(pixels=8)
            mapping_matrix = pix.create_mapping_matrix(sub_to_source, sub_to_image, image_pixels=5, sub_grid_size=2)

            assert (mapping_matrix == np.array(
                [[0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0],
                 [0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0],
                 [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
                 [0.25, 0.25, 0, 0.25, 0, 0, 0, 0.25],
                 [0, 0, 0.25, 0, 0.25, 0, 0.25, 0.25]])).all()

        def test__5_image_pixels__8_source_pixels__sub_grid_2x2__include_overlapping_pixels(self):

            sub_to_source = np.array([0, 0, 0, 1, 1, 1, 0, 0, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2])
            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

            pix = pixelization.Pixelization(pixels=8)
            mapping_matrix = pix.create_mapping_matrix(sub_to_source, sub_to_image, image_pixels=5, sub_grid_size=2)

            assert (mapping_matrix == np.array(
                [[0.75, 0.25, 0, 0, 0, 0, 0, 0],
                 [0.5, 0.5, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
                 [0.25, 0.25, 0, 0.25, 0, 0, 0, 0.25],
                 [0, 0, 0.25, 0, 0.25, 0, 0.25, 0.25]])).all()

        def test__3_image_pixels__6_source_pixels__sub_grid_4x4(self):

            sub_to_source = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                      0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3])


            sub_to_image = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

            pix = pixelization.Pixelization(pixels=6)
            mapping_matrix = pix.create_mapping_matrix(sub_to_source, sub_to_image, image_pixels=3, sub_grid_size=4)

            assert (mapping_matrix == np.array(
                [[0.75, 0.25, 0, 0, 0, 0],
                 [0, 0, 1.0, 0, 0, 0],
                 [0.1875, 0.1875, 0.1875, 0.1875, 0.125, 0.125]])).all()

    class TestComputeSourceSignals:

        def test__x3_image_pixels_signals_1s__source_scale_1__source_signals_all_1s(self):
            pix = pixelization.Pixelization(pixels=3, source_signal_scale=1.0)

            image_to_source = np.array([0, 1, 2])
            galaxy_image = np.array([1.0, 1.0, 1.0])

            source_signals = pix.compute_source_signals(image_to_source, galaxy_image)

            assert (source_signals == np.array([1.0, 1.0, 1.0])).all()

        def test__x4_image_pixels_signals_1s__source_signals_still_all_1s(self):
            pix = pixelization.Pixelization(pixels=3, source_signal_scale=1.0)

            image_to_source = np.array([0, 1, 2, 0])
            galaxy_image = np.array([1.0, 1.0, 1.0, 1.0])

            source_signals = pix.compute_source_signals(image_to_source, galaxy_image)

            assert (source_signals == np.array([1.0, 1.0, 1.0])).all()

        def test__galaxy_flux_in_a_source_pixel_is_double_the_others__source_signal_is_1_others_a_half(self):
            pix = pixelization.Pixelization(pixels=3, source_signal_scale=1.0)

            image_to_source = np.array([0, 1, 2])
            galaxy_image = np.array([2.0, 1.0, 1.0])

            source_signals = pix.compute_source_signals(image_to_source, galaxy_image)

            assert (source_signals == np.array([1.0, 0.5, 0.5])).all()

        def test__same_as_above_but_source_scale_2__scales_source_signals(self):
            pix = pixelization.Pixelization(pixels=3, source_signal_scale=2.0)

            image_to_source = np.array([0, 1, 2])
            galaxy_image = np.array([2.0, 1.0, 1.0])

            source_signals = pix.compute_source_signals(image_to_source, galaxy_image)

            assert (source_signals == np.array([1.0, 0.25, 0.25])).all()

    class TestComputeRegularizationWeights(object):

        def test__source_signals_all_1s__coefficients_all_1s__weights_all_1s(self):
            pix = pixelization.Pixelization(pixels=3, regularization_coefficients=(1.0, 1.0))

            source_signals = np.array([1.0, 1.0, 1.0])

            weights = pix.compute_regularization_weights(source_signals)

            assert (weights == np.array([1.0, 1.0, 1.0])).all()

        def test__source_signals_vary__coefficents_all_1s__weights_still_all_1s(self):
            pix = pixelization.Pixelization(pixels=3, regularization_coefficients=(1.0, 1.0))

            source_signals = np.array([0.25, 0.5, 0.75])

            weights = pix.compute_regularization_weights(source_signals)

            assert (weights == np.array([1.0, 1.0, 1.0])).all()

        def test__source_signals_vary__coefficents_1_and_0__weights_are_source_signals_squared(self):
            pix = pixelization.Pixelization(pixels=3, regularization_coefficients=(1.0, 0.0))

            source_signals = np.array([0.25, 0.5, 0.75])

            weights = pix.compute_regularization_weights(source_signals)

            assert (weights == np.array([0.25 ** 2.0, 0.5 ** 2.0, 0.75 ** 2.0])).all()

        def test__source_signals_vary__coefficents_0_and_1__weights_are_1_minus_source_signals_squared(self):
            pix = pixelization.Pixelization(pixels=3, regularization_coefficients=(0.0, 1.0))

            source_signals = np.array([0.25, 0.5, 0.75])

            weights = pix.compute_regularization_weights(source_signals)

            assert (weights == np.array([0.75 ** 2.0, 0.5 ** 2.0, 0.25 ** 2.0])).all()

    class TestRegularizationMatrix(object):
        # The regularization matrix, H, is calculated by defining a set of B matrices which describe how source-plane
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

        # B_x = [-1,  1,  0,  0,  0,  0,  0,  0,  0] # [0->1] This, row 0, correspodns to pixel 0 (signified by the -1). The 1's in columns 1 is saying we want to regularize pixel 0 with pixel 1.
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
        # of each source-pixel cluster. This means the number of B matrices we compute is equal to the the number of
        # Voronoi vertices in the source-pixel with the most Voronoi vertices (i.e. the most neighbours a source-pixel has).

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

        #### SKIPPING THE B MATRIX CALCULATION ####

        # The create_regularization_matrix routines in pixelization don't use the B matrices to compute H!.

        # This is because, if you know all the pairs between source pixels (which the Voronoi gridding can tell you), you
        # can bypass the B matrix multiplicaion entirely and enter the values directly into the H matrix (making the
        # calculation significantly faster).

        #### POSITIVE DEFINITE MATRIX ####

        # The regularization matrix must be positive-definite, to ensure that its determinant is 0. To ensure this
        # criteria is met, two forms of regularization schemes are applied in slightly different ways.

        #### CONSTANT REGULARIZATION ####

        # For the constant regularization scheme, there is only 1 regularization coefficient that is applied to all
        # source-pair regularizations equally. This means that we when write B, we only need to regularize pixels
        # in one direction (e.g. source pixel 0 regularizes source pixel 1, but NOT visa versa). For example:

        # B = [-1, 1]  [0->1]
        #     [0, -1]  1 does not regularization with 0

        #### WEIGHTED REGULARIZATION ####

        # For the weighted regularization scheme, each sourceis given an 'effective regularization weight', instead
        # of applying one constant regularization coefficient to each source-pixel pair. scheme overall. This is
        # because different regions of a source-plane want different levels of regularization
        # (see Nightingale, Dye and Massey) 2018.

        # Unlike the constant regularization scheme, regularization pairs are now defined such that all pixels are
        # regularized with one another (e.g. if source pixel 0 regularizes source pixel 1, source pixel 1 also
        # regularizes source pixel 0). For example :
        #
        # B = [-1, 1]  [0->1]
        #     [-1, -1]  1 now also regularizes 0
        #
        # For a constant regularization coefficient this would NOT produce a positive-definite matrix. However, for
        # the weighted scheme, it does!

        # The regularize weights change the B matrix as shown below, we simply multiply each source-pixel's effective
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

        class TestConstantRegularization:

            def test__1_B_matrix_size_3x3__weights_all_1s__makes_correct_regularization_matrix(self):
                # Here, we define the source_neighbors first here and make the B matrices based on them.

                # You'll notice that actually, the B Matrix doesn't have to have the -1's going down the diagonal and we don't
                # have to have as many B matrices as we do the source pixel with the most  vertices. We can combine the
                # rows of each B matrix wherever we like ;0.

                source_neighbors = np.array([[1, 2], [0], [0]])

                test_b_matrix = np.array([[-1, 1, 0],  # Pair 1
                                          [-1, 0, 1],  # Pair 2
                                          [0, 0, 0, ]])  # Pair 1 flip

                test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix) + 1e-8 * np.identity(3)

                pix = pixelization.Pixelization(pixels=3, regularization_coefficients=(1.0,))
                regularization_matrix = pix.create_constant_regularization_matrix(source_neighbors)

                assert (regularization_matrix == test_regularization_matrix).all()
                assert (abs(np.linalg.det(regularization_matrix)) > 1e-8)

            def test__1_B_matrix_size_4x4__weights_all_1s__makes_correct_regularization_matrix(self):
                test_b_matrix = np.array([[-1, 1, 0, 0],
                                          [0, -1, 1, 0],
                                          [0, 0, -1, 1],
                                          [1, 0, 0, -1]])

                test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix) + 1e-8 * np.identity(4)

                source_neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])

                pix = pixelization.Pixelization(pixels=4, regularization_coefficients=(1.0,))
                regularization_matrix = pix.create_constant_regularization_matrix(source_neighbors)

                assert (regularization_matrix == test_regularization_matrix).all()
                assert (abs(np.linalg.det(regularization_matrix)) > 1e-8)

            def test__1_B_matrix_size_4x4__coefficient_2__makes_correct_regularization_matrix(self):
                source_neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])

                test_b_matrix = 2.0 * np.array([[-1, 1, 0, 0],
                                                [0, -1, 1, 0],
                                                [0, 0, -1, 1],
                                                [1, 0, 0, -1]])

                test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix) + 1e-8 * np.identity(4)

                pix = pixelization.Pixelization(pixels=4, regularization_coefficients=(2.0,))
                regularization_matrix = pix.create_constant_regularization_matrix(source_neighbors)

                assert (regularization_matrix == test_regularization_matrix).all()
                assert (abs(np.linalg.det(regularization_matrix)) > 1e-8)

            def test__1_B_matrix_size_9x9__coefficient_2__makes_correct_regularization_matrix(self):
                source_neighbors = [[1, 3], [4, 2, 0], [1, 5], [4, 6, 0], [7, 1, 5, 3], [4, 2, 8], [7, 3], [4, 8, 6],
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

                test_regularization_matrix = test_regularization_matrix_0 + test_regularization_matrix_1 + \
                                             1e-8 * np.identity(9)

                pix = pixelization.Pixelization(pixels=9, regularization_coefficients=(1.0,))
                regularization_matrix = pix.create_constant_regularization_matrix(source_neighbors)

                assert (regularization_matrix == test_regularization_matrix).all()
                assert (abs(np.linalg.det(regularization_matrix)) > 1e-8)

        class TestWeightedRegularization:

            def test__1_B_matrix_size_4x4__weights_all_1s__makes_correct_regularization_matrix(self):
                source_neighbors = np.array([[2], [3], [0], [1]])

                test_b_matrix = np.array([[-1, 0, 1, 0],
                                          [0, -1, 0, 1],
                                          [1, 0, -1, 0],
                                          [0, 1, 0, -1]])

                test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix)

                regularization_weights = np.ones((4))

                pix = pixelization.Pixelization(pixels=4)
                regularization_matrix = pix.create_weighted_regularization_matrix(regularization_weights,
                                                                                  source_neighbors)

                assert (regularization_matrix == test_regularization_matrix).all()

            def test__2_B_matrices_size_3x3__weights_all_1s__makes_correct_regularization_matrix(self):
                # Here, we define the source_neighbors first here and make the B matrices based on them.

                # You'll notice that actually, the B Matrix doesn't have to have the -1's going down the diagonal and we don't
                # have to have as many B matrices as we do the source pixel with the most  vertices. We can combine the
                # rows of each B matrix wherever we like ;0.

                source_neighbors = np.array([[1, 2], [0], [0]])

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

                pix = pixelization.Pixelization(pixels=3)
                regularization_matrix = pix.create_weighted_regularization_matrix(regularization_weights,
                                                                                  source_neighbors)

                assert (regularization_matrix == test_regularization_matrix).all()

            def test__2_B_matrices_size_4x4__weights_all_1s__makes_correct_regularization_matrix(self):
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

                source_neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])
                regularization_weights = np.ones((4))

                pix = pixelization.Pixelization(pixels=4)
                regularization_matrix = pix.create_weighted_regularization_matrix(regularization_weights,
                                                                                  source_neighbors)

                assert (regularization_matrix == test_regularization_matrix).all()

            def test__4_B_matrices_size_6x6__weights_all_1s__makes_correct_regularization_matrix(self):
                # Again, lets exploit the freedom we have when setting up our B matrices to make matching it to pairs a lot less
                # Stressful.

                source_neighbors = [[2, 3, 4], [2, 5], [0, 1, 3, 5], [0, 2], [5, 0], [4, 1, 2]]

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

                pix = pixelization.Pixelization(pixels=6)
                regularization_matrix = pix.create_weighted_regularization_matrix(regularization_weights,
                                                                                  source_neighbors)

                assert (regularization_matrix == test_regularization_matrix).all()

            def test__1_B_matrix_size_3x3_variables_regularization_weights__makes_correct_regularization_matrix(self):
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

                source_neighbors = [[1], [0], []]

                pix = pixelization.Pixelization(pixels=3)
                regularization_matrix = pix.create_weighted_regularization_matrix(regularization_weights,
                                                                                  source_neighbors)

                assert (regularization_matrix == test_regularization_matrix).all()

            def test__2_B_matrices_size_4x4_variables_regularization_weights__makes_correct_regularization_matrix(self):
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

                source_neighbors = [[1, 2], [0, 2, 3], [0, 1], [1]]

                pix = pixelization.Pixelization(pixels=4)
                regularization_matrix = pix.create_weighted_regularization_matrix(regularization_weights,
                                                                                  source_neighbors)

                assert (regularization_matrix == test_regularization_matrix).all()

            def test__4_B_matrices_size_6x6_with_regularization_weights__makes_correct_regularization_matrix(self):
                source_neighbors = [[1, 4], [2, 4, 0], [3, 4, 5, 1], [5, 2], [5, 0, 1, 2], [2, 3, 4]]
                regularization_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

                # I'm inputting the regularizationo weights directly thiss time, as it'd be a pain to multiply with a loop.

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

                pix = pixelization.Pixelization(pixels=6)
                regularization_matrix = pix.create_weighted_regularization_matrix(regularization_weights,
                                                                                  source_neighbors)

                assert (regularization_matrix == test_regularization_matrix).all()


class TestRectangularPixelization:

    class TestConstructor:

        def test__number_of_pixels_and_regularization_set_up_correctly(self):
            pix = pixelization.RectangularPixelization(shape=(3, 3), regularization_coefficients=(2.0,))

            assert pix.shape == (3, 3)
            assert pix.pixels == 9
            assert pix.regularization_coefficients == (2.0,)

    class TestSetupGeometry:

        def test__3x3_grid__buffer_is_small__coordinates_give_min_minus_1_max_1__sets_up_geometry_correctly(self):

            pix = pixelization.RectangularPixelization(shape=(3,3))

            source_coordinates = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                           [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                           [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]])

            geometry = pix.compute_geometry(source_coordinates, buffer=1e-8)

            assert geometry.y_min == -1.0 - 1e-8
            assert geometry.y_max == 1.0 + 1e-8
            assert geometry.x_min == -1.0 - 1e-8
            assert geometry.x_max == 1.0 + 1e-8
            assert geometry.y_pixel_scale == (geometry.y_max - geometry.y_min) / 3
            assert geometry.x_pixel_scale == (geometry.x_max - geometry.x_min) / 3

        def test__3x3_grid__same_as_above_change_buffer(self):

            pix = pixelization.RectangularPixelization(shape=(3,3))

            source_coordinates = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                           [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                           [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]])

            geometry = pix.compute_geometry(source_coordinates, buffer=1e-4)

            assert geometry.y_min == -1.0 - 1e-4
            assert geometry.y_max == 1.0 + 1e-4
            assert geometry.x_min == -1.0 - 1e-4
            assert geometry.x_max == 1.0 + 1e-4
            assert geometry.y_pixel_scale == (geometry.y_max - geometry.y_min) / 3
            assert geometry.x_pixel_scale == (geometry.x_max - geometry.x_min) / 3

        def test__5x4_grid__buffer_is_small(self):

            pix = pixelization.RectangularPixelization(shape=(5,4))

            source_coordinates = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                           [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                           [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]])

            geometry = pix.compute_geometry(source_coordinates, buffer=1e-8)

            assert geometry.y_min == -1.0 - 1e-8
            assert geometry.y_max == 1.0 + 1e-8
            assert geometry.x_min == -1.0 - 1e-8
            assert geometry.x_max == 1.0 + 1e-8
            assert geometry.y_pixel_scale == (geometry.y_max - geometry.y_min) / 5
            assert geometry.x_pixel_scale == (geometry.x_max - geometry.x_min) / 4

        def test__3x3_grid__larger_range_of_coordinates(self):

            pix = pixelization.RectangularPixelization(shape=(3,3))

            source_coordinates = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

            geometry = pix.compute_geometry(source_coordinates, buffer=1e-8)

            assert geometry.y_min == 1.0 - 1e-8
            assert geometry.y_max == 7.0 + 1e-8
            assert geometry.x_min == 2.0 - 1e-8
            assert geometry.x_max == 8.0 + 1e-8
            assert geometry.y_pixel_scale == (geometry.y_max - geometry.y_min) / 3
            assert geometry.x_pixel_scale == (geometry.x_max - geometry.x_min) / 3

    class TestComputeSourceNeighbors:

        def test__compute_source_neighbors__3x3_grid(self):
            # |0|1|2|
            # |3|4|5|
            # |6|7|8|

            pix = pixelization.RectangularPixelization(shape=(3, 3), regularization_coefficients=(1.0,))

            source_neighbors = pix.compute_source_neighbors()

            assert source_neighbors[0] == [1, 3]
            assert source_neighbors[1] == [0, 2, 4]
            assert source_neighbors[2] == [1, 5]
            assert source_neighbors[3] == [0, 4, 6]
            assert source_neighbors[4] == [1, 3, 5, 7]
            assert source_neighbors[5] == [2, 4, 8]
            assert source_neighbors[6] == [3, 7]
            assert source_neighbors[7] == [4, 6, 8]
            assert source_neighbors[8] == [5, 7]

        def test__compute_source_neighbors__3x4_grid(self):
            # |0|1| 2| 3|
            # |4|5| 6| 7|
            # |8|9|10|11|

            pix = pixelization.RectangularPixelization(shape=(3, 4), regularization_coefficients=(1.0,))

            source_neighbors = pix.compute_source_neighbors()

            assert source_neighbors[0] == [1, 4]
            assert source_neighbors[1] == [0, 2, 5]
            assert source_neighbors[2] == [1, 3, 6]
            assert source_neighbors[3] == [2, 7]
            assert source_neighbors[4] == [0, 5, 8]
            assert source_neighbors[5] == [1, 4, 6, 9]
            assert source_neighbors[6] == [2, 5, 7, 10]
            assert source_neighbors[7] == [3, 6, 11]
            assert source_neighbors[8] == [4, 9]
            assert source_neighbors[9] == [5, 8, 10]
            assert source_neighbors[10] == [6, 9, 11]
            assert source_neighbors[11] == [7, 10]

        def test__compute_source_neighbors__4x3_grid(self):
            # |0| 1| 2|
            # |3| 4| 5|
            # |6| 7| 8|
            # |9|10|11|

            pix = pixelization.RectangularPixelization(shape=(4, 3), regularization_coefficients=(1.0,))

            source_neighbors = pix.compute_source_neighbors()

            assert source_neighbors[0] == [1, 3]
            assert source_neighbors[1] == [0, 2, 4]
            assert source_neighbors[2] == [1, 5]
            assert source_neighbors[3] == [0, 4, 6]
            assert source_neighbors[4] == [1, 3, 5, 7]
            assert source_neighbors[5] == [2, 4, 8]
            assert source_neighbors[6] == [3, 7, 9]
            assert source_neighbors[7] == [4, 6, 8, 10]
            assert source_neighbors[8] == [5, 7, 11]
            assert source_neighbors[9] == [6, 10]
            assert source_neighbors[10] == [7, 9, 11]
            assert source_neighbors[11] == [8, 10]

        def test__compute_source_neighbors__4x4_grid(self):
            # |0 | 1| 2| 3|
            # |4 | 5| 6| 7|
            # |8 | 9|10|11|
            # |12|13|14|15|

            pix = pixelization.RectangularPixelization(shape=(4, 4), regularization_coefficients=(1.0,))

            source_neighbors = pix.compute_source_neighbors()

            assert source_neighbors[0] == [1, 4]
            assert source_neighbors[1] == [0, 2, 5]
            assert source_neighbors[2] == [1, 3, 6]
            assert source_neighbors[3] == [2, 7]
            assert source_neighbors[4] == [0, 5, 8]
            assert source_neighbors[5] == [1, 4, 6, 9]
            assert source_neighbors[6] == [2, 5, 7, 10]
            assert source_neighbors[7] == [3, 6, 11]
            assert source_neighbors[8] == [4, 9, 12]
            assert source_neighbors[9] == [5, 8, 10, 13]
            assert source_neighbors[10] == [6, 9, 11, 14]
            assert source_neighbors[11] == [7, 10, 15]
            assert source_neighbors[12] == [8, 13]
            assert source_neighbors[13] == [9, 12, 14]
            assert source_neighbors[14] == [10, 13, 15]
            assert source_neighbors[15] == [11, 14]

    class TestComputeImageToSource:

        def test__3x3_grid_of_source_coordinates__1_coordinate_per_square_source_pixel__in_centre_of_pixels(self):

            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

            source_coordinates = np.array([[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                                           [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                                           [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]])

            pix = pixelization.RectangularPixelization(shape=(3,3), regularization_coefficients=(1.0,))

            image_to_source = pix.compute_pixel_to_source(source_coordinates, pix.compute_geometry(source_coordinates))

            assert (image_to_source == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

        def test__3x3_grid_of_source_coordinates__1_coordinate_per_square_source_pixel__near_edges_of_pixels(self):

            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

            source_coordinates = np.array([[-0.34, -0.34], [-0.34, 0.325], [-1.0, 1.0],
                                           [ -0.32, -1.0], [-0.32, 0.32], [ 0.0, 1.0],
                                           [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]])

            pix = pixelization.RectangularPixelization(shape=(3,3), regularization_coefficients=(1.0,))

            image_to_source = pix.compute_pixel_to_source(source_coordinates, pix.compute_geometry(source_coordinates))

            assert (image_to_source == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

        def test__3x3_grid_of_source_coordinates__add_multiple_coordinates_to_1_source_pixel(self):

            #                  _ _ _
            # -1.0 to -(1/3)  |_|_|_|
            # -(1/3) to (1/3) |_|_|_|
            #  (1/3) to 1.0   |_|_|_|

            source_coordinates = np.array([[-1.0, -1.0], [0.0, 0.0], [-1.0, 1.0],
                                           [ 0.0,  0.0], [0.0, 0.0], [ 0.0, 0.0],
                                           [ 1.0, -1.0], [0.0, 0.0], [ 1.0, 1.0]])

            pix = pixelization.RectangularPixelization(shape=(3,3), regularization_coefficients=(1.0,))

            image_to_source = pix.compute_pixel_to_source(source_coordinates, pix.compute_geometry(source_coordinates))

            assert (image_to_source == np.array([0, 4, 2, 4, 4, 4, 6, 4, 8])).all()

        def test__4x3_grid_of_source_coordinates__1_coordinate_in_each_pixel(self):

            #   _ _ _
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|

            # Boundaries for column pixel 0 -1.0 to -(1/3)
            # Boundaries for column pixel 1 -(1/3) to (1/3)
            # Boundaries for column pixel 2  (1/3) to 1.0

            # Bounadries for row pixel 0 -1.0 to -0.5
            # Bounadries for row pixel 1 -0.5 to 0.0
            # Bounadries for row pixel 2  0.0 to 0.5
            # Bounadries for row pixel 3  0.5 to 1.0

            source_coordinates = np.array([[-1.0, -1.0], [-1.0, -0.32], [-1.0, 0.34], [-0.49, -1.0],
                                           [0.01, 0.34], [1.0, 1.0]])

            pix = pixelization.RectangularPixelization(shape=(4,3), regularization_coefficients=(1.0,))

            image_to_source = pix.compute_pixel_to_source(source_coordinates, pix.compute_geometry(source_coordinates))

            assert (image_to_source == np.array([0, 1, 2, 3, 8, 11])).all()

        def test__3x4_grid_of_source_coordinates__1_coordinate_in_each_pixel(self):

            #   _ _ _ _
            #  |_|_|_|_|
            #  |_|_|_|_|
            #  |_|_|_|_|

            # Boundaries for row pixel 0 -1.0 to -(1/3)
            # Boundaries for row pixel 1 -(1/3) to (1/3)
            # Boundaries for row pixel 2  (1/3) to 1.0

            # Bounadries for column pixel 0 -1.0 to -0.5
            # Bounadries for column pixel 1 -0.5 to 0.0
            # Bounadries for column pixel 2  0.0 to 0.5
            # Bounadries for column pixel 3  0.5 to 1.0

            source_coordinates = np.array([[-1.0, -1.0], [-1.0, -0.49], [-1.0, 0.01], [-0.32, 0.01],
                                           [0.34, -0.01], [1.0, 1.0]])

            pix = pixelization.RectangularPixelization(shape=(3,4), regularization_coefficients=(1.0,))

            image_to_source = pix.compute_pixel_to_source(source_coordinates, pix.compute_geometry(source_coordinates))

            assert (image_to_source == np.array([0, 1, 2, 6, 9, 11])).all()

        def test__3x3_grid__change_arcsecond_dimensions_size__grid_adapts_accordingly(self):

            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.5 to -0.5
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 -0.5 to 0.5
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2  0.5 to 1.5

            source_coordinates = np.array([[-1.5, -1.5], [-1.0, 0.0], [-1.0, 0.6], [1.4, 0.0], [1.5, 1.5]])

            pix = pixelization.RectangularPixelization(shape=(3,3), regularization_coefficients=(1.0,))

            image_to_source = pix.compute_pixel_to_source(source_coordinates, pix.compute_geometry(source_coordinates))

            assert (image_to_source == np.array([0, 1, 2, 7, 8])).all()

        def test__3x3_grid__change_arcsecond_dimensions__not_symmetric(self):

            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.5 to -0.5
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 -0.5 to 0.5
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2  0.5 to 1.5

            source_coordinates = np.array([[-1.0, -1.5], [-1.0, -0.49], [-0.32, -1.5], [-0.32, 0.51], [1.0, 1.5]])

            pix = pixelization.RectangularPixelization(shape=(3,3), regularization_coefficients=(1.0,))

            image_to_source = pix.compute_pixel_to_source(source_coordinates, pix.compute_geometry(source_coordinates))

            assert (image_to_source == np.array([0, 1, 3, 5, 8])).all()

        def test__4x3_grid__change_arcsecond_dimensions__not_symmetric(self):

            #   _ _ _
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|

            source_coordinates = np.array([[-1.0, -1.5], [-1.0, -0.49], [-0.49, -1.5], [0.6, 0.0], [1.0, 1.5]])

            pix = pixelization.RectangularPixelization(shape=(4,3), regularization_coefficients=(1.0,))

            image_to_source = pix.compute_pixel_to_source(source_coordinates, pix.compute_geometry(source_coordinates))

            assert (image_to_source == np.array([0, 1, 3, 10, 11])).all()

        def test__3x4_grid__change_arcsecond_dimensions__not_symmetric(self):

            #   _ _ _ _
            #  |_|_|_|_|
            #  |_|_|_|_|
            #  |_|_|_|_|

            source_coordinates = np.array([[-1.0, -1.5], [-1.0, -0.49], [-0.32, -1.5], [0.34, 0.49], [1.0, 1.5]])

            pix = pixelization.RectangularPixelization(shape=(3,4), regularization_coefficients=(1.0,))

            image_to_source = pix.compute_pixel_to_source(source_coordinates, pix.compute_geometry(source_coordinates))

            assert (image_to_source == np.array([0, 1, 4, 10, 11])).all()

    class TestComputePixelizationMatrices:

        def test__5_simple_coordinates__no_sub_grid__sets_up_correct_pix_matrices(self):

            # Source-plane comprises 5 coordinates, so 5 image pixels traced to the source-plane.
            source_coordinates = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]])

            # There is no sub-grid, so our sub_coordinates are just the image coordinates (note the NumPy data structure
            # ensures this has no sub-gridding)
            source_sub_coordinates = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]])
            sub_to_image = np.array([0, 1, 2, 3, 4])

            pix = pixelization.RectangularPixelization(shape=(3,3), regularization_coefficients=(1.0,))

            pix_matrices = pix.compute_pixelization_matrices(source_coordinates, source_sub_coordinates, sub_to_image,
                                                             image_pixels=5, sub_grid_size=1)

            assert (pix_matrices.mapping == np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])).all()

            assert (pix_matrices.regularization == np.array([[2.00000001, -1.0,  0.0, -1.0,  0.0,   0.0,  0.0,  0.0,  0.0],
                                                             [-1.0,  3.00000001, -1.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0],
                                                             [ 0.0, -1.0,  2.00000001,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0],
                                                             [-1.0,  0.0,  0.0,  3.00000001, -1.0,  0.0, -1.0,  0.0,  0.0],
                                                             [ 0.0, -1.0,  0.0, -1.0,  4.00000001, -1.0,  0.0, -1.0,  0.0],
                                                             [ 0.0,  0.0, -1.0,  0.0, -1.0,  3.00000001,  0.0,  0.0, -1.0],
                                                             [ 0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  2.00000001, -1.0,  0.0],
                                                             [ 0.0,  0.0,  0.0,  0.0, -1.0,  0.0, -1.0,  3.00000001, -1.0],
                                                             [ 0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0, -1.0,  2.00000001]])).all()

        def test__15_coordinates__no_sub_grid__sets_up_correct_pix_matrices(self):

            # Source-plane comprises 15 coordinates, so 15 image pixels traced to the source-plane.

            source_coordinates = np.array([[-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1],
                                           [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                           [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                           [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                           [0.9, 0.9], [1.0, 1.0], [1.1, 1.1]])

            # There is no sub-grid, so our sub_coordinates are just the image coordinates (note the NumPy data structure
            # ensures this has no sub-gridding)
            source_sub_coordinates = np.array([[-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1],
                                               [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                               [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                               [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                               [0.9, 0.9], [1.0, 1.0], [1.1, 1.1]])

            sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

            pix = pixelization.RectangularPixelization(shape=(3,3), regularization_coefficients=(1.0,))

            pix_matrices = pix.compute_pixelization_matrices(source_coordinates, source_sub_coordinates,
                                                             sub_to_image, image_pixels=15,
                                                             sub_grid_size=1)

            assert (pix_matrices.mapping == np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])).all()


            assert (pix_matrices.regularization == np.array([[2.00000001, -1.0,  0.0, -1.0,  0.0,   0.0,  0.0,  0.0,  0.0],
                                                             [-1.0,  3.00000001, -1.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0],
                                                             [ 0.0, -1.0,  2.00000001,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0],
                                                             [-1.0,  0.0,  0.0,  3.00000001, -1.0,  0.0, -1.0,  0.0,  0.0],
                                                             [ 0.0, -1.0,  0.0, -1.0,  4.00000001, -1.0,  0.0, -1.0,  0.0],
                                                             [ 0.0,  0.0, -1.0,  0.0, -1.0,  3.00000001,  0.0,  0.0, -1.0],
                                                             [ 0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  2.00000001, -1.0,  0.0],
                                                             [ 0.0,  0.0,  0.0,  0.0, -1.0,  0.0, -1.0,  3.00000001, -1.0],
                                                             [ 0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0, -1.0,  2.00000001]])).all()

        def test__5_simple_coordinates__include_sub_grid__sets_up_correct_pix_matrices(self):

            # Source-plane comprises 5 coordinates, so 5 image pixels traced to the source-plane.
            source_coordinates = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]])

            # Assume a 2x2 sub-grid, so each of our 5 image-pixels are split into 4.
            # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
            # happen for a real lensing calculation. This is to make a mapping matrix which explicitly tests the sub-grid.
            source_sub_coordinates = np.array([[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0],
                                               [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
                                               [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                               [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                               [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])

            sub_to_image = np.array([0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 2])

            pix = pixelization.RectangularPixelization(shape=(3,3), regularization_coefficients=(1.0,))

            pix_matrices = pix.compute_pixelization_matrices(source_coordinates, source_sub_coordinates,
                                                             sub_to_image, image_pixels=5, sub_grid_size=2)

            assert (pix_matrices.mapping == np.array([[0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75]])).all()

            assert (pix_matrices.regularization == np.array([[2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                             [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                             [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                                                             [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                                                             [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                                                             [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                                                             [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001]])).all()


class TestVoronoiPixelization:
    class TestComputeVoronoi:

        def test__points_in_x_cross_shape__sets_up_diamond_voronoi_vertices(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            points = np.array([[-1.0, 1.0], [1.0, 1.0],
                               [0.0, 0.0],
                               [-1.0, -1.0], [1.0, -1.0]])

            pix = pixelization.VoronoiPixelization(pixels=5, regularization_coefficients=1.0)
            voronoi = pix.compute_voronoi_grid(points)

            voronoi.vertices = list(map(lambda x: list(x), voronoi.vertices))

            assert [0, 1.] in voronoi.vertices
            assert [-1., 0.] in voronoi.vertices
            assert [1., 0.] in voronoi.vertices
            assert [0., -1.] in voronoi.vertices

        def test__9_points_in_square___sets_up_square_of_voronoi_vertices(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                               [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
                               [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])

            pix = pixelization.VoronoiPixelization(pixels=9, regularization_coefficients=1.0)
            voronoi = pix.compute_voronoi_grid(points)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            voronoi.vertices = list(map(lambda x: list(x), voronoi.vertices))

            assert [0.5, 1.5] in voronoi.vertices
            assert [1.5, 0.5] in voronoi.vertices
            assert [0.5, 0.5] in voronoi.vertices
            assert [1.5, 1.5] in voronoi.vertices

        def test__points_in_x_cross_shape__sets_up_pairs_of_voronoi_cells(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            points = np.array([[-1.0, 1.0], [1.0, 1.0],
                               [0.0, 0.0],
                               [-1.0, -1.0], [1.0, -1.0]])

            pix = pixelization.VoronoiPixelization(pixels=5, regularization_coefficients=1.0)
            voronoi = pix.compute_voronoi_grid(points)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            voronoi.ridge_points = list(map(lambda x: list(x), voronoi.ridge_points))

            assert len(voronoi.ridge_points) == 8

            assert [2, 0] in voronoi.ridge_points or [0, 2] in voronoi.ridge_points
            assert [2, 1] in voronoi.ridge_points or [1, 2] in voronoi.ridge_points
            assert [2, 3] in voronoi.ridge_points or [3, 2] in voronoi.ridge_points
            assert [2, 4] in voronoi.ridge_points or [4, 2] in voronoi.ridge_points
            assert [0, 1] in voronoi.ridge_points or [1, 0] in voronoi.ridge_points
            assert [0.3] in voronoi.ridge_points or [3, 0] in voronoi.ridge_points
            assert [3, 4] in voronoi.ridge_points or [4, 3] in voronoi.ridge_points
            assert [4, 1] in voronoi.ridge_points or [1, 4] in voronoi.ridge_points

        def test__9_points_in_square___sets_up_pairs_of_voronoi_cells(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                               [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
                               [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])

            pix = pixelization.VoronoiPixelization(pixels=9, regularization_coefficients=1.0)
            voronoi = pix.compute_voronoi_grid(points)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            voronoi.ridge_points = list(map(lambda x: list(x), voronoi.ridge_points))

            assert len(voronoi.ridge_points) == 12

            assert [0, 1] in voronoi.ridge_points or [1, 0] in voronoi.ridge_points
            assert [1, 2] in voronoi.ridge_points or [2, 1] in voronoi.ridge_points
            assert [3, 4] in voronoi.ridge_points or [4, 3] in voronoi.ridge_points
            assert [4, 5] in voronoi.ridge_points or [5, 4] in voronoi.ridge_points
            assert [6, 7] in voronoi.ridge_points or [7, 6] in voronoi.ridge_points
            assert [7, 8] in voronoi.ridge_points or [8, 7] in voronoi.ridge_points

            assert [0, 3] in voronoi.ridge_points or [3, 0] in voronoi.ridge_points
            assert [1, 4] in voronoi.ridge_points or [4, 1] in voronoi.ridge_points
            assert [4, 7] in voronoi.ridge_points or [7, 4] in voronoi.ridge_points
            assert [2, 5] in voronoi.ridge_points or [5, 2] in voronoi.ridge_points
            assert [5, 8] in voronoi.ridge_points or [8, 5] in voronoi.ridge_points
            assert [3, 6] in voronoi.ridge_points or [6, 3] in voronoi.ridge_points

    class TestComputeNeighbors:

        def test__points_in_x_cross_shape__neighbors_of_each_source_pixel_correct(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            points = np.array([[-1.0, 1.0], [1.0, 1.0],
                               [0.0, 0.0],
                               [-1.0, -1.0], [1.0, -1.0]])

            pix = pixelization.VoronoiPixelization(pixels=5, regularization_coefficients=1.0)
            voronoi = pix.compute_voronoi_grid(points)
            neighbors = pix.compute_source_neighbors(voronoi.ridge_points)

            assert set(neighbors[0]) == set([2, 1, 3])
            assert set(neighbors[1]) == set([2, 0, 4])
            assert set(neighbors[2]) == set([0, 1, 3, 4])
            assert set(neighbors[3]) == set([2, 0, 4])
            assert set(neighbors[4]) == set([2, 1, 3])

        def test__9_points_in_square___neighbors_of_each_source_pixel_correct(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                               [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
                               [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])

            pix = pixelization.VoronoiPixelization(pixels=9, regularization_coefficients=1.0)
            voronoi = pix.compute_voronoi_grid(points)
            neighbors = pix.compute_source_neighbors(voronoi.ridge_points)

            assert set(neighbors[0]) == set([1, 3])
            assert set(neighbors[1]) == set([0, 2, 4])
            assert set(neighbors[2]) == set([1, 5])
            assert set(neighbors[3]) == set([0, 4, 6])
            assert set(neighbors[4]) == set([1, 3, 5, 7])
            assert set(neighbors[5]) == set([2, 4, 8])
            assert set(neighbors[6]) == set([3, 7])
            assert set(neighbors[7]) == set([4, 6, 8])
            assert set(neighbors[8]) == set([5, 7])

    class TestImageToSourceViaNearestNeighborsForTesting:

        def test__image_coordinates_to_source_pixels_via_nearest_neighbour__case1__correct_pairs(self):
            source_pixels = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            image_coordinates = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1]])

            image_to_source = image_coordinates_to_source_pixels_via_nearest_neighbour(image_coordinates, source_pixels)

            assert image_to_source[0] == 0
            assert image_to_source[1] == 1
            assert image_to_source[2] == 2
            assert image_to_source[3] == 3

        def test__image_coordinates_to_source_pixels_via_nearest_neighbour___case2__correct_pairs(self):
            source_pixels = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            image_coordinates = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1],
                                          [0.9, -0.9], [-0.9, -0.9], [-0.9, 0.9], [0.9, 0.9]])

            image_to_source = image_coordinates_to_source_pixels_via_nearest_neighbour(image_coordinates, source_pixels)

            assert image_to_source[0] == 0
            assert image_to_source[1] == 1
            assert image_to_source[2] == 2
            assert image_to_source[3] == 3
            assert image_to_source[4] == 3
            assert image_to_source[5] == 2
            assert image_to_source[6] == 1
            assert image_to_source[7] == 0

        def test__image_coordinates_to_source_pixels_via_nearest_neighbour___case3__correct_pairs(self):
            source_pixels = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 0.0], [2.0, 2.0]])
            image_coordinates = np.array([[0.1, 0.1], [-0.1, -0.1], [0.49, 0.49],
                                          [0.51, 0.51], [1.01, 1.01], [1.51, 1.51]])

            image_to_source = image_coordinates_to_source_pixels_via_nearest_neighbour(image_coordinates, source_pixels)

            assert image_to_source[0] == 4
            assert image_to_source[1] == 4
            assert image_to_source[2] == 4
            assert image_to_source[3] == 0
            assert image_to_source[4] == 0
            assert image_to_source[5] == 5

    class TestSubToSourceViaNearestNeighborsForTesting:

        def test__sub_coordinates_to_source_pixels_via_nearest_neighbour__case1__correct_pairs(self):
            source_pixels = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            sub_coordinates = np.array([[[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1]]])

            image_sub_to_source = sub_coordinates_to_source_pixels_via_nearest_neighbour(sub_coordinates, source_pixels)

            assert image_sub_to_source[0, 0] == 0
            assert image_sub_to_source[0, 1] == 1
            assert image_sub_to_source[0, 2] == 2
            assert image_sub_to_source[0, 3] == 3

        def test__sub_coordinates_to_source_pixels_via_nearest_neighbour___case2__correct_pairs(self):
            source_pixels = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            sub_coordinates = np.array([[[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1]],
                                        [[0.9, -0.9], [-0.9, -0.9], [-0.9, 0.9], [0.9, 0.9]]])

            image_sub_to_source = sub_coordinates_to_source_pixels_via_nearest_neighbour(sub_coordinates, source_pixels)

            assert image_sub_to_source[0, 0] == 0
            assert image_sub_to_source[0, 1] == 1
            assert image_sub_to_source[0, 2] == 2
            assert image_sub_to_source[0, 3] == 3
            assert image_sub_to_source[1, 0] == 3
            assert image_sub_to_source[1, 1] == 2
            assert image_sub_to_source[1, 2] == 1
            assert image_sub_to_source[1, 3] == 0

        def test__sub_coordinates_to_source_pixels_via_nearest_neighbour___case3__correct_pairs(self):
            source_pixels = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 0.0], [2.0, 2.0]])
            sub_coordinates = np.array([[[0.1, 0.1], [-0.1, -0.1], [0.49, 0.49]],
                                        [[0.51, 0.51], [1.01, 1.01], [1.51, 1.51]]])

            image_sub_to_source = sub_coordinates_to_source_pixels_via_nearest_neighbour(sub_coordinates, source_pixels)

            assert image_sub_to_source[0, 0] == 4
            assert image_sub_to_source[0, 1] == 4
            assert image_sub_to_source[0, 2] == 4
            assert image_sub_to_source[1, 0] == 0
            assert image_sub_to_source[1, 1] == 0
            assert image_sub_to_source[1, 2] == 5

    class TestImageToSource:

        def test__coordinates_to_source_pixels_via_cluster_pairs__source_pixels_in_x_shape__correct_pairs(self):
            # The cluster_grid coordinates are not required by the pairing routine routine below,
            # but included here for clarity
            image_coordinates = np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
            cluster_coordinates = np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
            image_to_cluster = np.array([0, 1, 2])

            source_coordinates = np.array([[-1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]])
            source_centers = np.array([[-1.0, 1.0], [1.0, 1.0],
                                       [0.0, 0.0],
                                       [-1.0, -1.0], [1.0, -1.0]])
            cluster_to_source = np.array([0, 2, 4])

            pix = pixelization.VoronoiPixelization(pixels=5, regularization_coefficients=1.0)
            voronoi = pix.compute_voronoi_grid(source_centers)
            source_neighbors = pix.compute_source_neighbors(voronoi.ridge_points)

            image_to_source_via_nearest_neighbour = coordinates_to_source_pixels_via_nearest_neighbour(
                source_coordinates, source_centers)

            image_to_source_via_pairs = pix.compute_image_to_source(source_coordinates, source_centers,
                                                                    source_neighbors, image_to_cluster,
                                                                    cluster_to_source)

            assert (image_to_source_via_nearest_neighbour == image_to_source_via_pairs).all()

        def test__image_coordinates_to_source_pixels_via_cluster_pairs__grid_of_source_pixels__correct_pairs(self):

            source_centers = np.array([[0.1, 0.1], [1.1, 0.1], [2.1, 0.1],
                                       [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]])

            source_coordinates = np.array([[0.1, 0.1], [1.1, 0.1], [2.1, 0.1],
                                           [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]])

            pix = pixelization.VoronoiPixelization(pixels=6, regularization_coefficients=1.0)
            voronoi = pix.compute_voronoi_grid(source_centers)
            source_neighbors = pix.compute_source_neighbors(voronoi.ridge_points)

            image_image_to_source_via_nearest_neighbour = coordinates_to_source_pixels_via_nearest_neighbour(
                source_coordinates, source_centers)

            # The cluster_grid coordinates are not required by the pairing routine routine below, but included here for clarity
            cluster_coordinates = np.array([[-0.9, -0.9], [1.0, 1.0], [2.0, 1.0]])

            image_to_cluster = np.array([0, 1, 2, 1, 1, 2])
            cluster_to_source = np.array([3, 4, 5])

            image_image_to_source_via_pairs = pix.compute_image_to_source(
                source_coordinates, source_centers, source_neighbors, image_to_cluster, cluster_to_source)

            assert (image_image_to_source_via_nearest_neighbour == image_image_to_source_via_pairs).all()

    class TestComputeSubToSource:

        def test__sub_coordinates_to_source_pixels_via_cluster_pairs__source_pixels_in_x_shape__correct_pairs(self):

            source_centers = np.array([[-1.0, 1.0], [1.0, 1.0],
                                       [0.0, 0.0],
                                       [-1.0, -1.0], [1.0, -1.0]])

            # Make it so the central top, left, right and bottom coordinate all pair with the central source_pixel (index=2)

            source_sub_coordinates = np.array([[-1.1, 1.1], [-0.9, 1.1], [-1.1, 0.9], [-0.9, 0.9],
                                               [-0.1, 0.1], [0.1, 0.1], [-0.1, -0.1], [0.1, -0.1],
                                               [0.9, 1.1], [1.1, 1.1], [0.9, 0.9], [1.1, 0.9]])

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

            source_coordinates = np.array([[-1.0, 1.0],
                                           [0.0, 0.0],
                                           [1.0, 1.0]])

            pix = pixelization.VoronoiPixelization(pixels=5, regularization_coefficients=1.0)
            voronoi = pix.compute_voronoi_grid(source_centers)
            source_neighbors = pix.compute_source_neighbors(voronoi.ridge_points)

            sub_to_source_via_nearest_neighbour = coordinates_to_source_pixels_via_nearest_neighbour(
                source_sub_coordinates, source_centers)

            # The cluster_grid coordinates are not required by the pairing routine routine below,
            # but included here for clarity
            cluster_coordinates = np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])

            image_to_cluster = np.array([0, 1, 2])

            source_to_cluster = np.array([1, 2, 4])

            sub_to_source_via_pairs = pix.compute_sub_to_source(source_sub_coordinates, source_centers,
                                         source_neighbors, sub_to_image, image_to_cluster, source_to_cluster)

            assert (sub_to_source_via_nearest_neighbour == sub_to_source_via_pairs).all()

        def test__sub_coordinates_to_source_pixels_via_cluster_pairs__grid_of_source_pixels__correct_pairs(self):

            source_centers = np.array([[0.1, 0.1], [1.1, 0.1], [2.1, 0.1],
                                       [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]])

            source_sub_coordinates = np.array([[0.05, 0.15], [0.15, 0.15], [0.05, 0.05], [0.15, 0.05],
                                               [1.05, 0.15], [1.15, 0.15], [1.05, 0.05], [1.15, 0.05],
                                               [2.05, 0.15], [2.15, 0.15], [2.05, 0.05], [2.15, 0.05],
                                               [0.05, 1.15], [0.15, 1.15], [0.05, 1.05], [0.15, 1.05],
                                               [1.05, 1.15], [1.15, 1.15], [1.05, 1.05], [1.15, 1.05],
                                               [2.05, 1.15], [2.15, 1.15], [2.05, 1.05], [2.15, 1.05]])

            source_coordinates = np.array([[0.1, 0.1], [1.1, 0.1], [2.1, 0.1],
                                           [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]])

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])

            pix = pixelization.VoronoiPixelization(pixels=6, regularization_coefficients=1.0)
            voronoi = pix.compute_voronoi_grid(source_centers)
            source_neighbors = pix.compute_source_neighbors(voronoi.ridge_points)

            sub_to_source_via_nearest_neighbour = coordinates_to_source_pixels_via_nearest_neighbour(
                source_sub_coordinates, source_centers)

            # The cluster_grid coordinates are not required by the pairing routine routine below, but included here for clarity
            cluster_coordinates = np.array([[-0.9, -0.9], [1.0, 1.0], [2.0, 1.0]])

            image_to_cluster = np.array([0, 1, 2, 1, 1, 2])
            cluster_to_source = np.array([3, 4, 5])

            sub_to_source_via_pairs = pix.compute_sub_to_source(source_sub_coordinates, source_centers, source_neighbors,
                                                                sub_to_image, image_to_cluster, cluster_to_source)

            assert (sub_to_source_via_nearest_neighbour == sub_to_source_via_pairs).all()


class TestClusterPixelization:
    
    class TestComputePixelizationMatrices:

        def test__5_simple_coordinates__no_sub_grid__sets_up_correct_pix_matrices(self):
            # Source-plane comprises 5 coordinates, so 5 image pixels traced to the source-plane.
    
            source_coordinates = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
    
            # We'll use 5 cluster_pixels (and therefore source pixels), which map onto (1.0, 1.0), (-1.0, 1.0),
            # (0.0, 0.0), (1.0, -1.0) and (-1.0, -1.0).
            cluster_to_image = np.array([0, 1, 2, 3, 4])
            # We will assume that source-pixels with similar coordinates are near each other in the image.
            image_to_cluster = np.array([0, 1, 2, 3, 4])
    
            # There is no sub-grid, so our sub_coordinates are just the image coordinates (note the NumPy data structure
            # ensures this has no sub-gridding)
            source_sub_coordinates = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            sub_to_image = np.array([0, 1, 2, 3, 4])
    
            mapper_cluster = MockMapperCluster(cluster_to_image=cluster_to_image, image_to_cluster=image_to_cluster)
    
            pix = pixelization.ClusterPixelization(pixels=5, regularization_coefficients=(1.0,))

            pix_matrices = pix.compute_pixelization_matrices(source_coordinates, source_sub_coordinates,
                                                                    mapper_cluster, sub_to_image, image_pixels=5,
                                                                    sub_grid_size=1)
    
            assert (pix_matrices.mapping == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

            assert (pix_matrices.regularization == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                             [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                             [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()

        def test__15_coordinates__no_sub_grid__sets_up_correct_pix_matrices(self):
            # Source-plane comprises 15 coordinates, so 15 image pixels traced to the source-plane.
    
            source_coordinates = np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                           [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                           [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                           [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                           [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1]])
    
            # We'll use 5 cluster_pixels (and therefore source pixels), which map onto (1.0, 1.0), (-1.0, 1.0),
            # (0.0, 0.0), (1.0, -1.0) and (-1.0, -1.0).
            cluster_to_image = np.array([1, 4, 7, 10, 13])
            # We will assume that source-pixels with similar coordinates are near each other in the image.
            image_to_cluster = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    
            # There is no sub-grid, so our sub_coordinates are just the image coordinates (note the NumPy data structure
            # ensures this has no sub-gridding)
            source_sub_coordinates = np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                               [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                               [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                               [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                               [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1]])
    
            sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
            mapper_cluster = MockMapperCluster(cluster_to_image=cluster_to_image, image_to_cluster=image_to_cluster)
    
            pix = pixelization.ClusterPixelization(pixels=5, regularization_coefficients=(1.0,))
    
            pix_matrices = pix.compute_pixelization_matrices(source_coordinates, source_sub_coordinates,
                                                                    mapper_cluster, sub_to_image, image_pixels=15,
                                                                    sub_grid_size=1)
    
            assert (pix_matrices.mapping == np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                [1.0, 0.0, 0.0, 0.0, 0.0],
                                                [1.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 1.0],
                                                [0.0, 0.0, 0.0, 0.0, 1.0],
                                                [0.0, 0.0, 0.0, 0.0, 1.0]])).all()

            assert (pix_matrices.regularization == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                             [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                             [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()
    
        def test__5_simple_coordinates__include_sub_grid__sets_up_correct_pix_matrices(self):
            # Source-plane comprises 5 coordinates, so 5 image pixels traced to the source-plane.
            source_coordinates = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
    
            # We'll use 5 cluster_pixels (and therefore source pixels), which map onto (1.0, 1.0), (-1.0, 1.0),
            # (0.0, 0.0), (1.0, -1.0) and (-1.0, -1.0).
            cluster_to_image = np.array([0, 1, 2, 3, 4])
            # We will assume that source-pixels with similar coordinates are near each other in the image.
            image_to_cluster = np.array([0, 1, 2, 3, 4])
    
            # Assume a 2x2 sub-grid, so each of our 5 image-pixels are split into 4.
            # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
            # happen for a real lensing calculation. This is to make a mapping matrix which explicitly tests the sub-grid.
            source_sub_coordinates = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0],
                                               [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
                                               [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                               [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                               [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0]])
    
            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    
            mapper_cluster = MockMapperCluster(cluster_to_image=cluster_to_image, image_to_cluster=image_to_cluster)
    
            pix = pixelization.ClusterPixelization(pixels=5, regularization_coefficients=(1.0,))
    
            pix_matrices = pix.compute_pixelization_matrices(source_coordinates, source_sub_coordinates,
                                                              mapper_cluster, sub_to_image, image_pixels=5,
                                                                    sub_grid_size=2)

            assert (pix_matrices.mapping == np.array([[0.75, 0.0, 0.25, 0.0, 0.0],
                                                      [0.0, 0.75, 0.25, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.25, 0.75, 0.0],
                                                      [0.0, 0.0, 0.25, 0.0, 0.75]])).all()

            assert (pix_matrices.regularization == np.array([[3.00000001, -1.0, -1.0, -1.0, 0.0],
                                                             [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                                                             [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                                                             [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                                                             [0.0, -1.0, -1.0, -1.0, 3.00000001]])).all()


class TestAmorphousPixelization:

    class TestKMeans:

        def test__simple_points__sets_up_two_clusters(self):
            cluster_coordinates = np.array([[0.99, 0.99], [1.0, 1.0], [1.01, 1.01],
                                            [1.99, 1.99], [2.0, 2.0], [2.01, 2.01]])

            pix = pixelization.AmorphousPixelization(pixels=2)

            source_centers, source_to_image = pix.kmeans_cluster(cluster_coordinates)

            assert [2.0, 2.0] in source_centers
            assert [1.0, 1.0] in source_centers

            assert list(source_to_image).count(0) == 3
            assert list(source_to_image).count(1) == 3

        def test__simple_points__sets_up_three_clusters(self):
            cluster_coordinates = np.array([[-0.99, -0.99], [-1.0, -1.0], [-1.01, -1.01],
                                            [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],
                                            [1.99, 1.99], [2.0, 2.0], [2.01, 2.01]])

            pix = pixelization.AmorphousPixelization(pixels=3)

            source_centers, source_to_image = pix.kmeans_cluster(cluster_coordinates)

            assert [2.0, 2.0] in source_centers
            assert [1.0, 1.0] in source_centers
            assert [-1.0, -1.0] in source_centers

            assert list(source_to_image).count(0) == 3
            assert list(source_to_image).count(1) == 3
            assert list(source_to_image).count(2) == 3

        def test__simple_points__sets_up_three_clusters_more_points_in_third_cluster(self):
            cluster_coordinates = np.array([[-0.99, -0.99], [-1.0, -1.0], [-1.01, -1.01],

                                            [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],
                                            [0.99, 0.99], [1.0, 1.0], [1.01, 1.01],

                                            [1.99, 1.99], [2.0, 2.0], [2.01, 2.01],
                                            [1.99, 1.99], [2.0, 2.0], [2.01, 2.01],
                                            [1.99, 1.99], [2.0, 2.0], [2.01, 2.01],
                                            [1.99, 1.99], [2.0, 2.0], [2.01, 2.01]])

            pix = pixelization.AmorphousPixelization(pixels=3)

            source_centers, source_to_image = pix.kmeans_cluster(cluster_coordinates)

            source_centers = list(map(lambda x: pytest.approx(list(x), 1e-3), source_centers))

            assert [2.0, 2.0] in source_centers
            assert [1.0, 1.0] in source_centers
            assert [-1.0, -1.0] in source_centers

            assert list(source_to_image).count(0) == 3 or 6 or 12
            assert list(source_to_image).count(1) == 3 or 6 or 12
            assert list(source_to_image).count(2) == 3 or 6 or 12

            assert list(source_to_image).count(0) != list(source_to_image).count(1) != list(source_to_image).count(2)

    class TestComputePixelizationMatrices:

        def test__5_simple_coordinates__no_sub_grid__sets_up_correct_pix_matrices(self):
            # Source-plane comprises 5 coordinates, so 5 image pixels traced to the source-plane.

            source_coordinates = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])

            # We'll use 5 cluster_pixels (and therefore source pixels), which map onto (1.0, 1.0), (-1.0, 1.0),
            # (0.0, 0.0), (1.0, -1.0) and (-1.0, -1.0).
            cluster_to_image = np.array([0, 1, 2, 3, 4])
            # We will assume that source-pixels with similar coordinates are near each other in the image.
            image_to_cluster = np.array([0, 1, 2, 3, 4])

            # There is no sub-grid, so our sub_coordinates are just the image coordinates (note the NumPy data structure
            # ensures this has no sub-gridding)
            source_sub_coordinates = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])
            sub_to_image = np.array([0, 1, 2, 3, 4])

            mapper_cluster = MockMapperCluster(cluster_to_image=cluster_to_image, image_to_cluster=image_to_cluster)

            pix = pixelization.AmorphousPixelization(pixels=5, regularization_coefficients=(1.0,))

            pix_matrices = pix.compute_pixelization_matrices(source_coordinates, source_sub_coordinates,
                                                                    mapper_cluster, sub_to_image, image_pixels=15,
                                                                    sub_grid_size=1)

            assert np.sum(pix_matrices.mapping) == 5.0
            assert np.sum(pix_matrices.mapping[:, 0]) == 1.0
            assert np.sum(pix_matrices.mapping[:, 1]) == 1.0
            assert np.sum(pix_matrices.mapping[:, 2]) == 1.0
            assert np.sum(pix_matrices.mapping[:, 3]) == 1.0
            assert np.sum(pix_matrices.mapping[:, 4]) == 1.0
            assert np.sum(pix_matrices.mapping[0, :]) == 1.0
            assert np.sum(pix_matrices.mapping[1, :]) == 1.0
            assert np.sum(pix_matrices.mapping[2, :]) == 1.0
            assert np.sum(pix_matrices.mapping[3, :]) == 1.0
            assert np.sum(pix_matrices.mapping[4, :]) == 1.0

            assert np.sum(np.diag(pix_matrices.regularization)) == 16.00000005
            assert np.sum(pix_matrices.regularization) - np.sum(np.diag(pix_matrices.regularization)) == -16.0

        def test__15_coordinates__no_sub_grid__sets_up_correct_pix_matrices(self):
            # Source-plane comprises 15 coordinates, so 15 image pixels traced to the source-plane.

            source_coordinates = np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                           [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                           [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                           [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                           [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1]])

            # We'll use 5 cluster_pixels (and therefore source pixels), which map onto (1.0, 1.0), (-1.0, 1.0),
            # (0.0, 0.0), (1.0, -1.0) and (-1.0, -1.0).
            cluster_to_image = np.array([1, 4, 7, 10, 13])
            # We will assume that source-pixels with similar coordinates are near each other in the image.
            image_to_cluster = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

            # There is no sub-grid, so our sub_coordinates are just the image coordinates (note the NumPy data structure
            # ensures this has no sub-gridding)
            source_sub_coordinates = np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.1],
                                               [-0.9, 0.9], [-1.0, 1.0], [-1.1, 1.1],
                                               [-0.01, 0.01], [0.0, 0.0], [0.01, 0.01],
                                               [0.9, -0.9], [1.0, -1.0], [1.1, -1.1],
                                               [-0.9, -0.9], [-1.0, -1.0], [-1.1, -1.1]])

            sub_to_image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

            mapper_cluster = MockMapperCluster(cluster_to_image=cluster_to_image, image_to_cluster=image_to_cluster)

            pix = pixelization.AmorphousPixelization(pixels=5, regularization_coefficients=(1.0,))

            pix_matrices = pix.compute_pixelization_matrices(source_coordinates, source_sub_coordinates,
                                                                    mapper_cluster, sub_to_image, image_pixels=15,
                                                                    sub_grid_size=1)

            assert np.sum(pix_matrices.mapping) == 15.0

            assert np.sum(pix_matrices.mapping[:, 0]) == 3.0
            assert np.sum(pix_matrices.mapping[:, 1]) == 3.0
            assert np.sum(pix_matrices.mapping[:, 2]) == 3.0
            assert np.sum(pix_matrices.mapping[:, 3]) == 3.0
            assert np.sum(pix_matrices.mapping[:, 4]) == 3.0

            assert np.sum(pix_matrices.mapping[0, :]) == 1.0
            assert np.sum(pix_matrices.mapping[1, :]) == 1.0
            assert np.sum(pix_matrices.mapping[2, :]) == 1.0
            assert np.sum(pix_matrices.mapping[3, :]) == 1.0
            assert np.sum(pix_matrices.mapping[4, :]) == 1.0
            assert np.sum(pix_matrices.mapping[5, :]) == 1.0
            assert np.sum(pix_matrices.mapping[6, :]) == 1.0
            assert np.sum(pix_matrices.mapping[7, :]) == 1.0
            assert np.sum(pix_matrices.mapping[8, :]) == 1.0
            assert np.sum(pix_matrices.mapping[9, :]) == 1.0
            assert np.sum(pix_matrices.mapping[10, :]) == 1.0
            assert np.sum(pix_matrices.mapping[11, :]) == 1.0
            assert np.sum(pix_matrices.mapping[12, :]) == 1.0
            assert np.sum(pix_matrices.mapping[13, :]) == 1.0
            assert np.sum(pix_matrices.mapping[14, :]) == 1.0

            assert np.sum(np.diag(pix_matrices.regularization)) == 16.00000005
            assert np.sum(pix_matrices.regularization) - np.sum(np.diag(pix_matrices.regularization)) == -16.0

        def test__5_simple_coordinates__include_sub_grid__sets_up_correct_mapping_matrix(self):
            # Source-plane comprises 5 coordinates, so 5 image pixels traced to the source-plane.
            source_coordinates = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0]])

            # We'll use 5 cluster_pixels (and therefore source pixels), which map onto (1.0, 1.0), (-1.0, 1.0),
            # (0.0, 0.0), (1.0, -1.0) and (-1.0, -1.0).
            cluster_to_image = np.array([0, 1, 2, 3, 4])
            # We will assume that source-pixels with similar coordinates are near each other in the image.
            image_to_cluster = np.array([0, 1, 2, 3, 4])

            # Assume a 2x2 sub-grid, so each of our 5 image-pixels are split into 4.
            # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
            # happen for a real lensing calculation. This is to make a mapping matrix which explicitly tests the sub-grid.
            source_sub_coordinates = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0],
                                               [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
                                               [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                               [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
                                               [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.0, 0.0]])

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

            mapper_cluster = MockMapperCluster(cluster_to_image=cluster_to_image, image_to_cluster=image_to_cluster)

            pix = pixelization.AmorphousPixelization(pixels=5, regularization_coefficients=(1.0,))

            pix_matrices = pix.compute_pixelization_matrices(source_coordinates, source_sub_coordinates,
                                                                    mapper_cluster, sub_to_image, image_pixels=5,
                                                                    sub_grid_size=2)

            assert np.sum(pix_matrices.mapping) == 5.0

            assert np.sum(pix_matrices.mapping[0, :]) == 1.0
            assert np.sum(pix_matrices.mapping[1, :]) == 1.0
            assert np.sum(pix_matrices.mapping[2, :]) == 1.0
            assert np.sum(pix_matrices.mapping[3, :]) == 1.0
            assert np.sum(pix_matrices.mapping[4, :]) == 1.0

            assert np.sum(pix_matrices.mapping[:, 0]) or np.sum(pix_matrices.mapping[:, 1]) or np.sum(pix_matrices.mapping[:, 2]) or \
                   np.sum(pix_matrices.mapping[:, 3]) or np.sum(pix_matrices.mapping[:, 4]) == 0.75

            assert np.sum(np.diag(pix_matrices.regularization)) == 16.00000005
            assert np.sum(pix_matrices.regularization) - np.sum(np.diag(pix_matrices.regularization)) == -16.0