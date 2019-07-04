import numpy as np

from autolens.model.inversion import regularization
from autolens.model.inversion.util import regularization_util as reg_util


class TestRegularizationConstant:

    def test__regularization_matrix__compare_to_regularization_util(self):

        pixel_neighbors = np.array([[1, 3, 7, 2],
                                   [4, 2, 0, -1],
                                   [1, 5, 3, -1],
                                   [4, 6, 0, -1],
                                   [7, 1, 5, 3],
                                   [4, 2, 8, -1],
                                   [7, 3, 0, -1],
                                   [4, 8, 6, -1],
                                   [7, 5, -1, -1]])

        pixel_neighbors_size = np.array([4, 3, 3, 3, 4, 3, 3, 3, 2])

        reg = regularization.Constant(coefficients=(1.0,))
        regularization_matrix = reg.regularization_matrix_from_pixel_neighbors(pixel_neighbors, pixel_neighbors_size)

        regularization_matrix_util = reg_util.constant_regularization_matrix_from_pixel_neighbors(
            coefficients=(1.0,), pixel_neighbors=pixel_neighbors,  pixel_neighbors_size=pixel_neighbors_size)

        assert (regularization_matrix == regularization_matrix_util).all()


class TestRegularizationWeighted:

    def test__pixel_signals__compare_to_regularization_util(self):
        
        reg = regularization.AdaptiveBrightness(signal_scale=2.0)

        sub_to_pix = np.array([0, 1, 2, 3, 3, 4, 5])
        sub_to_regular = np.array([0, 1, 2, 3, 3, 4, 5])
        galaxy_image = np.array([2.0, 1.0, 1.0, 4.0, 5.0, 6.0, 1.0])

        pixel_signals = reg.pixel_signals_from_images(
            pixels=6, sub_to_pix=sub_to_pix, sub_to_regular=sub_to_regular, hyper_image=galaxy_image)
        
        pixel_signals_util = reg_util.adaptive_pixel_signals_from_images(
            pixels=6, signal_scale=2.0, sub_to_pix=sub_to_pix, sub_to_regular=sub_to_regular, hyper_image=galaxy_image)

        assert (pixel_signals == pixel_signals_util).all()

    def test__weights__compare_to_regularization_util(self):

        reg = regularization.AdaptiveBrightness(coefficients=(10.0, 15.0))

        pixel_signals = np.array([0.21, 0.586, 0.45])

        weights = reg.regularization_weights_from_pixel_signals(pixel_signals)

        weights_util = reg_util.adaptive_regularization_weights_from_pixel_signals(coefficients=(10.0, 15.0),
                                                                                   pixel_signals=pixel_signals)

        assert (weights == weights_util).all()

    def test__regularization_matrix__compare_to_regularization_util(self):

        reg = regularization.AdaptiveBrightness()

        pixel_neighbors = np.array([[1, 4, -1, -1],
                                    [2, 4, 0, -1],
                                    [3, 4, 5, 1],
                                    [5, 2, -1, -1],
                                    [5, 0, 1, 2],
                                    [2, 3, 4, -1]])

        pixel_neighbors_size = np.array([2, 3, 4, 2, 4, 3])
        regularization_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        regularization_matrix = reg.regularization_matrix_from_regularization_weights_and_pixel_neighbors(
            regularization_weights=regularization_weights, pixel_neighbors=pixel_neighbors,
            pixel_neighbors_size=pixel_neighbors_size)

        regularization_matrix_util = reg_util.weighted_regularization_matrix_from_pixel_neighbors(
            regularization_weights=regularization_weights, pixel_neighbors=pixel_neighbors,
            pixel_neighbors_size=pixel_neighbors_size)

        assert (regularization_matrix == regularization_matrix_util).all()
