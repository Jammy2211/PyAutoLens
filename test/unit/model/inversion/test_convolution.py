import numpy as np
import pytest

from autolens.model.inversion import convolution


class TestConvolveMappingMatrix(object):

    def test__asymetric_convolver__matrix_blurred_correctly(self):
        shape = (4, 4)
        mask = np.full(shape, False)

        asymmetric_kernel = np.array([[0, 0.0, 0],
                                      [0.4, 0.2, 0.3],
                                      [0, 0.1, 0]])

        convolver = convolution.ConvolverMappingMatrix(mask=mask, psf=asymmetric_kernel)

        mapping = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 1, 0],  # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
                            [0, 0, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

        blurred_mapping = convolver.convolve_mapping_matrix(mapping)

        assert (blurred_mapping == np.array([[0, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0],
                                             [0, 0.4, 0],
                                             [0, 0.2, 0],
                                             [0.4, 0, 0],
                                             [0.2, 0, 0.4],
                                             [0.3, 0, 0.2],
                                             [0, 0.1, 0.3],
                                             [0, 0, 0],
                                             [0.1, 0, 0],
                                             [0, 0, 0.1],
                                             [0, 0, 0]])).all()

    def test__asymetric_convolver__multiple_overlapping_blurred_entires_in_matrix(self):
        shape = (4, 4)
        mask = np.full(shape, False)

        asymmetric_kernel = np.array([[0, 0.0, 0],
                                      [0.4, 0.2, 0.3],
                                      [0, 0.1, 0]])

        convolver = convolution.ConvolverMappingMatrix(mask=mask, psf=asymmetric_kernel)

        mapping = np.array([[0, 1, 0],
                            [0, 1, 0],
                            [0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 1, 0],  # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

        blurred_mapping = convolver.convolve_mapping_matrix(mapping)

        assert blurred_mapping == pytest.approx(np.array([[0, 0.6, 0],
                                                          [0, 0.9, 0],
                                                          [0, 0.5, 0],
                                                          [0, 0.3, 0],
                                                          [0, 0.1, 0],
                                                          [0, 0.1, 0],
                                                          [0, 0.5, 0],
                                                          [0, 0.2, 0],
                                                          [0.6, 0, 0],
                                                          [0.5, 0, 0.4],
                                                          [0.3, 0, 0.2],
                                                          [0, 0.1, 0.3],
                                                          [0.1, 0, 0],
                                                          [0.1, 0, 0],
                                                          [0, 0, 0.1],
                                                          [0, 0, 0]]), 1e-4)