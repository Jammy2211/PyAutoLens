import numpy as np
import frame_convolution


class TestFrameConvolution(object):
    def test_simple_numbering(self):
        shape = (3, 3)
        number_array = frame_convolution.number_array_for_mask(np.ones(shape))

        assert number_array.shape == shape
        assert (number_array == np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])).all()

    def test_simple_mask(self):
        shape = (3, 3)
        mask = np.ones(shape)

        mask[0, 0] = 0
        mask[0, 2] = 0
        mask[2, 2] = 0
        mask[2, 0] = 0

        number_array = frame_convolution.number_array_for_mask(mask)

        assert (number_array == np.array([[-1, 0, -1], [1, 2, 3], [-1, 4, -1]])).all()
