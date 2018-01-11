import numpy as np
import frame_convolution
import pytest


@pytest.fixture(name="simple_number_array")
def make_simple_number_array():
    return np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])


@pytest.fixture(name="cross_mask")
def make_cross_mask():
    mask = np.ones((3, 3))

    mask[0, 0] = 0
    mask[0, 2] = 0
    mask[2, 2] = 0
    mask[2, 0] = 0

    return mask


@pytest.fixture(name="cross_number_array")
def make_cross_number_array():
    return np.array([[-1, 0, -1], [1, 2, 3], [-1, 4, -1]])


@pytest.fixture(name="simple_frame_array")
def make_simple_frame_array(simple_number_array):
    return frame_convolution.make_frame_array(simple_number_array, kernel_shape=(3, 3))


@pytest.fixture(name="cross_frame_array")
def make_cross_frame_array(cross_number_array):
    return frame_convolution.make_frame_array(cross_number_array, kernel_shape=(3, 3))


@pytest.fixture(name="simple_kernel")
def make_simple_kernel():
    return np.array([[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]])


class TestNumbering(object):
    def test_simple_numbering(self, simple_number_array):
        shape = (3, 3)
        number_array = frame_convolution.number_array_for_mask(np.ones(shape))

        assert number_array.shape == shape
        # noinspection PyUnresolvedReferences
        assert (number_array == simple_number_array).all()

    def test_simple_mask(self, cross_mask):
        number_array = frame_convolution.number_array_for_mask(cross_mask)

        assert (number_array == np.array([[-1, 0, -1], [1, 2, 3], [-1, 4, -1]])).all()


class TestFrameExtraction(object):
    def test_frame_at_coords(self, simple_number_array):
        kernel_shape = (3, 3)

        # noinspection PyUnresolvedReferences
        assert (simple_number_array == frame_convolution.frame_at_coords(simple_number_array, coords=(1, 1),
                                                                         kernel_shape=kernel_shape)).all()

        corner_array = np.array([[-1, -1, -1], [-1, 0, 1], [-1, 3, 4]])

        corner_frame = frame_convolution.frame_at_coords(simple_number_array, coords=(0, 0), kernel_shape=kernel_shape)

        assert (corner_array == corner_frame).all()

    def test_simple_square(self, simple_number_array):
        frame_array = frame_convolution.make_frame_array(simple_number_array, kernel_shape=(3, 3))

        assert 9 == len(frame_array)

        assert frame_array[4].shape == simple_number_array.shape
        # noinspection PyUnresolvedReferences
        assert (frame_array[4] == simple_number_array).all()

    def test_masked_square(self, cross_number_array):
        frame_array = frame_convolution.make_frame_array(cross_number_array, kernel_shape=(3, 3))

        assert 5 == len(frame_array)

        assert (np.array([[1, 2, 3], [-1, 4, -1], [-1, -1, -1]]) == frame_array[4]).all()


class TestConvolution(object):
    def test_simple_convolution(self, simple_frame_array, simple_number_array, simple_kernel):
        pixel_vector = [0, 0, 0, 0, 1, 0, 0, 0, 0]

        convolver = frame_convolution.Convolver(pixel_vector, simple_frame_array, simple_number_array, simple_kernel)

        result = convolver.convolution_for_pixel(4)

        # noinspection PyUnresolvedReferences
        assert (result == [0.0, 0.1, 0.0, 0.1, 0.6, 0.1, 0.0, 0.1, 0.0]).all()

    def test_full_convolution(self, simple_frame_array, simple_number_array):
        pixel_vector = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        kernel = np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 0, 0]])

        convolver = frame_convolution.Convolver(pixel_vector, simple_frame_array, simple_number_array, kernel)

        result = convolver.convolution

        # noinspection PyUnresolvedReferences
        assert (result == [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5]).all()

    def test_cross_mask_convolution(self, cross_frame_array, cross_number_array):
        pixel_vector = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        kernel = np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 0, 0]])

        print(cross_frame_array)

        convolver = frame_convolution.Convolver(pixel_vector, cross_frame_array, cross_number_array, kernel)

        result = convolver.convolution

        print(result)

        # noinspection PyUnresolvedReferences
        assert (result == [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]).all()
