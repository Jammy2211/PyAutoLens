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
def make_simple_frame_array(simple_frame_maker):
    return simple_frame_maker.make_frame_array((3, 3))


@pytest.fixture(name="cross_frame_array")
def make_cross_frame_array(cross_frame_maker):
    return cross_frame_maker.make_frame_array((3, 3))


@pytest.fixture(name="simple_frame_maker")
def make_simple_frame_maker():
    return frame_convolution.FrameMaker(np.ones((3, 3)))


@pytest.fixture(name="cross_frame_maker")
def make_cross_frame_maker(cross_mask):
    return frame_convolution.FrameMaker(cross_mask)


@pytest.fixture(name="simple_kernel")
def make_simple_kernel():
    return np.array([[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]])


class TestNumbering(object):
    def test_simple_numbering(self, simple_number_array):
        shape = (3, 3)

        frame_maker = frame_convolution.FrameMaker(np.ones(shape))

        number_array = frame_maker.number_array

        assert number_array.shape == shape
        # noinspection PyUnresolvedReferences
        assert (number_array == simple_number_array).all()

    def test_simple_mask(self, cross_mask):
        number_array = frame_convolution.FrameMaker(cross_mask).number_array

        assert (number_array == np.array([[-1, 0, -1], [1, 2, 3], [-1, 4, -1]])).all()


class TestFrameExtraction(object):
    def test_frame_at_coords(self, simple_frame_maker):
        kernel_shape = (3, 3)

        # noinspection PyUnresolvedReferences
        assert (simple_frame_maker.number_array == simple_frame_maker.frame_at_coords(coords=(1, 1),
                                                                                      kernel_shape=kernel_shape)).all()

        corner_array = np.array([[-1, -1, -1], [-1, 0, 1], [-1, 3, 4]])

        corner_frame = simple_frame_maker.frame_at_coords(coords=(0, 0), kernel_shape=kernel_shape)

        assert (corner_array == corner_frame).all()

    def test_simple_square(self, simple_frame_maker):
        frame_array = simple_frame_maker.make_frame_array(kernel_shape=(3, 3))

        assert 9 == len(frame_array)

        assert frame_array[4].shape == simple_frame_maker.number_array.shape
        # noinspection PyUnresolvedReferences
        assert (frame_array[4] == simple_frame_maker.number_array).all()

    def test_masked_square(self, cross_frame_maker):
        frame_array = cross_frame_maker.make_frame_array(kernel_shape=(3, 3))

        assert 5 == len(frame_array)

        assert (np.array([[-1, -1, -1],
                          [-1, 0, -1],
                          [1, 2, 3]]) == frame_array[0]).all()

        assert (np.array([[-1, -1, 0],
                          [-1, 1, 2],
                          [-1, -1, 4]]) == frame_array[1]).all()

        # noinspection PyUnresolvedReferences
        assert (cross_frame_maker.number_array == frame_array[2]).all()

        assert (np.array([[0, -1, -1],
                          [2, 3, -1],
                          [4, -1, -1]]) == frame_array[3]).all()

        assert (np.array([[1, 2, 3],
                          [-1, 4, -1],
                          [-1, -1, -1]]) == frame_array[4]).all()


class TestConvolution(object):
    def test_simple_convolution(self, simple_frame_array, simple_number_array, simple_kernel):
        pixel_vector = [0, 0, 0, 0, 1, 0, 0, 0, 0]

        convolver = frame_convolution.Convolver(simple_frame_array, simple_number_array)

        result = convolver.convolution_for_pixel_index_vector_and_kernel(4, pixel_vector, simple_kernel)

        # noinspection PyUnresolvedReferences
        assert (result == [0.0, 0.1, 0.0, 0.1, 0.6, 0.1, 0.0, 0.1, 0.0]).all()

    def test_full_convolution(self, simple_frame_array, simple_number_array):
        pixel_vector = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        kernel = np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 0, 0]])

        convolver = frame_convolution.Convolver(simple_frame_array, simple_number_array)

        result = convolver.convolve_vector_with_kernel(pixel_vector, kernel)

        # noinspection PyUnresolvedReferences
        assert (result == [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5]).all()

    def test_is_frame_for_index(self, cross_frame_array, cross_number_array):
        convolver = frame_convolution.Convolver(cross_frame_array, cross_number_array)

        # noinspection PyTypeChecker
        assert [False, True, False, True, True, True, False, True, False] == map(convolver.is_frame_for_index,
                                                                                 range(9))

    def test_frame_for_index(self, cross_frame_array, cross_number_array, cross_frame_maker):
        convolver = frame_convolution.Convolver(cross_frame_array, cross_number_array)

        assert (np.array([[-1, -1, -1],
                          [-1, 0, -1],
                          [1, 2, 3]]) == convolver.frame_for_index(1)).all()

        assert (np.array([[-1, -1, 0],
                          [-1, 1, 2],
                          [-1, -1, 4]]) == convolver.frame_for_index(3)).all()

        # noinspection PyUnresolvedReferences
        assert (cross_frame_maker.number_array == convolver.frame_for_index(4)).all()

        assert (np.array([[0, -1, -1],
                          [2, 3, -1],
                          [4, -1, -1]]) == convolver.frame_for_index(5)).all()

        assert (np.array([[1, 2, 3],
                          [-1, 4, -1],
                          [-1, -1, -1]]) == convolver.frame_for_index(7)).all()

    def test_cross_mask_convolution(self, cross_frame_array, cross_number_array):
        pixel_vector = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        kernel = np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 0, 0]])

        print(cross_frame_array)

        convolver = frame_convolution.Convolver(cross_frame_array, cross_number_array)

        result = convolver.convolve_vector_with_kernel(pixel_vector, kernel)

        print(result)

        # noinspection PyUnresolvedReferences
        assert (result == [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]).all()
