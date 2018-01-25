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

    def test_even_failure(self):
        with pytest.raises(frame_convolution.KernelException):
            frame_convolution.FrameMaker(np.ones((3, 3))).convolver_for_kernel_shape((2, 2))


class TestFrameExtraction(object):
    def test_trivial_frame_at_coords(self, simple_frame_maker):
        assert ({i: i for i in range(9)} == simple_frame_maker.frame_at_coords(coords=(1, 1),
                                                                               kernel_shape=(3, 3)))

    def test_corner_frame(self, simple_frame_maker):
        corner_dict = {4: 0, 5: 1, 7: 3, 8: 4}

        corner_frame = simple_frame_maker.frame_at_coords(coords=(0, 0), kernel_shape=(3, 3))

        assert corner_dict == corner_frame

    def test_simple_square(self, simple_frame_maker):
        frame_array = simple_frame_maker.make_frame_array(kernel_shape=(3, 3))

        assert 9 == len(frame_array)

        assert {i: i for i in range(9)} == frame_array[4]

    def test_masked_square(self, cross_frame_maker):
        frame_array = cross_frame_maker.make_frame_array(kernel_shape=(3, 3))

        assert 5 == len(frame_array)

        assert {4: 0, 6: 1, 7: 2, 8: 3} == frame_array[0]

        assert {2: 0, 4: 1, 5: 2, 8: 4} == frame_array[1]

        assert {0: 0, 3: 2, 4: 3, 6: 4} == frame_array[3]

        assert {0: 1, 1: 2, 2: 3, 4: 4} == frame_array[4]


class TestConvolution(object):
    def test_simple_convolution(self, simple_frame_array, simple_kernel):
        pixel_dict = {4: 1}

        convolver = frame_convolution.Convolver(simple_frame_array)

        result = convolver.convolver_for_kernel(simple_kernel).convolution_for_pixel_index_vector(4, pixel_dict)

        assert result == {1: 0.1, 3: 0.1, 4: 0.6, 5: 0.1, 7: 0.1}

    def test_full_convolution(self, simple_frame_array):
        pixel_dict = {0: 1, 4: 1, 8: 1}

        kernel = np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 0, 0]])

        convolver = frame_convolution.Convolver(simple_frame_array)

        result = convolver.convolver_for_kernel(kernel).convolve_vector(pixel_dict)

        assert result == {0: 0.5, 1: 0.5, 4: 0.5, 5: 0.5, 8: 0.5}

    def test_cross_mask_convolution(self, cross_frame_array):
        pixel_dict = {2: 1}
        kernel = np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 0, 0]])

        convolver = frame_convolution.Convolver(cross_frame_array)

        result = convolver.convolver_for_kernel(kernel).convolve_vector(pixel_dict)

        assert result == {2: 0.5, 3: 0.5}


@pytest.fixture(name="convolver_4_simple")
def make_convolver_4_simple():
    shape = (4, 4)
    mask = np.ones(shape)

    frame_maker = frame_convolution.FrameMaker(mask)
    return frame_maker.convolver_for_kernel_shape((3, 3))


@pytest.fixture(name="convolver_4_edges")
def make_convolver_4_edges():
    mask = np.array(
        [[0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]]
    )

    frame_maker = frame_convolution.FrameMaker(mask)
    return frame_maker.convolver_for_kernel_shape((3, 3))


class TestNonTrivialExamples(object):
    def test_larger_mask(self, convolver_4_simple):
        kernel = np.array([[0, 0.2, 0],
                           [0.2, 0.4, 0.2],
                           [0, 0.2, 0]])

        pixel_dict = {9: 1}

        kernel_convolver = convolver_4_simple.convolver_for_kernel(kernel)

        result = kernel_convolver.convolve_vector(pixel_dict)

        assert result == {5: 0.2, 8: 0.2, 9: 0.4, 10: 0.2, 13: 0.2}

    def test_asymmetric_kernel(self, convolver_4_simple):
        asymmetric_kernel = np.array([[0, 0.0, 0],
                                      [0.4, 0.2, 0.3],
                                      [0, 0.1, 0]])

        pixel_dict = {9: 1}

        kernel_convolver = convolver_4_simple.convolver_for_kernel(asymmetric_kernel)
        result = kernel_convolver.convolve_vector(pixel_dict)

        assert result == {8: 0.4, 9: 0.2, 10: 0.3, 13: 0.1}

    def test_two_pixel_sum(self, convolver_4_simple):
        kernel = np.array([[0, 0.2, 0],
                           [0.2, 0.4, 0.2],
                           [0, 0.2, 0]])

        pixel_dict = {6: 1, 9: 1}

        kernel_convolver = convolver_4_simple.convolver_for_kernel(kernel)

        result = kernel_convolver.convolve_vector(pixel_dict)

        assert result == {2: 0.2, 5: 0.4, 6: 0.4, 7: 0.2, 8: 0.2, 9: 0.4, 10: 0.4, 13: 0.2}

    def test_two_pixel_sum_masked(self, convolver_4_edges):
        kernel = np.array([[0, 0.2, 0],
                           [0.2, 0.4, 0.2],
                           [0, 0.2, 0]])

        pixel_dict = {1: 1, 2: 1}

        kernel_convolver = convolver_4_edges.convolver_for_kernel(kernel)

        result = kernel_convolver.convolve_vector(pixel_dict)

        assert result == {0: 0.4, 1: 0.4, 2: 0.4, 3: 0.4}


class TestSubConvolution(object):
    def test_calculate_limits(self):
        limits = frame_convolution.calculate_limits((5, 5), (3, 3))
        assert limits == (1, 1, 4, 4)

    def test_is_in_sub_shape(self):
        assert not frame_convolution.is_in_sub_shape(0, (1, 1, 4, 4), (5, 5))
        assert not frame_convolution.is_in_sub_shape(4, (1, 1, 4, 4), (5, 5))
        assert not frame_convolution.is_in_sub_shape(5, (1, 1, 4, 4), (5, 5))
        assert not frame_convolution.is_in_sub_shape(9, (1, 1, 4, 4), (5, 5))
        assert frame_convolution.is_in_sub_shape(6, (1, 1, 4, 4), (5, 5))
        assert frame_convolution.is_in_sub_shape(8, (1, 1, 4, 4), (5, 5))
        assert frame_convolution.is_in_sub_shape(16, (1, 1, 4, 4), (5, 5))
        assert frame_convolution.is_in_sub_shape(18, (1, 1, 4, 4), (5, 5))
        assert not frame_convolution.is_in_sub_shape(21, (1, 1, 4, 4), (5, 5))
        assert not frame_convolution.is_in_sub_shape(24, (1, 1, 4, 4), (5, 5))

    def test_simple_convolution(self):
        convolver = frame_convolution.FrameMaker(mask=np.ones((5, 5))).convolver_for_kernel_shape(
            (5, 5)).convolver_for_kernel(np.ones((5, 5)))

        # pixel_vector = [0, 0, 0, 0, 0,
        #                 0, 0, 0, 0, 0,
        #                 0, 0, 1, 0, 0,
        #                 0, 0, 0, 0, 0,
        #                 0, 0, 0, 0, 0]

        pixel_dict = {16: 1}

        convolved_vector = convolver.convolve_vector(pixel_dict, sub_shape=(3, 3))

        # print(convolved_vector)

        # assert (np.array([0, 0, 0, 0, 0,
        #                   0, 1, 1, 1, 0,
        #                   0, 1, 1, 1, 0,
        #                   0, 1, 1, 1, 0,
        #                   0, 0, 0, 0, 0]) == convolved_vector).all()

        assert {6: 1, 7: 1, 8: 1, 11: 1, 12: 1, 13: 1, 16: 1, 17: 1, 18: 1} == convolved_vector
