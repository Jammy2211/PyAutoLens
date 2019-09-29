import numpy as np
import pytest

import autolens as al
from autolens import exc


@pytest.fixture(name="simple_mask_7x7")
def make_simple_mask_7x7():

    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return al.Mask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=1)

@pytest.fixture(name="simple_mask_5x5")
def make_simple_mask_5x5():

    mask = np.array([[True, True, True, True, True],
                     [True, False, False, False, True],
                     [True, False, False, False, True],
                     [True, False, False, False, True],
                     [True, True, True, True, True]])

    return al.Mask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=1)

@pytest.fixture(name="simple_mask_index_array")
def make_simple_mask_index_array():
    return np.array([[6, 7, 8], [11, 12, 13], [16, 17, 18]])


@pytest.fixture(name="cross_mask")
def make_cross_mask():
    mask = np.full((5, 5), True)

    mask[2, 2] = False
    mask[1, 2] = False
    mask[3, 2] = False
    mask[2, 1] = False
    mask[2, 3] = False

    return al.Mask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=1)


@pytest.fixture(name="cross_mask_index_array")
def make_cross_mask_index_array():
    return np.array([[-1, 0, -1], [1, 2, 3], [-1, 4, -1]])


@pytest.fixture(name="simple_image_frame_indexes")
def make_simple_image_frame_indexes(simple_convolver):
    return simple_convolver.make_image_frame_indexes((3, 3))


@pytest.fixture(name="cross_image_frame_indexes")
def make_cross_image_frame_indexes(cross_convolver):
    return cross_convolver.make_image_frame_indexes((3, 3))


@pytest.fixture(name="cross_mask_image_frame_indexes")
def make_cross_mask_image_frame_indexes(cross_convolver):
    return cross_convolver.make_blurring_image_frame_indexes(
        (3, 3),
    )


@pytest.fixture(name="simple_convolver")
def make_simple_convolver(simple_mask_5x5):

    return al.Convolver(
        mask=simple_mask_5x5,
        kernel=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    )


@pytest.fixture(name="cross_convolver")
def make_cross_convolver(cross_mask):
    return al.Convolver(
        mask=cross_mask,
        kernel=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    )


@pytest.fixture(name="simple_kernel")
def make_simple_kernel():
    return np.array([[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]])


class TestNumbering(object):
    def test_simple_numbering(self, simple_mask_5x5, simple_mask_index_array):

        convolver = al.Convolver(
            mask=simple_mask_5x5,
            kernel=np.ones((1, 1)),
        )

        mask_index_array = convolver.mask_index_array

        assert mask_index_array.shape == (5,5)
        # noinspection PyUnresolvedReferences
        assert (mask_index_array == np.array([[-1, -1, -1, -1, -1],
                                              [-1, 0, 1, 2, -1],
                                              [-1, 3, 4, 5, -1],
                                              [-1, 6, 7, 8, -1],
                                              [-1, -1, -1, -1, -1]])).all()

    def test__cross_mask(self, cross_mask):
        convolver = al.Convolver(
            mask=cross_mask,
            kernel=np.ones((1, 1)),
        )

        assert (
            convolver.mask_index_array
            == np.array([[-1, -1, -1, -1, -1],
                         [-1, -1, 0, -1, -1],
                         [-1, 1, 2, 3, -1],
                         [-1, -1, 4, -1, -1],
                         [-1, -1, -1, -1, -1]])
        ).all()

    def test__even_kernel_failure(self):
        with pytest.raises(exc.ConvolutionException):
            al.Convolver(
                mask=np.full((3, 3), False),
                kernel=np.ones((2, 2)),
            )


class TestFrameExtraction(object):
    def test__frame_at_coords(self, simple_mask_5x5, simple_convolver):
        frame, kernel_frame = simple_convolver.frame_at_coordinates_jit(
            coordinates=(2, 2),
            mask=simple_mask_5x5,
            mask_index_array=simple_convolver.mask_index_array,
            kernel=simple_convolver.kernel,
        )

        assert (frame == np.array([i for i in range(9)])).all()

        corner_frame = np.array([0, 1, 3, 4, -1, -1, -1, -1, -1])

        frame, kernel_frame = simple_convolver.frame_at_coordinates_jit(
            coordinates=(1, 1),
            mask=simple_mask_5x5,
            mask_index_array=simple_convolver.mask_index_array,
            kernel=simple_convolver.kernel,
        )

        assert (frame == corner_frame).all()

    def test__kernel_frame_at_coords(self, simple_mask_5x5, simple_convolver):

        frame, kernel_frame = simple_convolver.frame_at_coordinates_jit(
            coordinates=(2, 2),
            mask=simple_mask_5x5,
            mask_index_array=simple_convolver.mask_index_array,
            kernel=simple_convolver.kernel,
        )

        assert (
            kernel_frame == np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        ).all()

        frame, kernel_frame = simple_convolver.frame_at_coordinates_jit(
            coordinates=(1, 1),
            mask=simple_mask_5x5,
            mask_index_array=simple_convolver.mask_index_array,
            kernel=simple_convolver.kernel,
        )

        assert (
            kernel_frame == np.array([5.0, 6.0, 8.0, 9.0, -1, -1, -1, -1, -1])
        ).all()

    def test__simple_square(self, simple_convolver):
        assert 9 == len(simple_convolver.image_frame_1d_indexes)

        assert (
            simple_convolver.image_frame_1d_indexes[4]
            == np.array([i for i in range(9)])
        ).all()

    def test__frame_5x5_kernel__at_coords(self, simple_mask_7x7):
        convolver = al.Convolver(
            mask=simple_mask_7x7,
            kernel=np.array(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0, 25.0],
                ]
            ),
        )

        frame, kernel_frame = convolver.frame_at_coordinates_jit(
            coordinates=(2, 2),
            mask=simple_mask_7x7,
            mask_index_array=convolver.mask_index_array,
            kernel=convolver.kernel,
        )

        assert (
            kernel_frame
            == np.array(
                [
                    13.0,
                    14.0,
                    15.0,
                    18.0,
                    19.0,
                    20.0,
                    23.0,
                    24.0,
                    25.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ]
            )
        ).all()

        frame, kernel_frame = convolver.frame_at_coordinates_jit(
            coordinates=(3, 2),
            mask=simple_mask_7x7,
            mask_index_array=convolver.mask_index_array,
            kernel=convolver.kernel,
        )

        assert (
            kernel_frame
            == np.array(
                [
                    8.0,
                    9.0,
                    10.0,
                    13.0,
                    14.0,
                    15.0,
                    18.0,
                    19.0,
                    20.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1,
                ]
            )
        ).all()

        frame, kernel_frame = convolver.frame_at_coordinates_jit(
            coordinates=(3, 3),
            mask=simple_mask_7x7,
            mask_index_array=convolver.mask_index_array,
            kernel=convolver.kernel,
        )

        assert (
            kernel_frame
            == np.array(
                [
                    7.0,
                    8.0,
                    9.0,
                    12.0,
                    13.0,
                    14.0,
                    17.0,
                    18.0,
                    19.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1,
                ]
            )
        ).all()


class TestImageFrameIndexes(object):
    def test__masked_cross__3x3_kernel(self, cross_convolver):
        assert 5 == len(cross_convolver.image_frame_1d_indexes)

        assert (
            cross_convolver.image_frame_1d_indexes[0]
            == np.array([0, 1, 2, 3, -1, -1, -1, -1, -1])
        ).all()
        assert (
            cross_convolver.image_frame_1d_indexes[1]
            == np.array([0, 1, 2, 4, -1, -1, -1, -1, -1])
        ).all()
        assert (
            cross_convolver.image_frame_1d_indexes[2]
            == np.array([0, 1, 2, 3, 4, -1, -1, -1, -1])
        ).all()
        assert (
            cross_convolver.image_frame_1d_indexes[3]
            == np.array([0, 2, 3, 4, -1, -1, -1, -1, -1])
        ).all()
        assert (
            cross_convolver.image_frame_1d_indexes[4]
            == np.array([1, 2, 3, 4, -1, -1, -1, -1, -1])
        ).all()

    def test__masked_square__3x5_kernel__loses_edge_of_top_and_bottom_rows(self, simple_mask_7x7):
        convolver = al.Convolver(
            mask=simple_mask_7x7,
            kernel=np.ones((3, 5)),
        )

        assert (
            convolver.image_frame_1d_indexes[0]
            == np.array([0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[1]
            == np.array([0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[2]
            == np.array([0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[3]
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[4]
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[5]
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[6]
            == np.array([3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[7]
            == np.array([3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[8]
            == np.array([3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()

    def test__masked_square__5x3_kernel__loses_edge_of_left_and_right_columns(self, simple_mask_7x7):
        convolver = al.Convolver(
            mask=simple_mask_7x7,
            kernel=np.ones((5, 3)),
        )

        assert (
            convolver.image_frame_1d_indexes[0]
            == np.array([0, 1, 3, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[1]
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[2]
            == np.array([1, 2, 4, 5, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[3]
            == np.array([0, 1, 3, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[4]
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[5]
            == np.array([1, 2, 4, 5, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[6]
            == np.array([0, 1, 3, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[7]
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[8]
            == np.array([1, 2, 4, 5, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()

    def test__masked_square__7x7_kernel(self, simple_mask_7x7):
        convolver = al.Convolver(
            mask=simple_mask_7x7,
            kernel=np.ones((5, 5)),
        )

        assert (
            convolver.image_frame_1d_indexes[0]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[1]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[2]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[3]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[4]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[5]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[6]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[7]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[8]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()


class TestImageFrameKernels(object):
    def test_simple_square(self, simple_convolver):
        assert 9 == len(simple_convolver.image_frame_1d_indexes)

        assert (
            simple_convolver.image_frame_1d_kernels[4]
            == np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        ).all()

    def test_masked_square__3x5_kernel__loses_edge_of_top_and_bottom_rows(self, simple_mask_7x7):
        convolver = al.Convolver(
            mask=simple_mask_7x7,
            kernel=np.array(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0],
                ]
            ),
        )

        assert (
            convolver.image_frame_1d_kernels[0]
            == np.array(
                [8.0, 9.0, 10.0, 13.0, 14.0, 15.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[1]
            == np.array(
                [7.0, 8.0, 9.0, 12.0, 13.0, 14.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[2]
            == np.array(
                [6.0, 7.0, 8.0, 11.0, 12.0, 13.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[3]
            == np.array(
                [
                    3.0,
                    4.0,
                    5.0,
                    8.0,
                    9.0,
                    10.0,
                    13.0,
                    14.0,
                    15.0,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[4]
            == np.array(
                [2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 12.0, 13.0, 14.0, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[5]
            == np.array(
                [1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[6]
            == np.array(
                [3.0, 4.0, 5.0, 8.0, 9.0, 10.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[7]
            == np.array(
                [2.0, 3.0, 4.0, 7.0, 8.0, 9.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[8]
            == np.array(
                [1.0, 2.0, 3.0, 6.0, 7.0, 8.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()

    def test_masked_square__5x3_kernel__loses_edge_of_left_and_right_columns(self, simple_mask_7x7):
        convolver = al.Convolver(
            mask=simple_mask_7x7,
            kernel=np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                ]
            ),
        )

        assert (
            convolver.image_frame_1d_kernels[0]
            == np.array(
                [8.0, 9.0, 11.0, 12.0, 14.0, 15.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[1]
            == np.array(
                [
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[2]
            == np.array(
                [7.0, 8.0, 10.0, 11.0, 13.0, 14.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[3]
            == np.array(
                [5.0, 6.0, 8.0, 9.0, 11.0, 12.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[4]
            == np.array(
                [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[5]
            == np.array(
                [4.0, 5.0, 7.0, 8.0, 10.0, 11.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[6]
            == np.array(
                [2.0, 3.0, 5.0, 6.0, 8.0, 9.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[7]
            == np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[8]
            == np.array(
                [1.0, 2.0, 4.0, 5.0, 7.0, 8.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()


class TestBlurringFrameIndxes(object):
    def test__blurring_region_3x3_kernel(self, cross_mask):

        convolver = al.Convolver(
            mask=cross_mask,  kernel=np.ones((3, 3))
        )

        print(convolver.blurring_frame_1d_indexes)

        assert (
            convolver.blurring_frame_1d_indexes[4]
            == np.array([0, 1, 2, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.blurring_frame_1d_indexes[5]
            == np.array([0, 2, 3, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.blurring_frame_1d_indexes[10]
            == np.array([1, 2, 4, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.blurring_frame_1d_indexes[11]
            == np.array([2, 3, 4, -1, -1, -1, -1, -1, -1])
        ).all()


class TestBlurringFrameKernels(object):
    def test__blurring_region_3x3_kernel(self, cross_mask):

        convolver = al.Convolver(
            mask=cross_mask,
            
            kernel=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        )

        assert (
            convolver.blurring_frame_1d_kernels[4]
            == np.array([6.0, 8.0, 9.0, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.blurring_frame_1d_kernels[5]
            == np.array([4.0, 7.0, 8.0, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.blurring_frame_1d_kernels[10]
            == np.array([2.0, 3.0, 6.0, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.blurring_frame_1d_kernels[11]
            == np.array([1.0, 2.0, 4.0, -1, -1, -1, -1, -1, -1])
        ).all()


class TestFrameLengths(object):
    def test__frames_are_from_examples_above__lengths_are_right(self, simple_mask_7x7):
        convolver = al.Convolver(
            mask=simple_mask_7x7,
            kernel=np.ones((3, 5)),
        )

        # convolver_image.image_frame_indexes[0] == np.array([0, 1, 2, 3, 4, 5])
        # convolver_image.image_frame_indexes[1] == np.array([0, 1, 2, 3, 4, 5])
        # convolver_image.image_frame_indexes[2] == np.array([0, 1, 2, 3, 4, 5])
        # convolver_image.image_frame_indexes[3] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # (convolver_image.image_frame_indexes[4] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # convolver_image.image_frame_indexes[5] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # convolver_image.image_frame_indexes[6] == np.array([3, 4, 5, 6, 7, 8])
        # convolver_image.image_frame_indexes[7] == np.array([3, 4, 5, 6, 7, 8])
        # convolver_image.image_frame_indexes[8] == np.array([3, 4, 5, 6, 7, 8])

        assert (
            convolver.image_frame_1d_lengths == np.array([6, 6, 6, 9, 9, 9, 6, 6, 6])
        ).all()

class TestConvolveMappingMatrix(object):
    def test__asymetric_convolver__matrix_blurred_correctly(self):

        mask = np.array([[True, True, True, True, True, True],
            [True, False, False, False, False, True],
            [True, False, False, False, False, True],
            [True, False, False, False, False, True],
            [True, False, False, False, False, True],
            [True, True, True, True, True, True],])

        asymmetric_kernel = np.array([[0, 0.0, 0], [0.4, 0.2, 0.3], [0, 0.1, 0]])

        convolver = al.Convolver(mask=mask, kernel=asymmetric_kernel)

        mapping = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [
                    0,
                    1,
                    0,
                ],  # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        blurred_mapping = convolver.convolve_mapping_matrix(mapping)

        assert (
            blurred_mapping
            == np.array(
                [
                    [0, 0, 0],
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
                    [0, 0, 0],
                ]
            )
        ).all()

    def test__asymetric_convolver__multiple_overlapping_blurred_entires_in_matrix(self):

        mask = np.array([[True, True, True, True, True, True],
            [True, False, False, False, False, True],
            [True, False, False, False, False, True],
            [True, False, False, False, False, True],
            [True, False, False, False, False, True],
            [True, True, True, True, True, True],])


        asymmetric_kernel = np.array([[0, 0.0, 0], [0.4, 0.2, 0.3], [0, 0.1, 0]])

        convolver = al.Convolver(mask=mask, kernel=asymmetric_kernel)

        mapping = np.array(
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [
                    0,
                    1,
                    0,
                ],  # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
                [1, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        blurred_mapping = convolver.convolve_mapping_matrix(mapping)

        assert blurred_mapping == pytest.approx(
            np.array(
                [
                    [0, 0.6, 0],
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
                    [0, 0, 0],
                ]
            ),
            1e-4,
        )


class TestConvolution(object):
    def test_cross_mask_with_blurring_entries(self, cross_mask):
        kernel = np.array([[0, 0.2, 0], [0.2, 0.4, 0.2], [0, 0.2, 0]])

        convolver = al.Convolver(
            mask=cross_mask, kernel=kernel
        )

        image_array = np.array([1, 0, 0, 0, 0])
        blurring_array = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        result = convolver.convolved_image_1d_from_image_array_and_blurring_array(
            image_array=image_array, blurring_array=blurring_array)

        assert (np.round(result, 1) == np.array([0.6, 0.2, 0.2, 0.0, 0.0])).all()


class TestCompareToFull2dConv:
    def test__compare_convolver_to_2d_convolution(self):
        # Setup a blurred datas_, using the PSF to perform the convolution in 2D, then masks it to make a 1d array.

        import scipy.signal

        im = np.arange(900).reshape(30, 30)
        kernel = np.arange(49).reshape(7, 7)
        blurred_im = scipy.signal.convolve2d(im, kernel, mode="same")
        mask = al.Mask.circular(
            shape=(30, 30), pixel_scales=(1.0, 1.0), sub_size=1, radius_arcsec=4.0
        )
        blurred_masked_im_0 = mask.scaled_array_from_array_2d(blurred_im)

        # Now reproduce this datas_ using the frame convolver_image

        blurring_mask = mask.blurring_mask_from_kernel_shape(kernel.shape)
        convolver = al.Convolver(mask=mask, kernel=kernel)
        im_1d = mask.scaled_array_from_array_2d(im)
        blurring_im_1d = blurring_mask.scaled_array_from_array_2d(im)
        blurred_masked_im_1 = convolver.convolved_image_1d_from_image_array_and_blurring_array(
            image_array=im_1d, blurring_array=blurring_im_1d
        )

        assert blurred_masked_im_0 == pytest.approx(blurred_masked_im_1, 1e-4)
