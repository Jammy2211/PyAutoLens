import numpy as np
import pytest

from autolens import exc
from autolens.data.imaging import image, convolution
from autolens.data.array import mask


@pytest.fixture(name="simple_mask_index_array")
def make_simple_mask_index_array():
    return np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])


@pytest.fixture(name="cross_mask")
def make_cross_mask():
    mask = np.full((3, 3), False)

    mask[0, 0] = True
    mask[0, 2] = True
    mask[2, 2] = True
    mask[2, 0] = True

    return mask


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
    return cross_convolver.make_blurring_image_frame_indexes((3, 3), blurring_region_mask=np.full((3, 3), False))


@pytest.fixture(name="simple_convolver")
def make_simple_convolver():
    return convolution.ConvolverImage(mask=np.full((3, 3), False), blurring_mask=(np.full((3, 3), False)),
                                      psf=np.array([[1.0, 2.0, 3.0],
                                                    [4.0, 5.0, 6.0],
                                                    [7.0, 8.0, 9.0]]))


@pytest.fixture(name="cross_convolver")
def make_cross_convolver(cross_mask):
    return convolution.ConvolverImage(mask=cross_mask, blurring_mask=(np.full((3, 3), False)),
                                      psf=np.array([[1.0, 2.0, 3.0],
                                                    [4.0, 5.0, 6.0],
                                                    [7.0, 8.0, 9.0]]))


@pytest.fixture(name="simple_kernel")
def make_simple_kernel():
    return np.array([[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]])


class TestNumbering(object):

    def test_simple_numbering(self, simple_mask_index_array):
        shape = (3, 3)

        convolver = convolution.ConvolverImage(mask=np.full(shape, False), blurring_mask=np.full(shape, False),
                                               psf=np.ones((1, 1)))

        mask_index_array = convolver.mask_index_array

        assert mask_index_array.shape == shape
        # noinspection PyUnresolvedReferences
        assert (mask_index_array == simple_mask_index_array).all()

    def test_simple_mask(self, cross_mask):
        convolver = convolution.ConvolverImage(mask=cross_mask, blurring_mask=np.full(cross_mask.shape, False),
                                               psf=np.ones((1, 1)))

        assert (convolver.mask_index_array == np.array([[-1, 0, -1], [1, 2, 3], [-1, 4, -1]])).all()

    def test_even_failure(self):
        with pytest.raises(exc.KernelException):
            convolution.ConvolverImage(mask=np.full((3, 3), False), blurring_mask=np.full((3, 3), False),
                                       psf=np.ones((2, 2)))

    def test_mismatching_masks_failure(self):
        with pytest.raises(exc.KernelException):
            convolution.ConvolverImage(mask=np.full((3, 3), False),
                                       blurring_mask=np.full((3, 4), False), psf=np.ones((1, 1)))


class TestFrameExtraction(object):

    def test_trivial_frame_at_coords(self, simple_convolver):
        frame, psf_frame = simple_convolver.frame_at_coordinates_jit(coordinates=(1, 1), mask=np.full((3, 3), False),
                                                                     mask_index_array=simple_convolver.mask_index_array,
                                                                     psf=simple_convolver.psf)

        assert (frame == np.array([i for i in range(9)])).all()

    def test_corner_frame(self, simple_convolver):
        corner_frame = np.array([0, 1, 3, 4, -1, -1, -1, -1, -1])

        frame, psf_frame = simple_convolver.frame_at_coordinates_jit(coordinates=(0, 0), mask=np.full((3, 3), False),
                                                                     mask_index_array=simple_convolver.mask_index_array,
                                                                     psf=simple_convolver.psf)

        assert (frame == corner_frame).all()

    def test_trivial_psf_frame_at_coords(self, simple_convolver):
        frame, psf_frame = simple_convolver.frame_at_coordinates_jit(coordinates=(1, 1), mask=np.full((3, 3), False),
                                                                     mask_index_array=simple_convolver.mask_index_array,
                                                                     psf=simple_convolver.psf)

        assert (psf_frame == np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])).all()

    def test_corner_psf_frame(self, simple_convolver):
        corner_frame = [0, 1, 3, 4]

        frame, psf_frame = simple_convolver.frame_at_coordinates_jit(coordinates=(0, 0), mask=np.full((3, 3), False),
                                                                     mask_index_array=simple_convolver.mask_index_array,
                                                                     psf=simple_convolver.psf)

        assert (psf_frame == np.array([5.0, 6.0, 8.0, 9.0, -1, -1, -1, -1, -1])).all()

    def test_simple_square(self, simple_convolver):
        assert 9 == len(simple_convolver.image_frame_indexes)

        assert (simple_convolver.image_frame_indexes[4] == np.array([i for i in range(9)])).all()

    def test_frame_5x5_kernel__at_coords(self):
        convolver = convolution.ConvolverImage(mask=np.full((3, 3), False), blurring_mask=(np.full((3, 3), False)),
                                               psf=np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                                                             [6.0, 7.0, 8.0, 9.0, 10.0],
                                                             [11.0, 12.0, 13.0, 14.0, 15.0],
                                                             [16.0, 17.0, 18.0, 19.0, 20.0],
                                                             [21.0, 22.0, 23.0, 24.0, 25.0]]))

        frame, psf_frame = convolver.frame_at_coordinates_jit(coordinates=(0, 0), mask=np.full((3, 3), False),
                                                              mask_index_array=convolver.mask_index_array,
                                                              psf=convolver.psf)

        assert (psf_frame == np.array([13.0, 14.0, 15.0, 18.0, 19.0, 20.0, 23.0, 24.0, 25.0, -1., -1., -1., -1., -1.,
                                       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])).all()

        frame, psf_frame = convolver.frame_at_coordinates_jit(coordinates=(1, 0), mask=np.full((3, 3), False),
                                                              mask_index_array=convolver.mask_index_array,
                                                              psf=convolver.psf)

        assert (psf_frame == np.array([8.0, 9.0, 10.0, 13.0, 14.0, 15.0, 18.0, 19.0, 20.0, -1., -1., -1., -1., -1.,
                                       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1])).all()

        frame, psf_frame = convolver.frame_at_coordinates_jit(coordinates=(1, 1), mask=np.full((3, 3), False),
                                                              mask_index_array=convolver.mask_index_array,
                                                              psf=convolver.psf)

        assert (psf_frame == np.array([7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 17.0, 18.0, 19.0, -1., -1., -1., -1., -1.,
                                       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1])).all()


class TestImageFrameIndexes(object):

    def test_masked_cross__3x3_kernel(self, cross_convolver):
        assert 5 == len(cross_convolver.image_frame_indexes)

        assert (cross_convolver.image_frame_indexes[0] == np.array([0, 1, 2, 3, -1, -1, -1, -1, -1])).all()
        assert (cross_convolver.image_frame_indexes[1] == np.array([0, 1, 2, 4, -1, -1, -1, -1, -1])).all()
        assert (cross_convolver.image_frame_indexes[2] == np.array([0, 1, 2, 3, 4, -1, -1, -1, -1])).all()
        assert (cross_convolver.image_frame_indexes[3] == np.array([0, 2, 3, 4, -1, -1, -1, -1, -1])).all()
        assert (cross_convolver.image_frame_indexes[4] == np.array([1, 2, 3, 4, -1, -1, -1, -1, -1])).all()

    def test_masked_square__3x5_kernel__loses_edge_of_top_and_bottom_rows(self):
        convolver = convolution.ConvolverImage(mask=np.full((3, 3), False), blurring_mask=(np.full((3, 3), False)),
                                               psf=np.ones((3, 5)))

        assert (convolver.image_frame_indexes[0] == np.array(
            [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[1] == np.array(
            [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[2] == np.array(
            [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[3] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[4] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[5] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[6] == np.array(
            [3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[7] == np.array(
            [3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[8] == np.array(
            [3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])).all()

    def test_masked_square__5x3_kernel__loses_edge_of_left_and_right_columns(self):
        convolver = convolution.ConvolverImage(mask=np.full((3, 3), False), blurring_mask=(np.full((3, 3), False)),
                                               psf=np.ones((5, 3)))

        assert (convolver.image_frame_indexes[0] == np.array(
            [0, 1, 3, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[1] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[2] == np.array(
            [1, 2, 4, 5, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[3] == np.array(
            [0, 1, 3, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[4] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[5] == np.array(
            [1, 2, 4, 5, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[6] == np.array(
            [0, 1, 3, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[7] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_indexes[8] == np.array(
            [1, 2, 4, 5, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])).all()

    def test_masked_square__5x5_kernel(self):
        convolver = convolution.ConvolverImage(mask=np.full((3, 3), False), blurring_mask=(np.full((3, 3), False)),
                                               psf=np.ones((5, 5)))

        assert (convolver.image_frame_indexes[0] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()
        assert (convolver.image_frame_indexes[1] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()
        assert (convolver.image_frame_indexes[2] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()
        assert (convolver.image_frame_indexes[3] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()
        assert (convolver.image_frame_indexes[4] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()
        assert (convolver.image_frame_indexes[5] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()
        assert (convolver.image_frame_indexes[6] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()
        assert (convolver.image_frame_indexes[7] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()
        assert (convolver.image_frame_indexes[8] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()


class TestImageFrameKernels(object):

    def test_simple_square(self, simple_convolver):
        assert 9 == len(simple_convolver.image_frame_indexes)

        assert (simple_convolver.image_frame_psfs[4] == np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])).all()

    def test_masked_square__3x5_kernel__loses_edge_of_top_and_bottom_rows(self):
        convolver = convolution.ConvolverImage(mask=np.full((3, 3), False), blurring_mask=(np.full((3, 3), False)),
                                               psf=np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                                                             [6.0, 7.0, 8.0, 9.0, 10.0],
                                                             [11.0, 12.0, 13.0, 14.0, 15.0]]))

        assert (convolver.image_frame_psfs[0] == np.array([8.0, 9.0, 10.0, 13.0, 14.0, 15.0, -1, -1, -1,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[1] == np.array([7.0, 8.0, 9.0, 12.0, 13.0, 14.0, -1, -1, -1,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[2] == np.array([6.0, 7.0, 8.0, 11.0, 12.0, 13.0, -1, -1, -1,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[3] == np.array([3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 13.0, 14.0, 15.0,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[4] == np.array([2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 12.0, 13.0, 14.0,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[5] == np.array([1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[6] == np.array([3.0, 4.0, 5.0, 8.0, 9.0, 10.0, -1, -1, -1,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[7] == np.array([2.0, 3.0, 4.0, 7.0, 8.0, 9.0, -1, -1, -1,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[8] == np.array([1.0, 2.0, 3.0, 6.0, 7.0, 8.0, -1, -1, -1,
                                                           -1, -1, -1, -1, -1, -1])).all()

    def test_masked_square__5x3_kernel__loses_edge_of_left_and_right_columns(self):
        convolver = convolution.ConvolverImage(mask=np.full((3, 3), False), blurring_mask=(np.full((3, 3), False)),
                                               psf=np.array([[1.0, 2.0, 3.0],
                                                             [4.0, 5.0, 6.0],
                                                             [7.0, 8.0, 9.0],
                                                             [10.0, 11.0, 12.0],
                                                             [13.0, 14.0, 15.0]]))

        assert (convolver.image_frame_psfs[0] == np.array([8.0, 9.0, 11.0, 12.0, 14.0, 15.0, -1, -1, -1,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[1] == np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[2] == np.array([7.0, 8.0, 10.0, 11.0, 13.0, 14.0, -1, -1, -1,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[3] == np.array([5.0, 6.0, 8.0, 9.0, 11.0, 12.0, -1, -1, -1,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[4] == np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[5] == np.array([4.0, 5.0, 7.0, 8.0, 10.0, 11.0, -1, -1, -1,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[6] == np.array([2.0, 3.0, 5.0, 6.0, 8.0, 9.0, -1, -1, -1,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[7] == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                                                           -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.image_frame_psfs[8] == np.array([1.0, 2.0, 4.0, 5.0, 7.0, 8.0, -1, -1, -1,
                                                           -1, -1, -1, -1, -1, -1])).all()


class TestBlurringFrameIndxes(object):

    def test__blurring_region_3x3_kernel(self, cross_mask):
        blurring_mask = np.array([[False, True, False],
                                  [True, True, True],
                                  [False, True, False]])

        convolver = convolution.ConvolverImage(mask=cross_mask, blurring_mask=blurring_mask, psf=np.ones((3, 3)))

        assert (convolver.blurring_frame_indexes[0] == np.array([0, 1, 2, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.blurring_frame_indexes[1] == np.array([0, 2, 3, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.blurring_frame_indexes[2] == np.array([1, 2, 4, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.blurring_frame_indexes[3] == np.array([2, 3, 4, -1, -1, -1, -1, -1, -1])).all()

    def test__blurring_region_5x5_kernel(self, cross_mask):
        blurring_mask = np.array([[False, True, False],
                                  [True, True, True],
                                  [False, True, False]])

        convolver = convolution.ConvolverImage(mask=cross_mask, blurring_mask=blurring_mask, psf=np.ones((5, 5)))

        assert (convolver.blurring_frame_indexes[0] == np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1,
                                                                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                                 -1])).all()
        assert (convolver.blurring_frame_indexes[1] == np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1,
                                                                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                                 -1])).all()
        assert (convolver.blurring_frame_indexes[2] == np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1,
                                                                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                                 -1])).all()
        assert (convolver.blurring_frame_indexes[3] == np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1,
                                                                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                                 -1])).all()


class TestBlurringFrameKernels(object):

    def test__blurring_region_3x3_kernel(self, cross_mask):
        blurring_mask = np.array([[False, True, False],
                                  [True, True, True],
                                  [False, True, False]])

        convolver = convolution.ConvolverImage(mask=cross_mask, blurring_mask=blurring_mask,
                                               psf=np.array([[1.0, 2.0, 3.0],
                                                             [4.0, 5.0, 6.0],
                                                             [7.0, 8.0, 9.0]]))

        assert (convolver.blurring_frame_psfs[0] == np.array([6.0, 8.0, 9.0, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.blurring_frame_psfs[1] == np.array([4.0, 7.0, 8.0, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.blurring_frame_psfs[2] == np.array([2.0, 3.0, 6.0, -1, -1, -1, -1, -1, -1])).all()
        assert (convolver.blurring_frame_psfs[3] == np.array([1.0, 2.0, 4.0, -1, -1, -1, -1, -1, -1])).all()

    def test__blurring_region_5x5_kernel(self, cross_mask):
        blurring_mask = np.array([[False, True, False],
                                  [True, True, True],
                                  [False, True, False]])

        convolver = convolution.ConvolverImage(mask=cross_mask, blurring_mask=blurring_mask,
                                               psf=np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                                                             [6.0, 7.0, 8.0, 9.0, 10.0],
                                                             [11.0, 12.0, 13.0, 14.0, 15.0],
                                                             [16.0, 17.0, 18.0, 19.0, 20.0],
                                                             [21.0, 22.0, 23.0, 24.0, 25.0]]))

        assert (convolver.blurring_frame_psfs[0] == np.array([14.0, 18.0, 19.0, 20.0, 24.0, -1, -1, -1, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()
        assert (convolver.blurring_frame_psfs[1] == np.array([12.0, 16.0, 17.0, 18.0, 22.0, -1, -1, -1, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()
        assert (convolver.blurring_frame_psfs[2] == np.array([4.0, 8.0, 9.0, 10.0, 14.0, -1, -1, -1, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()
        assert (convolver.blurring_frame_psfs[3] == np.array([2.0, 6.0, 7.0, 8.0, 12.0, -1, -1, -1, -1, -1, -1,
                                                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                              -1])).all()


class TestFrameLengths(object):

    def test__frames_are_from_examples_above__lengths_are_right(self):
        convolver = convolution.ConvolverImage(mask=np.full((3, 3), False), blurring_mask=(np.full((3, 3), False)),
                                               psf=np.ones((3, 5)))

        # convolver_image.image_frame_indexes[0] == np.array([0, 1, 2, 3, 4, 5])
        # convolver_image.image_frame_indexes[1] == np.array([0, 1, 2, 3, 4, 5])
        # convolver_image.image_frame_indexes[2] == np.array([0, 1, 2, 3, 4, 5])
        # convolver_image.image_frame_indexes[3] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # (convolver_image.image_frame_indexes[4] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # convolver_image.image_frame_indexes[5] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # convolver_image.image_frame_indexes[6] == np.array([3, 4, 5, 6, 7, 8])
        # convolver_image.image_frame_indexes[7] == np.array([3, 4, 5, 6, 7, 8])
        # convolver_image.image_frame_indexes[8] == np.array([3, 4, 5, 6, 7, 8])

        assert (convolver.image_frame_lengths == np.array([6, 6, 6, 9, 9, 9, 6, 6, 6])).all()

    def test__blurring_frames_from_example_above__lengths_are_right(self, cross_mask):
        blurring_mask = np.array([[False, True, False],
                                  [True, True, True],
                                  [False, True, False]])

        convolver = convolution.ConvolverImage(mask=cross_mask, blurring_mask=blurring_mask,
                                               psf=np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                                                             [6.0, 7.0, 8.0, 9.0, 10.0],
                                                             [11.0, 12.0, 13.0, 14.0, 15.0],
                                                             [16.0, 17.0, 18.0, 19.0, 20.0],
                                                             [21.0, 22.0, 23.0, 24.0, 25.0]]))

        # assert (convolver_image.blurring_frame_psfs[0] == np.array([14.0, 18.0, 19.0, 20.0, 24.0])).all()
        # assert (convolver_image.blurring_frame_psfs[1] == np.array([12.0, 16.0, 17.0, 18.0, 22.0])).all()
        # assert (convolver_image.blurring_frame_psfs[2] == np.array([4.0, 8.0, 9.0, 10.0, 14.0])).all()
        # assert (convolver_image.blurring_frame_psfs[3] == np.array([2.0, 6.0, 7.0, 8.0, 12.0])).all()

        assert (convolver.blurring_frame_lengths == np.array([5, 5, 5, 5])).all()


@pytest.fixture(name="convolver_4_edges")
def make_convolver_4_edges():
    mask = np.array(
        [[True, True, True, True],
         [True, False, False, True],
         [True, False, False, True],
         [True, True, True, True]]
    )

    convolver = convolution.ConvolverImage(mask)
    return convolver.convolver_for_kernel_shape((3, 3), mask)


class TestConvolution(object):

    def test_cross_mask_with_blurring_entries(self, cross_mask):
        kernel = np.array([[0, 0.2, 0],
                           [0.2, 0.4, 0.2],
                           [0, 0.2, 0]])

        blurring_mask = np.array([[False, True, False],
                                  [True, True, True],
                                  [False, True, False]])

        convolver = convolution.ConvolverImage(mask=cross_mask, blurring_mask=blurring_mask, psf=kernel)

        pixel_array = np.array([1, 0, 0, 0, 0])
        blurring_array = np.array([1, 0, 0, 0])

        result = convolver.convolve_image(pixel_array, blurring_array)

        assert (np.round(result, 1) == np.array([0.6, 0.2, 0.2, 0., 0.])).all()


@pytest.fixture(name='sim_image_31x31', scope='function')
def sim_grid_9x9():
    sim_grid_9x9.ma = mask.Mask.padded_mask_unmasked_psf_edges(shape_arc_seconds=(5.5, 5.5), pixel_scale=0.5,
                                                               pad_size=(3, 3))
    sim_grid_9x9.image_grid = sim_grid_9x9.ma.coordinates_collection_for_subgrid_size_and_blurring_shape(
        sub_grid_size=1,
        blurring_shape=(3, 3))
    sim_grid_9x9.mapping = sim_grid_9x9.ma.grid_mapping_with_sub_grid_size(sub_grid_size=1, cluster_grid_size=1)
    return sim_grid_9x9


class TestCompareToFull2dConv:

    def test__compare_convolver_to_2d_convolution(self):
        # Setup a blurred datas_, using the PSF to perform the convolution in 2D, then masks it to make a 1d array.

        im = np.arange(900).reshape(30, 30)
        psf = image.PSF(array=np.arange(49).reshape(7, 7), pixel_scale=1.0)
        blurred_im = psf.convolve(im)
        msk = mask.Mask.circular(shape=(30, 30), pixel_scale=1.0, radius_arcsec=4.0)
        blurred_masked_im_0 = msk.map_2d_array_to_masked_1d_array(blurred_im)

        # Now reproduce this datas_ using the frame convolver_image

        blurring_mask = msk.blurring_mask_for_psf_shape(psf.shape)
        convolver = convolution.ConvolverImage(mask=msk, blurring_mask=blurring_mask, psf=psf)
        im_1d = msk.map_2d_array_to_masked_1d_array(im)
        blurring_im_1d = blurring_mask.map_2d_array_to_masked_1d_array(im)
        blurred_masked_im_1 = convolver.convolve_image(image_array=im_1d, blurring_array=blurring_im_1d)

        assert blurred_masked_im_0 == pytest.approx(blurred_masked_im_1, 1e-4)
