import numpy as np
import pytest

from autolens.imaging import mask
from autolens.inversion.util import pixelization_util


class TestTotalMaskedPixels:

    def test__mask_full_false__full_pixelization_grid_pixels_in_mask(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 4

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [2 ,1]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 6

    def test__mask_is_cross__only_pixelization_grid_pixels_in_mask_are_counted(self):

        ma = mask.Mask(array=np.array([[True,  False, True],
                                       [False, False, False],
                                       [True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 2

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [2 ,1]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 4

    def test__same_as_above_but_3x4_mask(self):

        ma = mask.Mask(array=np.array([[True,  True,  False, True],
                                       [False, False, False, False],
                                       [True,  True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 2

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [1 ,3], [2 ,2]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 6

    def test__same_as_above_but_4x3_mask(self):

        ma = mask.Mask(array=np.array([[True,  False, True],
                                       [True,  False, True],
                                       [False, False, False],
                                       [True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,1]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 2

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,1], [2 ,0], [2 ,1], [2 ,2], [3 ,1]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 6


class TestPixToAllPix:

    def test__mask_full_false__image_mask_and_pixel_centres_fully_overlap__each_pix_maps_to_unmaked_pix(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array \
            ([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [2 ,0], [2 ,1], [2 ,2]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        pix_to_full_pix = pixelization_util.pix_to_full_pix(total_masked_pixels=total_masked_pixels, mask=ma,
                                                            full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (pix_to_full_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__same_as_above__but_remove_some_centre_pixels_and_change_order__order_does_not_change_mapping(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [2 ,2], [1 ,1], [0 ,2], [2 ,0], [0 ,2]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        pix_to_full_pix = pixelization_util.pix_to_full_pix(total_masked_pixels=total_masked_pixels, mask=ma,
                                                            full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (pix_to_full_pix == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    def test__mask_is_cross__some_pix_pixels_are_masked__omitted_from_mapping(self):

        ma = mask.Mask(array=np.array([[True, False, True],
                                       [False, False, False],
                                       [True, False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array \
            ([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [2 ,0], [2 ,1], [2 ,2]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        pix_to_full_pix = pixelization_util.pix_to_full_pix(total_masked_pixels=total_masked_pixels, mask=ma,
                                                            full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (pix_to_full_pix == np.array([1, 3, 4, 5, 7])).all()

    def test__same_as_above__different_mask_and_centres(self):

        ma = mask.Mask(array=np.array([[False, False, True],
                                       [False, False, False],
                                       [True, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        pix_to_full_pix = pixelization_util.pix_to_full_pix(total_masked_pixels=total_masked_pixels, mask=ma,
                                                            full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (pix_to_full_pix == np.array([0, 1, 5])).all()

    def test__same_as_above__but_3x4_mask(self):

        ma = mask.Mask(array=np.array([[True,  True,  False, True],
                                       [False, False, False, False],
                                       [True,  True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1], [2 ,3], [2, 2]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        pix_to_full_pix = pixelization_util.pix_to_full_pix(total_masked_pixels=total_masked_pixels, mask=ma,
                                                            full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (pix_to_full_pix == np.array([2, 3, 4, 5, 7])).all()

    def test__same_as_above__but_4x3_mask(self):

        ma = mask.Mask(array=np.array([[True,  False, True],
                                       [True,  False, True],
                                       [False, False, False],
                                       [True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1], [2 ,2], [3, 1]])

        total_masked_pixels = pixelization_util.total_masked_pixels(mask=ma,
                                                                    full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        pix_to_full_pix = pixelization_util.pix_to_full_pix(total_masked_pixels=total_masked_pixels, mask=ma,
                                                            full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (pix_to_full_pix == np.array([1, 5, 6, 7])).all()


class TestAllPixToPix:

    def test__mask_full_false__image_mask_and_pixel_centres_fully_overlap__each_pix_maps_to_unmaked_pix(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array \
            ([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [2 ,0], [2 ,1], [2 ,2]])

        full_pix_to_pix = pixelization_util.full_pix_to_pix(mask=ma,
                                                            full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (full_pix_to_pix  == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__same_as_above__but_remove_some_centre_pixels_and_change_order__order_does_not_change_mapping(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [2 ,2], [1 ,1], [0 ,2], [2 ,0], [0 ,2]])

        full_pix_to_pix = pixelization_util.full_pix_to_pix(mask=ma,
                                                            full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (full_pix_to_pix  == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    def test__mask_is_cross__some_pix_pixels_are_masked__omitted_from_mapping(self):

        ma = mask.Mask(array=np.array([[True, False, True],
                                       [False, False, False],
                                       [True, False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array \
            ([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [2 ,0], [2 ,1], [2 ,2]])

        full_pix_to_pix = pixelization_util.full_pix_to_pix(mask=ma,
                                                            full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (full_pix_to_pix  == np.array([0, 0, 1, 1, 2, 3, 4, 4, 5])).all()

    def test__same_as_above__different_mask_and_centres(self):

        ma = mask.Mask(array=np.array([[False, False, True],
                                       [False, False, False],
                                       [True, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1]])

        full_pix_to_pix = pixelization_util.full_pix_to_pix(mask=ma,
                                                            full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (full_pix_to_pix  == np.array([0, 1, 2, 2, 2, 2])).all()

    def test__same_as_above__but_3x4_mask(self):

        ma = mask.Mask(array=np.array([[True,  True,  False, True],
                                       [False, False, False, False],
                                       [True,  True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1], [2 ,3], [0, 2]])

        full_pix_to_pix = pixelization_util.full_pix_to_pix(mask=ma,
                                                            full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (full_pix_to_pix  == np.array([0, 0, 0, 1, 2, 3, 4, 4])).all()


    def test__same_as_above__but_4x3_mask(self):

        ma = mask.Mask(array=np.array([[True,  False, True],
                                       [True,  False, True],
                                       [False, False, False],
                                       [True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1], [2 ,2], [3, 1]])

        full_pix_to_pix = pixelization_util.full_pix_to_pix(mask=ma,
                                                            full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (full_pix_to_pix  == np.array([0, 0, 1, 1, 1, 1, 2, 3])).all()