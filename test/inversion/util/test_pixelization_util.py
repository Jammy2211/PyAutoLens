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

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 4

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [2 ,1]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 6

    def test__mask_is_cross__only_pixelization_grid_pixels_in_mask_are_counted(self):

        ma = mask.Mask(array=np.array([[True,  False, True],
                                       [False, False, False],
                                       [True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 2

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [2 ,1]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 4

    def test__same_as_above_but_3x4_mask(self):

        ma = mask.Mask(array=np.array([[True,  True,  False, True],
                                       [False, False, False, False],
                                       [True,  True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 2

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [1 ,3], [2 ,2]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 6

    def test__same_as_above_but_4x3_mask(self):

        ma = mask.Mask(array=np.array([[True,  False, True],
                                       [True,  False, True],
                                       [False, False, False],
                                       [True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,1]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 2

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,1], [2 ,0], [2 ,1], [2 ,2], [3 ,1]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert total_masked_pixels == 6


class TestPixToAllPix:

    def test__mask_full_false__image_mask_and_pixel_centres_fully_overlap__each_pix_maps_to_unmaked_pix(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array \
            ([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [2 ,0], [2 ,1], [2 ,2]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        pix_to_full_pix = pixelization_util.pix_to_full_pix_from_mask_and_pixel_centres(total_pix_pixels=total_masked_pixels, mask=ma,
                                                                                        full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (pix_to_full_pix == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__same_as_above__but_remove_some_centre_pixels_and_change_order__order_does_not_change_mapping(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [2 ,2], [1 ,1], [0 ,2], [2 ,0], [0 ,2]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        pix_to_full_pix = pixelization_util.pix_to_full_pix_from_mask_and_pixel_centres(total_pix_pixels=total_masked_pixels, mask=ma,
                                                                                        full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (pix_to_full_pix == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    def test__mask_is_cross__some_pix_pixels_are_masked__omitted_from_mapping(self):

        ma = mask.Mask(array=np.array([[True, False, True],
                                       [False, False, False],
                                       [True, False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array \
            ([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [2 ,0], [2 ,1], [2 ,2]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        pix_to_full_pix = pixelization_util.pix_to_full_pix_from_mask_and_pixel_centres(total_pix_pixels=total_masked_pixels, mask=ma,
                                                                                        full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (pix_to_full_pix == np.array([1, 3, 4, 5, 7])).all()

    def test__same_as_above__different_mask_and_centres(self):

        ma = mask.Mask(array=np.array([[False, False, True],
                                       [False, False, False],
                                       [True, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        pix_to_full_pix = pixelization_util.pix_to_full_pix_from_mask_and_pixel_centres(total_pix_pixels=total_masked_pixels, mask=ma,
                                                                                        full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (pix_to_full_pix == np.array([0, 1, 5])).all()

    def test__same_as_above__but_3x4_mask(self):

        ma = mask.Mask(array=np.array([[True,  True,  False, True],
                                       [False, False, False, False],
                                       [True,  True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1], [2 ,3], [2, 2]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        pix_to_full_pix = pixelization_util.pix_to_full_pix_from_mask_and_pixel_centres(total_pix_pixels=total_masked_pixels, mask=ma,
                                                                                        full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (pix_to_full_pix == np.array([2, 3, 4, 5, 7])).all()

    def test__same_as_above__but_4x3_mask(self):

        ma = mask.Mask(array=np.array([[True,  False, True],
                                       [True,  False, True],
                                       [False, False, False],
                                       [True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1], [2 ,2], [3, 1]])

        total_masked_pixels = pixelization_util.total_pix_pixels_from_mask(mask=ma,
                                                                           full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        pix_to_full_pix = pixelization_util.pix_to_full_pix_from_mask_and_pixel_centres(total_pix_pixels=total_masked_pixels, mask=ma,
                                                                                        full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (pix_to_full_pix == np.array([1, 5, 6, 7])).all()


class TestAllPixToPix:

    def test__mask_full_false__image_mask_and_pixel_centres_fully_overlap__each_pix_maps_to_unmaked_pix(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array \
            ([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [2 ,0], [2 ,1], [2 ,2]])

        full_pix_to_pix = pixelization_util.full_pix_to_pix_from_mask_and_pixel_centres(mask=ma,
                                                                                        full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (full_pix_to_pix  == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__same_as_above__but_remove_some_centre_pixels_and_change_order__order_does_not_change_mapping(self):

        ma = mask.Mask(array=np.array([[False, False, False],
                                       [False, False, False],
                                       [False, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [2 ,2], [1 ,1], [0 ,2], [2 ,0], [0 ,2]])

        full_pix_to_pix = pixelization_util.full_pix_to_pix_from_mask_and_pixel_centres(mask=ma,
                                                                                        full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (full_pix_to_pix  == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    def test__mask_is_cross__some_pix_pixels_are_masked__omitted_from_mapping(self):

        ma = mask.Mask(array=np.array([[True, False, True],
                                       [False, False, False],
                                       [True, False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array \
            ([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [2 ,0], [2 ,1], [2 ,2]])

        full_pix_to_pix = pixelization_util.full_pix_to_pix_from_mask_and_pixel_centres(mask=ma,
                                                                                        full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (full_pix_to_pix  == np.array([0, 0, 1, 1, 2, 3, 4, 4, 5])).all()

    def test__same_as_above__different_mask_and_centres(self):

        ma = mask.Mask(array=np.array([[False, False, True],
                                       [False, False, False],
                                       [True, False, False]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1]])

        full_pix_to_pix = pixelization_util.full_pix_to_pix_from_mask_and_pixel_centres(mask=ma,
                                                                                        full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (full_pix_to_pix  == np.array([0, 1, 2, 2, 2, 2])).all()

    def test__same_as_above__but_3x4_mask(self):

        ma = mask.Mask(array=np.array([[True,  True,  False, True],
                                       [False, False, False, False],
                                       [True,  True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1], [2 ,3], [0, 2]])

        full_pix_to_pix = pixelization_util.full_pix_to_pix_from_mask_and_pixel_centres(mask=ma,
                                                                                        full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (full_pix_to_pix  == np.array([0, 0, 0, 1, 2, 3, 4, 4])).all()

    def test__same_as_above__but_4x3_mask(self):

        ma = mask.Mask(array=np.array([[True,  False, True],
                                       [True,  False, True],
                                       [False, False, False],
                                       [True,  False, True]]), pixel_scale=1.0)

        full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [0 ,2], [0 ,2], [1 ,1], [2 ,2], [3, 1]])

        full_pix_to_pix = pixelization_util.full_pix_to_pix_from_mask_and_pixel_centres(mask=ma,
                                                                                        full_pix_grid_pixel_centres=full_pix_grid_pixel_centres)

        assert (full_pix_to_pix  == np.array([0, 0, 1, 1, 1, 1, 2, 3])).all()


class TestImageToPix:

    def test__simple_cases_for_image_to_full_pix_and__full_pix_to_pix(self):

        image_to_full_pix = np.array([0, 1, 2, 3, 4])
        full_pix_to_pix = np.array([0, 1, 2, 3, 4])
        image_to_pix = pixelization_util.image_to_pix_from_pix_mappings(image_to_full_pix=image_to_full_pix,
                                                                        full_pix_to_pix=full_pix_to_pix)

        assert (image_to_pix == np.array([0, 1, 2, 3, 4])).all()

        image_to_full_pix = np.array([0, 1, 2, 3, 4])
        full_pix_to_pix = np.array([0, 1, 5, 7, 18])
        image_to_pix = pixelization_util.image_to_pix_from_pix_mappings(image_to_full_pix=image_to_full_pix,
                                                                        full_pix_to_pix=full_pix_to_pix)

        assert (image_to_pix == np.array([0, 1, 5, 7, 18])).all()

        image_to_full_pix = np.array([1, 1, 1, 1, 2])
        full_pix_to_pix = np.array([0, 10, 15, 3, 4])
        image_to_pix = pixelization_util.image_to_pix_from_pix_mappings(image_to_full_pix=image_to_full_pix,
                                                                        full_pix_to_pix=full_pix_to_pix)

        assert (image_to_pix == np.array([10, 10, 10, 10, 15])).all()

        image_to_full_pix = np.array([5, 6, 7, 8, 9])
        full_pix_to_pix = np.array([0, 1, 2, 3, 4, 19, 18, 17, 16, 15])
        image_to_pix = pixelization_util.image_to_pix_from_pix_mappings(image_to_full_pix=image_to_full_pix,
                                                                        full_pix_to_pix=full_pix_to_pix)

        assert (image_to_pix == np.array([19, 18, 17, 16, 15])).all()


class TestPixGridFromFullPixGrid:

    def test__simple_full_pix_grid__full_grid_pix_grid_same_size__straightforward_mappings(self):

        full_pix_grid = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        pix_to_full_pix = np.array([0, 1, 2, 3])
        pix_grid = pixelization_util.pix_grid_from_full_pix_grid(full_pix_grid=full_pix_grid,
                                                                 pix_to_full_pix=pix_to_full_pix)

        assert (pix_grid == np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])).all()

        full_pix_grid = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])
        pix_to_full_pix = np.array([0, 1, 2, 3])
        pix_grid = pixelization_util.pix_grid_from_full_pix_grid(full_pix_grid=full_pix_grid,
                                                                 pix_to_full_pix=pix_to_full_pix)

        assert (pix_grid == np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])).all()

        full_pix_grid = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])
        pix_to_full_pix = np.array([1, 0, 3, 2])
        pix_grid = pixelization_util.pix_grid_from_full_pix_grid(full_pix_grid=full_pix_grid,
                                                                 pix_to_full_pix=pix_to_full_pix)

        assert (pix_grid == np.array([[4.0, 5.0], [0.0, 0.0], [8.0, 7.0], [2.0, 2.0]])).all()

    def test__simple_full_pix_grid__full_grid_pix_bigger_than_pix__straightforward_mappings(self):

        full_pix_grid = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        pix_to_full_pix = np.array([1, 2])
        pix_grid = pixelization_util.pix_grid_from_full_pix_grid(full_pix_grid=full_pix_grid,
                                                                 pix_to_full_pix=pix_to_full_pix)

        assert (pix_grid == np.array([[1.0, 1.0], [2.0, 2.0]])).all()

        full_pix_grid = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])
        pix_to_full_pix = np.array([2, 2, 3])
        pix_grid = pixelization_util.pix_grid_from_full_pix_grid(full_pix_grid=full_pix_grid,
                                                                 pix_to_full_pix=pix_to_full_pix)

        assert (pix_grid == np.array([[2.0, 2.0], [2.0, 2.0], [8.0, 7.0]])).all()

        full_pix_grid = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0], [11.0, 11.0], [-20.0, -15.0]])
        pix_to_full_pix = np.array([1, 0, 5, 2])
        pix_grid = pixelization_util.pix_grid_from_full_pix_grid(full_pix_grid=full_pix_grid,
                                                                 pix_to_full_pix=pix_to_full_pix)

        assert (pix_grid == np.array([[4.0, 5.0], [0.0, 0.0], [-20.0, -15.0], [2.0, 2.0]])).all()