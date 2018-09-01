import numpy as np
from autolens.imaging import imaging_util
from autolens.imaging import mask
import pytest


class TestMask(object):

    def test__constructor(self):

        msk = np.array([[True, True, True, True],
                        [True, False, False, True],
                        [True, True, True, True]])

        msk = mask.Mask(msk, pixel_scale=1)

        assert (msk == np.array([[True, True, True, True],
                                 [True, False, False, True],
                                 [True, True, True, True]])).all()
        assert msk.pixel_scale == 1.0
        assert msk.central_pixel_coordinates == (1.0, 1.5)
        assert msk.shape == (3, 4)
        assert msk.shape_arc_seconds == (3.0, 4.0)

    def test__mask_circular__compare_to_array_util(self):

        msk_util = imaging_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(5, 4), pixel_scale=2.7,
                                                                                radius_arcsec=3.5, centre=(1.0, 1.0))

        msk = mask.Mask.circular(shape=(5, 4), pixel_scale=2.7, radius_mask_arcsec=3.5, centre=(1.0, 1.0))

        assert (msk == msk_util).all()

    def test__mask_annulus__compare_to_array_util(self):

        msk_util = imaging_util.mask_annular_from_shape_pixel_scale_and_radii(shape=(5, 4), pixel_scale=2.7,
                                                                              inner_radius_arcsec=0.8,
                                                                              outer_radius_arcsec=3.5,
                                                                              centre=(1.0, 1.0))

        msk = mask.Mask.annular(shape=(5, 4), pixel_scale=2.7, inner_radius_arcsec=0.8, outer_radius_arcsec=3.5,
                                 centre=(1.0, 1.0))

        assert (msk == msk_util).all()

    def test__mask_unmasked__5x5__input__all_are_false(self):

        msk = mask.Mask.unmasked(shape_arc_seconds=(5, 5), pixel_scale=1)

        assert msk.shape == (5, 5)
        assert (msk == np.array([[False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False]])).all()

    def test__grid_to_pixel__compare_to_array_utill(self):

        msk = np.array([[True, True, True],
                        [True, False, False],
                        [True, True, False]])

        msk = mask.Mask(msk, pixel_scale=7.0)

        grid_to_pixel_util = imaging_util.grid_to_pixel_from_mask(msk)

        assert msk.grid_to_pixel == pytest.approx(grid_to_pixel_util, 1e-4)

    def test__map_2d_array_to_masked_1d_array__compare_to_array_util(self):

        array_2d = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9],
                             [10, 11, 12]])

        msk = np.array([[True, False, True],
                        [False, False, False],
                        [True, False, True],
                        [True, True, True]])

        array_1d_util = imaging_util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(msk, array_2d)

        msk = mask.Mask(msk, pixel_scale=3.0)

        array_1d = msk.map_2d_array_to_masked_1d_array(array_2d)

        assert (array_1d == array_1d_util).all()

    def test__map_masked_1d_array_to_2d_array__compare_to_array_util(self):

        array_1d = np.array([1.0, 6.0, 4.0, 5.0, 2.0])

        msk = np.array([[True, False, True],
                        [False, False, False],
                        [True, False, True],
                        [True, True, True]])

        one_to_two = np.array([[0,1], [1,0], [1,1], [1,2], [2,1]])

        array_2d_util = imaging_util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(array_1d=array_1d,
                                                                                                        shape=(4,3), one_to_two=one_to_two)

        msk = mask.Mask(msk, pixel_scale=3.0)

        array_2d = msk.map_masked_1d_array_to_2d_array(array_1d)

        assert (array_2d == array_2d_util).all()

    def test__masked_image_grid_from_mask__compare_to_array_util(self):

        msk = np.array([[True, True, False, False],
                        [True, False, True, True],
                        [True, True, False, False]])

        image_grid_util = imaging_util.image_grid_masked_from_mask_and_pixel_scale(mask=msk, pixel_scale=2.0)

        msk = mask.Mask(msk, pixel_scale=2.0)

        assert msk.masked_image_grid == pytest.approx(image_grid_util, 1e-4)

    def test__blurring_mask_for_psf_shape__compare_to_array_util(self):

        msk = np.array([[True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True]])

        blurring_mask_util = imaging_util.mask_blurring_from_mask_and_psf_shape(mask=msk, psf_shape=(3, 3))

        msk = mask.Mask(msk, pixel_scale=1.0)
        blurring_mask = msk.blurring_mask_for_psf_shape(psf_shape=(3, 3))

        assert (blurring_mask == blurring_mask_util).all()

    def test__border_image_pixels__compare_to_array_util(self):

        msk = np.array([[True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, False, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True]])

        border_pixels_util = imaging_util.border_pixels_from_mask(msk)

        msk = mask.Mask(msk, pixel_scale=3.0)

        border_pixels = msk.border_pixel_indices

        assert border_pixels == pytest.approx(border_pixels_util, 1e-4)

    def test__border_sub_pixels__compare_to_array_util(self):

        msk = np.array([[True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, False, False, False, True, True],
                        [True, True, False, False, False, True, True],
                        [True, True, False, False, False, True, True],
                        [True, True, True, True, False, False, True],
                        [True, True, True, True, True, False, True]])

        border_sub_pixels_util = imaging_util.border_sub_pixels_from_mask_pixel_scale_and_sub_grid_size(mask=msk,
                                                                                                        pixel_scale=3.0,
                                                                                                        sub_grid_size=2)

        msk = mask.Mask(msk, pixel_scale=3.0)

        border_sub_pixels = msk.border_sub_pixel_indices(sub_grid_size=2)

        assert border_sub_pixels == pytest.approx(border_sub_pixels_util, 1e-4)