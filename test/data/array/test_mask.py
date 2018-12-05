import numpy as np
import pytest

from autolens.data.array.util import mapping_util, mask_util
from autolens.data.array import mask


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

    def test__mask_padded__5x5__input__all_are_false(self):

        msk = mask.Mask.padded_for_shape_and_pixel_scale(shape=(5, 5), pixel_scale=1.5)

        assert msk.shape == (5, 5)
        assert (msk == np.array([[False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False]])).all()

    def test__mask_masked__5x5__input__all_are_false(self):
        msk = mask.Mask.masked_for_shape_and_pixel_scale(shape=(5, 5), pixel_scale=1)

        assert msk.shape == (5, 5)
        assert (msk == np.array([[True, True, True, True, True],
                                 [True, True, True, True, True],
                                 [True, True, True, True, True],
                                 [True, True, True, True, True],
                                 [True, True, True, True, True]])).all()

    def test__mask_circular__compare_to_array_util(self):
        msk_util = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(5, 4), pixel_scale=2.7,
                                                                             radius_arcsec=3.5, centre=(1.0, 1.0))

        msk = mask.Mask.circular(shape=(5, 4), pixel_scale=2.7, radius_arcsec=3.5, centre=(1.0, 1.0))

        assert (msk == msk_util).all()

    def test__mask_annulus__compare_to_array_util(self):
        msk_util = mask_util.mask_circular_annular_from_shape_pixel_scale_and_radii(shape=(5, 4), pixel_scale=2.7,
                                                                                    inner_radius_arcsec=0.8,
                                                                                    outer_radius_arcsec=3.5,
                                                                                    centre=(1.0, 1.0))

        msk = mask.Mask.circular_annular(shape=(5, 4), pixel_scale=2.7, inner_radius_arcsec=0.8, outer_radius_arcsec=3.5,
                                         centre=(1.0, 1.0))

        assert (msk == msk_util).all()

    def test__mask_anti_annulus__compare_to_array_util(self):
        msk_util = mask_util.mask_circular_anti_annular_from_shape_pixel_scale_and_radii(shape=(9, 9), pixel_scale=1.2,
                                                                                         inner_radius_arcsec=0.8,
                                                                                         outer_radius_arcsec=2.2,
                                                                                         outer_radius_2_arcsec=3.0,
                                                                                         centre=(1.0, 1.0))

        msk = mask.Mask.circular_anti_annular(shape=(9, 9), pixel_scale=1.2, inner_radius_arcsec=0.8,
                                              outer_radius_arcsec=2.2, outer_radius_2_arcsec=3.0, origin=(1.0, 1.0))

        assert (msk == msk_util).all()

    def test__mask_elliptical__compare_to_array_util(self):

        msk_util = mask_util.mask_elliptical_from_shape_pixel_scale_and_radius(shape=(8, 5), pixel_scale=2.7,
                        major_axis_radius_arcsec=5.7, axis_ratio=0.4, phi=40.0, centre=(1.0, 1.0))

        msk = mask.Mask.elliptical(shape=(8, 5), pixel_scale=2.7,
                        major_axis_radius_arcsec=5.7, axis_ratio=0.4, phi=40.0, centre=(1.0, 1.0))

        assert (msk == msk_util).all()

    def test__mask_elliptical_annular__compare_to_array_util(self):

        msk_util = mask_util.mask_elliptical_annular_from_shape_pixel_scale_and_radius(shape=(8, 5), pixel_scale=2.7,
                        inner_major_axis_radius_arcsec=2.1, inner_axis_ratio=0.6, inner_phi=20.0,
                        outer_major_axis_radius_arcsec=5.7, outer_axis_ratio=0.4, outer_phi=40.0, centre=(1.0, 1.0))

        msk = mask.Mask.elliptical_annular(shape=(8, 5), pixel_scale=2.7,
                        inner_major_axis_radius_arcsec=2.1, inner_axis_ratio=0.6, inner_phi=20.0,
                        outer_major_axis_radius_arcsec=5.7, outer_axis_ratio=0.4, outer_phi=40.0, centre=(1.0, 1.0))

        assert (msk == msk_util).all()

    def test__grid_to_pixel__compare_to_array_utill(self):
        msk = np.array([[True, True, True],
                        [True, False, False],
                        [True, True, False]])

        msk = mask.Mask(msk, pixel_scale=7.0)

        grid_to_pixel_util = mask_util.masked_grid_1d_index_to_2d_pixel_index_from_mask(msk)

        assert msk.masked_grid_index_to_pixel == pytest.approx(grid_to_pixel_util, 1e-4)

    def test__map_2d_array_to_masked_1d_array__compare_to_array_util(self):
        array_2d = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9],
                             [10, 11, 12]])

        msk = np.array([[True, False, True],
                        [False, False, False],
                        [True, False, True],
                        [True, True, True]])

        array_1d_util = mapping_util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(msk, array_2d)

        msk = mask.Mask(msk, pixel_scale=3.0)

        array_1d = msk.map_2d_array_to_masked_1d_array(array_2d)

        assert (array_1d == array_1d_util).all()

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

        blurring_mask_util = mask_util.mask_blurring_from_mask_and_psf_shape(mask=msk, psf_shape=(3, 3))

        msk = mask.Mask(msk, pixel_scale=1.0)
        blurring_mask = msk.blurring_mask_for_psf_shape(psf_shape=(3, 3))

        assert (blurring_mask == blurring_mask_util).all()

    def test__edge_image_pixels__compare_to_array_util(self):
        msk = np.array([[True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, False, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True]])

        edge_pixels_util = mask_util.edge_pixels_from_mask(msk)

        msk = mask.Mask(msk, pixel_scale=3.0)

        assert msk.edge_pixels == pytest.approx(edge_pixels_util, 1e-4)

    def test__border_image_pixels__compare_to_array_util(self):

        msk = np.array([[False, False, False, False, False, False, False, True],
                         [False,  True,  True,  True,  True,  True, False, True],
                         [False,  True, False, False, False,  True, False, True],
                         [False,  True, False,  True, False,  True, False, True],
                         [False,  True, False, False, False,  True, False, True],
                         [False,  True,  True,  True,  True,  True, False, True],
                         [False, False, False, False, False, False, False, True]])

        border_pixels_util = mask_util.border_pixels_from_mask(msk)

        msk = mask.Mask(msk, pixel_scale=3.0)

        assert msk.border_pixels == pytest.approx(border_pixels_util, 1e-4)