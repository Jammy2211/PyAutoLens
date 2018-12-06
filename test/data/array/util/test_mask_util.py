import os

import numpy as np
import pytest

from autolens import exc
from autolens.data.array import mask
from autolens.data.array.util import mask_util

test_data_dir = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestTotalPixels:

    def test__total_image_pixels_from_mask(self):
        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        assert mask_util.total_regular_pixels_from_mask(mask) == 5

    def test__total_sub_pixels_from_mask(self):
        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        assert mask_util.total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size=2) == 20

    def test__total_edge_pixels_from_mask(self):

        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, False, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        assert mask_util.total_edge_pixels_from_mask(mask) == 8

    class TestTotalSparsePixels:
    
        def test__mask_full_false__full_pixelization_grid_pixels_in_mask(self):
    
            ma = mask.Mask(array=np.array([[False, False, False],
                                           [False, False, False],
                                           [False, False, False]]), pixel_scale=1.0)
    
            full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0]])
    
            total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                          unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres)
    
            assert total_masked_pixels == 4
    
            full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [2 ,1]])
    
            total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                          unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres)
    
            assert total_masked_pixels == 6
    
        def test__mask_is_cross__only_pixelization_grid_pixels_in_mask_are_counted(self):
    
            ma = mask.Mask(array=np.array([[True, False, True],
                                           [False, False, False],
                                           [True,  False, True]]), pixel_scale=1.0)
    
            full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0]])
    
            total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                          unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres)
    
            assert total_masked_pixels == 2
    
            full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [2 ,1]])
    
            total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                          unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres)
    
            assert total_masked_pixels == 4
    
        def test__same_as_above_but_3x4_mask(self):
    
            ma = mask.Mask(array=np.array([[True, True, False, True],
                                           [False, False, False, False],
                                           [True,  True,  False, True]]), pixel_scale=1.0)
    
            full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0]])
    
            total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                          unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres)
    
            assert total_masked_pixels == 2
    
            full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,0], [1 ,1], [1 ,2], [1 ,3], [2 ,2]])
    
            total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                          unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres)
    
            assert total_masked_pixels == 6
    
        def test__same_as_above_but_4x3_mask(self):
    
            ma = mask.Mask(array=np.array([[True, False, True],
                                           [True,  False, True],
                                           [False, False, False],
                                           [True,  False, True]]), pixel_scale=1.0)
    
            full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,1]])
    
            total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                          unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres)
    
            assert total_masked_pixels == 2
    
            full_pix_grid_pixel_centres = np.array([[0 ,0], [0 ,1], [0 ,2], [1 ,1], [2 ,0], [2 ,1], [2 ,2], [3 ,1]])
    
            total_masked_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                          unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres)
    
            assert total_masked_pixels == 6


class TestMaskCircular(object):

    def test__input_big_mask__mask(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=1.0,
                                                                         radius_arcsec=5.0)

        assert mask.shape == (3, 3)
        assert (mask == np.array([[False, False, False],
                                  [False, False, False],
                                  [False, False, False]])).all()

    def test__3x3_mask_input_radius_small__mask(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=1.0,
                                                                         radius_arcsec=0.5)

        assert (mask == np.array([[True, True, True],
                                  [True, False, True],
                                  [True, True, True]])).all()

    def test__3x3_mask_input_radius_medium__mask(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=1.0,
                                                                         radius_arcsec=1.3)

        assert (mask == np.array([[True,  False, True],
                                  [False, False, False],
                                  [True,  False, True]])).all()

    def test__3x3_mask_input_radius_large__mask(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=1.0,
                                                                         radius_arcsec=3.0)

        assert (mask == np.array([[False, False, False],
                                  [False, False, False],
                                  [False, False, False]])).all()

    def test__4x3_mask_input_radius_small__mask(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(4, 3), pixel_scale=1.0,
                                                                         radius_arcsec=0.5)

        assert (mask == np.array([[True, True, True],
                                  [True, False, True],
                                  [True, False, True],
                                  [True, True, True]])).all()

    def test__4x3_mask_input_radius_medium__mask(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(4, 3), pixel_scale=1.0,
                                                                         radius_arcsec=1.5001)

        assert (mask == np.array([[True, False, True],
                                  [False, False, False],
                                  [False, False, False],
                                  [True, False, True]])).all()

    def test__4x3_mask_input_radius_large__mask(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(4, 3), pixel_scale=1.0,
                                                                         radius_arcsec=3.0)

        assert (mask == np.array([[False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [False, False, False]])).all()

    def test__4x4_mask_input_radius_small__mask(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(4, 4), pixel_scale=1.0,
                                                                         radius_arcsec=0.72)

        assert (mask == np.array([[True, True, True, True],
                                  [True, False, False, True],
                                  [True, False, False, True],
                                  [True, True, True, True]])).all()

    def test__4x4_mask_input_radius_medium__mask(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(4, 4), pixel_scale=1.0,
                                                                         radius_arcsec=1.7)

        assert (mask == np.array([[True, False, False, True],
                                  [False, False, False, False],
                                  [False, False, False, False],
                                  [True, False, False, True]])).all()

    def test__4x4_mask_input_radius_large__mask(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(4, 4), pixel_scale=1.0,
                                                                         radius_arcsec=3.0)

        assert (mask == np.array([[False, False, False, False],
                                  [False, False, False, False],
                                  [False, False, False, False],
                                  [False, False, False, False]])).all()

    def test__origin_shift__simple_shift_downwards(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=3.0,
                                                                         radius_arcsec=0.5, centre=(-3, 0))

        assert mask.shape == (3, 3)
        assert (mask == np.array([[True, True, True],
                                  [True, True, True],
                                  [True, False, True]])).all()

    def test__origin_shift__simple_shift_right(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=3.0,
                                                                         radius_arcsec=0.5, centre=(0.0, 3.0))

        assert mask.shape == (3, 3)
        assert (mask == np.array([[True, True, True],
                                  [True, True, False],
                                  [True, True, True]])).all()

    def test__origin_shift__diagonal_shift(self):
        mask = mask_util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=3.0,
                                                                         radius_arcsec=0.5, centre=(3, 3))

        assert (mask == np.array([[True, True, False],
                                  [True, True, True],
                                  [True, True, True]])).all()


class TestMaskAnnular(object):

    def test__3x3_mask_inner_radius_zero_outer_radius_small__mask(self):
        mask = mask_util.mask_annular_from_shape_pixel_scale_and_radii(shape=(3, 3), pixel_scale=1.0,
                                                                       inner_radius_arcsec=0.0, outer_radius_arcsec=0.5)

        assert (mask == np.array([[True, True, True],
                                  [True, False, True],
                                  [True, True, True]])).all()

    def test__3x3_mask_inner_radius_small_outer_radius_large__mask(self):
        mask = mask_util.mask_annular_from_shape_pixel_scale_and_radii(shape=(3, 3), pixel_scale=1.0,
                                                                       inner_radius_arcsec=0.5, outer_radius_arcsec=3.0)

        assert (mask == np.array([[False, False, False],
                                  [False, True, False],
                                  [False, False, False]])).all()

    def test__4x3_mask_inner_radius_small_outer_radius_medium__mask(self):
        mask = mask_util.mask_annular_from_shape_pixel_scale_and_radii(shape=(4, 3), pixel_scale=1.0,
                                                                       inner_radius_arcsec=0.51, outer_radius_arcsec=1.51)

        assert (mask == np.array([[True, False, True],
                                  [False, True, False],
                                  [False, True, False],
                                  [True, False, True]])).all()

    def test__4x3_mask_inner_radius_medium_outer_radius_large__mask(self):
        mask = mask_util.mask_annular_from_shape_pixel_scale_and_radii(shape=(4, 3), pixel_scale=1.0,
                                                                       inner_radius_arcsec=1.51, outer_radius_arcsec=3.0)

        assert (mask == np.array([[False, True, False],
                                  [True, True, True],
                                  [True, True, True],
                                  [False, True, False]])).all()

    def test__3x3_mask_inner_radius_small_outer_radius_medium__mask(self):
        mask = mask_util.mask_annular_from_shape_pixel_scale_and_radii(shape=(4, 4), pixel_scale=1.0,
                                                                       inner_radius_arcsec=0.81, outer_radius_arcsec=2.0)

        assert (mask == np.array([[True, False, False, True],
                                  [False, True, True, False],
                                  [False, True, True, False],
                                  [True, False, False, True]])).all()

    def test__4x4_mask_inner_radius_medium_outer_radius_large__mask(self):
        mask = mask_util.mask_annular_from_shape_pixel_scale_and_radii(shape=(4, 4), pixel_scale=1.0,
                                                                       inner_radius_arcsec=1.71, outer_radius_arcsec=3.0)

        assert (mask == np.array([[False, True, True, False],
                                  [True, True, True, True],
                                  [True, True, True, True],
                                  [False, True, True, False]])).all()

    def test__origin_shift__simple_shift_upwards(self):
        mask = mask_util.mask_annular_from_shape_pixel_scale_and_radii(shape=(3, 3), pixel_scale=3.0,
                                                                       inner_radius_arcsec=0.5,
                                                                       outer_radius_arcsec=9.0, centre=(3.0, 0.0))

        assert mask.shape == (3, 3)
        assert (mask == np.array([[False, True, False],
                                  [False, False, False],
                                  [False, False, False]])).all()

    def test__origin_shift__simple_shift_forward(self):
        mask = mask_util.mask_annular_from_shape_pixel_scale_and_radii(shape=(3, 3), pixel_scale=3.0,
                                                                       inner_radius_arcsec=0.5,
                                                                       outer_radius_arcsec=9.0, centre=(0.0, 3.0))

        assert mask.shape == (3, 3)
        assert (mask == np.array([[False, False, False],
                                  [False, False, True],
                                  [False, False, False]])).all()

    def test__origin_shift__diagonal_shift(self):
        mask = mask_util.mask_annular_from_shape_pixel_scale_and_radii(shape=(3, 3), pixel_scale=3.0,
                                                                       inner_radius_arcsec=0.5,
                                                                       outer_radius_arcsec=9.0, centre=(-3.0, 3.0))

        assert mask.shape == (3, 3)
        assert (mask == np.array([[False, False, False],
                                  [False, False, False],
                                  [False, False, True]])).all()


class TestMaskAntiAnnular(object):

    def test__5x5_mask_inner_radius_includes_central_pixel__outer_extended_beyond_radius(self):

        mask = mask_util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape=(5, 5), pixel_scale=1.0,
                                                                            inner_radius_arcsec=0.5, outer_radius_arcsec=10.0,
                                                                            outer_radius_2_arcsec=20.0)

        assert (mask == np.array([[True, True, True, True, True],
                                  [True, True, True, True, True],
                                  [True, True, False, True, True],
                                  [True, True, True, True, True],
                                  [True, True, True, True, True]])).all()

    def test__5x5_mask_inner_radius_includes_3x3_central_pixels__outer_extended_beyond_radius(self):

        mask = mask_util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape=(5, 5), pixel_scale=1.0,
                                                                            inner_radius_arcsec=1.5, outer_radius_arcsec=10.0,
                                                                            outer_radius_2_arcsec=20.0)

        assert (mask == np.array([[True,  True,  True,  True, True],
                                  [True, False, False, False, True],
                                  [True, False, False, False, True],
                                  [True, False, False, False, True],
                                  [True,  True,  True,  True, True]])).all()

    def test__5x5_mask_inner_radius_includes_central_pixel__outer_radius_includes_outer_pixels(self):

        mask = mask_util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape=(5, 5), pixel_scale=1.0,
                                                                            inner_radius_arcsec=0.5, outer_radius_arcsec=1.5,
                                                                            outer_radius_2_arcsec=20.0)

        assert (mask == np.array([[False, False, False, False, False],
                                  [False, True,  True,  True,  False],
                                  [False, True, False,  True,  False],
                                  [False, True,  True,  True,  False],
                                  [False, False, False, False, False]])).all()

    def test__7x7_second_outer_radius_mask_works_too(self):

        mask = mask_util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape=(7, 7), pixel_scale=1.0,
                                                                            inner_radius_arcsec=0.5, outer_radius_arcsec=1.5,
                                                                            outer_radius_2_arcsec=2.9)

        assert (mask == np.array([[True,  True,  True,  True,  True,  True, True],
                                  [True, False, False, False, False, False, True],
                                  [True, False, True,   True,  True, False, True],
                                  [True, False, True,  False,  True, False, True],
                                  [True, False, True,   True,  True, False, True],
                                  [True, False, False, False, False, False, True],
                                  [True,  True,  True,  True,  True,  True, True]])).all()

    def test__origin_shift__diagonal_shift(self):

        mask = mask_util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape=(7, 7), pixel_scale=3.0,
                                                                            inner_radius_arcsec=1.5, outer_radius_arcsec=4.5,
                                                                            outer_radius_2_arcsec=8.7, centre=(-3.0, 3.0))

        assert (mask == np.array([[True,  True,  True,  True,  True,  True,  True],
                                  [True,  True,  True,  True,  True,  True,  True],
                                  [True,  True, False, False, False, False, False],
                                  [True,  True, False, True,   True,  True, False],
                                  [True,  True, False, True,  False,  True, False],
                                  [True,  True, False, True,   True,  True, False],
                                  [True,  True, False, False, False, False, False]])).all()


class TestMaskBlurring(object):

    def test__size__3x3_small_mask(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 3))

        assert (blurring_mask == np.array([[False, False, False],
                                           [False, True, False],
                                           [False, False, False]])).all()

    def test__size__3x3__large_mask(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 3))

        assert (blurring_mask == np.array([[True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True],
                                           [True, True, False, False, False, True, True],
                                           [True, True, False, True, False, True, True],
                                           [True, True, False, False, False, True, True],
                                           [True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True]])).all()

    def test__size__5x5__large_mask(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(5, 5))

        assert (blurring_mask == np.array([[True, True, True, True, True, True, True],
                                           [True, False, False, False, False, False, True],
                                           [True, False, False, False, False, False, True],
                                           [True, False, False, True, False, False, True],
                                           [True, False, False, False, False, False, True],
                                           [True, False, False, False, False, False, True],
                                           [True, True, True, True, True, True, True]])).all()

    def test__size__5x3__large_mask(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(5, 3))

        assert (blurring_mask == np.rot90(np.array([[True, True, True, True, True, True, True],
                                                    [True, True, True, True, True, True, True],
                                                    [True, False, False, False, False, False, True],
                                                    [True, False, False, True, False, False, True],
                                                    [True, False, False, False, False, False, True],
                                                    [True, True, True, True, True, True, True],
                                                    [True, True, True, True, True, True, True]]))).all()

    def test__size__3x5__large_mask(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 5))

        assert (blurring_mask == np.rot90(np.array([[True, True, True, True, True, True, True],
                                                    [True, True, False, False, False, True, True],
                                                    [True, True, False, False, False, True, True],
                                                    [True, True, False, True, False, True, True],
                                                    [True, True, False, False, False, True, True],
                                                    [True, True, False, False, False, True, True],
                                                    [True, True, True, True, True, True, True]]))).all()

    def test__size__3x3__multiple_points(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True],
                         [True, True, True, True, True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 3))

        assert (blurring_mask == np.array([[False, False, False, True, False, False, False],
                                           [False, True, False, True, False, True, False],
                                           [False, False, False, True, False, False, False],
                                           [True, True, True, True, True, True, True],
                                           [False, False, False, True, False, False, False],
                                           [False, True, False, True, False, True, False],
                                           [False, False, False, True, False, False, False]])).all()

    def test__size__5x5__multiple_points(self):
        mask = np.array([[True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(5, 5))

        assert (blurring_mask == np.array([[False, False, False, False, False, False, False, False, False],
                                           [False, False, False, False, False, False, False, False, False],
                                           [False, False, True, False, False, False, True, False, False],
                                           [False, False, False, False, False, False, False, False, False],
                                           [False, False, False, False, False, False, False, False, False],
                                           [False, False, False, False, False, False, False, False, False],
                                           [False, False, True, False, False, False, True, False, False],
                                           [False, False, False, False, False, False, False, False, False],
                                           [False, False, False, False, False, False, False, False,
                                            False]])).all()

    def test__size__5x3__multiple_points(self):
        mask = np.array([[True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(5, 3))

        assert (blurring_mask == np.rot90(np.array([[True, True, True, True, True, True, True, True, True],
                                                    [False, False, False, False, False, False, False, False, False],
                                                    [False, False, True, False, False, False, True, False, False],
                                                    [False, False, False, False, False, False, False, False, False],
                                                    [True, True, True, True, True, True, True, True, True],
                                                    [False, False, False, False, False, False, False, False, False],
                                                    [False, False, True, False, False, False, True, False, False],
                                                    [False, False, False, False, False, False, False, False, False],
                                                    [True, True, True, True, True, True, True, True, True]]))).all()

    def test__size__3x5__multiple_points(self):
        mask = np.array([[True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 5))

        assert (blurring_mask == np.rot90(np.array([[True, False, False, False, True, False, False, False, True],
                                                    [True, False, False, False, True, False, False, False, True],
                                                    [True, False, True, False, True, False, True, False, True],
                                                    [True, False, False, False, True, False, False, False, True],
                                                    [True, False, False, False, True, False, False, False, True],
                                                    [True, False, False, False, True, False, False, False, True],
                                                    [True, False, True, False, True, False, True, False, True],
                                                    [True, False, False, False, True, False, False, False, True],
                                                    [True, False, False, False, True, False, False, False,
                                                     True]]))).all()

    def test__size__3x3__even_sized_image(self):
        mask = np.array([[True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 3))

        assert (blurring_mask == np.array([[False, False, False, True, False, False, False, True],
                                           [False, True, False, True, False, True, False, True],
                                           [False, False, False, True, False, False, False, True],
                                           [True, True, True, True, True, True, True, True],
                                           [False, False, False, True, False, False, False, True],
                                           [False, True, False, True, False, True, False, True],
                                           [False, False, False, True, False, False, False, True],
                                           [True, True, True, True, True, True, True, True]])).all()

    def test__size__5x5__even_sized_image(self):
        mask = np.array([[True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(5, 5))

        assert (blurring_mask == np.array([[True, True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True, True],
                                           [True, True, True, False, False, False, False, False],
                                           [True, True, True, False, False, False, False, False],
                                           [True, True, True, False, False, True, False, False],
                                           [True, True, True, False, False, False, False, False],
                                           [True, True, True, False, False, False, False, False]])).all()

    def test__size__3x3__rectangular_8x9_image(self):
        mask = np.array([[True, True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 3))

        assert (blurring_mask == np.array([[False, False, False, True, False, False, False, True, True],
                                           [False, True, False, True, False, True, False, True, True],
                                           [False, False, False, True, False, False, False, True, True],
                                           [True, True, True, True, True, True, True, True, True],
                                           [False, False, False, True, False, False, False, True, True],
                                           [False, True, False, True, False, True, False, True, True],
                                           [False, False, False, True, False, False, False, True, True],
                                           [True, True, True, True, True, True, True, True, True]])).all()

    def test__size__3x3__rectangular_9x8_image(self):
        mask = np.array([[True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True]])

        blurring_mask = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 3))

        assert (blurring_mask == np.array([[False, False, False, True, False, False, False, True],
                                           [False, True, False, True, False, True, False, True],
                                           [False, False, False, True, False, False, False, True],
                                           [True, True, True, True, True, True, True, True],
                                           [False, False, False, True, False, False, False, True],
                                           [False, True, False, True, False, True, False, True],
                                           [False, False, False, True, False, False, False, True],
                                           [True, True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True, True]])).all()

    def test__size__5x5__multiple_points__mask_extends_beyond_edge_so_raises_mask_exception(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True],
                         [True, True, True, True, True, True, True]])

        with pytest.raises(exc.MaskException):
            mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(5, 5))


class TestGridToMaskedPixel(object):

    def test__setup_3x3_image_one_pixel(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        grid_to_pixel = mask_util.masked_grid_1d_index_to_2d_pixel_index_from_mask(mask)

        assert (grid_to_pixel == np.array([[1, 1]])).all()

    def test__setup_3x3_image__five_pixels(self):
        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        grid_to_pixel = mask_util.masked_grid_1d_index_to_2d_pixel_index_from_mask(mask)

        assert (grid_to_pixel == np.array([[0, 1],
                                           [1, 0], [1, 1], [1, 2],
                                           [2, 1]])).all()

    def test__setup_3x4_image__six_pixels(self):
        mask = np.array([[True, False, True, True],
                         [False, False, False, True],
                         [True, False, True, False]])

        grid_to_pixel = mask_util.masked_grid_1d_index_to_2d_pixel_index_from_mask(mask)

        assert (grid_to_pixel == np.array([[0, 1],
                                           [1, 0], [1, 1], [1, 2],
                                           [2, 1], [2, 3]])).all()

    def test__setup_4x3_image__six_pixels(self):
        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True],
                         [True, True, False]])

        grid_to_pixel = mask_util.masked_grid_1d_index_to_2d_pixel_index_from_mask(mask)

        assert (grid_to_pixel == np.array([[0, 1],
                                           [1, 0], [1, 1], [1, 2],
                                           [2, 1],
                                           [3, 2]])).all()


class TestEdgePixels(object):

    def test__7x7_mask_one_central_pixel__is_entire_edge(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        edge_pixels = mask_util.edge_pixels_from_mask(mask)

        assert (edge_pixels == np.array([0])).all()

    def test__7x7_mask_nine_central_pixels__is_edge(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        edge_pixels = mask_util.edge_pixels_from_mask(mask)

        assert (edge_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

    def test__7x7_mask_rectangle_of_fifteen_central_pixels__is_edge(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True]])

        edge_pixels = mask_util.edge_pixels_from_mask(mask)

        assert (edge_pixels == np.array([0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14])).all()

    def test__8x7_mask_add_edge_pixels__also_in_edge(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, False, False, False, False, False, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True]])

        edge_pixels = mask_util.edge_pixels_from_mask(mask)

        assert (edge_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17])).all()

    def test__8x7_mask_big_square(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, True, True, True, True, True, True]])

        edge_pixels = mask_util.edge_pixels_from_mask(mask)

        assert (edge_pixels == np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 26, 27, 28, 29])).all()

    def test__7x8_mask_add_edge_pixels__also_in_edge(self):
        mask = np.array([[True, True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True, True],
                         [True, True, False, False, False, True, True, True],
                         [True, True, False, False, False, True, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, True, False, False, False, True, True, True],
                         [True, True, True, True, True, True, True, True]])

        edge_pixels = mask_util.edge_pixels_from_mask(mask)

        assert (edge_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14])).all()

    def test__7x8_mask_big_square(self):
        mask = np.array([[True, True, True, True, True, True, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, True, True, True, True, True, True, True]])

        edge_pixels = mask_util.edge_pixels_from_mask(mask)

        assert (edge_pixels == np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24])).all()


class TestBorderPixels(object):

    def test__7x7_mask_one_central_pixel__border_is_central_pixel(self):

        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        border_pixels = mask_util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0])).all()

    def test__7x7_mask_three_pixel__border_pixels_all_of_them_is_central_pixel(self):

        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, False, False, True, True, True, True],
                         [True, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        border_pixels = mask_util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0, 1, 2])).all()

    def test__7x7_mask_nine_central_pixels__central_pixel_is_not_border_pixel(self):

        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        border_pixels = mask_util.border_pixels_from_mask(mask)

        print(border_pixels)

        assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

    def test__7x7_annulus_mask__inner_pixels_excluded(self):

        mask = np.array([[False, False, False, False, False, False, False],
                         [False,  True,  True,  True,  True,  True, False],
                         [False,  True, False, False, False,  True, False],
                         [False,  True, False,  True, False,  True, False],
                         [False,  True, False, False, False,  True, False],
                         [False,  True,  True,  True,  True,  True, False],
                         [False, False, False, False, False, False, False]])

        border_pixels = mask_util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 17, 18, 22, 23, 24, 25,
                                           26, 27, 28, 29, 30, 31])).all()

    def test__same_as_above_but_8x7_annulus_mask__true_values_at_top_or_bottom(self):

        mask = np.array([[False, False, False, False, False, False, False],
                         [False,  True,  True,  True,  True,  True, False],
                         [False,  True, False, False, False,  True, False],
                         [False,  True, False,  True, False,  True, False],
                         [False,  True, False, False, False,  True, False],
                         [False,  True,  True,  True,  True,  True, False],
                         [False, False, False, False, False, False, False],
                         [ True,  True,  True,  True,  True,  True,  True]])

        border_pixels = mask_util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 17, 18, 22, 23, 24, 25,
                                           26, 27, 28, 29, 30, 31])).all()

        mask = np.array([[ True,  True,  True,  True,  True,  True,  True],
                         [False, False, False, False, False, False, False],
                         [False,  True,  True,  True,  True,  True, False],
                         [False,  True, False, False, False,  True, False],
                         [False,  True, False,  True, False,  True, False],
                         [False,  True, False, False, False,  True, False],
                         [False,  True,  True,  True,  True,  True, False],
                         [False, False, False, False, False, False, False]])

        border_pixels = mask_util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 17, 18, 22, 23, 24, 25,
                                           26, 27, 28, 29, 30, 31])).all()

    def test__same_as_above_but_7x8_annulus_mask__true_values_at_right_or_left(self):

        mask = np.array([[False, False, False, False, False, False, False, True],
                         [False,  True,  True,  True,  True,  True, False, True],
                         [False,  True, False, False, False,  True, False, True],
                         [False,  True, False,  True, False,  True, False, True],
                         [False,  True, False, False, False,  True, False, True],
                         [False,  True,  True,  True,  True,  True, False, True],
                         [False, False, False, False, False, False, False, True]])

        border_pixels = mask_util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 17, 18, 22, 23, 24, 25,
                                           26, 27, 28, 29, 30, 31])).all()

        mask = np.array([[True, False, False, False, False, False, False, False],
                         [True, False,  True,  True,  True,  True,  True, False],
                         [True, False,  True, False, False, False,  True, False],
                         [True, False,  True, False,  True, False,  True, False],
                         [True, False,  True, False, False, False,  True, False],
                         [True, False,  True,  True,  True,  True,  True, False],
                         [True, False, False, False, False, False, False, False]])

        border_pixels = mask_util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 17, 18, 22, 23, 24, 25,
                                           26, 27, 28, 29, 30, 31])).all()