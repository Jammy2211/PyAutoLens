import shutil
import os
import numpy as np
import pytest

from autolens.data.array.util import mask_util as util
from autolens.data.array.util import binning_util
from autolens.data.array.util import mapping_util
from autolens.data.array import mask as msk
from autolens.data.array.util import grid_util

test_data_dir = "{}/../../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestMask:

    def test__constructor(self):

        mask = np.array([[True, True, True, True],
                        [True, False, False, True],
                        [True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1)

        assert (mask == np.array([[True, True, True, True],
                                 [True, False, False, True],
                                 [True, True, True, True]])).all()
        assert mask.pixel_scale == 1.0
        assert mask.central_pixel_coordinates == (1.0, 1.5)
        assert mask.shape == (3, 4)
        assert mask.shape_arcsec == (3.0, 4.0)

    def test__array_finalize__masks_pass_attributes(self):

        mask = np.array([[True, True, True, True],
                        [True, False, False, True],
                        [True, True, True, True]])

        mask = msk.Mask(array=mask, pixel_scale=1.0)

        mask_new = mask + mask

        assert mask_new.pixel_scale == 1.0
        assert mask_new.origin == (0.0, 0.0)
        assert mask_new.centre == (0.0, 0.0)

    def test__centring__adapts_to_max_and_min_of_mask(self):

        mask = np.array([[True, True, True, True],
                        [True, False, False, True],
                        [True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        assert mask.centre == (0.0, 0.0)

        mask = np.array([[True, True, True, True],
                         [True, False, False, False],
                         [True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        assert mask.centre == (0.0, 0.5)

        mask = np.array([[True, True, False, True],
                         [True, False, False, True],
                         [True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        assert mask.centre == (0.5, 0.0)

        mask = np.array([[True, True, True, True],
                         [False, False, False, True],
                         [True, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        assert mask.centre == (0.0, -0.5)

        mask = np.array([[True, True, True, True],
                         [True, False, False, True],
                         [True, False, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        assert mask.centre == (-0.5, 0.0)

        mask = np.array([[True, True, True, True],
                         [True, False, False, True],
                         [False, True, True, True]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        assert mask.centre == (-0.5, -0.5)


class TestMaskShapes:

    def test__mask_all_unmasked__5x5__input__all_are_false(self):

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(5, 5), pixel_scale=1.5, invert=False)

        assert mask.shape == (5, 5)
        assert (mask == np.array([[False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False]])).all()

        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)

    def test__mask_all_unmasked_inverted__5x5__input__all_are_true(self):

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(5, 5), pixel_scale=1, invert=True)

        assert mask.shape == (5, 5)
        assert (mask == np.array([[True, True, True, True, True],
                                 [True, True, True, True, True],
                                 [True, True, True, True, True],
                                 [True, True, True, True, True],
                                 [True, True, True, True, True]])).all()

        assert mask.origin == (0.0, 0.0)

    def test__mask_circular__compare_to_array_util(self):

        mask_util = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(5, 4), pixel_scale=2.7,
                                                                             radius_arcsec=3.5, centre=(0.0, 0.0))

        mask = msk.Mask.circular(shape=(5, 4), pixel_scale=2.7, radius_arcsec=3.5, centre=(0.0, 0.0))

        assert (mask == mask_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)

    def test__mask_circular__inverted__compare_to_array_util(self):

        mask_util = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(5, 4), pixel_scale=2.7,
                                                                             radius_arcsec=3.5, centre=(0.0, 0.0))

        mask = msk.Mask.circular(shape=(5, 4), pixel_scale=2.7, radius_arcsec=3.5, centre=(0.0, 0.0), invert=True)

        assert (mask == np.invert(mask_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)

    def test__mask_annulus__compare_to_array_util(self):
        
        mask_util = util.mask_circular_annular_from_shape_pixel_scale_and_radii(shape=(5, 4), pixel_scale=2.7,
                                                                                    inner_radius_arcsec=0.8,
                                                                                    outer_radius_arcsec=3.5,
                                                                                    centre=(0.0, 0.0))

        mask = msk.Mask.circular_annular(shape=(5, 4), pixel_scale=2.7, inner_radius_arcsec=0.8, outer_radius_arcsec=3.5,
                                         centre=(0.0, 0.0))

        assert (mask == mask_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)

    def test__mask_annulus_inverted__compare_to_array_util(self):
        mask_util = util.mask_circular_annular_from_shape_pixel_scale_and_radii(shape=(5, 4), pixel_scale=2.7,
                                                                                inner_radius_arcsec=0.8,
                                                                                outer_radius_arcsec=3.5,
                                                                                centre=(0.0, 0.0))

        mask = msk.Mask.circular_annular(shape=(5, 4), pixel_scale=2.7, inner_radius_arcsec=0.8,
                                         outer_radius_arcsec=3.5,
                                         centre=(0.0, 0.0), invert=True)

        assert (mask == np.invert(mask_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)

    def test__mask_anti_annulus__compare_to_array_util(self):
        mask_util = util.mask_circular_anti_annular_from_shape_pixel_scale_and_radii(shape=(9, 9), pixel_scale=1.2,
                                                                                         inner_radius_arcsec=0.8,
                                                                                         outer_radius_arcsec=2.2,
                                                                                         outer_radius_2_arcsec=3.0,
                                                                                         centre=(0.0, 0.0))

        mask = msk.Mask.circular_anti_annular(shape=(9, 9), pixel_scale=1.2, inner_radius_arcsec=0.8,
                                              outer_radius_arcsec=2.2, outer_radius_2_arcsec=3.0, centre=(0.0, 0.0))

        assert (mask == mask_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)

    def test__mask_anti_annulus_inverted__compare_to_array_util(self):
        mask_util = util.mask_circular_anti_annular_from_shape_pixel_scale_and_radii(shape=(9, 9), pixel_scale=1.2,
                                                                                         inner_radius_arcsec=0.8,
                                                                                         outer_radius_arcsec=2.2,
                                                                                         outer_radius_2_arcsec=3.0,
                                                                                         centre=(0.0, 0.0))

        mask = msk.Mask.circular_anti_annular(shape=(9, 9), pixel_scale=1.2, inner_radius_arcsec=0.8,
                                              outer_radius_arcsec=2.2, outer_radius_2_arcsec=3.0, centre=(0.0, 0.0),
                                              invert=True)

        assert (mask == np.invert(mask_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)

    def test__mask_elliptical__compare_to_array_util(self):

        mask_util = util.mask_elliptical_from_shape_pixel_scale_and_radius(shape=(8, 5), pixel_scale=2.7,
                        major_axis_radius_arcsec=5.7, axis_ratio=0.4, phi=40.0, centre=(0.0, 0.0))

        mask = msk.Mask.elliptical(shape=(8, 5), pixel_scale=2.7,
                        major_axis_radius_arcsec=5.7, axis_ratio=0.4, phi=40.0, centre=(0.0, 0.0))

        assert (mask == mask_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)

    def test__mask_elliptical_inverted__compare_to_array_util(self):

        mask_util = util.mask_elliptical_from_shape_pixel_scale_and_radius(shape=(8, 5), pixel_scale=2.7,
                        major_axis_radius_arcsec=5.7, axis_ratio=0.4, phi=40.0, centre=(0.0, 0.0))

        mask = msk.Mask.elliptical(shape=(8, 5), pixel_scale=2.7,
                        major_axis_radius_arcsec=5.7, axis_ratio=0.4, phi=40.0, centre=(0.0, 0.0), invert=True)

        assert (mask == np.invert(mask_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)

    def test__mask_elliptical_annular__compare_to_array_util(self):

        mask_util = util.mask_elliptical_annular_from_shape_pixel_scale_and_radius(shape=(8, 5), pixel_scale=2.7,
                        inner_major_axis_radius_arcsec=2.1, inner_axis_ratio=0.6, inner_phi=20.0,
                        outer_major_axis_radius_arcsec=5.7, outer_axis_ratio=0.4, outer_phi=40.0, centre=(0.0, 0.0))

        mask = msk.Mask.elliptical_annular(shape=(8, 5), pixel_scale=2.7,
                        inner_major_axis_radius_arcsec=2.1, inner_axis_ratio=0.6, inner_phi=20.0,
                        outer_major_axis_radius_arcsec=5.7, outer_axis_ratio=0.4, outer_phi=40.0, centre=(0.0, 0.0))

        assert (mask == mask_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)

    def test__mask_elliptical_annular_inverted__compare_to_array_util(self):

        mask_util = util.mask_elliptical_annular_from_shape_pixel_scale_and_radius(shape=(8, 5), pixel_scale=2.7,
                        inner_major_axis_radius_arcsec=2.1, inner_axis_ratio=0.6, inner_phi=20.0,
                        outer_major_axis_radius_arcsec=5.7, outer_axis_ratio=0.4, outer_phi=40.0, centre=(0.0, 0.0))

        mask = msk.Mask.elliptical_annular(shape=(8, 5), pixel_scale=2.7,
                        inner_major_axis_radius_arcsec=2.1, inner_axis_ratio=0.6, inner_phi=20.0,
                        outer_major_axis_radius_arcsec=5.7, outer_axis_ratio=0.4, outer_phi=40.0, centre=(0.0, 0.0),
                                           invert=True)

        assert (mask == np.invert(mask_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)


class TestMaskMappings:

    def test__grid_to_pixel__compare_to_array_util(self):
        mask = np.array([[True, True, True],
                        [True, False, False],
                        [True, True, False]])

        mask = msk.Mask(mask, pixel_scale=7.0)

        grid_to_pixel_util = util.masked_grid_1d_index_to_2d_pixel_index_from_mask(mask)

        assert mask.masked_grid_index_to_pixel == pytest.approx(grid_to_pixel_util, 1e-4)

    def test__sub_grid_to_pixel__compare_to_array_util(self):

        mask = np.array([[True, True, True],
                        [True, False, False],
                        [True, True, False]])

        mask = msk.Mask(mask, pixel_scale=7.0)

        sub_grid_to_pixel_util = util.masked_sub_grid_1d_index_to_2d_sub_pixel_index_from_mask(mask=mask,
                                                                                               sub_grid_size=2)

        assert mask.masked_sub_grid_index_to_sub_pixel(sub_grid_size=2) == pytest.approx(sub_grid_to_pixel_util, 1e-4)

    def test__array_1d_from_array_2d__compare_to_array_util(self):

        array_2d = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9],
                             [10, 11, 12]])

        mask = np.array([[True, False, True],
                        [False, False, False],
                        [True, False, True],
                        [True, True, True]])

        array_1d_util = mapping_util.map_array_2d_to_masked_array_1d_from_array_2d_and_mask(mask, array_2d)

        mask = msk.Mask(mask, pixel_scale=3.0)

        array_1d = mask.array_1d_from_array_2d(
            array_2d=array_2d)

        assert (array_1d == array_1d_util).all()

    def test__scaled_array_2d_from_array_1d__compare_to_util(self):

        mask = np.array([[True, True, False, False],
                         [True, False, True, True],
                         [True, True, False, False]])

        array_1d = np.array([1.0, 6.0, 4.0, 5.0, 2.0])
        one_to_two = np.array([[0, 2], [0, 3], [1, 1], [2, 2], [2, 3]])

        array_2d_util = mapping_util.map_masked_array_1d_to_array_2d_from_array_1d_shape_and_one_to_two(
            array_1d=array_1d, shape=(3, 4), one_to_two=one_to_two)

        mask = msk.Mask(array=mask, pixel_scale=3.0)

        array_2d = mask.array_2d_from_array_1d(array_1d=array_1d)

        assert (array_2d == array_2d_util).all()

        scaled_array_2d = mask.scaled_array_2d_from_array_1d(array_1d=array_1d)

        assert (scaled_array_2d == array_2d_util).all()
        assert (scaled_array_2d.xticks == np.array([-6.0, -2.0, 2.0, 6.0])).all()
        assert (scaled_array_2d.yticks == np.array([-4.5, -1.5, 1.5, 4.5])).all()
        assert scaled_array_2d.shape_arcsec == (9.0, 12.0)
        assert scaled_array_2d.pixel_scale == 3.0
        assert scaled_array_2d.origin == (0.0, 0.0)

    def test__grid_2d_from_grid_1d__compare_to_util(self):

        mask = np.array([[True, True, False, False],
                         [True, False, True, True],
                         [True, True, False, False]])

        grid_1d = np.array([[1.0, 1.0], [6.0, 6.0], [4.0, 4.0], [5.0, 5.0], [2.0, 2.0]])
        one_to_two = np.array([[0, 2], [0, 3], [1, 1], [2, 2], [2, 3]])

        grid_2d_util = mapping_util.map_masked_1d_grid_to_2d_grid_from_grid_1d_shape_and_one_to_two(
            grid_1d=grid_1d, shape=(3, 4), one_to_two=one_to_two)

        mask = msk.Mask(array=mask, pixel_scale=2.0)
        grid_2d = mask.grid_2d_from_grid_1d(grid_1d=grid_1d)

        assert (grid_2d_util == grid_2d).all()

    def test___grid_1d_from_grid_2d__compare_to_util(self):

        grid_2d = np.array([[[1,1], [2,2], [3,3], [4,4]],
                             [[5,5], [6,6], [7,7], [8,8]],
                             [[9,9], [10,10], [11,11], [12,12]]])

        mask = np.array([[True, False, True, True],
                         [False, False, False, True],
                         [True, False, True, False]])

        grid_1d_util = mapping_util.map_grid_2d_to_masked_grid_1d_from_grid_2d_and_mask(
            grid_2d=grid_2d, mask=mask)

        mask = msk.Mask(array=mask, pixel_scale=2.0)
        grid_1d = mask.grid_1d_from_grid_2d(grid_2d=grid_2d)

        assert (grid_1d_util == grid_1d).all()

    def test__sub_array_2d_from_sub_array_1d__use_real_mask_and_grid(self):

        mask = np.array([[False, True],
                         [False, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_array_1d = np.array([1.0, 2.0, 3.0, 4.0,
                                 9.0, 10.0, 11.0, 12.0,
                                 13.0, 14.0, 15.0, 16.0])

        sub_array_2d = mask.sub_array_2d_from_sub_array_1d_and_sub_grid_size(
            sub_array_1d=sub_array_1d, sub_grid_size=2)

        assert (sub_array_2d == np.array([[1.0, 2.0, 0.0, 0.0],
                                          [3.0, 4.0, 0.0, 0.0],
                                          [9.0, 10.0, 13.0, 14.0],
                                          [11.0, 12.0, 15.0, 16.0]])).all()

    def test__sub_array_2d_from_sub_array_1d__use_2x3_mask(self):

        mask = np.array([[False, False, True],
                         [False, True, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_array_1d = np.array([1.0, 1.0, 1.0, 1.0,
                                 2.0, 2.0, 2.0, 2.0,
                                 3.0, 3.0, 3.0, 3.0,
                                 4.0, 4.0, 4.0, 4.0])

        sub_array_2d = mask.sub_array_2d_from_sub_array_1d_and_sub_grid_size(sub_array_1d=sub_array_1d, sub_grid_size=2)

        assert (sub_array_2d == np.array([[1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                                          [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                                          [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                                          [3.0, 3.0, 0.0, 0.0, 4.0, 4.0]])).all()

    def test__scaled_sub_array_2d_from_sub_array_1d(self):

        mask = np.array([[False, False, True],
                         [False, True, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_array_1d = np.array([1.0, 1.0, 1.0, 1.0,
                                 2.0, 2.0, 2.0, 2.0,
                                 3.0, 3.0, 3.0, 3.0,
                                 4.0, 4.0, 4.0, 4.0])

        scaled_sub_array_2d = mask.scaled_array_2d_with_sub_dimensions_from_sub_array_1d_and_sub_grid_size(
            sub_array_1d=sub_array_1d, sub_grid_size=2)

        assert (scaled_sub_array_2d == np.array([[1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                                                 [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                                                 [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                                                 [3.0, 3.0, 0.0, 0.0, 4.0, 4.0]])).all()

        assert scaled_sub_array_2d.pixel_scales == (1.5, 1.5)
        assert scaled_sub_array_2d.origin == (0.0, 0.0)

    def test__scaled_array_from_sub_array_1d_by_binning_up(self):

        mask = np.array([[False, False, True],
                         [False, True, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_array_1d = np.array([1.0, 10.0, 2.0, 1.0,
                                 2.0, 2.0, 2.0, 2.0,
                                 3.0, 3.0, 3.0, 3.0,
                                 4.0, 0.0, 0.0, 4.0])

        scaled_array_2d = mask.scaled_array_2d_binned_up_from_sub_array_1d_and_sub_grid_size(
            sub_array_1d=sub_array_1d, sub_grid_size=2)

        assert (scaled_array_2d == np.array([[3.5, 2.0, 0.0],
                                             [3.0, 0.0, 2.0]])).all()

        assert scaled_array_2d.pixel_scales == (3.0, 3.0)
        assert scaled_array_2d.origin == (0.0, 0.0)

    def test__sub_grid_2d_from_sub_grid_1d__use_real_mask_and_grid(self):

        mask = np.array([[False, True],
                         [False, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_grid_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],
                                [9.0, 9.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0],
                                [13.0, 13.0], [14.0, 14.0], [15.0, 15.0], [16.0, 16.0]])

        sub_grid_2d = mask.sub_grid_2d_from_sub_grid_1d_and_sub_grid_size(
            sub_grid_1d=sub_grid_1d, sub_grid_size=2)

        assert (sub_grid_2d == np.array([[[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                                         [[3.0, 3.0], [4.0,4.0], [0.0, 0.0], [0.0, 0.0]],
                                         [[9.0, 9.0],  [10.0, 10.0], [13.0, 13.0], [14.0, 14.0]],
                                         [[11.0, 11.0], [12.0, 12.0], [15.0, 15.0], [16.0, 16.0]]])).all()

    def test__sub_grid_2d_from_sub_grid_1d__use_2x3_mask(self):

        mask = np.array([[False, False, True],
                         [False, True, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_grid_1d = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
                                 [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0],
                                 [3.0, 3.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0],
                                 [4.0, 4.0], [4.0, 4.0], [4.0, 4.0], [4.0, 4.0]])

        sub_grid_2d = mask.sub_grid_2d_from_sub_grid_1d_and_sub_grid_size(
            sub_grid_1d=sub_grid_1d, sub_grid_size=2)

        assert (sub_grid_2d == np.array([[[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                                          [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                                          [[3.0, 3.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0], [4.0, 4.0], [4.0, 4.0]],
                                          [[3.0, 3.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0], [4.0, 4.0], [4.0, 4.0]]])).all()

    def test__sub_array_1d_from_sub_array_2d__numerical_values(self):

        sub_array_2d = np.array([[1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                                 [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                                 [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                                 [3.0, 3.0, 0.0, 0.0, 4.0, 4.0]])

        mask = np.array([[False, False, True],
                         [False, True, False]])

        mask = msk.Mask(array=mask, pixel_scale=2.0)

        sub_array_1d = mask.sub_array_1d_from_sub_array_2d_and_sub_grid_size(
            sub_array_2d=sub_array_2d, sub_grid_size=2)

        assert (sub_array_1d == np.array([1.0, 1.0, 1.0, 1.0,
                                          2.0, 2.0, 2.0, 2.0,
                                          3.0, 3.0, 3.0, 3.0,
                                          4.0, 4.0, 4.0, 4.0])).all()

    def test__sub_array_1d_from_sub_array_2d__compare_to_util(self):

        sub_array_2d = np.array([[1.0, 1.0, 2.0, 2.0, 3.0, 10.0],
                                 [1.0, 1.0, 2.0, 2.0, 3.0, 10.0],
                                 [3.0, 3.0, 8.0, 1.0, 4.0, 4.0],
                                 [3.0, 3.0, 7.0, 2.0, 4.0, 4.0]])

        mask = np.array([[False, False, False],
                         [True, True, False]])

        mask = msk.Mask(array=mask, pixel_scale=2.0)

        sub_array_1d = mask.sub_array_1d_from_sub_array_2d_and_sub_grid_size(
            sub_array_2d=sub_array_2d, sub_grid_size=2)

        sub_array_1d_util = mapping_util.map_sub_array_2d_to_masked_sub_array_1d_from_sub_array_2d_mask_and_sub_grid_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_grid_size=2)

        assert (sub_array_1d == sub_array_1d_util).all()

    def test__map_sub_array_to_1d_and_2d_and_back__returns_original_array(self):

        mask = np.array([[False, False, True],
                         [False, True, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_array_1d = np.array([1.0, 10.0, 2.0, 1.0,
                                 2.0, 2.0, 2.0, 2.0,
                                 3.0, 3.0, 3.0, 3.0,
                                 4.0, 0.0, 0.0, 4.0])

        sub_array_2d = mask.sub_array_2d_from_sub_array_1d_and_sub_grid_size(
            sub_array_1d=sub_array_1d, sub_grid_size=2)

        sub_array_1d_new = mask.sub_array_1d_from_sub_array_2d_and_sub_grid_size(
            sub_array_2d=sub_array_2d, sub_grid_size=2)

        assert (sub_array_1d == sub_array_1d_new).all()

    def test__sub_to_regular__compare_to_util(self):

        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, False]])

        sub_to_regular_util = mapping_util.sub_to_regular_from_mask(
            mask=mask, sub_grid_size=2)

        mask = msk.Mask(mask, pixel_scale=3.0)

        assert (mask.sub_to_regular_from_sub_grid_size(sub_grid_size=2) == sub_to_regular_util).all()


class TestMaskRegions:

    def test__blurring_mask_for_psf_shape__compare_to_array_util(self):
        
        mask = np.array([[True, True, True, True, True, True, True, True],
                        [True, False, True, True, True, False, True, True],
                        [True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True],
                        [True, False, True, True, True, False, True, True],
                        [True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True]])

        blurring_mask_util = util.blurring_mask_from_mask_and_psf_shape(mask=mask, psf_shape=(3, 3))

        mask = msk.Mask(mask, pixel_scale=1.0)
        blurring_mask = mask.blurring_mask_for_psf_shape(psf_shape=(3, 3))

        assert (blurring_mask == blurring_mask_util).all()

    def test__edge_image_pixels__compare_to_array_util(self):
        mask = np.array([[True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, False, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True]])

        edge_pixels_util = util.edge_pixels_from_mask(mask)

        mask = msk.Mask(mask, pixel_scale=3.0)

        assert mask.edge_pixels == pytest.approx(edge_pixels_util, 1e-4)

    def test__border_image_pixels__compare_to_array_util(self):

        mask = np.array([[False, False, False, False, False, False, False, True],
                         [False,  True,  True,  True,  True,  True, False, True],
                         [False,  True, False, False, False,  True, False, True],
                         [False,  True, False,  True, False,  True, False, True],
                         [False,  True, False, False, False,  True, False, True],
                         [False,  True,  True,  True,  True,  True, False, True],
                         [False, False, False, False, False, False, False, True]])

        border_pixels_util = util.border_pixels_from_mask(mask)

        mask = msk.Mask(mask, pixel_scale=3.0)

        assert mask.border_pixels == pytest.approx(border_pixels_util, 1e-4)


class TestMaskedGrid1d:

    def test__simple_grids(self):

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(3,3), pixel_scale=1.0)

        assert (mask.masked_grid_1d == np.array([[1., -1.], [1., 0.], [1., 1.],
                                                [0., -1.], [0., 0.], [0., 1.],
                                                [-1., -1.], [-1., 0.], [-1., 1.]])).all()

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(3,3), pixel_scale=1.0)
        mask[1,1] = True

        assert (mask.masked_grid_1d == np.array([[1., -1.], [1., 0.], [1., 1.],
                                                [0., -1.],            [0., 1.],
                                                [-1., -1.], [-1., 0.], [-1., 1.]])).all()

        mask = msk.Mask(array=np.array([[False, True],
                                        [True, False],
                                        [True, False]]), pixel_scale=1.0, origin=(3.0, -2.0))

        assert (mask.masked_grid_1d == np.array([[4., -2.5],
                                                             [3., -1.5],
                                                             [2., -1.5]])).all()

    def test__compare_to_grid_util(self):

        mask = msk.Mask.circular(shape=(4,7), radius_arcsec=4.0, pixel_scale=2.0, centre=(1.0, 5.0))

        masked_grid_1d_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask,
                                                                                                 pixel_scales=(2.0, 2.0))

        assert (mask.masked_grid_1d == masked_grid_1d_util).all()


class TestZoomCentreAndOffet:

    def test__odd_sized_false_mask__centre_is_0_0__pixels_from_centre_are_0_0(self):

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(3,3), pixel_scale=1.0)
        assert mask.zoom_centre == (1.0, 1.0)
        assert mask.zoom_offset_pixels == (0, 0)

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(5,5), pixel_scale=1.0)
        assert mask.zoom_centre == (2.0, 2.0)
        assert mask.zoom_offset_pixels == (0, 0)

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(3,5), pixel_scale=1.0)
        assert mask.zoom_centre == (1.0, 2.0)
        assert mask.zoom_offset_pixels == (0, 0)

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(5,3), pixel_scale=1.0)
        assert mask.zoom_centre == (2.0, 1.0)
        assert mask.zoom_offset_pixels == (0, 0)

    def test__even_sized_false_mask__centre_is_0_0__pixels_from_centre_are_0_0(self):

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(4,4), pixel_scale=1.0)
        assert mask.zoom_centre == (1.5, 1.5)
        assert mask.zoom_offset_pixels == (0, 0)

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(6,6), pixel_scale=1.0)
        assert mask.zoom_centre == (2.5, 2.5)
        assert mask.zoom_offset_pixels == (0, 0)

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(4,6), pixel_scale=1.0)
        assert mask.zoom_centre == (1.5, 2.5)
        assert mask.zoom_offset_pixels == (0, 0)

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(6,4), pixel_scale=1.0)
        assert mask.zoom_centre == (2.5, 1.5)
        assert mask.zoom_offset_pixels == (0, 0)

    def test__mask_is_single_false__extraction_centre_is_central_pixel(self):

        mask = msk.Mask(array=np.array([[False, True, True],
                                        [True, True, True],
                                        [True, True, True]]), pixel_scale=1.0)
        assert mask.zoom_centre == (0, 0)
        assert mask.zoom_offset_pixels == (-1, -1)

        mask = msk.Mask(array=np.array([[True, True, False],
                                        [True, True, True],
                                        [True, True, True]]), pixel_scale=1.0)
        assert mask.zoom_centre == (0, 2)
        assert mask.zoom_offset_pixels == (-1, 1)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, True, True],
                                        [False, True, True]]), pixel_scale=1.0)
        assert mask.zoom_centre == (2, 0)
        assert mask.zoom_offset_pixels == (1, -1)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, True, True],
                                        [True, True, False]]), pixel_scale=1.0)
        assert mask.zoom_centre == (2, 2)
        assert mask.zoom_offset_pixels == (1, 1)

        mask = msk.Mask(array=np.array([[True, False, True],
                                        [True, True, True],
                                        [True, True, True]]), pixel_scale=1.0)
        assert mask.zoom_centre == (0, 1)
        assert mask.zoom_offset_pixels == (-1, 0)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [False, True, True],
                                        [True, True, True]]), pixel_scale=1.0)
        assert mask.zoom_centre == (1, 0)
        assert mask.zoom_offset_pixels == (0, -1)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, True, False],
                                        [True, True, True]]), pixel_scale=1.0)
        assert mask.zoom_centre == (1, 2)
        assert mask.zoom_offset_pixels == (0, 1)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, True, True],
                                        [True, False, True]]), pixel_scale=1.0)
        assert mask.zoom_centre == (2, 1)
        assert mask.zoom_offset_pixels == (1, 0)

    def test__mask_is_x2_false__extraction_centre_is_central_pixel(self):

        mask = msk.Mask(array=np.array([[False, True, True],
                                        [True, True, True],
                                        [True, True, False]]), pixel_scale=1.0)
        assert mask.zoom_centre == (1, 1)
        assert mask.zoom_offset_pixels == (0, 0)

        mask = msk.Mask(array=np.array([[False, True, True],
                                        [True, True, True],
                                        [False, True, True]]), pixel_scale=1.0)
        assert mask.zoom_centre == (1, 0)
        assert mask.zoom_offset_pixels == (0, -1)

        mask = msk.Mask(array=np.array([[False, True, False],
                                        [True, True, True],
                                        [True, True, True]]), pixel_scale=1.0)
        assert mask.zoom_centre == (0, 1)
        assert mask.zoom_offset_pixels == (-1, 0)

        mask = msk.Mask(array=np.array([[False, False, True],
                                        [True, True, True],
                                        [True, True, True]]), pixel_scale=1.0)
        assert mask.zoom_centre == (0, 0.5)
        assert mask.zoom_offset_pixels == (-1, -0.5)

    def test__rectangular_mask(self):

        mask = msk.Mask(array=np.array([[False, True, True, True],
                                        [True, True, True, True],
                                        [True, True, True, True]]), pixel_scale=1.0)

        assert mask.zoom_centre == (0, 0)
        assert mask.zoom_offset_pixels == (-1.0, -1.5)

        mask = msk.Mask(array=np.array([[True, True, True, True],
                                        [True, True, True, True],
                                        [True, True, True, False]]), pixel_scale=1.0)

        assert mask.zoom_centre == (2, 3)
        assert mask.zoom_offset_pixels == (1.0, 1.5)

        mask = msk.Mask(array=np.array([[True, True, True, True, True],
                                        [True, True, True, True, True],
                                        [True, True, True, True, False]]), pixel_scale=1.0)

        assert mask.zoom_centre == (2, 4)
        assert mask.zoom_offset_pixels == (1, 2)

        mask = msk.Mask(array=np.array([[True, True, True, True, True, True, True],
                                        [True, True, True, True, True, True, True],
                                        [True, True, True, True, True, True, False]]), pixel_scale=1.0)

        assert mask.zoom_centre == (2, 6)
        assert mask.zoom_offset_pixels == (1, 3)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, True, True],
                                        [True, True, True],
                                        [True, True, True],
                                        [True, True, False]]), pixel_scale=1.0)

        assert mask.zoom_centre == (4, 2)
        assert mask.zoom_offset_pixels == (2, 1)

        mask = msk.Mask(array=np.array([[True, True, True],
                                        [True, True, True],
                                        [True, True, True],
                                        [True, True, True],
                                        [True, True, True],
                                        [True, True, True],
                                        [True, True, False]]), pixel_scale=1.0)

        assert mask.zoom_centre == (6, 2)
        assert mask.zoom_offset_pixels == (3, 1)


class TestMaskExtractor:

    def test__mask_extract_region__uses_the_limits_of_the_mask(self):

        mask = msk.Mask(array=np.array([[True,  True,  True, True],
                                        [True, False, False, True],
                                        [True, False, False, True],
                                        [True,  True,  True, True]]), pixel_scale=1.0)

        assert mask.zoom_region == [1, 3, 1, 3]

        mask = msk.Mask(array=np.array([[True,  True,  True, True],
                                        [True, False, False, True],
                                        [True, False, False, False],
                                        [True,  True,  True, True]]), pixel_scale=1.0)

        assert mask.zoom_region == [1, 3, 1, 4]

        mask = msk.Mask(array=np.array([[True,  True,  True, True],
                                        [True, False, False, True],
                                        [True, False, False, True],
                                        [True,  True, False, True]]), pixel_scale=1.0)

        assert mask.zoom_region == [1, 4, 1, 3]


        mask = msk.Mask(array=np.array([[True,  True,   True, True],
                                        [True, False,  False, True],
                                        [False, False, False, True],
                                        [True,  True,  True, True]]), pixel_scale=1.0)

        assert mask.zoom_region == [1, 3, 0, 3]

        mask = msk.Mask(array=np.array([[True, False,  True, True],
                                        [True, False, False, True],
                                        [True, False, False, True],
                                        [True,  True,  True, True]]), pixel_scale=1.0)

        assert mask.zoom_region == [0, 3, 1, 3]


class TestParse:

    def test__load_mask_from_fits__loads_mask(self):

        mask = msk.load_mask_from_fits(mask_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1)

        assert (mask == np.ones((3,3))).all()
        assert mask.pixel_scale == 0.1

    def test__output_mask_to_fits__outputs_mask(self):

        mask = msk.load_mask_from_fits(mask_path=test_data_dir + '3x3_ones.fits', pixel_scale=0.1)

        output_data_dir = "{}/../../test_files/array/output_test/".format(os.path.dirname(os.path.realpath(__file__)))
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        msk.output_mask_to_fits(mask=mask, mask_path=output_data_dir + 'mask.fits')

        mask = msk.load_mask_from_fits(mask_path=output_data_dir + 'mask.fits', pixel_scale=0.1)

        assert (mask == np.ones((3,3))).all()
        assert mask.pixel_scale == 0.1


class TestBinnedMaskFromMask:

    def test__compare_to_mask_util(self):

        mask = np.full(shape=(14, 19), fill_value=True)
        mask[1, 5] = False
        mask[6, 5] = False
        mask[4, 9] = False
        mask[11, 10] = False

        binned_up_mask_util = binning_util.binned_up_mask_2d_from_mask_2d_and_bin_up_factor(mask_2d=mask, bin_up_factor=2)

        mask = msk.Mask(array=mask, pixel_scale=1.0)
        mask = mask.binned_up_mask_from_mask(bin_up_factor=2)
        assert (mask == binned_up_mask_util).all()
        assert mask.pixel_scale == 2.0

        binned_up_mask_util = binning_util.binned_up_mask_2d_from_mask_2d_and_bin_up_factor(mask_2d=mask, bin_up_factor=3)

        mask = msk.Mask(array=mask, pixel_scale=2.0)
        mask = mask.binned_up_mask_from_mask(bin_up_factor=3)
        assert (mask == binned_up_mask_util).all()
        assert mask.pixel_scale == 6.0


class TestRescaledMaskFromMask(object):

    def test__mask_5x5_central_pixel__rescale_factor_is_1__returns_same_mask(self):

        mask = np.array([[True, True, True, True, True],
                         [True, True, True, True, True],
                         [True, True, False, True, True],
                         [True, True, True, True, True],
                         [True, True, True, True, True]])

        rescaled_mask = util.rescaled_mask_2d_from_mask_2d_and_rescale_factor(mask_2d=mask, rescale_factor=1.0)

        assert (rescaled_mask == np.array([[True, True, True, True, True],
                                           [True, True, True, True, True],
                                           [True, True, False, True, True],
                                           [True, True, True, True, True],
                                           [True, True, True, True, True]])).all()

    def test__mask_5x5_central_pixel__rescale_factor_is_2__returns_10x10_mask_4_central_values(self):

        mask = np.array([[True, True, True, True, True],
                         [True, True, True, True, True],
                         [True, True, False, True, True],
                         [True, True, True, True, True],
                         [True, True, True, True, True]])

        rescaled_mask = util.rescaled_mask_2d_from_mask_2d_and_rescale_factor(mask_2d=mask, rescale_factor=2.0)

        assert (rescaled_mask == np.array([[True, True, True, True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True,  True, True, True],
                                           [True, True, True, True, True, True, True, True, True, True],
                                           [True, True, True, False, False, False, False, True, True, True],
                                           [True, True, True, False, False, False, False, True, True, True],
                                           [True, True, True, False, False, False, False, True, True, True],
                                           [True, True, True, False, False, False, False, True, True, True],
                                           [True, True, True, True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True, True, True, True]])).all()

    def test__same_as_above__off_centre_pixels(self):

        mask = np.array([[True, True, True, True, True],
                         [True, False, True, True, True],
                         [True, True, True, True, True],
                         [True, True, True, False, True],
                         [True, True, True, True, True]])

        rescaled_mask = util.rescaled_mask_2d_from_mask_2d_and_rescale_factor(mask_2d=mask, rescale_factor=2.0)

        assert (rescaled_mask == np.array([[True, True, True, True, True, True, True, True, True, True],
                                           [True, False, False, False, False, True, True,  True, True, True],
                                           [True, False, False, False, False, True, True, True, True, True],
                                           [True, False, False, False, False, True, True, True, True, True],
                                           [True, False, False, False, False, True, True, True, True, True],
                                           [True, True, True, True, True, False, False, False, False, True],
                                           [True, True, True, True, True, False, False, False, False, True],
                                           [True, True, True, True, True, False, False, False, False, True],
                                           [True, True, True, True, True, False, False, False, False, True],
                                           [True, True, True, True, True, True, True, True, True, True]])).all()

    def test__mask_4x3_two_central_pixels__rescale_near_1__returns_slightly_different_masks(self):

        mask = np.array([[True, True, True, True, True],
                         [True, True, True, True, True],
                         [True, True, False, True, True],
                         [True, True, False, True, True],
                         [True, True, True, True, True],
                         [True, True, True, True, True]])

        rescaled_mask = util.rescaled_mask_2d_from_mask_2d_and_rescale_factor(mask_2d=mask, rescale_factor=1.2)

        assert (rescaled_mask == np.array([[True, True, True, True, True, True],
                                           [True, True, True, True, True, True],
                                           [True, True, False, False, True, True],
                                           [True, True, False, False, True, True],
                                           [True, True, False, False, True, True],
                                           [True, True, True, True, True, True],
                                           [True, True, True, True, True, True]])).all()

        rescaled_mask = util.rescaled_mask_2d_from_mask_2d_and_rescale_factor(mask_2d=mask, rescale_factor=0.8)

        assert (rescaled_mask == np.array([[True, True, True, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, False, False, True],
                                           [True, True, True, True]])).all()

    def test__mask_3x4_two_central_pixels__rescale_near_1__returns_slightly_different_masks(self):

        mask = np.array([[True, True, True, True, True, True],
                         [True, True, True, True, True, True],
                         [True, True, False, False, True, True],
                         [True, True, True, True, True, True],
                         [True, True, True, True, True, True]])

        rescaled_mask = util.rescaled_mask_2d_from_mask_2d_and_rescale_factor(mask_2d=mask, rescale_factor=1.2)

        assert (rescaled_mask == np.array([[True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True],
                                           [True, True, False, False, False, True, True],
                                           [True, True, False, False, False, True, True],
                                           [True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True]])).all()

        rescaled_mask = util.rescaled_mask_2d_from_mask_2d_and_rescale_factor(mask_2d=mask, rescale_factor=0.8)

        assert (rescaled_mask == np.array([[True, True, True, True, True],
                                           [True, False, False, False, True],
                                           [True, False, False, False, True],
                                           [True, True, True, True, True]])).all()


class TestEdgeBuffedMaskFromMask(object):

    def test__5x5_mask_false_centre_pixel__3x3_falses_in_centre_of_edge_buffed_mask(self):

        mask = np.array([[True, True, True, True, True],
                         [True, True, True, True, True],
                         [True, True, False, True, True],
                         [True, True, True, True, True],
                         [True, True, True, True, True]])

        edge_buffed_mask = util.edge_buffed_mask_from_mask(mask=mask)

        assert (edge_buffed_mask == np.array([[True, True, True, True, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True, False, False, False, True],
                                              [True, True, True, True, True]])).all()

    def test__5x5_mask_false_offset_pixel__3x3_falses_in_centre_of_edge_buffed_mask(self):

        mask = np.array([[True, True, True, True, True],
                         [True, True, True, True, True],
                         [True, True, True, True, True],
                         [True, True, True, False, True],
                         [True, True, True, True, True]])

        edge_buffed_mask = util.edge_buffed_mask_from_mask(mask=mask)

        assert (edge_buffed_mask == np.array([[True, True, True, True, True],
                                              [True, True, True, True, True],
                                              [True, True, False, False, False],
                                              [True, True, False, False, False],
                                              [True, True, False, False, False]])).all()

    def test__mask_4x3__buffed_mask_same_shape(self):

        mask = np.array([[True, True, True, True, True],
                         [True, True, True, True, True],
                         [True, True, False, True, True],
                         [True, True, False, True, True],
                         [True, True, True, True, True],
                         [True, True, True, True, True]])

        edge_buffed_mask = util.edge_buffed_mask_from_mask(mask=mask)

        assert (edge_buffed_mask == np.array([[True, True, True, True, True],
                                           [True,  False, False, False, True],
                                           [True,  False, False, False, True],
                                           [True,  False, False, False, True],
                                           [True,  False, False, False, True],
                                           [True, True, True, True, True]])).all()

    def test__mask_3x4_two_central_pixels__rescale_near_1__returns_slightly_different_masks(self):

        mask = np.array([[True, True, True, True, True, True],
                         [True, True, True, True, True, True],
                         [True, True, False, False, True, True],
                         [True, True, True, True, True, True],
                         [True, True, True, True, True, True]])

        edge_buffed_mask = util.edge_buffed_mask_from_mask(mask=mask)

        assert (edge_buffed_mask == np.array([[True, True, True, True, True, True],
                                              [True, False, False, False, False, True],
                                              [True, False, False, False, False, True],
                                              [True, False, False, False, False, True],
                                              [True, True, True, True, True, True]])).all()