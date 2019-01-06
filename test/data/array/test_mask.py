import shutil
import os
import numpy as np
import pytest

from autolens.data.array.util import mask_util as util
from autolens.data.array.util import mapping_util
from autolens.data.array import mask as msk

test_data_dir = "{}/../../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))

class TestMask(object):

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
        assert mask.shape_arc_seconds == (3.0, 4.0)

    def test__mask_padded__5x5__input__all_are_false(self):

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=(5, 5), pixel_scale=1.5)

        assert mask.shape == (5, 5)
        assert (mask == np.array([[False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False]])).all()

        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)

    def test__mask_masked__5x5__input__all_are_false(self):
        mask = msk.Mask.masked_for_shape_and_pixel_scale(shape=(5, 5), pixel_scale=1)

        assert mask.shape == (5, 5)
        assert (mask == np.array([[True, True, True, True, True],
                                 [True, True, True, True, True],
                                 [True, True, True, True, True],
                                 [True, True, True, True, True],
                                 [True, True, True, True, True]])).all()

        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (0.0, 0.0)

    def test__mask_circular__compare_to_array_util(self):
        mask_util = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(5, 4), pixel_scale=2.7,
                                                                             radius_arcsec=3.5, centre=(1.0, 1.0))

        mask = msk.Mask.circular(shape=(5, 4), pixel_scale=2.7, radius_arcsec=3.5, centre=(1.0, 1.0))

        assert (mask == mask_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (1.0, 1.0)

    def test__mask_annulus__compare_to_array_util(self):
        
        mask_util = util.mask_circular_annular_from_shape_pixel_scale_and_radii(shape=(5, 4), pixel_scale=2.7,
                                                                                    inner_radius_arcsec=0.8,
                                                                                    outer_radius_arcsec=3.5,
                                                                                    centre=(1.0, 1.0))

        mask = msk.Mask.circular_annular(shape=(5, 4), pixel_scale=2.7, inner_radius_arcsec=0.8, outer_radius_arcsec=3.5,
                                         centre=(1.0, 1.0))

        assert (mask == mask_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (1.0, 1.0)

    def test__mask_anti_annulus__compare_to_array_util(self):
        mask_util = util.mask_circular_anti_annular_from_shape_pixel_scale_and_radii(shape=(9, 9), pixel_scale=1.2,
                                                                                         inner_radius_arcsec=0.8,
                                                                                         outer_radius_arcsec=2.2,
                                                                                         outer_radius_2_arcsec=3.0,
                                                                                         centre=(1.0, 1.0))

        mask = msk.Mask.circular_anti_annular(shape=(9, 9), pixel_scale=1.2, inner_radius_arcsec=0.8,
                                              outer_radius_arcsec=2.2, outer_radius_2_arcsec=3.0, centre=(1.0, 1.0))

        assert (mask == mask_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (1.0, 1.0)

    def test__mask_elliptical__compare_to_array_util(self):

        mask_util = util.mask_elliptical_from_shape_pixel_scale_and_radius(shape=(8, 5), pixel_scale=2.7,
                        major_axis_radius_arcsec=5.7, axis_ratio=0.4, phi=40.0, centre=(1.0, 1.0))

        mask = msk.Mask.elliptical(shape=(8, 5), pixel_scale=2.7,
                        major_axis_radius_arcsec=5.7, axis_ratio=0.4, phi=40.0, centre=(1.0, 1.0))

        assert (mask == mask_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (1.0, 1.0)

    def test__mask_elliptical_annular__compare_to_array_util(self):

        mask_util = util.mask_elliptical_annular_from_shape_pixel_scale_and_radius(shape=(8, 5), pixel_scale=2.7,
                        inner_major_axis_radius_arcsec=2.1, inner_axis_ratio=0.6, inner_phi=20.0,
                        outer_major_axis_radius_arcsec=5.7, outer_axis_ratio=0.4, outer_phi=40.0, centre=(1.0, 1.0))

        mask = msk.Mask.elliptical_annular(shape=(8, 5), pixel_scale=2.7,
                        inner_major_axis_radius_arcsec=2.1, inner_axis_ratio=0.6, inner_phi=20.0,
                        outer_major_axis_radius_arcsec=5.7, outer_axis_ratio=0.4, outer_phi=40.0, centre=(1.0, 1.0))

        assert (mask == mask_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.centre == (1.0, 1.0)

    def test__grid_to_pixel__compare_to_array_utill(self):
        mask = np.array([[True, True, True],
                        [True, False, False],
                        [True, True, False]])

        mask = msk.Mask(mask, pixel_scale=7.0)

        grid_to_pixel_util = util.masked_grid_1d_index_to_2d_pixel_index_from_mask(mask)

        assert mask.masked_grid_index_to_pixel == pytest.approx(grid_to_pixel_util, 1e-4)

    def test__map_2d_array_to_masked_1d_array__compare_to_array_util(self):
        array_2d = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9],
                             [10, 11, 12]])

        mask = np.array([[True, False, True],
                        [False, False, False],
                        [True, False, True],
                        [True, True, True]])

        array_1d_util = mapping_util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(mask, array_2d)

        mask = msk.Mask(mask, pixel_scale=3.0)

        array_1d = mask.map_2d_array_to_masked_1d_array(array_2d)

        assert (array_1d == array_1d_util).all()

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

        blurring_mask_util = util.mask_blurring_from_mask_and_psf_shape(mask=mask, psf_shape=(3, 3))

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