import numpy as np
import pytest

from autolens.data import ccd
from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.data.array.util import grid_util
from autolens.data.array.util import mapping_util, mask_util
from autolens.model.profiles import mass_profiles as mp


@pytest.fixture(name="mask")
def make_mask():
    return msk.Mask(np.array([[True, False, True],
                              [False, False, False],
                              [True, False, True]]), pixel_scale=1.0)


@pytest.fixture(name="centre_mask")
def make_centre_mask():
    return msk.Mask(np.array([[True, True, True],
                              [True, False, True],
                              [True, True, True]]), pixel_scale=1.0)


@pytest.fixture(name="sub_grid")
def make_sub_grid(mask):
    return grids.SubGrid.from_mask_and_sub_grid_size(mask, sub_grid_size=1)


@pytest.fixture(name="grid_stack")
def make_grid_stack(centre_mask):
    return grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(centre_mask, 2, (3, 3))


class TestRegularGrid:

    def test__regular_grid_from_mask__compare_to_array_util(self):

        mask = np.array([[True, True, False, False],
                         [True, False, True, True],
                         [True, True, False, False]])
        mask = msk.Mask(array=mask, pixel_scale=2.0)

        regular_grid_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask,
                                                                                               pixel_scales=(2.0, 2.0))

        regular_grid = grids.RegularGrid.from_mask(mask=mask)

        assert type(regular_grid) == grids.RegularGrid
        assert regular_grid == pytest.approx(regular_grid_util, 1e-4)
        assert regular_grid.pixel_scale == 2.0
        assert (regular_grid.mask.masked_grid_index_to_pixel == mask.masked_grid_index_to_pixel).all()
        assert regular_grid.interpolator == None

    def test__regular_grid_unlensed_grid_properties_compare_to_array_util(self):
        mask = np.array([[True, True, False, False],
                         [True, False, True, True],
                         [True, True, False, False]])
        mask = msk.Mask(array=mask, pixel_scale=2.0)

        regular_grid = grids.RegularGrid(arr=np.array([[1.0, 1.0], [1.0, 1.0]]), mask=mask)

        regular_grid_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask,
                                                                                               pixel_scales=(2.0, 2.0))

        assert type(regular_grid) == grids.RegularGrid
        assert regular_grid.unlensed_grid == pytest.approx(regular_grid_util, 1e-4)

        regular_grid_util = grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=(3, 4),
                                                                                         pixel_scales=(2.0, 2.0))

        assert type(regular_grid) == grids.RegularGrid
        assert regular_grid.unlensed_unmasked_grid == pytest.approx(regular_grid_util, 1e-4)

    def test__regular_grid_from_shape_and_pixel_scale__compare_to_array_util(self):
        mask = np.array([[False, False, False, False],
                         [False, False, False, False],
                         [False, False, False, False]])
        mask = msk.Mask(array=mask, pixel_scale=2.0)

        regular_grid_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask,
                                                                                               pixel_scales=(2.0, 2.0))

        regular_grid = grids.RegularGrid.from_shape_and_pixel_scale(shape=(3, 4), pixel_scale=2.0)

        assert type(regular_grid) == grids.RegularGrid
        assert regular_grid == pytest.approx(regular_grid_util, 1e-4)
        assert regular_grid.pixel_scale == 2.0
        assert (regular_grid.mask.masked_grid_index_to_pixel == mask.masked_grid_index_to_pixel).all()

    def test__blurring_grid_from_mask__compare_to_array_util(self):
        mask = np.array([[True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True]])
        mask = msk.Mask(array=mask, pixel_scale=2.0)

        blurring_mask_util = mask_util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 5))
        blurring_grid_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(blurring_mask_util,
                                                                                                pixel_scales=(2.0, 2.0))

        mask = msk.Mask(mask, pixel_scale=2.0)
        blurring_grid = grids.RegularGrid.blurring_grid_from_mask_and_psf_shape(mask=mask, psf_shape=(3, 5))

        blurring_mask = mask.blurring_mask_for_psf_shape(psf_shape=(3, 5))

        assert type(blurring_grid) == grids.RegularGrid
        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scale == 2.0
        assert (blurring_grid.mask.masked_grid_index_to_pixel == blurring_mask.masked_grid_index_to_pixel).all()

    def test__map_to_2d__compare_to_util(self):
        mask = np.array([[True, True, False, False],
                         [True, False, True, True],
                         [True, True, False, False]])
        array_1d = np.array([1.0, 6.0, 4.0, 5.0, 2.0])
        one_to_two = np.array([[0, 2], [0, 3], [1, 1], [2, 2], [2, 3]])
        array_2d_util = mapping_util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(
            array_1d=array_1d, shape=(3, 4), one_to_two=one_to_two)

        mask = msk.Mask(array=mask, pixel_scale=2.0)
        regular_grid = grids.RegularGrid.from_mask(mask)
        array_2d_grid = regular_grid.scaled_array_2d_from_array_1d(array_1d)

        assert (array_2d_util == array_2d_grid).all()
        assert array_2d_grid.pixel_scale == 2.0
        assert array_2d_grid.origin == (0.0, 0.0)

    def test__scaled_array_from_array_1d__compare_to_util(self):
        mask = np.array([[True, True, False, False],
                         [True, False, True, True],
                         [True, True, False, False]])
        array_1d = np.array([1.0, 6.0, 4.0, 5.0, 2.0])
        one_to_two = np.array([[0, 2], [0, 3], [1, 1], [2, 2], [2, 3]])
        array_2d_util = mapping_util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(
            array_1d=array_1d, shape=(3, 4), one_to_two=one_to_two)

        mask = msk.Mask(array=mask, pixel_scale=3.0)
        regular_grid = grids.RegularGrid.from_mask(mask)
        scaled_array_2d = regular_grid.scaled_array_2d_from_array_1d(array_1d)

        assert (scaled_array_2d == array_2d_util).all()
        assert (scaled_array_2d.xticks == np.array([-6.0, -2.0, 2.0, 6.0])).all()
        assert (scaled_array_2d.yticks == np.array([-4.5, -1.5, 1.5, 4.5])).all()
        assert scaled_array_2d.shape_arcsec == (9.0, 12.0)
        assert scaled_array_2d.pixel_scale == 3.0
        assert scaled_array_2d.origin == (0.0, 0.0)

    def test__new_grid_with_interpolator__returns_grid_with_interpolator(self):

        mask = np.array([[True, True, False, False],
                         [True, False, True, True],
                         [True, True, False, False]])
        mask = msk.Mask(array=mask, pixel_scale=2.0)

        regular_grid = grids.RegularGrid.from_mask(mask=mask)

        regular_grid_with_interp = regular_grid.new_grid_with_interpolator(interp_pixel_scale=1.0)

        assert (regular_grid[:,:] == regular_grid_with_interp[:,:]).all()
        assert regular_grid.mask == regular_grid_with_interp.mask

        interpolator_manual = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(mask=mask, grid=regular_grid,
                                                                                  interp_pixel_scale=1.0)
        assert (regular_grid.interpolator.vtx == interpolator_manual.vtx).all()
        assert (regular_grid.interpolator.wts == interpolator_manual.wts).all()

    def test__yticks(self):
        sca = grids.RegularGrid(arr=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=None)
        assert sca.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        sca = grids.RegularGrid(arr=np.array([[3.0, 1.0], [-3.0, -1.0]]), mask=None)
        assert sca.yticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        sca = grids.RegularGrid(arr=np.array([[5.0, 3.5], [2.0, -1.0]]), mask=None)
        assert sca.yticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__xticks(self):
        sca = grids.RegularGrid(arr=np.array([[1.0, 1.5], [-1.0, -1.5]]), mask=None)
        assert sca.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        sca = grids.RegularGrid(arr=np.array([[1.0, 3.0], [-1.0, -3.0]]), mask=None)
        assert sca.xticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        sca = grids.RegularGrid(arr=np.array([[3.5, 2.0], [-1.0, 5.0]]), mask=None)
        assert sca.xticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__masked_shape_arcsec(self):
        sca = grids.RegularGrid(arr=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=None)
        assert sca.masked_shape_arcsec == (3.0, 2.0)

        sca = grids.RegularGrid(arr=np.array([[1.5, 1.0], [-1.5, -1.0], [0.1, 0.1]]), mask=None)
        assert sca.masked_shape_arcsec == (3.0, 2.0)

        sca = grids.RegularGrid(arr=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0]]), mask=None)
        assert sca.masked_shape_arcsec == (4.5, 4.0)

        sca = grids.RegularGrid(arr=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0], [7.0, -5.0]]), mask=None)
        assert sca.masked_shape_arcsec == (8.5, 8.0)


class TestSubGrid(object):

    def test_sub_grid(self, sub_grid):
        assert type(sub_grid) == grids.SubGrid
        assert sub_grid.shape == (5, 2)
        assert sub_grid.pixel_scale == 1.0
        assert (sub_grid == np.array([[1, 0], [0, -1], [0, 0], [0, 1], [-1, 0]])).all()

    def test__regular_grid_unlensed_grid_properties_compare_to_array_util(self, mask, sub_grid):
        assert type(sub_grid) == grids.SubGrid
        assert sub_grid.unlensed_grid == \
               pytest.approx(np.array([[1, 0], [0, -1], [0, 0], [0, 1], [-1, 0]]), 1e-4)

        sub_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((3, 3), False), pixel_scales=(1.0, 1.0), sub_grid_size=1)

        assert type(sub_grid) == grids.SubGrid
        assert sub_grid.unlensed_unmasked_grid == pytest.approx(sub_grid_util, 1e-4)

        sub_grid = grids.SubGrid.from_mask_and_sub_grid_size(mask, sub_grid_size=2)

        sub_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((3, 3), False), pixel_scales=(1.0, 1.0), sub_grid_size=2)

        assert type(sub_grid) == grids.SubGrid
        assert sub_grid.unlensed_unmasked_grid == pytest.approx(sub_grid_util, 1e-4)

    def test_sub_to_pixel(self, sub_grid):
        assert (sub_grid.sub_to_regular == np.array(range(5))).all()

    def test__from_mask(self):
        mask = np.array([[True, True, True],
                         [True, False, False],
                         [True, True, False]])

        sub_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask,
                                                                                              pixel_scales=(
                                                                                                  3.0, 3.0),
                                                                                              sub_grid_size=2)

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_grid = grids.SubGrid.from_mask_and_sub_grid_size(mask, sub_grid_size=2)

        assert sub_grid == pytest.approx(sub_grid_util, 1e-4)

    def test__from_shape_and_pixel_scale(self):
        mask = np.array([[False, False, False],
                         [False, False, False],
                         [False, False, False]])

        sub_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask,
                                                                                              pixel_scales=(
                                                                                                  3.0, 3.0),
                                                                                              sub_grid_size=2)

        sub_grid = grids.SubGrid.from_shape_pixel_scale_and_sub_grid_size(shape=(3, 3), pixel_scale=3.0,
                                                                          sub_grid_size=2)

        assert sub_grid == pytest.approx(sub_grid_util, 1e-4)

    def test__sub_array_2d_from_sub_array_1d__use_real_mask_and_grid(self):

        mask = np.array([[False, True],
                         [False, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_array_1d = np.array([1.0, 2.0, 3.0, 4.0,
                                 9.0, 10.0, 11.0, 12.0,
                                 13.0, 14.0, 15.0, 16.0])

        sub_grid = grids.SubGrid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        sub_array_2d = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=sub_array_1d)

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

        sub_grid = grids.SubGrid.from_mask_and_sub_grid_size(mask, sub_grid_size=2)

        sub_array_2d = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=sub_array_1d)

        assert (sub_array_2d == np.array([[1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                                          [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                                          [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                                          [3.0, 3.0, 0.0, 0.0, 4.0, 4.0]])).all()


    def test__scaled_sub_array_from_sub_array_1d(self):

        mask = np.array([[False, False, True],
                         [False, True, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_array_1d = np.array([1.0, 1.0, 1.0, 1.0,
                                 2.0, 2.0, 2.0, 2.0,
                                 3.0, 3.0, 3.0, 3.0,
                                 4.0, 4.0, 4.0, 4.0])

        sub_grid = grids.SubGrid.from_mask_and_sub_grid_size(mask, sub_grid_size=2)

        scaled_sub_array_2d = sub_grid.scaled_array_2d_with_sub_dimensions_from_sub_array_1d(sub_array_1d=sub_array_1d)

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

        sub_grid = grids.SubGrid.from_mask_and_sub_grid_size(mask, sub_grid_size=2)

        scaled_array_2d = sub_grid.scaled_array_2d_with_regular_dimensions_from_binned_up_sub_array_1d(sub_array_1d=sub_array_1d)

        assert (scaled_array_2d == np.array([[3.5, 2.0, 0.0],
                                             [3.0, 0.0, 2.0]])).all()

        assert scaled_array_2d.pixel_scales == (3.0, 3.0)
        assert scaled_array_2d.origin == (0.0, 0.0)

    def test__map_to_2d__compare_to_util(self):
        mask = np.array([[True, True, False, False],
                         [True, False, True, True],
                         [True, True, False, False]])
        array_1d = np.array([1.0, 6.0, 4.0, 5.0, 2.0])
        one_to_two = np.array([[0, 2], [0, 3], [1, 1], [2, 2], [2, 3]])
        array_2d_util = mapping_util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(
            array_1d=array_1d,
            shape=(3, 4), one_to_two=one_to_two)

        mask = msk.Mask(array=mask, pixel_scale=2.0)
        regular_grid = grids.SubGrid.from_mask_and_sub_grid_size(mask, sub_grid_size=2)
        array_2d_grid = regular_grid.scaled_array_2d_from_array_1d(array_1d)

        assert (array_2d_util == array_2d_grid).all()
        assert array_2d_grid.pixel_scale == 2.0
        assert array_2d_grid.origin == (0.0, 0.0)

    def test_sub_data_to_image(self, sub_grid):
        assert (sub_grid.regular_data_1d_from_sub_data_1d(np.array(range(5))) == np.array(range(5))).all()

    def test_sub_to_image__compare_to_array_util(self):
        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, False]])

        sub_to_image_util = mapping_util.sub_to_regular_from_mask(mask, sub_grid_size=2)

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_grid = grids.SubGrid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)
        assert sub_grid.sub_grid_size == 2
        assert sub_grid.sub_grid_fraction == (1.0 / 4.0)
        assert (sub_grid.sub_to_regular == sub_to_image_util).all()


class TestPixGrid:

    def test_pix_regular_grid__attributes(self):
        pix_regular_grid = grids.PixGrid(arr=np.array([[1.0, 1.0], [2.0, 2.0]]),
                                         regular_to_nearest_pix=np.array([0, 1]))

        assert type(pix_regular_grid) == grids.PixGrid
        assert (pix_regular_grid == np.array([[1.0, 1.0], [2.0, 2.0]])).all()
        assert (pix_regular_grid.regular_to_nearest_pix == np.array([0, 1])).all()


class TestSparseToRegularGrid:

    def test__properties_consistent_with_mapping_util(self):
        ma = msk.Mask(array=np.array([[True, False, True],
                                      [False, False, False],
                                      [True, False, True]]), pixel_scale=0.5)

        regular_grid = grids.RegularGrid.from_mask(mask=ma)

        pix_grid = grids.SparseToRegularGrid(unmasked_sparse_grid_shape=(10, 10), pixel_scales=(0.16, 0.16),
                                             regular_grid=regular_grid)

        full_pix_grid_pixel_centres = regular_grid.mask.grid_arcsec_to_grid_pixel_centres(pix_grid.unmasked_sparse_grid)
        total_pix_pixels = mask_util.total_sparse_pixels_from_mask(mask=ma,
                                                                   unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres)

        image_to_full_pix_util = pix_grid.grid_arcsec_to_grid_pixel_indexes(grid_arcsec=regular_grid)

        pix_to_full_pix_util = mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_pix_pixels, mask=ma,
            unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres).astype('int')

        full_pix_to_pix_util = mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(mask=ma,
                                                                                                  unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres,
                                                                                                  total_sparse_pixels=pix_grid.total_sparse_pixels).astype(
            'int')

        image_to_pix_util = mapping_util.regular_to_sparse_from_sparse_mappings(
            regular_to_unmasked_sparse=image_to_full_pix_util,
            unmasked_sparse_to_sparse=full_pix_to_pix_util)

        pix_grid_util = mapping_util.sparse_grid_from_unmasked_sparse_grid(unmasked_sparse_grid=pix_grid.grid_1d,
                                                                           sparse_to_unmasked_sparse=pix_to_full_pix_util)

        assert pix_grid.total_sparse_pixels == total_pix_pixels
        assert (pix_grid.sparse_to_unmasked_sparse == pix_to_full_pix_util).all()
        assert (pix_grid.unmasked_sparse_to_sparse == full_pix_to_pix_util).all()
        assert (pix_grid.regular_to_unmasked_sparse == image_to_full_pix_util).all()
        assert (pix_grid.regular_to_sparse == image_to_pix_util).all()
        assert (pix_grid.sparse_grid == pix_grid_util).all()

    def test__pixelization_grid_overlaps_mask_perfectly__masked_pixels_in_masked_pixelization_grid(self):
        ma = msk.Mask(array=np.array([[True, False, True],
                                      [False, False, False],
                                      [True, False, True]]), pixel_scale=1.0)

        regular_grid = grids.RegularGrid.from_mask(mask=ma)

        pix_grid = grids.SparseToRegularGrid(unmasked_sparse_grid_shape=(3, 3), pixel_scales=(1.0, 1.0),
                                             regular_grid=regular_grid)

        assert pix_grid.total_sparse_pixels == 5
        assert (pix_grid.sparse_to_unmasked_sparse == np.array([1, 3, 4, 5, 7])).all()
        assert (pix_grid.unmasked_sparse_to_sparse == np.array([0, 0, 1, 1, 2, 3, 4, 4, 4])).all()
        assert (pix_grid.regular_to_unmasked_sparse == np.array([1, 3, 4, 5, 7])).all()
        assert (pix_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4])).all()
        assert (pix_grid.sparse_grid == np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                  [-1.0, 0.0]])).all()

    def test__same_as_above_but_4x3_grid_and_mask(self):
        ma = msk.Mask(array=np.array([[True, False, True],
                                      [False, False, False],
                                      [False, False, False],
                                      [True, False, True]]), pixel_scale=1.0)

        regular_grid = grids.RegularGrid.from_mask(mask=ma)

        pix_grid = grids.SparseToRegularGrid(unmasked_sparse_grid_shape=(4, 3), pixel_scales=(1.0, 1.0),
                                             regular_grid=regular_grid)

        assert pix_grid.total_sparse_pixels == 8
        assert (pix_grid.sparse_to_unmasked_sparse == np.array([1, 3, 4, 5, 6, 7, 8, 10])).all()
        assert (pix_grid.unmasked_sparse_to_sparse == np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 7])).all()
        assert (pix_grid.regular_to_unmasked_sparse == np.array([1, 3, 4, 5, 6, 7, 8, 10])).all()
        assert (pix_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7])).all()
        assert (pix_grid.sparse_grid == np.array([[1.5, 0.0],
                                                  [0.5, -1.0], [0.5, 0.0], [0.5, 1.0],
                                                  [-0.5, -1.0], [-0.5, 0.0], [-0.5, 1.0],
                                                  [-1.5, 0.0]])).all()

    def test__same_as_above_but_3x4_grid_and_mask(self):
        ma = msk.Mask(array=np.array([[True, False, True, True],
                                      [False, False, False, False],
                                      [True, False, True, True]]), pixel_scale=1.0)

        regular_grid = grids.RegularGrid.from_mask(mask=ma)

        pix_grid = grids.SparseToRegularGrid(unmasked_sparse_grid_shape=(3, 4), pixel_scales=(1.0, 1.0),
                                             regular_grid=regular_grid)

        assert pix_grid.total_sparse_pixels == 6
        assert (pix_grid.sparse_to_unmasked_sparse == np.array([1, 4, 5, 6, 7, 9])).all()
        assert (pix_grid.unmasked_sparse_to_sparse == np.array([0, 0, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5])).all()
        assert (pix_grid.regular_to_unmasked_sparse == np.array([1, 4, 5, 6, 7, 9])).all()
        assert (pix_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4, 5])).all()
        assert (pix_grid.sparse_grid == np.array([[1.0, -0.5],
                                                  [0.0, -1.5], [0.0, -0.5], [0.0, 0.5], [0.0, 1.5],
                                                  [-1.0, -0.5]])).all()

    def test__mask_with_offset_centre__changing_origin_of_sparse_to_regular_grid_ensures_same_pairings(self):
        ma = msk.Mask(array=np.array([[True, True, True, False, True],
                                      [True, True, False, False, False],
                                      [True, True, True, False, True],
                                      [True, True, True, True, True],
                                      [True, True, True, True, True]]), pixel_scale=1.0, centre=(1.0, 1.0))

        regular_grid = grids.RegularGrid.from_mask(mask=ma)

        # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
        # the central (3x3) pixels only.

        pix_grid = grids.SparseToRegularGrid(unmasked_sparse_grid_shape=(3, 3), pixel_scales=(1.0, 1.0),
                                             regular_grid=regular_grid)

        assert pix_grid.total_sparse_pixels == 3
        assert (pix_grid.sparse_to_unmasked_sparse == np.array([1, 2, 5])).all()
        assert (pix_grid.unmasked_sparse_to_sparse == np.array([0, 0, 1, 2, 2, 2, 2, 2, 2])).all()
        assert (pix_grid.regular_to_unmasked_sparse == np.array([2, 1, 2, 3, 5])).all()
        assert (pix_grid.regular_to_sparse == np.array([1, 0, 1, 2, 2])).all()
        assert (pix_grid.sparse_grid == np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])).all()

        pix_grid = grids.SparseToRegularGrid(unmasked_sparse_grid_shape=(3, 3), pixel_scales=(1.0, 1.0),
                                             regular_grid=regular_grid, origin=(1.0, 1.0))

        assert pix_grid.total_sparse_pixels == 5
        assert (pix_grid.sparse_to_unmasked_sparse == np.array([1, 3, 4, 5, 7])).all()
        assert (pix_grid.unmasked_sparse_to_sparse == np.array([0, 0, 1, 1, 2, 3, 4, 4, 4])).all()
        assert (pix_grid.regular_to_unmasked_sparse == np.array([1, 3, 4, 5, 7])).all()
        assert (pix_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4])).all()
        assert (pix_grid.sparse_grid == np.array([[2.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
                                                  [0.0, 1.0]])).all()

    def test__same_as_above_but_different_offset(self):
        ma = msk.Mask(array=np.array([[True, True, True, True, True],
                                      [True, True, True, False, True],
                                      [True, True, False, False, False],
                                      [True, True, True, False, True],
                                      [True, True, True, True, True]]), pixel_scale=2.0)

        regular_grid = grids.RegularGrid.from_mask(mask=ma)

        # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
        # the central (3x3) pixels only.

        pix_grid = grids.SparseToRegularGrid(unmasked_sparse_grid_shape=(3, 3), pixel_scales=(2.0, 2.0),
                                             regular_grid=regular_grid, origin=(0.0, 2.0))

        assert pix_grid.total_sparse_pixels == 5
        assert (pix_grid.sparse_to_unmasked_sparse == np.array([1, 3, 4, 5, 7])).all()
        assert (pix_grid.unmasked_sparse_to_sparse == np.array([0, 0, 1, 1, 2, 3, 4, 4, 4])).all()
        assert (pix_grid.regular_to_unmasked_sparse == np.array([1, 3, 4, 5, 7])).all()
        assert (pix_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4])).all()
        assert (pix_grid.sparse_grid == np.array([[2.0, 2.0], [0.0, 0.0], [0.0, 2.0], [0.0, 4.0],
                                                  [-2.0, 2.0]])).all()


class TestPaddedGrids:
    class TestPaddedImageGridFromShapes:

        def test__3x3_array__psf_size_is_1x1__mask_is_all_false(self):
            regular_padded_grid = grids.PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(shape=(3, 3),
                                                                                                           psf_shape=(
                                                                                                               1, 1),
                                                                                                           pixel_scale=1.0)

            assert len(regular_padded_grid) == 9
            assert regular_padded_grid.mask == np.array([[False, False, False],
                                                         [False, False, False],
                                                         [False, False, False]])

        def test__3x3_array__psf_size_is_1x1__no_padding(self):
            regular_padded_grid = grids.PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(shape=(3, 3),
                                                                                                           psf_shape=(
                                                                                                               1, 1),
                                                                                                           pixel_scale=1.0)

            assert len(regular_padded_grid) == 9
            assert regular_padded_grid.pixel_scale == 1.0
            assert regular_padded_grid.image_shape == (3, 3)
            assert regular_padded_grid.padded_shape == (3, 3)

        def test__3x3_image__5x5_psf_size__7x7_regular_padded_grid_made(self):
            regular_padded_grid = grids.PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(shape=(3, 3),
                                                                                                           psf_shape=(
                                                                                                               5, 5),
                                                                                                           pixel_scale=1.0)

            assert len(regular_padded_grid) == 49
            assert regular_padded_grid.image_shape == (3, 3)
            assert regular_padded_grid.padded_shape == (7, 7)

        def test__3x3_image__7x7_psf_size__9x9_regular_padded_grid_made(self):
            regular_padded_grid = grids.PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(shape=(3, 3),
                                                                                                           psf_shape=(
                                                                                                               7, 7),
                                                                                                           pixel_scale=1.0)
            assert len(regular_padded_grid) == 81
            assert regular_padded_grid.image_shape == (3, 3)
            assert regular_padded_grid.padded_shape == (9, 9)

        def test__4x3_image__3x3_psf_size__6x5_regular_padded_grid_made(self):
            regular_padded_grid = grids.PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(shape=(4, 3),
                                                                                                           psf_shape=(
                                                                                                               3, 3),
                                                                                                           pixel_scale=1.0)
            assert len(regular_padded_grid) == 30
            assert regular_padded_grid.image_shape == (4, 3)
            assert regular_padded_grid.padded_shape == (6, 5)

        def test__3x4_image__3x3_psf_size__5x6_regular_padded_grid_made(self):
            regular_padded_grid = grids.PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(shape=(3, 4),
                                                                                                           psf_shape=(
                                                                                                               3, 3),
                                                                                                           pixel_scale=1.0)

            assert len(regular_padded_grid) == 30
            assert regular_padded_grid.image_shape == (3, 4)
            assert regular_padded_grid.padded_shape == (5, 6)

        def test__4x4_image__3x3_psf_size__6x6_regular_padded_grid_made(self):
            regular_padded_grid = grids.PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(shape=(4, 4),
                                                                                                           psf_shape=(
                                                                                                               3, 3),
                                                                                                           pixel_scale=1.0)

            assert len(regular_padded_grid) == 36
            assert regular_padded_grid.image_shape == (4, 4)
            assert regular_padded_grid.padded_shape == (6, 6)

        def test__regular_padded_grid_coordinates__match_grid_2d_after_padding(self):
            regular_padded_grid = grids.PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(shape=(4, 4),
                                                                                                           psf_shape=(
                                                                                                               3, 3),
                                                                                                           pixel_scale=3.0)

            regular_padded_grid_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
                mask=np.full((6, 6), False), pixel_scales=(3.0, 3.0))
            assert (regular_padded_grid == regular_padded_grid_util).all()
            assert regular_padded_grid.image_shape == (4, 4)
            assert regular_padded_grid.padded_shape == (6, 6)

            regular_padded_grid = grids.PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(shape=(4, 5),
                                                                                                           psf_shape=(
                                                                                                               3, 3),
                                                                                                           pixel_scale=2.0)
            regular_padded_grid_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
                mask=np.full((6, 7), False), pixel_scales=(2.0, 2.0))
            assert (regular_padded_grid == regular_padded_grid_util).all()
            assert regular_padded_grid.image_shape == (4, 5)
            assert regular_padded_grid.padded_shape == (6, 7)

            regular_padded_grid = grids.PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(shape=(5, 4),
                                                                                                           psf_shape=(
                                                                                                               3, 3),
                                                                                                           pixel_scale=1.0)
            regular_padded_grid_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
                mask=np.full((7, 6), False), pixel_scales=(1.0, 1.0))
            assert (regular_padded_grid == regular_padded_grid_util).all()
            assert regular_padded_grid.image_shape == (5, 4)
            assert regular_padded_grid.padded_shape == (7, 6)

            regular_padded_grid = grids.PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(shape=(2, 5),
                                                                                                           psf_shape=(
                                                                                                               5, 5),
                                                                                                           pixel_scale=8.0)
            regular_padded_grid_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
                mask=np.full((6, 9), False), pixel_scales=(8.0, 8.0))
            assert (regular_padded_grid == regular_padded_grid_util).all()
            assert regular_padded_grid.image_shape == (2, 5)
            assert regular_padded_grid.padded_shape == (6, 9)

    class TestPaddedSubGridFromMask:

        def test__3x3_array__psf_size_is_1x1__no_padding__mask_is_all_false(self):
            mask = msk.Mask(array=np.full((3, 3), False), pixel_scale=1.0)

            sub_padded_grid = grids.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=3,
                                                                                                    psf_shape=(1, 1))

            assert len(sub_padded_grid) == 9 * 3 ** 2
            assert sub_padded_grid.mask == np.array([[False, False, False],
                                                     [False, False, False],
                                                     [False, False, False]])

        def test__3x3_array__psf_size_is_1x1__no_padding(self):
            mask = msk.Mask(array=np.full((3, 3), False), pixel_scale=1.0)

            sub_padded_grid = grids.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=3,
                                                                                                    psf_shape=(1, 1))

            assert len(sub_padded_grid) == 9 * 3 ** 2
            assert sub_padded_grid.image_shape == (3, 3)
            assert sub_padded_grid.padded_shape == (3, 3)

        def test__3x3_image__5x5_psf_size__7x7_regular_grid_made(self):
            mask = msk.Mask(array=np.full((3, 3), False), pixel_scale=1.0)

            sub_padded_grid = grids.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=2,
                                                                                                    psf_shape=(5, 5))

            assert len(sub_padded_grid) == 49 * 2 ** 2
            assert sub_padded_grid.image_shape == (3, 3)
            assert sub_padded_grid.padded_shape == (7, 7)

        def test__4x3_image__3x3_psf_size__6x5_regular_grid_made(self):
            mask = msk.Mask(array=np.full((4, 3), False), pixel_scale=1.0)

            sub_padded_grid = grids.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=2,
                                                                                                    psf_shape=(3, 3))

            assert len(sub_padded_grid) == 30 * 2 ** 2
            assert sub_padded_grid.image_shape == (4, 3)
            assert sub_padded_grid.padded_shape == (6, 5)

        def test__3x4_image__3x3_psf_size__5x6_regular_grid_made(self):
            mask = msk.Mask(array=np.full((3, 4), False), pixel_scale=1.0)

            sub_padded_grid = grids.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=2,
                                                                                                    psf_shape=(3, 3))

            assert len(sub_padded_grid) == 30 * 2 ** 2
            assert sub_padded_grid.image_shape == (3, 4)
            assert sub_padded_grid.padded_shape == (5, 6)

        def test__4x4_image__3x3_psf_size__6x6_regular_grid_made(self):
            mask = msk.Mask(array=np.full((4, 4), False), pixel_scale=1.0)

            sub_padded_grid = grids.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=4,
                                                                                                    psf_shape=(3, 3))

            assert len(sub_padded_grid) == 36 * 4 ** 2
            assert sub_padded_grid.image_shape == (4, 4)
            assert sub_padded_grid.padded_shape == (6, 6)

        def test__sub_padded_grid_coordinates__match_grid_2d_after_padding(self):
            mask = msk.Mask(array=np.full((4, 4), False), pixel_scale=3.0)

            sub_padded_grid = grids.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=3,
                                                                                                    psf_shape=(3, 3))

            sub_padded_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
                mask=np.full((6, 6), False), pixel_scales=(3.0, 3.0), sub_grid_size=3)

            assert sub_padded_grid == pytest.approx(sub_padded_grid_util, 1e-4)

            mask = msk.Mask(array=np.full((4, 5), False), pixel_scale=2.0)

            sub_padded_grid = grids.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=1,
                                                                                                    psf_shape=(3, 3))

            sub_padded_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
                mask=np.full((6, 7), False), pixel_scales=(2.0, 2.0), sub_grid_size=1)

            assert sub_padded_grid == pytest.approx(sub_padded_grid_util, 1e-4)

            mask = msk.Mask(array=np.full((5, 4), False), pixel_scale=2.0)

            sub_padded_grid = grids.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=2,
                                                                                                    psf_shape=(3, 3))

            sub_padded_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
                mask=np.full((7, 6), False), pixel_scales=(2.0, 2.0), sub_grid_size=2)

            assert sub_padded_grid == pytest.approx(sub_padded_grid_util, 1e-4)

            mask = msk.Mask(array=np.full((2, 5), False), pixel_scale=8.0)

            sub_padded_grid = grids.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                    sub_grid_size=4,
                                                                                                    psf_shape=(5, 5))

            sub_padded_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
                mask=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_grid_size=4)

            assert sub_padded_grid == pytest.approx(sub_padded_grid_util, 1e-4)

    class TestConvolve:

        def test__convolves_1d_array_with_psf(self):
            mask = msk.Mask(array=np.full((4, 4), False), pixel_scale=1.0)

            regular_padded_grid = grids.PaddedRegularGrid(arr=np.empty((0)), mask=mask, image_shape=(2, 2))

            array_1d = np.array([0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0])

            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = ccd.PSF(array=kernel, pixel_scale=1.0)

            blurred_array_1d = regular_padded_grid.convolve_array_1d_with_psf(array_1d, psf)

            assert (blurred_array_1d == np.array([0.0, 0.0, 0.0, 0.0,
                                                  0.0, 1.0, 0.0, 0.0,
                                                  1.0, 2.0, 1.0, 0.0,
                                                  0.0, 1.0, 0.0, 0.0])).all()

        def test__same_as_above_but_different_quantities(self):
            mask = msk.Mask(array=np.full((5, 4), False), pixel_scale=1.0)

            regular_padded_grid = grids.PaddedRegularGrid(arr=np.empty((0)), mask=mask, image_shape=(3, 2))

            array_1d = np.array([0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0,
                                 1.0, 0.0, 0.0, 0.0])

            kernel = np.array([[1.0, 1.0, 4.0],
                               [1.0, 3.0, 1.0],
                               [1.0, 1.0, 1.0]])

            psf = ccd.PSF(array=kernel, pixel_scale=1.0)

            blurred_array_1d = regular_padded_grid.convolve_array_1d_with_psf(array_1d, psf)

            assert (blurred_array_1d == np.array([0.0, 0.0, 0.0, 0.0,
                                                  1.0, 1.0, 4.0, 0.0,
                                                  1.0, 3.0, 1.0, 0.0,
                                                  2.0, 5.0, 1.0, 0.0,
                                                  3.0, 1.0, 0.0, 0.0])).all()

    class TestMapUnmaskedArrays:

        def test__map_to_2d_keep_padded__4x4_from_1d(self):
            mask = msk.Mask(array=np.full((4, 4), False), pixel_scale=1.0)

            regular_padded_grid = grids.PaddedRegularGrid(arr=np.empty((0)), mask=mask, image_shape=(2, 2))

            array_1d = np.array([6.0, 7.0, 9.0, 10.0,
                                 1.0, 2.0, 3.0, 4.0,
                                 5.0, 6.0, 7.0, 8.0,
                                 1.0, 2.0, 3.0, 4.0])

            array_2d = regular_padded_grid.map_to_2d_keep_padded(padded_array_1d=array_1d)

            assert (array_2d == np.array([[6.0, 7.0, 9.0, 10.0],
                                          [1.0, 2.0, 3.0, 4.0],
                                          [5.0, 6.0, 7.0, 8.0],
                                          [1.0, 2.0, 3.0, 4.0]])).all()

        def test__map_to_2d_keep_padded__5x3_from_1d(self):
            mask = msk.Mask(array=np.full((5, 3), False), pixel_scale=1.0)

            regular_padded_grid = grids.PaddedRegularGrid(arr=np.empty((0)), mask=mask, image_shape=(3, 1))

            array_1d = np.array([1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0,
                                 7.0, 8.0, 9.0,
                                 1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0])
            array_2d = regular_padded_grid.map_to_2d_keep_padded(padded_array_1d=array_1d)

            assert (array_2d == np.array([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0],
                                          [7.0, 8.0, 9.0],
                                          [1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]])).all()

        def test__map_to_2d_keep_padded__3x5__from_1d(self):
            mask = msk.Mask(array=np.full((3, 5), False), pixel_scale=1.0)

            regular_padded_grid = grids.PaddedRegularGrid(arr=np.empty((0)), mask=mask, image_shape=(1, 3))

            array_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0,
                                 6.0, 7.0, 8.0, 9.0, 1.0,
                                 2.0, 3.0, 4.0, 5.0, 6.0])
            array_2d = regular_padded_grid.map_to_2d_keep_padded(padded_array_1d=array_1d)

            assert (array_2d == np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                                          [6.0, 7.0, 8.0, 9.0, 1.0],
                                          [2.0, 3.0, 4.0, 5.0, 6.0]])).all()

        def test__map_to_2d_and_trim__4x4_to_2x2__from_1d(self):
            mask = msk.Mask(array=np.full((4, 4), False), pixel_scale=1.0)

            regular_padded_grid = grids.PaddedRegularGrid(arr=np.empty((0)), mask=mask, image_shape=(2, 2))

            array_1d = np.array([1.0, 2.0, 3.0, 4.0,
                                 5.0, 6.0, 7.0, 8.0,
                                 9.0, 1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0, 7.0])
            array_2d = regular_padded_grid.array_2d_from_array_1d(padded_array_1d=array_1d)

            assert (array_2d == np.array([[6.0, 7.0],
                                          [1.0, 2.0]])).all()

        def test__map_to_2d_and_trim__5x3_to_3x1__from_1d(self):
            mask = msk.Mask(array=np.full((5, 3), False), pixel_scale=1.0)

            regular_padded_grid = grids.PaddedRegularGrid(arr=np.empty((0)), mask=mask, image_shape=(3, 1))

            array_1d = np.array([1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0,
                                 7.0, 8.0, 9.0,
                                 1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0])
            array_2d = regular_padded_grid.array_2d_from_array_1d(padded_array_1d=array_1d)

            assert (array_2d == np.array([[5.0],
                                          [8.0],
                                          [2.0]])).all()

        def test__map_to_2d_and_trim__3x5_to_1x3__from_1d(self):
            mask = msk.Mask(array=np.full((3, 5), False), pixel_scale=1.0)

            regular_padded_grid = grids.PaddedRegularGrid(arr=np.empty((0)), mask=mask, image_shape=(1, 3))

            array_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0,
                                 6.0, 7.0, 8.0, 9.0, 1.0,
                                 2.0, 3.0, 4.0, 5.0, 6.0])
            array_2d = regular_padded_grid.array_2d_from_array_1d(padded_array_1d=array_1d)

            assert (array_2d == np.array([[7.0, 8.0, 9.0]])).all()

    class TestUnmaskedBlurredImage:

        def test__convolve_1d_array_with_psf_and_trims_to_original_size(self):
            mask = msk.Mask(array=np.full((4, 4), False), pixel_scale=1.0)

            regular_padded_grid = grids.PaddedRegularGrid(arr=np.empty((0)), mask=mask, image_shape=(2, 2))

            array_1d = np.array([0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0])

            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = ccd.PSF(array=kernel, pixel_scale=1.0)

            blurred_array_2d = regular_padded_grid.padded_blurred_image_2d_from_padded_image_1d_and_psf(
                padded_image_1d=array_1d, psf=psf)

            assert (blurred_array_2d == np.array([[1.0, 0.0],
                                                  [2.0, 1.0]])).all()

        def test__same_as_above_but_different_quantities(self):
            mask = msk.Mask(array=np.full((5, 4), False), pixel_scale=1.0)

            regular_padded_grid = grids.PaddedRegularGrid(arr=np.empty((0)), mask=mask, image_shape=(3, 2))

            array_1d = np.array([0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0,
                                 1.0, 0.0, 0.0, 0.0])

            kernel = np.array([[1.0, 1.0, 4.0],
                               [1.0, 3.0, 1.0],
                               [1.0, 1.0, 1.0]])

            psf = ccd.PSF(array=kernel, pixel_scale=1.0)

            blurred_array_2d = regular_padded_grid.padded_blurred_image_2d_from_padded_image_1d_and_psf(
                padded_image_1d=array_1d, psf=psf)

            assert (blurred_array_2d == np.array([[1.0, 4.0],
                                                  [3.0, 1.0],
                                                  [5.0, 1.0]])).all()


class TestGridStack(object):

    def test__grids(self, grid_stack):
        assert (grid_stack.regular == np.array([[0., 0.]])).all()
        np.testing.assert_almost_equal(grid_stack.sub, np.array([[0.16666667, -0.16666667],
                                                                 [0.16666667, 0.16666667],
                                                                 [-0.16666667, -0.16666667],
                                                                 [-0.16666667, 0.16666667]]))
        assert (grid_stack.blurring == np.array([[1., -1.],
                                                 [1., 0.],
                                                 [1., 1.],
                                                 [0., -1.],
                                                 [0., 1.],
                                                 [-1., -1.],
                                                 [-1., 0.],
                                                 [-1., 1.]])).all()
        assert (grid_stack.pix == np.array([[0.0, 0.0]])).all()

    def test__from_shape_and_pixel_scale(self):
        ma = msk.Mask(np.array([[False, False, False],
                                [False, False, False],
                                [False, False, False]]), pixel_scale=2.0)

        grid_stack_mask = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=2,
                                                                                           psf_shape=(1, 1))

        grid_stack_shape = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=(3, 3), pixel_scale=2.0,
                                                                                    sub_grid_size=2)

        assert (grid_stack_mask.regular == grid_stack_shape.regular).all()
        assert (grid_stack_mask.sub == grid_stack_shape.sub).all()
        assert (grid_stack_mask.pix == np.array([[0.0, 0.0]])).all()

    def test__padded_grids(self):
        mask = np.array([[False, False],
                         [False, False]])

        mask = msk.Mask(mask, pixel_scale=1.0)

        padded_grids = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask, sub_grid_size=2,
                                                                                               psf_shape=(3, 3))

        regular_padded_grid_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
            mask=np.full((4, 4), False),
            pixel_scales=(1.0, 1.0))

        sub_padded_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((4, 4), False),
            pixel_scales=(1.0, 1.0), sub_grid_size=2)

        assert (padded_grids.regular == regular_padded_grid_util).all()
        assert padded_grids.regular.image_shape == (2, 2)
        assert padded_grids.regular.padded_shape == (4, 4)

        assert padded_grids.sub == pytest.approx(sub_padded_grid_util, 1e-4)
        assert padded_grids.sub.image_shape == (2, 2)
        assert padded_grids.sub.padded_shape == (4, 4)

        assert (padded_grids.blurring == np.array([0.0, 0.0])).all()

        assert (padded_grids.pix == np.array([[0.0, 0.0]])).all()

    def test__for_simulation(self):
        padded_grids = grids.GridStack.grid_stack_for_simulation(shape=(2, 2), pixel_scale=1.0, sub_grid_size=2,
                                                                 psf_shape=(3, 3))

        regular_padded_grid_util = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
            mask=np.full((4, 4), False),
            pixel_scales=(1.0, 1.0))

        sub_padded_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((4, 4), False),
            pixel_scales=(1.0, 1.0), sub_grid_size=2)

        assert (padded_grids.regular == regular_padded_grid_util).all()
        assert padded_grids.regular.image_shape == (2, 2)
        assert padded_grids.regular.padded_shape == (4, 4)

        assert padded_grids.sub == pytest.approx(sub_padded_grid_util, 1e-4)
        assert padded_grids.sub.image_shape == (2, 2)
        assert padded_grids.sub.padded_shape == (4, 4)

        assert (padded_grids.blurring == np.array([0.0, 0.0])).all()

        assert (padded_grids.pix == np.array([[0.0, 0.0]])).all()

    def test__scaled_array_from_array_1d(self, grid_stack):
        scaled_array_from_grid_stack = grid_stack.scaled_array_2d_from_array_1d(array_1d=np.ones(5))
        scaled_array_from_regular = grid_stack.regular.scaled_array_2d_from_array_1d(array_1d=np.ones(5))

        assert (scaled_array_from_grid_stack == scaled_array_from_regular).all()

    def test__apply_function_retains_attributes(self, grid_stack):
        grid_stack.pix = grid_stack.regular

        def add_one(coords):
            return np.add(1, coords)

        assert isinstance(grid_stack.regular, grids.RegularGrid)
        assert grid_stack.regular.mask is not None

        assert isinstance(grid_stack.sub, grids.SubGrid)
        assert grid_stack.sub.mask is not None
        assert grid_stack.sub.sub_grid_size is not None
        assert grid_stack.sub.sub_grid_length is not None
        assert grid_stack.sub.sub_grid_fraction is not None

        new_collection = grid_stack.apply_function(add_one)

        assert new_collection.regular.mask is not None
        assert new_collection.sub.mask is not None
        assert new_collection.sub.sub_grid_size is not None
        assert new_collection.sub.sub_grid_length is not None
        assert new_collection.sub.sub_grid_fraction is not None

        assert isinstance(grid_stack.pix, grids.RegularGrid)
        assert grid_stack.pix.mask is not None

    def test__apply_function(self, grid_stack):
        grid_stack.pix = grid_stack.regular

        def add_one(coords):
            return np.add(1, coords)

        new_collection = grid_stack.apply_function(add_one)
        assert isinstance(new_collection, grids.GridStack)
        assert (new_collection.regular == np.add(1, np.array([[0., 0.]]))).all()
        np.testing.assert_almost_equal(new_collection.sub, np.add(1, np.array([[0.16666667, -0.16666667],
                                                                               [0.16666667, 0.16666667],
                                                                               [-0.16666667, -0.16666667],
                                                                               [-0.16666667, 0.16666667]])))
        assert (new_collection.blurring == np.add(1, np.array([[1., -1.],
                                                               [1., 0.],
                                                               [1., 1.],
                                                               [0., -1.],
                                                               [0., 1.],
                                                               [-1., -1.],
                                                               [-1., 0.],
                                                               [-1., 1.]]))).all()
        assert (new_collection.pix == np.add(1, np.array([[0., 0.]]))).all()

    def test__map_function(self, grid_stack):
        grid_stack.pix = grid_stack.regular

        def add_number(coords, number):
            return np.add(coords, number)

        new_collection = grid_stack.map_function(add_number, [1, 2, 3, 1])

        assert isinstance(new_collection, grids.GridStack)
        assert (new_collection.regular == np.add(1, np.array([[0., 0.]]))).all()
        np.testing.assert_almost_equal(new_collection.sub, np.add(2, np.array([[0.16666667, -0.16666667],
                                                                               [0.16666667, 0.16666667],
                                                                               [-0.16666667, -0.16666667],
                                                                               [-0.16666667, 0.16666667]])))
        assert (new_collection.blurring == np.add(3, np.array([[1., -1.],
                                                               [1., 0.],
                                                               [1., 1.],
                                                               [0., -1.],
                                                               [0., 1.],
                                                               [-1., -1.],
                                                               [-1., 0.],
                                                               [-1., 1.]]))).all()

        assert (new_collection.pix == np.add(1, np.array([[0., 0.]]))).all()

    def test__new_grid_stack_with_pix_grid(self, grid_stack):
        grid_stack = grid_stack.new_grid_stack_with_pix_grid_added(pix_grid=np.array([[5.0, 5.0], [6.0, 7.0]]),
                                                                   regular_to_nearest_pix=np.array([0, 1]))

        assert (grid_stack.regular == np.array([[0., 0.]])).all()
        np.testing.assert_almost_equal(grid_stack.sub, np.array([[0.16666667, -0.16666667],
                                                                 [0.16666667, 0.16666667],
                                                                 [-0.16666667, -0.16666667],
                                                                 [-0.16666667, 0.16666667]]))
        assert (grid_stack.blurring == np.array([[1., -1.],
                                                 [1., 0.],
                                                 [1., 1.],
                                                 [0., -1.],
                                                 [0., 1.],
                                                 [-1., -1.],
                                                 [-1., 0.],
                                                 [-1., 1.]])).all()
        assert (grid_stack.pix == np.array([[5.0, 5.0], [6.0, 7.0]])).all()
        assert (grid_stack.pix.regular_to_nearest_pix == np.array([0, 1])).all()

    def test__new_grid_stack_with_interpolator_added_to_each_grid(self):

        mask = np.array([[True, True, True, True, True, True],
                         [True, True, True, False, False, True],
                         [True, False, True, True, True, True],
                         [True, True, True, False, False, True],
                         [True, True, True, True, True, True]])
        mask = msk.Mask(array=mask, pixel_scale=2.0)

        grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
                        mask=mask, sub_grid_size=2, psf_shape=(3, 3))

        print(grid_stack.blurring)
        new_grid_stack = grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(interp_pixel_scale=1.0)

        regular_grid_manual = grids.RegularGrid.from_mask(mask=mask)
        sub_grid_manual = grids.SubGrid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)
        blurring_grid_manual = grids.RegularGrid.blurring_grid_from_mask_and_psf_shape(mask=mask, psf_shape=(3,3))

        assert (new_grid_stack.regular == regular_grid_manual).all()
        np.testing.assert_almost_equal(new_grid_stack.sub, sub_grid_manual)

        print(new_grid_stack.blurring)

        assert (new_grid_stack.blurring == blurring_grid_manual).all()

        regular_interpolator_manual = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=regular_grid_manual, interp_pixel_scale=1.0)
        sub_interpolator_manual = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=sub_grid_manual, interp_pixel_scale=1.0)
        blurring_interpolator_manual = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=blurring_grid_manual.mask, grid=blurring_grid_manual, interp_pixel_scale=1.0)

        assert (new_grid_stack.regular.interpolator.vtx == regular_interpolator_manual.vtx).all()
        assert (new_grid_stack.regular.interpolator.wts == regular_interpolator_manual.wts).all()

        assert (new_grid_stack.sub.interpolator.vtx == sub_interpolator_manual.vtx).all()
        assert (new_grid_stack.sub.interpolator.wts == sub_interpolator_manual.wts).all()

        assert (new_grid_stack.blurring.interpolator.vtx == blurring_interpolator_manual.vtx).all()
        assert (new_grid_stack.blurring.interpolator.wts == blurring_interpolator_manual.wts).all()

    def test__same_as_above_for_padded_grid_stack__blurring_grid_is_zeros__has_no_interpolator(self):

        mask = np.array([[True, True, True, True, True, True],
                         [True, True, True, False, False, True],
                         [True, False, True, True, True, True],
                         [True, True, True, False, False, True],
                         [True, True, True, True, True, True]])
        mask = msk.Mask(array=mask, pixel_scale=2.0)

        padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(
                        mask=mask, sub_grid_size=2, psf_shape=(3, 3))
        new_padded_grid_stack = padded_grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(interp_pixel_scale=1.0)

        regular_grid_manual = grids.PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(
            shape=mask.shape, pixel_scale=mask.pixel_scale, psf_shape=(3,3))
        sub_grid_manual = grids.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(
            mask=mask, sub_grid_size=2, psf_shape=(3,3))

        assert (new_padded_grid_stack.regular == regular_grid_manual).all()
        np.testing.assert_almost_equal(new_padded_grid_stack.sub, sub_grid_manual)

        regular_interpolator_manual = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=regular_grid_manual.mask, grid=regular_grid_manual, interp_pixel_scale=1.0)
        sub_interpolator_manual = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=sub_grid_manual.mask, grid=sub_grid_manual, interp_pixel_scale=1.0)

        assert (new_padded_grid_stack.regular.interpolator.vtx == regular_interpolator_manual.vtx).all()
        assert (new_padded_grid_stack.regular.interpolator.wts == regular_interpolator_manual.wts).all()

        assert (new_padded_grid_stack.sub.interpolator.vtx == sub_interpolator_manual.vtx).all()
        assert (new_padded_grid_stack.sub.interpolator.wts == sub_interpolator_manual.wts).all()

        assert (new_padded_grid_stack.blurring == np.array([[0.0, 0.0]])).all()


class TestImageGridBorder(object):
    class TestFromMask:

        def test__simple_mask_border_pixels_is_border(self):
            mask = np.array([[False, False, False, False, False, False, False, True],
                             [False, True, True, True, True, True, False, True],
                             [False, True, False, False, False, True, False, True],
                             [False, True, False, True, False, True, False, True],
                             [False, True, False, False, False, True, False, True],
                             [False, True, True, True, True, True, False, True],
                             [False, False, False, False, False, False, False, True]])

            mask = msk.Mask(mask, pixel_scale=3.0)

            border = grids.RegularGridBorder.from_mask(mask)

            assert (border == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 17, 18, 22, 23, 24, 25,
                                        26, 27, 28, 29, 30, 31])).all()

    class TestRelocateCoordinates(object):

        def test__inside_border_no_relocations(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 32)
            grid_circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))
            grid = grid_circle
            grid.append(np.array([0.1, 0.0]))
            grid.append(np.array([-0.2, -0.3]))
            grid.append(np.array([0.5, 0.4]))
            grid.append(np.array([0.7, -0.1]))
            regular_grid = np.asarray(grid)
            sub_grid = np.asarray(grid)
            sub_grid[35, 0] = 0.5
            sub_grid[35, 1] = 0.3
            grid_stack = grids.GridStack(regular=regular_grid, sub=sub_grid, blurring=None)

            border = grids.RegularGridBorder(arr=np.arange(32))
            relocated_grids = border.relocated_grid_stack_from_grid_stack(grid_stack)

            assert relocated_grids.regular[0:32] == pytest.approx(np.asarray(grid_circle)[0:32], 1e-3)
            assert relocated_grids.regular[32] == pytest.approx(np.array([0.1, 0.0]), 1e-3)
            assert relocated_grids.regular[33] == pytest.approx(np.array([-0.2, -0.3]), 1e-3)
            assert relocated_grids.regular[34] == pytest.approx(np.array([0.5, 0.4]), 1e-3)
            assert relocated_grids.regular[35] == pytest.approx(np.array([0.7, -0.1]), 1e-3)

            assert relocated_grids.sub[0:32] == pytest.approx(np.asarray(grid_circle)[0:32], 1e-3)
            assert relocated_grids.sub[32] == pytest.approx(np.array([0.1, 0.0]), 1e-3)
            assert relocated_grids.sub[33] == pytest.approx(np.array([-0.2, -0.3]), 1e-3)
            assert relocated_grids.sub[34] == pytest.approx(np.array([0.5, 0.4]), 1e-3)
            assert relocated_grids.sub[35] == pytest.approx(np.array([0.5, 0.3]), 1e-3)

        def test__8_points_with_border_as_circle__points_go_to_circle_edge(self):
            grid = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
                             [0.7071, 0.7071], [0.7071, -0.7071],
                             [-0.7071, 0.7071], [-0.7071, -0.7071],
                             [10., 10.], [10., -10.], [-10., 10], [-10., -10.]])
            grid_stack = grids.GridStack(regular=grid, sub=grid, blurring=None)

            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            border = grids.RegularGridBorder(border_pixels)

            relocated_grids = border.relocated_grid_stack_from_grid_stack(grid_stack)

            assert relocated_grids.regular[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.regular[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.regular[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.regular[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.regular[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.regular[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.regular[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.regular[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.regular[8] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)
            assert relocated_grids.regular[9] == pytest.approx(np.array([0.7071, -0.7071]), 1e-3)
            assert relocated_grids.regular[10] == pytest.approx(np.array([-0.7071, 0.7071]), 1e-3)
            assert relocated_grids.regular[11] == pytest.approx(np.array([-0.7071, -0.7071]), 1e-3)

            assert relocated_grids.sub[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.sub[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.sub[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.sub[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.sub[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.sub[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.sub[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.sub[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.sub[8] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)
            assert relocated_grids.sub[9] == pytest.approx(np.array([0.7071, -0.7071]), 1e-3)
            assert relocated_grids.sub[10] == pytest.approx(np.array([-0.7071, 0.7071]), 1e-3)
            assert relocated_grids.sub[11] == pytest.approx(np.array([-0.7071, -0.7071]), 1e-3)

        def test__same_as_above_but_ensure_positive_origin_moves_points(self):
            grid = np.array([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 0.0],
                             [1.0 + 0.7071, 1.0 + 0.7071], [1.0 + 0.7071, 1.0 - 0.7071],
                             [1.0 - 0.7071, 1.0 + 0.7071], [1.0 - 0.7071, 1.0 - 0.7071],
                             [11., 11.], [11., -9.], [-9., 11], [-9., -9.]])
            grid_stack = grids.GridStack(regular=grid, sub=grid, blurring=None)

            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            border = grids.RegularGridBorder(border_pixels)

            relocated_grids = border.relocated_grid_stack_from_grid_stack(grid_stack)

            assert relocated_grids.regular[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.regular[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.regular[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.regular[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.regular[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.regular[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.regular[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.regular[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.regular[8] == pytest.approx(np.array([1.0 + 0.7071, 1.0 + 0.7071]), 1e-3)
            assert relocated_grids.regular[9] == pytest.approx(np.array([1.0 + 0.7071, 1.0 - 0.7071]), 1e-3)
            assert relocated_grids.regular[10] == pytest.approx(np.array([1.0 - 0.7071, 1.0 + 0.7071]), 1e-3)
            assert relocated_grids.regular[11] == pytest.approx(np.array([1.0 - 0.7071, 1.0 - 0.7071]), 1e-3)

            assert relocated_grids.sub[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.sub[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.sub[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.sub[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.sub[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.sub[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.sub[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.sub[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.sub[8] == pytest.approx(np.array([1.0 + 0.7071, 1.0 + 0.7071]), 1e-3)
            assert relocated_grids.sub[9] == pytest.approx(np.array([1.0 + 0.7071, 1.0 - 0.7071]), 1e-3)
            assert relocated_grids.sub[10] == pytest.approx(np.array([1.0 - 0.7071, 1.0 + 0.7071]), 1e-3)
            assert relocated_grids.sub[11] == pytest.approx(np.array([1.0 - 0.7071, 1.0 - 0.7071]), 1e-3)

        def test__same_as_above_but_ensure_negative_origin_moves_points(self):
            grid = np.array([[0.0, -1.0], [-1.0, 0.0], [-2.0, -1.0], [-1.0, -2.0],
                             [-1.0 + 0.7071, -1.0 + 0.7071], [-1.0 + 0.7071, -1.0 - 0.7071],
                             [-1.0 - 0.7071, -1.0 + 0.7071], [-1.0 - 0.7071, -1.0 - 0.7071],
                             [9., 9.], [9., -11.], [-11., 9], [-11., -11.]])
            grid_stack = grids.GridStack(regular=grid, sub=grid, blurring=None)
            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            border = grids.RegularGridBorder(border_pixels)

            relocated_grids = border.relocated_grid_stack_from_grid_stack(grid_stack)

            assert relocated_grids.regular[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.regular[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.regular[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.regular[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.regular[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.regular[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.regular[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.regular[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.regular[8] == pytest.approx(np.array([-1.0 + 0.7071, -1.0 + 0.7071]), 1e-3)
            assert relocated_grids.regular[9] == pytest.approx(np.array([-1.0 + 0.7071, -1.0 - 0.7071]), 1e-3)
            assert relocated_grids.regular[10] == pytest.approx(np.array([-1.0 - 0.7071, -1.0 + 0.7071]), 1e-3)
            assert relocated_grids.regular[11] == pytest.approx(np.array([-1.0 - 0.7071, -1.0 - 0.7071]), 1e-3)

            assert relocated_grids.sub[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.sub[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.sub[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.sub[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.sub[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.sub[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.sub[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.sub[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.sub[8] == pytest.approx(np.array([-1.0 + 0.7071, -1.0 + 0.7071]), 1e-3)
            assert relocated_grids.sub[9] == pytest.approx(np.array([-1.0 + 0.7071, -1.0 - 0.7071]), 1e-3)
            assert relocated_grids.sub[10] == pytest.approx(np.array([-1.0 - 0.7071, -1.0 + 0.7071]), 1e-3)
            assert relocated_grids.sub[11] == pytest.approx(np.array([-1.0 - 0.7071, -1.0 - 0.7071]), 1e-3)

        def test__point_is_inside_border_but_further_than_minimum_border_point_radii__does_not_relocate(self):
            grid = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -0.9],
                             [0.7071, 0.7071], [0.7071, -0.7071],
                             [-0.7071, 0.7071], [-0.7071, -0.7071],
                             [0.02, 0.95]])

            grid_stack = grids.GridStack(regular=grid, sub=grid, blurring=None)

            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            border = grids.RegularGridBorder(border_pixels)

            relocated_grids = border.relocated_grid_stack_from_grid_stack(grid_stack)

            assert relocated_grids.regular[8] == pytest.approx(np.array([0.02, 0.95]), 1e-4)
            assert relocated_grids.sub[8] == pytest.approx(np.array([0.02, 0.95]), 1e-4)

        def test__inside_border_no_relocations__also_include_pix_grid(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 32)
            grid_circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))
            grid = grid_circle
            grid.append(np.array([0.1, 0.0]))
            grid.append(np.array([-0.2, -0.3]))
            grid.append(np.array([0.5, 0.4]))
            grid.append(np.array([0.7, -0.1]))
            regular_grid = np.asarray(grid)
            sub_grid = np.asarray(grid)
            sub_grid[35, 0] = 0.5
            sub_grid[35, 1] = 0.3
            grid_stack = grids.GridStack(regular=regular_grid, sub=sub_grid, blurring=None, pix=regular_grid)

            border = grids.RegularGridBorder(arr=np.arange(32))
            relocated_grids = border.relocated_grid_stack_from_grid_stack(grid_stack)

            assert relocated_grids.regular[0:32] == pytest.approx(np.asarray(grid_circle)[0:32], 1e-3)
            assert relocated_grids.regular[32] == pytest.approx(np.array([0.1, 0.0]), 1e-3)
            assert relocated_grids.regular[33] == pytest.approx(np.array([-0.2, -0.3]), 1e-3)
            assert relocated_grids.regular[34] == pytest.approx(np.array([0.5, 0.4]), 1e-3)
            assert relocated_grids.regular[35] == pytest.approx(np.array([0.7, -0.1]), 1e-3)

            assert relocated_grids.sub[0:32] == pytest.approx(np.asarray(grid_circle)[0:32], 1e-3)
            assert relocated_grids.sub[32] == pytest.approx(np.array([0.1, 0.0]), 1e-3)
            assert relocated_grids.sub[33] == pytest.approx(np.array([-0.2, -0.3]), 1e-3)
            assert relocated_grids.sub[34] == pytest.approx(np.array([0.5, 0.4]), 1e-3)
            assert relocated_grids.sub[35] == pytest.approx(np.array([0.5, 0.3]), 1e-3)

            assert relocated_grids.pix[0:32] == pytest.approx(np.asarray(grid_circle)[0:32], 1e-3)
            assert relocated_grids.pix[32] == pytest.approx(np.array([0.1, 0.0]), 1e-3)
            assert relocated_grids.pix[33] == pytest.approx(np.array([-0.2, -0.3]), 1e-3)
            assert relocated_grids.pix[34] == pytest.approx(np.array([0.5, 0.4]), 1e-3)
            assert relocated_grids.pix[35] == pytest.approx(np.array([0.7, -0.1]), 1e-3)


class TestInterpolator:

    def test_decorated_function__values_from_function_has_1_dimensions__returns_1d_result(self):

        # noinspection PyUnusedLocal
        @grids.grid_interpolate
        def func(profile, grid):
            result = np.zeros(grid.shape[0])
            result[0] = 1
            return result

        regular = grids.RegularGrid.from_mask(mask=msk.Mask.unmasked_for_shape_and_pixel_scale((3, 3), 1))

        values = func(None, regular)

        assert values.ndim == 1
        assert values.shape == (9,)
        assert (values == np.array([[1, 0, 0,
                                     0, 0, 0,
                                     0, 0, 0], ])).all()

        regular = grids.RegularGrid.from_mask(mask=msk.Mask.unmasked_for_shape_and_pixel_scale((3, 3), 1))
        regular.interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(regular.mask, regular,
                                                                                         interp_pixel_scale=0.5)
        interp_values = func(None, regular)
        assert interp_values.ndim == 1
        assert interp_values.shape == (9,)
        assert (interp_values != np.array([[1, 0, 0,
                                            0, 0, 0,
                                            0, 0, 0]])).any()

    def test_decorated_function__values_from_function_has_2_dimensions__returns_2d_result(self):

        # noinspection PyUnusedLocal
        @grids.grid_interpolate
        def func(profile, grid):
            result = np.zeros((grid.shape[0], 2))
            result[0,:] = 1
            return result

        regular = grids.RegularGrid.from_mask(mask=msk.Mask.unmasked_for_shape_and_pixel_scale((3, 3), 1))

        values = func(None, regular)

        assert values.ndim == 2
        assert values.shape == (9, 2)
        assert (values == np.array([[1,1], [0,0], [0,0],
                                    [0,0], [0,0], [0,0],
                                    [0,0], [0,0], [0,0]])).all()

        regular = grids.RegularGrid.from_mask(mask=msk.Mask.unmasked_for_shape_and_pixel_scale((3, 3), 1))
        regular.interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(regular.mask, regular,
                                                                                         interp_pixel_scale=0.5)

        interp_values = func(None, regular)
        assert interp_values.ndim == 2
        assert interp_values.shape == (9, 2)
        assert (interp_values != np.array(np.array([[1,1], [0,0], [0,0],
                                                    [0,0], [0,0], [0,0],
                                                    [0,0], [0,0], [0,0]]))).any()

    def test__20x20_deflection_angles_no_central_pixels__interpolated_accurately(self):

        shape = (20, 20)
        pixel_scale = 1.0

        mask = msk.Mask.circular_annular(shape=shape, pixel_scale=pixel_scale, inner_radius_arcsec=3.0,
                                         outer_radius_arcsec=8.0)

        grid = grids.RegularGrid.from_mask(mask=mask)

        isothermal = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        true_deflections = isothermal.deflections_from_grid(grid=grid)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=grid, interp_pixel_scale=1.0)

        interp_deflections_values = isothermal.deflections_from_grid(grid=interpolator.interp_grid)

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0])
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1])

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.001
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.001

    def test__move_centre_of_galaxy__interpolated_accurately(self):
        shape = (24, 24)
        pixel_scale = 1.0

        mask = msk.Mask.circular_annular(shape=shape, pixel_scale=pixel_scale, inner_radius_arcsec=3.0,
                                         outer_radius_arcsec=8.0, centre=(3.0, 3.0))

        grid = grids.RegularGrid.from_mask(mask=mask)

        isothermal = mp.SphericalIsothermal(centre=(3.0, 3.0), einstein_radius=1.0)

        true_deflections = isothermal.deflections_from_grid(grid=grid)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=grid, interp_pixel_scale=1.0)

        interp_deflections_values = isothermal.deflections_from_grid(grid=interpolator.interp_grid)

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0])
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1])

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.001
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.001

    def test__different_interpolation_pixel_scales_still_works(self):
        shape = (28, 28)
        pixel_scale = 1.0

        mask = msk.Mask.circular_annular(shape=shape, pixel_scale=pixel_scale, inner_radius_arcsec=3.0,
                                         outer_radius_arcsec=8.0, centre=(3.0, 3.0))

        grid = grids.RegularGrid.from_mask(mask=mask)

        isothermal = mp.SphericalIsothermal(centre=(3.0, 3.0), einstein_radius=1.0)

        true_deflections = isothermal.deflections_from_grid(grid=grid)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=grid, interp_pixel_scale=0.2)

        interp_deflections_values = isothermal.deflections_from_grid(grid=interpolator.interp_grid)

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0])
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1])

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.001
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.001

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=grid, interp_pixel_scale=0.5)

        interp_deflections_values = isothermal.deflections_from_grid(grid=interpolator.interp_grid)

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0])
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1])

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.01
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.01

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=grid, interp_pixel_scale=1.1)

        interp_deflections_values = isothermal.deflections_from_grid(grid=interpolator.interp_grid)

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0])
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1])

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.1
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.1
