import numpy as np
import pytest

from autolens.imaging import image
from autolens.imaging.util import grid_util
from autolens.imaging.util import mask_util
from autolens.imaging.util import mapping_util
from autolens.imaging import mask


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

        msk = mask.Mask.circular(shape=(5, 4), pixel_scale=2.7, radius_mask_arcsec=3.5, centre=(1.0, 1.0))

        assert (msk == msk_util).all()

    def test__mask_annulus__compare_to_array_util(self):
        msk_util = mask_util.mask_annular_from_shape_pixel_scale_and_radii(shape=(5, 4), pixel_scale=2.7,
                                                                           inner_radius_arcsec=0.8,
                                                                           outer_radius_arcsec=3.5,
                                                                           centre=(1.0, 1.0))

        msk = mask.Mask.annular(shape=(5, 4), pixel_scale=2.7, inner_radius_arcsec=0.8, outer_radius_arcsec=3.5,
                                centre=(1.0, 1.0))

        assert (msk == msk_util).all()

    def test__mask_anti_annulus__compare_to_array_util(self):
        msk_util = mask_util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape=(9, 9), pixel_scale=1.2,
                                                                                inner_radius_arcsec=0.8,
                                                                                outer_radius_arcsec=2.2,
                                                                                outer_radius_2_arcsec=3.0,
                                                                                centre=(1.0, 1.0))

        msk = mask.Mask.anti_annular(shape=(9, 9), pixel_scale=1.2, inner_radius_arcsec=0.8,
                                     outer_radius_arcsec=2.2, outer_radius_2_arcsec=3.0, origin=(1.0, 1.0))

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


@pytest.fixture(name="msk")
def make_mask():
    return mask.Mask(np.array([[True, False, True],
                               [False, False, False],
                               [True, False, True]]), pixel_scale=1.0)


@pytest.fixture(name="centre_mask")
def make_centre_mask():
    return mask.Mask(np.array([[True, True, True],
                               [True, False, True],
                               [True, True, True]]), pixel_scale=1.0)


@pytest.fixture(name="sub_grid")
def make_sub_grid(msk):
    return mask.SubGrid.from_mask_and_sub_grid_size(msk, sub_grid_size=1)


@pytest.fixture(name="imaging_grids")
def make_imaging_grids(centre_mask):
    return mask.ImagingGrids.grids_from_mask_sub_grid_size_and_psf_shape(centre_mask, 2, (3, 3))


class TestImageGrid:

    def test__image_grid_from_mask__compare_to_array_util(self):
        msk = np.array([[True, True, False, False],
                        [True, False, True, True],
                        [True, True, False, False]])
        msk = mask.Mask(array=msk, pixel_scale=2.0)

        image_grid_util = grid_util.image_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=msk,
                                                                                           pixel_scales=(2.0, 2.0))

        image_grid = mask.ImageGrid.from_mask(msk)

        assert type(image_grid) == mask.ImageGrid
        assert image_grid == pytest.approx(image_grid_util, 1e-4)
        assert image_grid.pixel_scale == 2.0
        assert (image_grid.mask.masked_grid_index_to_pixel == msk.masked_grid_index_to_pixel).all()

    def test__image_grid_unlensed_grid_properties_compare_to_array_util(self):
        msk = np.array([[True, True, False, False],
                        [True, False, True, True],
                        [True, True, False, False]])
        msk = mask.Mask(array=msk, pixel_scale=2.0)

        image_grid = mask.ImageGrid(arr=np.array([[1.0, 1.0], [1.0, 1.0]]), mask=msk)

        image_grid_util = grid_util.image_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=msk,
                                                                                           pixel_scales=(2.0, 2.0))

        assert type(image_grid) == mask.ImageGrid
        assert image_grid.unlensed_grid == pytest.approx(image_grid_util, 1e-4)

        image_grid_util = grid_util.image_grid_1d_from_shape_pixel_scales_and_origin(shape=(3, 4), pixel_scales=(2.0, 2.0))

        assert type(image_grid) == mask.ImageGrid
        assert image_grid.unlensed_unmasked_grid == pytest.approx(image_grid_util, 1e-4)

    def test__image_grid_from_shape_and_pixel_scale__compare_to_array_util(self):
        msk = np.array([[False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False]])
        msk = mask.Mask(array=msk, pixel_scale=2.0)

        image_grid_util = grid_util.image_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=msk,
                                                                                           pixel_scales=(2.0, 2.0))

        image_grid = mask.ImageGrid.from_shape_and_pixel_scale(shape=(3, 4), pixel_scale=2.0)

        assert type(image_grid) == mask.ImageGrid
        assert image_grid == pytest.approx(image_grid_util, 1e-4)
        assert image_grid.pixel_scale == 2.0
        assert (image_grid.mask.masked_grid_index_to_pixel == msk.masked_grid_index_to_pixel).all()

    def test__blurring_grid_from_mask__compare_to_array_util(self):
        msk = np.array([[True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True],
                        [True, True, False, True, True, True, False, True, True],
                        [True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True],
                        [True, True, False, True, True, True, False, True, True],
                        [True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True]])
        msk = mask.Mask(array=msk, pixel_scale=2.0)

        blurring_mask_util = mask_util.mask_blurring_from_mask_and_psf_shape(msk, psf_shape=(3, 5))
        blurring_grid_util = grid_util.image_grid_1d_masked_from_mask_pixel_scales_and_origin(blurring_mask_util,
                                                                                              pixel_scales=(2.0, 2.0))

        msk = mask.Mask(msk, pixel_scale=2.0)
        blurring_grid = mask.ImageGrid.blurring_grid_from_mask_and_psf_shape(mask=msk, psf_shape=(3, 5))

        blurring_msk = msk.blurring_mask_for_psf_shape(psf_shape=(3, 5))

        assert type(blurring_grid) == mask.ImageGrid
        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scale == 2.0
        assert (blurring_grid.mask.masked_grid_index_to_pixel == blurring_msk.masked_grid_index_to_pixel).all()

    def test__map_to_2d__compare_to_util(self):
        msk = np.array([[True, True, False, False],
                        [True, False, True, True],
                        [True, True, False, False]])
        array_1d = np.array([1.0, 6.0, 4.0, 5.0, 2.0])
        one_to_two = np.array([[0, 2], [0, 3], [1, 1], [2, 2], [2, 3]])
        array_2d_util = mapping_util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(
            array_1d=array_1d, shape=(3, 4), one_to_two=one_to_two)

        msk = mask.Mask(array=msk, pixel_scale=2.0)
        image_grid = mask.ImageGrid.from_mask(msk)
        array_2d_grid = image_grid.scaled_array_from_array_1d(array_1d)

        assert (array_2d_util == array_2d_grid).all()
        assert array_2d_grid.pixel_scale == 2.0
        assert array_2d_grid.origin == (0.0, 0.0)

    def test__scaled_array_from_array_1d__compare_to_util(self):
        msk = np.array([[True, True, False, False],
                        [True, False, True, True],
                        [True, True, False, False]])
        array_1d = np.array([1.0, 6.0, 4.0, 5.0, 2.0])
        one_to_two = np.array([[0, 2], [0, 3], [1, 1], [2, 2], [2, 3]])
        array_2d_util = mapping_util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(
            array_1d=array_1d, shape=(3, 4), one_to_two=one_to_two)

        msk = mask.Mask(array=msk, pixel_scale=3.0)
        image_grid = mask.ImageGrid.from_mask(msk)
        scaled_array_2d = image_grid.scaled_array_from_array_1d(array_1d)

        assert (scaled_array_2d == array_2d_util).all()
        assert (scaled_array_2d.xticks == np.array([-6.0, -2.0, 2.0, 6.0])).all()
        assert (scaled_array_2d.yticks == np.array([-4.5, -1.5, 1.5, 4.5])).all()
        assert scaled_array_2d.shape_arc_seconds == (9.0, 12.0)
        assert scaled_array_2d.pixel_scale == 3.0
        assert scaled_array_2d.origin == (0.0, 0.0)

    def test__yticks(self):

        sca = mask.ImageGrid(arr=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=None)
        assert sca.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        sca = mask.ImageGrid(arr=np.array([[3.0, 1.0], [-3.0, -1.0]]), mask=None)
        assert sca.yticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        sca = mask.ImageGrid(arr=np.array([[5.0, 3.5], [2.0, -1.0]]), mask=None)
        assert sca.yticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__xticks(self):

        sca = mask.ImageGrid(arr=np.array([[1.0, 1.5], [-1.0, -1.5]]), mask=None)
        assert sca.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        sca = mask.ImageGrid(arr=np.array([[1.0, 3.0], [-1.0, -3.0]]), mask=None)
        assert sca.xticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        sca = mask.ImageGrid(arr=np.array([[3.5, 2.0], [-1.0, 5.0]]), mask=None)
        assert sca.xticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__masked_shape_arcsec(self):

        sca = mask.ImageGrid(arr=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=None)
        assert sca.masked_shape_arcsec == (3.0, 2.0)

        sca = mask.ImageGrid(arr=np.array([[1.5, 1.0], [-1.5, -1.0], [0.1, 0.1]]), mask=None)
        assert sca.masked_shape_arcsec == (3.0, 2.0)

        sca = mask.ImageGrid(arr=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0]]), mask=None)
        assert sca.masked_shape_arcsec == (4.5, 4.0)

        sca = mask.ImageGrid(arr=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0], [7.0, -5.0]]), mask=None)
        assert sca.masked_shape_arcsec == (8.5, 8.0)


class TestSubGrid(object):

    def test_sub_grid(self, sub_grid):

        assert type(sub_grid) == mask.SubGrid
        assert sub_grid.shape == (5, 2)
        assert sub_grid.pixel_scale == 1.0
        assert (sub_grid == np.array([[1, 0], [0, -1], [0, 0], [0, 1], [-1, 0]])).all()

    def test__image_grid_unlensed_grid_properties_compare_to_array_util(self, msk, sub_grid):

        assert type(sub_grid) == mask.SubGrid
        assert sub_grid.unlensed_grid == \
               pytest.approx(np.array([[1, 0], [0, -1], [0, 0], [0, 1], [-1, 0]]), 1e-4)

        sub_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((3, 3), False), pixel_scales=(1.0, 1.0), sub_grid_size=1)

        assert type(sub_grid) == mask.SubGrid
        assert sub_grid.unlensed_unmasked_grid == pytest.approx(sub_grid_util, 1e-4)

        sub_grid = mask.SubGrid.from_mask_and_sub_grid_size(msk, sub_grid_size=2)

        sub_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((3, 3), False), pixel_scales=(1.0, 1.0), sub_grid_size=2)

        assert type(sub_grid) == mask.SubGrid
        assert sub_grid.unlensed_unmasked_grid == pytest.approx(sub_grid_util, 1e-4)

    def test_sub_to_pixel(self, sub_grid):
        assert (sub_grid.sub_to_image == np.array(range(5))).all()

    def test__from_mask(self):
        msk = np.array([[True, True, True],
                        [True, False, False],
                        [True, True, False]])

        sub_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=msk,
                                                                                              pixel_scales=(
                                                                                                     3.0, 3.0),
                                                                                              sub_grid_size=2)

        msk = mask.Mask(msk, pixel_scale=3.0)

        sub_grid = mask.SubGrid.from_mask_and_sub_grid_size(msk, sub_grid_size=2)

        assert sub_grid == pytest.approx(sub_grid_util, 1e-4)

    def test__from_shape_and_pixel_scale(self):
        msk = np.array([[False, False, False],
                        [False, False, False],
                        [False, False, False]])

        sub_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=msk,
                                                                                              pixel_scales=(
                                                                                                     3.0, 3.0),
                                                                                              sub_grid_size=2)

        sub_grid = mask.SubGrid.from_shape_pixel_scale_and_sub_grid_size(shape=(3, 3), pixel_scale=3.0, sub_grid_size=2)

        assert sub_grid == pytest.approx(sub_grid_util, 1e-4)

    def test__map_to_2d__compare_to_util(self):
        msk = np.array([[True, True, False, False],
                        [True, False, True, True],
                        [True, True, False, False]])
        array_1d = np.array([1.0, 6.0, 4.0, 5.0, 2.0])
        one_to_two = np.array([[0, 2], [0, 3], [1, 1], [2, 2], [2, 3]])
        array_2d_util = mapping_util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(
            array_1d=array_1d,
            shape=(3, 4), one_to_two=one_to_two)

        msk = mask.Mask(array=msk, pixel_scale=2.0)
        image_grid = mask.SubGrid.from_mask_and_sub_grid_size(msk, sub_grid_size=2)
        array_2d_grid = image_grid.scaled_array_from_array_1d(array_1d)

        assert (array_2d_util == array_2d_grid).all()
        assert array_2d_grid.pixel_scale == 2.0
        assert array_2d_grid.origin == (0.0, 0.0)

    def test_sub_data_to_image(self, sub_grid):

        assert (sub_grid.sub_data_to_image(np.array(range(5))) == np.array(range(5))).all()

    def test_sub_to_image__compare_to_array_util(self):

        msk = np.array([[True, False, True],
                        [False, False, False],
                        [True, False, False]])

        sub_to_image_util = mapping_util.sub_to_image_from_mask(msk, sub_grid_size=2)

        msk = mask.Mask(msk, pixel_scale=3.0)

        sub_grid = mask.SubGrid.from_mask_and_sub_grid_size(mask=msk, sub_grid_size=2)
        assert sub_grid.sub_grid_size == 2
        assert sub_grid.sub_grid_fraction == (1.0 / 4.0)
        assert (sub_grid.sub_to_image == sub_to_image_util).all()


class TestUnmaskedGrids:
    class TestPaddedImageGridFromShapes:

        def test__3x3_array__psf_size_is_1x1__no_padding(self):
            image_padded_grid = mask.PaddedImageGrid.padded_grid_from_shapes_and_pixel_scale(shape=(3, 3),
                                                                                             psf_shape=(1, 1),
                                                                                             pixel_scale=1.0)

            assert len(image_padded_grid) == 9
            assert image_padded_grid.pixel_scale == 1.0
            assert image_padded_grid.image_shape == (3, 3)
            assert image_padded_grid.padded_shape == (3, 3)

        def test__3x3_image__5x5_psf_size__7x7_image_padded_grid_made(self):
            image_padded_grid = mask.PaddedImageGrid.padded_grid_from_shapes_and_pixel_scale(shape=(3, 3),
                                                                                             psf_shape=(5, 5),
                                                                                             pixel_scale=1.0)

            assert len(image_padded_grid) == 49
            assert image_padded_grid.image_shape == (3, 3)
            assert image_padded_grid.padded_shape == (7, 7)

        def test__3x3_image__7x7_psf_size__9x9_image_padded_grid_made(self):
            image_padded_grid = mask.PaddedImageGrid.padded_grid_from_shapes_and_pixel_scale(shape=(3, 3),
                                                                                             psf_shape=(7, 7),
                                                                                             pixel_scale=1.0)
            assert len(image_padded_grid) == 81
            assert image_padded_grid.image_shape == (3, 3)
            assert image_padded_grid.padded_shape == (9, 9)

        def test__4x3_image__3x3_psf_size__6x5_image_padded_grid_made(self):
            image_padded_grid = mask.PaddedImageGrid.padded_grid_from_shapes_and_pixel_scale(shape=(4, 3),
                                                                                             psf_shape=(3, 3),
                                                                                             pixel_scale=1.0)
            assert len(image_padded_grid) == 30
            assert image_padded_grid.image_shape == (4, 3)
            assert image_padded_grid.padded_shape == (6, 5)

        def test__3x4_image__3x3_psf_size__5x6_image_padded_grid_made(self):
            image_padded_grid = mask.PaddedImageGrid.padded_grid_from_shapes_and_pixel_scale(shape=(3, 4),
                                                                                             psf_shape=(3, 3),
                                                                                             pixel_scale=1.0)

            assert len(image_padded_grid) == 30
            assert image_padded_grid.image_shape == (3, 4)
            assert image_padded_grid.padded_shape == (5, 6)

        def test__4x4_image__3x3_psf_size__6x6_image_padded_grid_made(self):
            image_padded_grid = mask.PaddedImageGrid.padded_grid_from_shapes_and_pixel_scale(shape=(4, 4),
                                                                                             psf_shape=(3, 3),
                                                                                             pixel_scale=1.0)

            assert len(image_padded_grid) == 36
            assert image_padded_grid.image_shape == (4, 4)
            assert image_padded_grid.padded_shape == (6, 6)

        def test__image_padded_grid_coordinates__match_grid_2d_after_padding(self):
            image_padded_grid = mask.PaddedImageGrid.padded_grid_from_shapes_and_pixel_scale(shape=(4, 4),
                                                                                             psf_shape=(3, 3),
                                                                                             pixel_scale=3.0)

            image_padded_grid_util = grid_util.image_grid_1d_masked_from_mask_pixel_scales_and_origin(
                mask=np.full((6, 6), False), pixel_scales=(3.0, 3.0))
            assert (image_padded_grid == image_padded_grid_util).all()
            assert image_padded_grid.image_shape == (4, 4)
            assert image_padded_grid.padded_shape == (6, 6)

            image_padded_grid = mask.PaddedImageGrid.padded_grid_from_shapes_and_pixel_scale(shape=(4, 5),
                                                                                             psf_shape=(3, 3),
                                                                                             pixel_scale=2.0)
            image_padded_grid_util = grid_util.image_grid_1d_masked_from_mask_pixel_scales_and_origin(
                mask=np.full((6, 7), False), pixel_scales=(2.0, 2.0))
            assert (image_padded_grid == image_padded_grid_util).all()
            assert image_padded_grid.image_shape == (4, 5)
            assert image_padded_grid.padded_shape == (6, 7)

            image_padded_grid = mask.PaddedImageGrid.padded_grid_from_shapes_and_pixel_scale(shape=(5, 4),
                                                                                             psf_shape=(3, 3),
                                                                                             pixel_scale=1.0)
            image_padded_grid_util = grid_util.image_grid_1d_masked_from_mask_pixel_scales_and_origin(
                mask=np.full((7, 6), False), pixel_scales=(1.0, 1.0))
            assert (image_padded_grid == image_padded_grid_util).all()
            assert image_padded_grid.image_shape == (5, 4)
            assert image_padded_grid.padded_shape == (7, 6)

            image_padded_grid = mask.PaddedImageGrid.padded_grid_from_shapes_and_pixel_scale(shape=(2, 5),
                                                                                             psf_shape=(5, 5),
                                                                                             pixel_scale=8.0)
            image_padded_grid_util = grid_util.image_grid_1d_masked_from_mask_pixel_scales_and_origin(
                mask=np.full((6, 9), False), pixel_scales=(8.0, 8.0))
            assert (image_padded_grid == image_padded_grid_util).all()
            assert image_padded_grid.image_shape == (2, 5)
            assert image_padded_grid.padded_shape == (6, 9)

    class TestPaddedSubGridFromMask:

        def test__3x3_array__psf_size_is_1x1__no_padding(self):
            msk = mask.Mask(array=np.full((3, 3), False), pixel_scale=1.0)

            sub_padded_grid = mask.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=msk,
                                                                                                   sub_grid_size=3,
                                                                                                   psf_shape=(1, 1))

            assert len(sub_padded_grid) == 9 * 3 ** 2
            assert sub_padded_grid.image_shape == (3, 3)
            assert sub_padded_grid.padded_shape == (3, 3)

        def test__3x3_image__5x5_psf_size__7x7_image_grid_made(self):
            msk = mask.Mask(array=np.full((3, 3), False), pixel_scale=1.0)

            sub_padded_grid = mask.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=msk,
                                                                                                   sub_grid_size=2,
                                                                                                   psf_shape=(5, 5))

            assert len(sub_padded_grid) == 49 * 2 ** 2
            assert sub_padded_grid.image_shape == (3, 3)
            assert sub_padded_grid.padded_shape == (7, 7)

        def test__4x3_image__3x3_psf_size__6x5_image_grid_made(self):
            msk = mask.Mask(array=np.full((4, 3), False), pixel_scale=1.0)

            sub_padded_grid = mask.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=msk,
                                                                                                   sub_grid_size=2,
                                                                                                   psf_shape=(3, 3))

            assert len(sub_padded_grid) == 30 * 2 ** 2
            assert sub_padded_grid.image_shape == (4, 3)
            assert sub_padded_grid.padded_shape == (6, 5)

        def test__3x4_image__3x3_psf_size__5x6_image_grid_made(self):
            msk = mask.Mask(array=np.full((3, 4), False), pixel_scale=1.0)

            sub_padded_grid = mask.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=msk,
                                                                                                   sub_grid_size=2,
                                                                                                   psf_shape=(3, 3))

            assert len(sub_padded_grid) == 30 * 2 ** 2
            assert sub_padded_grid.image_shape == (3, 4)
            assert sub_padded_grid.padded_shape == (5, 6)

        def test__4x4_image__3x3_psf_size__6x6_image_grid_made(self):
            msk = mask.Mask(array=np.full((4, 4), False), pixel_scale=1.0)

            sub_padded_grid = mask.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=msk,
                                                                                                   sub_grid_size=4,
                                                                                                   psf_shape=(3, 3))

            assert len(sub_padded_grid) == 36 * 4 ** 2
            assert sub_padded_grid.image_shape == (4, 4)
            assert sub_padded_grid.padded_shape == (6, 6)

        def test__sub_padded_grid_coordinates__match_grid_2d_after_padding(self):
            msk = mask.Mask(array=np.full((4, 4), False), pixel_scale=3.0)

            sub_padded_grid = mask.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=msk,
                                                                                                   sub_grid_size=3,
                                                                                                   psf_shape=(3, 3))

            sub_padded_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
                mask=np.full((6, 6), False), pixel_scales=(3.0, 3.0), sub_grid_size=3)

            assert sub_padded_grid == pytest.approx(sub_padded_grid_util, 1e-4)

            msk = mask.Mask(array=np.full((4, 5), False), pixel_scale=2.0)

            sub_padded_grid = mask.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=msk,
                                                                                                   sub_grid_size=1,
                                                                                                   psf_shape=(3, 3))

            sub_padded_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
                mask=np.full((6, 7), False), pixel_scales=(2.0, 2.0), sub_grid_size=1)

            assert sub_padded_grid == pytest.approx(sub_padded_grid_util, 1e-4)

            msk = mask.Mask(array=np.full((5, 4), False), pixel_scale=2.0)

            sub_padded_grid = mask.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=msk,
                                                                                                   sub_grid_size=2,
                                                                                                   psf_shape=(3, 3))

            sub_padded_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
                mask=np.full((7, 6), False), pixel_scales=(2.0, 2.0), sub_grid_size=2)

            assert sub_padded_grid == pytest.approx(sub_padded_grid_util, 1e-4)

            msk = mask.Mask(array=np.full((2, 5), False), pixel_scale=8.0)

            sub_padded_grid = mask.PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=msk,
                                                                                                   sub_grid_size=4,
                                                                                                   psf_shape=(5, 5))

            sub_padded_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
                mask=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_grid_size=4)

            assert sub_padded_grid == pytest.approx(sub_padded_grid_util, 1e-4)

    class TestConvolve:

        def test__convolve_1d_mapper_array_with_psf_and_trim_to_original_size(self):
            msk = mask.Mask(array=np.full((4, 4), False), pixel_scale=1.0)

            image_padded_grid = mask.PaddedImageGrid(arr=np.empty((0)), mask=msk, image_shape=(2, 2))

            array_1d = np.array([0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0])

            kernel = np.array([[0.0, 1.0, 0.0],
                               [1.0, 2.0, 1.0],
                               [0.0, 1.0, 0.0]])

            psf = image.PSF(array=kernel, pixel_scale=1.0)

            blurred_array_1d = image_padded_grid.convolve_array_1d_with_psf(array_1d, psf)

            assert (blurred_array_1d == np.array([0.0, 0.0, 0.0, 0.0,
                                                  0.0, 1.0, 0.0, 0.0,
                                                  1.0, 2.0, 1.0, 0.0,
                                                  0.0, 1.0, 0.0, 0.0])).all()

        def test__same_as_above_but_different_quantities(self):
            msk = mask.Mask(array=np.full((5, 4), False), pixel_scale=1.0)

            image_padded_grid = mask.PaddedImageGrid(arr=np.empty((0)), mask=msk, image_shape=(3, 2))

            array_1d = np.array([0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0,
                                 1.0, 0.0, 0.0, 0.0])

            kernel = np.array([[1.0, 1.0, 4.0],
                               [1.0, 3.0, 1.0],
                               [1.0, 1.0, 1.0]])

            psf = image.PSF(array=kernel, pixel_scale=1.0)

            blurred_array_1d = image_padded_grid.convolve_array_1d_with_psf(array_1d, psf)

            assert (blurred_array_1d == np.array([0.0, 0.0, 0.0, 0.0,
                                                  1.0, 1.0, 4.0, 0.0,
                                                  1.0, 3.0, 1.0, 0.0,
                                                  2.0, 5.0, 1.0, 0.0,
                                                  3.0, 1.0, 0.0, 0.0])).all()

    class TestMapUnmaskedArrays:

        def test__map_to_2d_keep_padded__4x4_from_1d(self):
            msk = mask.Mask(array=np.full((4, 4), False), pixel_scale=1.0)

            image_padded_grid = mask.PaddedImageGrid(arr=np.empty((0)), mask=msk, image_shape=(2, 2))

            array_1d = np.array([6.0, 7.0, 9.0, 10.0,
                                 1.0, 2.0, 3.0, 4.0,
                                 5.0, 6.0, 7.0, 8.0,
                                 1.0, 2.0, 3.0, 4.0])
            array_2d = image_padded_grid.map_to_2d_keep_padded(array_1d)

            assert (array_2d == np.array([[6.0, 7.0, 9.0, 10.0],
                                          [1.0, 2.0, 3.0, 4.0],
                                          [5.0, 6.0, 7.0, 8.0],
                                          [1.0, 2.0, 3.0, 4.0]])).all()

        def test__map_to_2d_keep_padded__5x3_from_1d(self):
            msk = mask.Mask(array=np.full((5, 3), False), pixel_scale=1.0)

            image_padded_grid = mask.PaddedImageGrid(arr=np.empty((0)), mask=msk, image_shape=(3, 1))

            array_1d = np.array([1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0,
                                 7.0, 8.0, 9.0,
                                 1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0])
            array_2d = image_padded_grid.map_to_2d_keep_padded(array_1d)

            assert (array_2d == np.array([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0],
                                          [7.0, 8.0, 9.0],
                                          [1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]])).all()

        def test__map_to_2d_keep_padded__3x5__from_1d(self):
            msk = mask.Mask(array=np.full((3, 5), False), pixel_scale=1.0)

            image_padded_grid = mask.PaddedImageGrid(arr=np.empty((0)), mask=msk, image_shape=(1, 3))

            array_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0,
                                 6.0, 7.0, 8.0, 9.0, 1.0,
                                 2.0, 3.0, 4.0, 5.0, 6.0])
            array_2d = image_padded_grid.map_to_2d_keep_padded(array_1d)

            assert (array_2d == np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                                          [6.0, 7.0, 8.0, 9.0, 1.0],
                                          [2.0, 3.0, 4.0, 5.0, 6.0]])).all()

        def test__map_to_2d_and_trim__4x4_to_2x2__from_1d(self):
            msk = mask.Mask(array=np.full((4, 4), False), pixel_scale=1.0)

            image_padded_grid = mask.PaddedImageGrid(arr=np.empty((0)), mask=msk, image_shape=(2, 2))

            array_1d = np.array([1.0, 2.0, 3.0, 4.0,
                                 5.0, 6.0, 7.0, 8.0,
                                 9.0, 1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0, 7.0])
            array_2d = image_padded_grid.map_to_2d(padded_array_1d=array_1d)

            assert (array_2d == np.array([[6.0, 7.0],
                                          [1.0, 2.0]])).all()

        def test__map_to_2d_and_trim__5x3_to_3x1__from_1d(self):
            msk = mask.Mask(array=np.full((5, 3), False), pixel_scale=1.0)

            image_padded_grid = mask.PaddedImageGrid(arr=np.empty((0)), mask=msk, image_shape=(3, 1))

            array_1d = np.array([1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0,
                                 7.0, 8.0, 9.0,
                                 1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0])
            array_2d = image_padded_grid.map_to_2d(array_1d)

            assert (array_2d == np.array([[5.0],
                                          [8.0],
                                          [2.0]])).all()

        def test__map_to_2d_and_trim__3x5_to_1x3__from_1d(self):
            msk = mask.Mask(array=np.full((3, 5), False), pixel_scale=1.0)

            image_padded_grid = mask.PaddedImageGrid(arr=np.empty((0)), mask=msk, image_shape=(1, 3))

            array_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0,
                                 6.0, 7.0, 8.0, 9.0, 1.0,
                                 2.0, 3.0, 4.0, 5.0, 6.0])
            array_2d = image_padded_grid.map_to_2d(array_1d)

            assert (array_2d == np.array([[7.0, 8.0, 9.0]])).all()


class TestImagingGrids(object):

    def test__grids(self, imaging_grids):
        assert (imaging_grids.image == np.array([[0., 0.]])).all()
        np.testing.assert_almost_equal(imaging_grids.sub, np.array([[0.16666667, -0.16666667],
                                                                    [0.16666667, 0.16666667],
                                                                    [-0.16666667, -0.16666667],
                                                                    [-0.16666667, 0.16666667]]))
        assert (imaging_grids.blurring == np.array([[1., -1.],
                                                    [1., 0.],
                                                    [1., 1.],
                                                    [0., -1.],
                                                    [0., 1.],
                                                    [-1., -1.],
                                                    [-1., 0.],
                                                    [-1., 1.]])).all()
        assert (imaging_grids.pix == np.array([[0.0, 0.0]])).all()

    def test__from_shape_and_pixel_scale(self):

        ma = mask.Mask(np.array([[False, False, False],
                                 [False, False, False],
                                 [False, False, False]]), pixel_scale=2.0)

        imaging_grids_mask = mask.ImagingGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=2,
                                                                                           psf_shape=(1, 1))

        imaging_grids_shape = mask.ImagingGrids.from_shape_and_pixel_scale(shape=(3, 3), pixel_scale=2.0,
                                                                           sub_grid_size=2)

        assert (imaging_grids_mask.image == imaging_grids_shape.image).all()
        assert (imaging_grids_mask.sub == imaging_grids_shape.sub).all()
        assert (imaging_grids_mask.pix == np.array([[0.0, 0.0]])).all()

    def test__padded_grids(self):
        msk = np.array([[False, False],
                        [False, False]])

        msk = mask.Mask(msk, pixel_scale=1.0)

        padded_grids = mask.ImagingGrids.padded_grids_from_mask_sub_grid_size_and_psf_shape(msk, sub_grid_size=2,
                                                                                            psf_shape=(3, 3))

        image_padded_grid_util = grid_util.image_grid_1d_masked_from_mask_pixel_scales_and_origin(
            mask=np.full((4, 4), False),
            pixel_scales=(1.0, 1.0))

        sub_padded_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((4, 4), False),
            pixel_scales=(1.0, 1.0), sub_grid_size=2)

        assert (padded_grids.image == image_padded_grid_util).all()
        assert padded_grids.image.image_shape == (2, 2)
        assert padded_grids.image.padded_shape == (4, 4)

        assert padded_grids.sub == pytest.approx(sub_padded_grid_util, 1e-4)
        assert padded_grids.sub.image_shape == (2, 2)
        assert padded_grids.sub.padded_shape == (4, 4)

        assert (padded_grids.blurring == np.array([0.0, 0.0])).all()

        assert (padded_grids.pix == np.array([[0.0, 0.0]])).all()

    def test__for_simulation(self):
        padded_grids = mask.ImagingGrids.grids_for_simulation(shape=(2, 2), pixel_scale=1.0, sub_grid_size=2,
                                                              psf_shape=(3, 3))

        image_padded_grid_util = grid_util.image_grid_1d_masked_from_mask_pixel_scales_and_origin(
            mask=np.full((4, 4), False),
            pixel_scales=(1.0, 1.0))

        sub_padded_grid_util = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full((4, 4), False),
            pixel_scales=(1.0, 1.0), sub_grid_size=2)

        assert (padded_grids.image == image_padded_grid_util).all()
        assert padded_grids.image.image_shape == (2, 2)
        assert padded_grids.image.padded_shape == (4, 4)

        assert padded_grids.sub == pytest.approx(sub_padded_grid_util, 1e-4)
        assert padded_grids.sub.image_shape == (2, 2)
        assert padded_grids.sub.padded_shape == (4, 4)

        assert (padded_grids.blurring == np.array([0.0, 0.0])).all()

        assert (padded_grids.pix == np.array([[0.0, 0.0]])).all()

    def test__apply_function_retains_attributes(self, imaging_grids):

        imaging_grids.pix = imaging_grids.image

        def add_one(coords):
            return np.add(1, coords)

        assert isinstance(imaging_grids.image, mask.ImageGrid)
        assert imaging_grids.image.mask is not None

        assert isinstance(imaging_grids.sub, mask.SubGrid)
        assert imaging_grids.sub.mask is not None
        assert imaging_grids.sub.sub_grid_size is not None
        assert imaging_grids.sub.sub_grid_length is not None
        assert imaging_grids.sub.sub_grid_fraction is not None

        new_collection = imaging_grids.apply_function(add_one)

        assert new_collection.image.mask is not None
        assert new_collection.sub.mask is not None
        assert new_collection.sub.sub_grid_size is not None
        assert new_collection.sub.sub_grid_length is not None
        assert new_collection.sub.sub_grid_fraction is not None

        assert isinstance(imaging_grids.pix, mask.ImageGrid)
        assert imaging_grids.pix.mask is not None

    def test__apply_function(self, imaging_grids):

        imaging_grids.pix = imaging_grids.image

        def add_one(coords):
            return np.add(1, coords)

        new_collection = imaging_grids.apply_function(add_one)
        assert isinstance(new_collection, mask.ImagingGrids)
        assert (new_collection.image == np.add(1, np.array([[0., 0.]]))).all()
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

    def test__map_function(self, imaging_grids):

        imaging_grids.pix = imaging_grids.image

        def add_number(coords, number):
            return np.add(coords, number)

        new_collection = imaging_grids.map_function(add_number, [1, 2, 3, 1])

        assert isinstance(new_collection, mask.ImagingGrids)
        assert (new_collection.image == np.add(1, np.array([[0., 0.]]))).all()
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

        print(new_collection.pix)
        assert (new_collection.pix == np.add(1, np.array([[0., 0.]]))).all()

    def test__imaging_grids_with_pix_grid(self, imaging_grids):

        imaging_grids = imaging_grids.imaging_grids_with_pix_grid(pix=np.array([[5.0, 5.0], [6.0, 7.0]]))

        assert (imaging_grids.image == np.array([[0., 0.]])).all()
        np.testing.assert_almost_equal(imaging_grids.sub, np.array([[0.16666667, -0.16666667],
                                                                    [0.16666667, 0.16666667],
                                                                    [-0.16666667, -0.16666667],
                                                                    [-0.16666667, 0.16666667]]))
        assert (imaging_grids.blurring == np.array([[1., -1.],
                                                    [1., 0.],
                                                    [1., 1.],
                                                    [0., -1.],
                                                    [0., 1.],
                                                    [-1., -1.],
                                                    [-1., 0.],
                                                    [-1., 1.]])).all()
        assert (imaging_grids.pix == np.array([[5.0, 5.0], [6.0, 7.0]])).all()


class TestImageGridBorder(object):

    class TestFromMask:

        def test__simple_mask_border_pixels_is_border(self):

            msk = np.array([[False, False, False, False, False, False, False, True],
                             [False, True, True, True, True, True, False, True],
                             [False, True, False, False, False, True, False, True],
                             [False, True, False, True, False, True, False, True],
                             [False, True, False, False, False, True, False, True],
                             [False, True, True, True, True, True, False, True],
                             [False, False, False, False, False, False, False, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border = mask.ImageGridBorder.from_mask(msk)

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
            image_grid = np.asarray(grid)
            sub_grid = np.asarray(grid)
            sub_grid[35,0] = 0.5
            sub_grid[35,1] = 0.3
            grids = mask.ImagingGrids(image=image_grid, sub=sub_grid, blurring=None)

            border = mask.ImageGridBorder(arr=np.arange(32))
            relocated_grids = border.relocated_grids_from_grids(grids)

            assert relocated_grids.image[0:32] == pytest.approx(np.asarray(grid_circle)[0:32], 1e-3)
            assert relocated_grids.image[32] == pytest.approx(np.array([0.1, 0.0]), 1e-3)
            assert relocated_grids.image[33] == pytest.approx(np.array([-0.2, -0.3]), 1e-3)
            assert relocated_grids.image[34] == pytest.approx(np.array([0.5, 0.4]), 1e-3)
            assert relocated_grids.image[35] == pytest.approx(np.array([0.7, -0.1]), 1e-3)

            assert relocated_grids.sub[0:32] == pytest.approx(np.asarray(grid_circle)[0:32], 1e-3)
            assert relocated_grids.sub[32] == pytest.approx(np.array([0.1, 0.0]), 1e-3)
            assert relocated_grids.sub[33] == pytest.approx(np.array([-0.2, -0.3]), 1e-3)
            assert relocated_grids.sub[34] == pytest.approx(np.array([0.5, 0.4]), 1e-3)
            assert relocated_grids.sub[35] == pytest.approx(np.array([0.5, 0.3]), 1e-3)

        def test__8_points_with_border_as_circle__points_go_to_circle_edge(self):

            grid = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0,-1.0],
                             [0.7071, 0.7071], [0.7071, -0.7071],
                             [-0.7071, 0.7071], [-0.7071, -0.7071],
                             [10., 10.], [10., -10.], [-10., 10], [-10., -10.]])
            grids = mask.ImagingGrids(image=grid, sub=grid, blurring=None)
            
            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            border = mask.ImageGridBorder(border_pixels)

            relocated_grids = border.relocated_grids_from_grids(grids)

            assert relocated_grids.image[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.image[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.image[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.image[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.image[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.image[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.image[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.image[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.image[8] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)
            assert relocated_grids.image[9] == pytest.approx(np.array([0.7071, -0.7071]), 1e-3)
            assert relocated_grids.image[10] == pytest.approx(np.array([-0.7071, 0.7071]), 1e-3)
            assert relocated_grids.image[11] == pytest.approx(np.array([-0.7071, -0.7071]), 1e-3)
            
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
            grids = mask.ImagingGrids(image=grid, sub=grid, blurring=None)
            
            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            border = mask.ImageGridBorder(border_pixels)

            relocated_grids = border.relocated_grids_from_grids(grids)

            assert relocated_grids.image[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.image[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.image[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.image[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.image[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.image[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.image[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.image[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.image[8] == pytest.approx(np.array([1.0 + 0.7071, 1.0 + 0.7071]), 1e-3)
            assert relocated_grids.image[9] == pytest.approx(np.array([1.0 + 0.7071, 1.0 - 0.7071]), 1e-3)
            assert relocated_grids.image[10] == pytest.approx(np.array([1.0 - 0.7071, 1.0 + 0.7071]), 1e-3)
            assert relocated_grids.image[11] == pytest.approx(np.array([1.0 - 0.7071, 1.0 - 0.7071]), 1e-3)

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

            grid = np.array([[0.0, -1.0], [-1.0, 0.0], [-2.0, -1.0], [-1.0,-2.0],
                             [-1.0 + 0.7071, -1.0 + 0.7071], [-1.0 + 0.7071, -1.0 - 0.7071],
                             [-1.0 - 0.7071, -1.0 + 0.7071], [-1.0 - 0.7071, -1.0 - 0.7071],
                             [9., 9.], [9., -11.], [-11., 9], [-11., -11.]])
            grids = mask.ImagingGrids(image=grid, sub=grid, blurring=None)
            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            border = mask.ImageGridBorder(border_pixels)

            relocated_grids = border.relocated_grids_from_grids(grids)

            assert relocated_grids.image[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.image[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.image[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.image[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.image[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.image[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.image[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.image[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.image[8] == pytest.approx(np.array([-1.0 + 0.7071, -1.0 + 0.7071]), 1e-3)
            assert relocated_grids.image[9] == pytest.approx(np.array([-1.0 + 0.7071, -1.0 - 0.7071]), 1e-3)
            assert relocated_grids.image[10] == pytest.approx(np.array([-1.0 - 0.7071, -1.0 + 0.7071]), 1e-3)
            assert relocated_grids.image[11] == pytest.approx(np.array([-1.0 - 0.7071, -1.0 - 0.7071]), 1e-3)

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

            grids = mask.ImagingGrids(image=grid, sub=grid, blurring=None)

            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            border = mask.ImageGridBorder(border_pixels)

            relocated_grids = border.relocated_grids_from_grids(grids)

            assert relocated_grids.image[8] == pytest.approx(np.array([0.02, 0.95]), 1e-4)
            assert relocated_grids.sub[8] == pytest.approx(np.array([0.02, 0.95]), 1e-4)