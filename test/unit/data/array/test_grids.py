import numpy as np
import pytest

from autolens import exc
from autolens.data import ccd
from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.data.array.util import grid_util, mapping_util, mask_util
from autolens.model.profiles import mass_profiles as mp


@pytest.fixture(name="grid")
def make_grid():

    mask = msk.Mask(
        np.array([[True, False, True], [False, False, False], [True, False, True]]),
        pixel_scale=1.0,
    )

    return grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=1)


@pytest.fixture(name="grid_stack")
def make_grid_stack():

    centre_mask = msk.Mask(
        np.array([[True, True, True], [True, False, True], [True, True, True]]),
        pixel_scale=1.0,
    )

    return grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
        mask=centre_mask, sub_grid_size=2, psf_shape=(3, 3)
    )


class TestGrid:
    def test__from_mask__compare_to_array_util(self):

        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = msk.Mask(array=mask, pixel_scale=2.0)

        grid_via_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=mask, sub_grid_size=1, pixel_scales=(2.0, 2.0)
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=1)

        assert type(grid) == grids.Grid
        assert grid == pytest.approx(grid_via_util, 1e-4)
        assert grid.pixel_scale == 2.0
        assert (grid.mask.one_to_two == mask.one_to_two).all()
        assert grid.interpolator == None

        grid_2d = mask.grid_2d_from_grid_1d(grid_1d=grid)

        assert (grid.in_2d == grid_2d).all()

        mask = np.array([[True, True, True], [True, False, False], [True, True, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        grid_via_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=mask, pixel_scales=(3.0, 3.0), sub_grid_size=2
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask, sub_grid_size=2)

        assert grid == pytest.approx(grid_via_util, 1e-4)

    def test__grid_unlensed_property__compare_to_grid_util(self):

        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )

        mask = msk.Mask(array=mask, pixel_scale=2.0)

        grid = grids.Grid(
            arr=np.array([[1.0, 1.0], [1.0, 1.0]]), mask=mask, sub_grid_size=1
        )

        grid_via_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=mask, sub_grid_size=1, pixel_scales=(2.0, 2.0)
        )

        assert grid.unlensed_grid_1d == pytest.approx(grid_via_util, 1e-4)

        grid_via_util = grid_util.grid_1d_from_shape_pixel_scales_sub_grid_size_and_origin(
            shape=(3, 4), sub_grid_size=1, pixel_scales=(2.0, 2.0)
        )

        assert grid.unlensed_unmasked_grid_1d == pytest.approx(grid_via_util, 1e-4)

        mask = msk.Mask(
            np.array([[True, False, True], [False, False, False], [True, False, True]]),
            pixel_scale=1.0,
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask, sub_grid_size=1)

        assert grid.unlensed_grid_1d == pytest.approx(
            np.array([[1, 0], [0, -1], [0, 0], [0, 1], [-1, 0]]), 1e-4
        )

        grid_via_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=np.full((3, 3), False), pixel_scales=(1.0, 1.0), sub_grid_size=1
        )

        assert grid.unlensed_unmasked_grid_1d == pytest.approx(grid_via_util, 1e-4)

        grid = grids.Grid.from_mask_and_sub_grid_size(mask, sub_grid_size=2)

        grid_via_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=np.full((3, 3), False), pixel_scales=(1.0, 1.0), sub_grid_size=2
        )

        assert grid.unlensed_unmasked_grid_1d == pytest.approx(grid_via_util, 1e-4)

    def test__from_shape_and_pixel_scale__compare_to_grid_util(self):

        mask = np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        )
        mask = msk.Mask(array=mask, pixel_scale=2.0)

        grid_via_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=mask, pixel_scales=(2.0, 2.0), sub_grid_size=1
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(3, 4), pixel_scale=2.0, sub_grid_size=1
        )

        assert type(grid) == grids.Grid
        assert grid == pytest.approx(grid_via_util, 1e-4)
        assert grid.pixel_scale == 2.0
        assert (grid.mask.one_to_two == mask.one_to_two).all()

        mask = np.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )

        grid_via_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=mask, pixel_scales=(3.0, 3.0), sub_grid_size=2
        )

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(3, 3), pixel_scale=3.0, sub_grid_size=2
        )

        assert grid == pytest.approx(grid_via_util, 1e-4)

    def test__from_unmasked_grid_2d(self):

        grid_2d = np.array(
            [
                [[2.0, -1.0], [2.0, 0.0], [2.0, 1.0]],
                [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]],
                [[-2.0, -1.0], [-2.0, 0.0], [-2.0, 1.0]],
            ]
        )

        grid = grids.Grid.from_unmasked_grid_2d(grid_2d=grid_2d)

        assert (
            grid
            == np.array(
                [
                    [2.0, -1.0],
                    [2.0, 0.0],
                    [2.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [-2.0, -1.0],
                    [-2.0, 0.0],
                    [-2.0, 1.0],
                ]
            )
        ).all()

        assert (grid.mask == np.full(fill_value=False, shape=(3, 3))).all()

        assert grid.sub_grid_size == 1

    def test__blurring_grid_from_mask__compare_to_array_util(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        mask = msk.Mask(array=mask, pixel_scale=2.0)

        blurring_mask_util = mask_util.blurring_mask_from_mask_and_psf_shape(
            mask=mask, psf_shape=(3, 5)
        )

        blurring_grid_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_grid_size=1
        )

        mask = msk.Mask(array=mask, pixel_scale=2.0)
        blurring_grid = grids.Grid.blurring_grid_from_mask_and_psf_shape(
            mask=mask, psf_shape=(3, 5)
        )

        blurring_mask = mask.blurring_mask_for_psf_shape(psf_shape=(3, 5))

        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scale == 2.0
        assert (blurring_grid.mask.one_to_two == blurring_mask.one_to_two).all()
        assert blurring_grid.sub_grid_size == 1

    def test__new_grid__with_interpolator__returns_grid_with_interpolator(self):

        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = msk.Mask(array=mask, pixel_scale=2.0)

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        grid_with_interp = grid.new_grid_with_interpolator(interp_pixel_scale=1.0)

        assert (grid[:, :] == grid_with_interp[:, :]).all()
        assert grid.mask == grid_with_interp.mask

        interpolator_manual = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=grid, interp_pixel_scale=1.0
        )

        assert (grid.interpolator.vtx == interpolator_manual.vtx).all()
        assert (grid.interpolator.wts == interpolator_manual.wts).all()

    def test__padded_grid_from_psf_shape__matches_grid_2d_after_padding(self):

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(4, 4), pixel_scale=3.0, sub_grid_size=1
        )

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(3, 3))

        padded_grid_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=np.full((6, 6), False), pixel_scales=(3.0, 3.0), sub_grid_size=1
        )

        assert padded_grid.shape == (36, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 6))).all()
        assert (padded_grid == padded_grid_util).all()
        assert padded_grid.interpolator is None

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(4, 5), pixel_scale=2.0
        )

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(3, 3))

        padded_grid_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=np.full((6, 7), False), pixel_scales=(2.0, 2.0), sub_grid_size=1
        )

        assert padded_grid.shape == (42, 2)
        assert (padded_grid == padded_grid_util).all()

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(5, 4), pixel_scale=1.0
        )

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(3, 3))

        padded_grid_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=np.full((7, 6), False), pixel_scales=(1.0, 1.0), sub_grid_size=1
        )

        assert padded_grid.shape == (42, 2)
        assert (padded_grid == padded_grid_util).all()

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(5, 5), pixel_scale=8.0
        )

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(2, 5))

        padded_grid_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_grid_size=1
        )

        assert padded_grid.shape == (54, 2)
        assert (padded_grid == padded_grid_util).all()

        mask = msk.Mask(array=np.full((5, 4), False), pixel_scale=2.0)

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(3, 3))

        padded_grid_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=np.full((7, 6), False), pixel_scales=(2.0, 2.0), sub_grid_size=2
        )

        assert padded_grid.shape == (168, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(7, 6))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)
        assert padded_grid.interpolator is None

        mask = msk.Mask(array=np.full((2, 5), False), pixel_scale=8.0)

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=4)

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(5, 5))

        padded_grid_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_grid_size=4
        )

        assert padded_grid.shape == (864, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 9))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)

    def test__padded_grid_from_psf_shape__has_interpolator_grid_if_had_one_before(self):

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(4, 4), pixel_scale=3.0
        )

        grid = grid.new_grid_with_interpolator(interp_pixel_scale=0.1)

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(3, 3))

        assert padded_grid.interpolator is not None
        assert padded_grid.interpolator.interp_pixel_scale == 0.1

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(
            shape=(6, 6), pixel_scale=3.0
        )

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=padded_grid, interp_pixel_scale=0.1
        )

        assert (padded_grid.interpolator.vtx == interpolator.vtx).all()
        assert (padded_grid.interpolator.wts == interpolator.wts).all()

        mask = msk.Mask(array=np.full((5, 4), False), pixel_scale=2.0)

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        grid = grid.new_grid_with_interpolator(interp_pixel_scale=0.1)

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(3, 3))

        assert padded_grid.interpolator is not None
        assert padded_grid.interpolator.interp_pixel_scale == 0.1

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(
            shape=(7, 6), pixel_scale=2.0
        )

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=padded_grid, interp_pixel_scale=0.1
        )

        assert (padded_grid.interpolator.vtx == interpolator.vtx).all()
        assert (padded_grid.interpolator.wts == interpolator.wts).all()

    def test__trimmed_array_2d_from_padded_array_1d_and_image_shape(self):

        mask = msk.Mask(array=np.full((4, 4), False), pixel_scale=1.0)

        grid = grids.Grid(arr=np.empty((0)), mask=mask)

        array_1d = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
            ]
        )

        array_2d = grid.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(2, 2)
        )

        assert (array_2d == np.array([[6.0, 7.0], [1.0, 2.0]])).all()

        mask = msk.Mask(array=np.full((5, 3), False), pixel_scale=1.0)

        grid = grids.Grid(arr=np.empty((0)), mask=mask)

        array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

        array_2d = grid.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(3, 1)
        )

        assert (array_2d == np.array([[5.0], [8.0], [2.0]])).all()

        mask = msk.Mask(array=np.full((3, 5), False), pixel_scale=1.0)

        grid = grids.Grid(arr=np.empty((0)), mask=mask)

        array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

        array_2d = grid.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(1, 3)
        )

        assert (array_2d == np.array([[7.0, 8.0, 9.0]])).all()

    def test_sub_to_pixel(self, grid):
        assert (grid.sub_to_regular == np.array(range(5))).all()

    def test__sub_mask__is_mask_at_sub_grid_resolution(self):

        mask = np.array([[False, True], [False, False]])

        mask = msk.Mask(array=mask, pixel_scale=3.0)

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        assert (
            grid.sub_mask
            == np.array(
                [
                    [False, False, True, True],
                    [False, False, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ]
            )
        ).all()

        mask = np.array([[False, False, True], [False, True, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        assert (
            grid.sub_mask
            == np.array(
                [
                    [False, False, False, False, True, True],
                    [False, False, False, False, True, True],
                    [False, False, True, True, False, False],
                    [False, False, True, True, False, False],
                ]
            )
        ).all()

    def test__masked_shape_arcsec(self):

        grid = grids.Grid(arr=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=None)
        assert grid.masked_shape_arcsec == (3.0, 2.0)

        grid = grids.Grid(
            arr=np.array([[1.5, 1.0], [-1.5, -1.0], [0.1, 0.1]]), mask=None
        )
        assert grid.masked_shape_arcsec == (3.0, 2.0)

        grid = grids.Grid(
            arr=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0]]), mask=None
        )
        assert grid.masked_shape_arcsec == (4.5, 4.0)

        grid = grids.Grid(
            arr=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0], [7.0, -5.0]]), mask=None
        )
        assert grid.masked_shape_arcsec == (8.5, 8.0)

    def test__yticks(self):

        grid = grids.Grid(arr=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=None)
        assert grid.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        grid = grids.Grid(arr=np.array([[3.0, 1.0], [-3.0, -1.0]]), mask=None)
        assert grid.yticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        grid = grids.Grid(arr=np.array([[5.0, 3.5], [2.0, -1.0]]), mask=None)
        assert grid.yticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__xticks(self):

        grid = grids.Grid(arr=np.array([[1.0, 1.5], [-1.0, -1.5]]), mask=None)
        assert grid.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        grid = grids.Grid(arr=np.array([[1.0, 3.0], [-1.0, -3.0]]), mask=None)
        assert grid.xticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        grid = grids.Grid(arr=np.array([[3.5, 2.0], [-1.0, 5.0]]), mask=None)
        assert grid.xticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)


class TestMappings:
    def test__scaled_array_from_array_1d__compare_to_mask(self):

        array_1d = np.array([1.0, 6.0, 4.0, 5.0, 2.0])

        mask = msk.Mask(
            array=np.array(
                [
                    [True, True, False, False],
                    [True, False, True, True],
                    [True, True, False, False],
                ]
            ),
            pixel_scale=3.0,
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        array_2d = grid.array_2d_from_array_1d(array_1d=array_1d)

        array_2d_mask = mask.array_2d_from_array_1d(array_1d=array_1d)

        assert (array_2d == array_2d_mask).all()

        scaled_array_2d = grid.scaled_array_2d_from_array_1d(array_1d=array_1d)

        assert (scaled_array_2d == array_2d_mask).all()
        assert (scaled_array_2d.xticks == np.array([-6.0, -2.0, 2.0, 6.0])).all()
        assert (scaled_array_2d.yticks == np.array([-4.5, -1.5, 1.5, 4.5])).all()
        assert scaled_array_2d.shape_arcsec == (9.0, 12.0)
        assert scaled_array_2d.pixel_scale == 3.0
        assert scaled_array_2d.origin == (0.0, 0.0)

    def test__array_1d_from_array_2d__compare_to_mask(self):
        array_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

        mask = msk.Mask(
            array=np.array(
                [
                    [True, False, True, True],
                    [False, False, False, True],
                    [True, False, True, False],
                ]
            ),
            pixel_scale=2.0,
        )

        array_1d_mask = mask.array_1d_from_array_2d(array_2d=array_2d)

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)
        array_1d = grid.array_1d_from_array_2d(array_2d=array_2d)

        assert (array_1d_mask == array_1d).all()

    def test__grid_2d_from_grid_1d__compare_to_util(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )

        grid_1d = np.array([[1.0, 1.0], [6.0, 6.0], [4.0, 4.0], [5.0, 5.0], [2.0, 2.0]])

        mask = msk.Mask(array=mask, pixel_scale=2.0)

        grid_2d_mask = mask.grid_2d_from_grid_1d(grid_1d=grid_1d)

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)
        grid_2d = grid.grid_2d_from_grid_1d(grid_1d=grid_1d)

        assert (grid_2d_mask == grid_2d).all()

    def test__grid_1d_from_grid_2d__compare_to_mask(self):
        grid_2d = np.array(
            [
                [[1, 1], [2, 2], [3, 3], [4, 4]],
                [[5, 5], [6, 6], [7, 7], [8, 8]],
                [[9, 9], [10, 10], [11, 11], [12, 12]],
            ]
        )

        mask = msk.Mask(
            array=np.array(
                [
                    [True, False, True, True],
                    [False, False, False, True],
                    [True, False, True, False],
                ]
            ),
            pixel_scale=2.0,
        )

        grid_1d_mask = mask.grid_1d_from_grid_2d(grid_2d=grid_2d)

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)
        grid_1d = grid.grid_1d_from_grid_2d(grid_2d=grid_2d)

        assert (grid_1d_mask == grid_1d).all()

    def test__sub_array_2d_from_sub_array_1d__compare_to_mask(self):
        mask = msk.Mask(
            array=np.array([[False, True], [False, False]]), pixel_scale=3.0
        )

        sub_array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        )

        sub_array_2d_mask = mask.sub_array_2d_from_sub_array_1d_and_sub_grid_size(
            sub_array_1d=sub_array_1d, sub_grid_size=2
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        sub_array_2d = grid.sub_array_2d_from_sub_array_1d(sub_array_1d=sub_array_1d)

        assert (sub_array_2d_mask == sub_array_2d).all()

    def test__scaled_sub_array_2d_with_sub_dimensions_from_sub_array_1d__compare_to_mask(
        self
    ):
        mask = msk.Mask(
            array=np.array([[False, False, True], [False, True, False]]),
            pixel_scale=3.0,
        )

        sub_array_1d = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                3.0,
                4.0,
                4.0,
                4.0,
                4.0,
            ]
        )

        scaled_array_2d_mask = mask.scaled_array_2d_with_sub_dimensions_from_sub_array_1d_and_sub_grid_size(
            sub_array_1d=sub_array_1d, sub_grid_size=2
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        scaled_sub_array_2d = grid.scaled_array_2d_with_sub_dimensions_from_sub_array_1d(
            sub_array_1d=sub_array_1d
        )

        assert (scaled_array_2d_mask == scaled_sub_array_2d).all()

        assert scaled_array_2d_mask.pixel_scales == scaled_sub_array_2d.pixel_scales
        assert scaled_array_2d_mask.origin == scaled_sub_array_2d.origin

        assert scaled_sub_array_2d.pixel_scales == (1.5, 1.5)
        assert scaled_sub_array_2d.origin == (0.0, 0.0)

    def test__scaled_array_2d_from_sub_array_1d_by_binning_up__compare_to_mask(self):
        mask = np.array([[False, False, True], [False, True, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_array_1d = np.array(
            [
                1.0,
                10.0,
                2.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                3.0,
                4.0,
                0.0,
                0.0,
                4.0,
            ]
        )

        scaled_array_2d_binned_mask = mask.scaled_array_2d_binned_from_sub_array_1d_and_sub_grid_size(
            sub_array_1d=sub_array_1d, sub_grid_size=2
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        scaled_array_2d_binned = grid.scaled_array_2d_binned_from_sub_array_1d(
            sub_array_1d=sub_array_1d
        )

        assert (scaled_array_2d_binned_mask == scaled_array_2d_binned).all()

        assert (
            scaled_array_2d_binned_mask.pixel_scales
            == scaled_array_2d_binned.pixel_scales
        )
        assert scaled_array_2d_binned_mask.origin == scaled_array_2d_binned.origin

        assert scaled_array_2d_binned.pixel_scales == (3.0, 3.0)
        assert scaled_array_2d_binned.origin == (0.0, 0.0)

    def test__sub_array_1d_from_sub_array_2d__compare_to_mask(self):
        sub_array_2d = np.array(
            [
                [1.0, 1.0, 2.0, 2.0, 3.0, 10.0],
                [1.0, 1.0, 2.0, 2.0, 3.0, 10.0],
                [3.0, 3.0, 8.0, 1.0, 4.0, 4.0],
                [3.0, 3.0, 7.0, 2.0, 4.0, 4.0],
            ]
        )

        mask = msk.Mask(
            array=np.array([[False, False, False], [True, True, False]]),
            pixel_scale=2.0,
        )

        sub_array_1d_mask = mask.sub_array_1d_with_sub_dimensions_from_sub_array_2d_and_sub_grid_size(
            sub_array_2d=sub_array_2d, sub_grid_size=2
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        sub_array_1d = grid.sub_array_1d_from_sub_array_2d(sub_array_2d=sub_array_2d)

        assert (sub_array_1d == sub_array_1d_mask).all()

    def test__sub_grid_1d_with_sub_dimensions_from_sub_grid_2d__compare_to_mask(self):
        mask = msk.Mask(
            array=np.array([[False, True], [False, False]]), pixel_scale=3.0
        )

        sub_grid_2d = np.array(
            [
                [[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-1.0, -1.0]],
                [[3.0, 3.0], [4.0, 4.0], [-1.0, -1.0], [-1.0, -1.0]],
                [[5.0, 5.0], [6.0, 6.0], [9.0, 9.0], [10.0, 10.0]],
                [[7.0, 7.0], [8.0, 8.0], [11.0, 11.0], [12.0, 12.0]],
            ]
        )

        sub_grid_1d_mask = mask.sub_grid_1d_with_sub_dimensions_from_sub_grid_2d_and_sub_grid_size(
            sub_grid_2d=sub_grid_2d, sub_grid_size=2
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        sub_grid_1d = grid.sub_grid_1d_with_sub_dimensions_from_sub_grid_2d(
            sub_grid_2d=sub_grid_2d
        )

        assert (sub_grid_1d_mask == sub_grid_1d).all()

    def test__sub_grid_2d_with_sub_dimensions_from_sub_grid_1d__compare_to_mask(self):
        mask = msk.Mask(
            array=np.array([[False, True], [False, False]]), pixel_scale=3.0
        )

        sub_grid_1d = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [9.0, 9.0],
                [10.0, 10.0],
                [11.0, 11.0],
                [12.0, 12.0],
                [13.0, 13.0],
                [14.0, 14.0],
                [15.0, 15.0],
                [16.0, 16.0],
            ]
        )

        sub_grid_2d_mask = mask.sub_grid_2d_with_sub_dimensions_from_sub_grid_1d_and_sub_grid_size(
            sub_grid_1d=sub_grid_1d, sub_grid_size=2
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        sub_grid_2d = grid.sub_grid_2d_with_sub_dimensions_from_sub_grid_1d(
            sub_grid_1d=sub_grid_1d
        )

        assert (sub_grid_2d_mask == sub_grid_2d).all()

    def test__grid_1d_binned_from_sub_grid_1d__compare_to_mask(self):
        mask = msk.Mask(
            array=np.array([[False, True], [False, False]]), pixel_scale=3.0
        )

        sub_grid_1d = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [9.0, 9.0],
                [10.0, 10.0],
                [11.0, 11.0],
                [12.0, 12.0],
                [13.0, 13.0],
                [14.0, 14.0],
                [15.0, 15.0],
                [16.0, 16.0],
            ]
        )

        grid_1d_binned_mask = mask.grid_1d_binned_from_sub_grid_1d_and_sub_grid_size(
            sub_grid_1d=sub_grid_1d, sub_grid_size=2
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        grid_1d_binned = grid.grid_1d_binned_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

        assert (grid_1d_binned_mask == grid_1d_binned).all()

    def test__grid_2d_binned_from_sub_grid_1d__compare_to_mask(self):
        mask = msk.Mask(
            array=np.array([[False, True], [False, False]]), pixel_scale=3.0
        )

        sub_grid_1d = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [9.0, 9.0],
                [10.0, 10.0],
                [11.0, 11.0],
                [12.0, 12.0],
                [13.0, 13.0],
                [14.0, 14.0],
                [15.0, 15.0],
                [16.0, 16.0],
            ]
        )

        grid_2d_binned_mask = mask.grid_2d_binned_from_sub_grid_1d_and_sub_grid_size(
            sub_grid_1d=sub_grid_1d, sub_grid_size=2
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)

        grid_2d_binned = grid.grid_2d_binned_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

        assert (grid_2d_binned_mask == grid_2d_binned).all()

    def test__map_sub_array_to_1d_and_2d_and_back__returns_original_array(self):
        mask = np.array([[False, False, True], [False, True, False]])

        mask = msk.Mask(mask, pixel_scale=3.0)

        sub_array_1d = np.array(
            [
                1.0,
                10.0,
                2.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                3.0,
                4.0,
                0.0,
                0.0,
                4.0,
            ]
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask, sub_grid_size=2)

        sub_array_2d = grid.sub_array_2d_from_sub_array_1d(sub_array_1d=sub_array_1d)
        sub_array_1d_new = grid.sub_array_1d_from_sub_array_2d(
            sub_array_2d=sub_array_2d
        )

        assert (sub_array_1d == sub_array_1d_new).all()

    def test__sub_data_to_image(self, grid):
        assert (
            grid.array_1d_binned_from_sub_array_1d(np.array(range(5)))
            == np.array(range(5))
        ).all()

    def test__sub_to_image__compare_to_util(self):
        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, False]]
        )

        sub_to_image_util = mapping_util.sub_to_regular_from_mask(mask, sub_grid_size=2)

        mask = msk.Mask(mask, pixel_scale=3.0)

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=2)
        assert grid.sub_grid_size == 2
        assert grid.sub_grid_fraction == (1.0 / 4.0)
        assert (grid.sub_to_regular == sub_to_image_util).all()


class TestClusterGrid:
    def test__from_mask_and_cluster_pixel_scale__correct_cluster_bin_up_calculated(
        self, mask_7x7, grid_7x7
    ):

        mask_7x7.pixel_scale = 1.0
        cluster_grid = grids.ClusterGrid.from_mask_and_cluster_pixel_scale(
            mask=mask_7x7, cluster_pixel_scale=1.0
        )

        assert (cluster_grid == grid_7x7).all()
        assert (cluster_grid.mask == mask_7x7).all()
        assert cluster_grid.bin_up_factor == 1
        assert (
            cluster_grid.cluster_to_regular_all
            == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
        ).all()

        mask_7x7.pixel_scale = 1.0
        cluster_grid = grids.ClusterGrid.from_mask_and_cluster_pixel_scale(
            mask=mask_7x7, cluster_pixel_scale=1.9
        )

        assert cluster_grid.bin_up_factor == 1
        assert (cluster_grid.mask == mask_7x7).all()
        assert (
            cluster_grid.cluster_to_regular_all
            == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
        ).all()

        mask_7x7.pixel_scale = 1.0
        cluster_grid = grids.ClusterGrid.from_mask_and_cluster_pixel_scale(
            mask=mask_7x7, cluster_pixel_scale=2.0
        )

        assert cluster_grid.bin_up_factor == 2
        assert (
            cluster_grid.mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

        assert (
            cluster_grid
            == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert (
            cluster_grid.cluster_to_regular_all
            == np.array([[0, -1, -1, -1], [1, 2, -1, -1], [3, 6, -1, -1], [4, 5, 7, 8]])
        ).all()

        mask_7x7.pixel_scale = 2.0
        cluster_grid = grids.ClusterGrid.from_mask_and_cluster_pixel_scale(
            mask=mask_7x7, cluster_pixel_scale=1.0
        )

        assert cluster_grid.bin_up_factor == 1

    def test__from_mask_and_cluster_pixel_scale__maximum_cluster_pixels_changes_bin_up_factor(
        self, mask_7x7, grid_7x7
    ):

        mask_7x7.pixel_scale = 1.0

        cluster_grid = grids.ClusterGrid.from_mask_and_cluster_pixel_scale(
            mask=mask_7x7, cluster_pixel_scale=4.0, cluster_pixels_limit=None
        )

        assert cluster_grid.bin_up_factor == 4

        cluster_grid = grids.ClusterGrid.from_mask_and_cluster_pixel_scale(
            mask=mask_7x7, cluster_pixel_scale=4.0, cluster_pixels_limit=9
        )

        assert cluster_grid.bin_up_factor == 1

        with pytest.raises(exc.DataException):

            grids.ClusterGrid.from_mask_and_cluster_pixel_scale(
                mask=mask_7x7, cluster_pixel_scale=4.0, cluster_pixels_limit=10
            )


class TestPixelizationGrid:
    def test_pix_grid__attributes(self):

        pix_grid = grids.PixelizationGrid(
            arr=np.array([[1.0, 1.0], [2.0, 2.0]]),
            regular_to_pixelization=np.array([0, 1]),
        )

        assert type(pix_grid) == grids.PixelizationGrid
        assert (pix_grid == np.array([[1.0, 1.0], [2.0, 2.0]])).all()
        assert (pix_grid.regular_to_pixelization == np.array([0, 1])).all()

    def test__from_unmasked_sparse_shape_and_grid(self):

        mask = msk.Mask(
            array=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scale=0.5,
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        sparse_to_grid = grids.SparseToRegularGrid.from_unmasked_2d_grid_shape_and_regular_grid(
            unmasked_sparse_shape=(10, 10), regular_grid=grid
        )

        pixelization_grid = grids.PixelizationGrid.from_unmasked_2d_grid_shape_and_regular_grid(
            unmasked_sparse_shape=(10, 10), regular_grid=grid
        )

        assert (sparse_to_grid.sparse == pixelization_grid).all()
        assert (
            sparse_to_grid.regular_to_sparse
            == pixelization_grid.regular_to_pixelization
        ).all()


class TestSparseToRegularGrid:
    class TestUnmaskedShape:
        def test__properties_consistent_with_mapping_util(self):

            mask = msk.Mask(
                array=np.array(
                    [[True, False, True], [False, False, False], [True, False, True]]
                ),
                pixel_scale=0.5,
            )

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=1)

            sparse_to_grid = grids.SparseToRegularGrid.from_unmasked_2d_grid_shape_and_regular_grid(
                unmasked_sparse_shape=(10, 10), regular_grid=grid
            )

            unmasked_sparse_grid_util = grid_util.grid_1d_from_shape_pixel_scales_sub_grid_size_and_origin(
                shape=(10, 10),
                pixel_scales=(0.15, 0.15),
                sub_grid_size=1,
                origin=(0.0, 0.0),
            )

            unmasked_sparse_grid_pixel_centres = grid.mask.grid_arcsec_to_grid_pixel_centres(
                grid_arcsec=unmasked_sparse_grid_util
            )

            total_sparse_pixels = mask_util.total_sparse_pixels_from_mask(
                mask=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            )

            regular_to_unmasked_sparse_util = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(
                grid_arcsec_1d=grid,
                shape=(10, 10),
                pixel_scales=(0.15, 0.15),
                origin=(0.0, 0.0),
            ).astype(
                "int"
            )

            sparse_to_unmasked_sparse_util = mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
                total_sparse_pixels=total_sparse_pixels,
                mask=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            ).astype(
                "int"
            )

            unmasked_sparse_to_sparse_util = mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(
                mask=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
                total_sparse_pixels=total_sparse_pixels,
            ).astype(
                "int"
            )

            regular_to_sparse_util = mapping_util.regular_to_sparse_from_sparse_mappings(
                regular_to_unmasked_sparse=regular_to_unmasked_sparse_util,
                unmasked_sparse_to_sparse=unmasked_sparse_to_sparse_util,
            )

            sparse_grid_util = mapping_util.sparse_grid_from_unmasked_sparse_grid(
                unmasked_sparse_grid=unmasked_sparse_grid_util,
                sparse_to_unmasked_sparse=sparse_to_unmasked_sparse_util,
            )

            assert (sparse_to_grid.regular_to_sparse == regular_to_sparse_util).all()
            assert (sparse_to_grid.sparse == sparse_grid_util).all()

        def test__sparse_grid_overlaps_mask_perfectly__masked_pixels_in_masked_sparse_grid(
            self
        ):

            mask = msk.Mask(
                array=np.array(
                    [[True, False, True], [False, False, False], [True, False, True]]
                ),
                pixel_scale=1.0,
            )

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

            sparse_to_grid = grids.SparseToRegularGrid.from_unmasked_2d_grid_shape_and_regular_grid(
                unmasked_sparse_shape=(3, 3), regular_grid=grid
            )

            assert (sparse_to_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4])).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
                )
            ).all()

        def test__same_as_above_but_4x3_grid_and_mask(self):

            mask = msk.Mask(
                array=np.array(
                    [
                        [True, False, True],
                        [False, False, False],
                        [False, False, False],
                        [True, False, True],
                    ]
                ),
                pixel_scale=1.0,
            )

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

            sparse_to_grid = grids.SparseToRegularGrid.from_unmasked_2d_grid_shape_and_regular_grid(
                unmasked_sparse_shape=(4, 3), regular_grid=grid
            )

            assert (
                sparse_to_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [
                        [1.5, 0.0],
                        [0.5, -1.0],
                        [0.5, 0.0],
                        [0.5, 1.0],
                        [-0.5, -1.0],
                        [-0.5, 0.0],
                        [-0.5, 1.0],
                        [-1.5, 0.0],
                    ]
                )
            ).all()

        def test__same_as_above_but_3x4_grid_and_mask(self):

            mask = msk.Mask(
                array=np.array(
                    [
                        [True, False, True, True],
                        [False, False, False, False],
                        [True, False, True, True],
                    ]
                ),
                pixel_scale=1.0,
            )

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

            sparse_to_grid = grids.SparseToRegularGrid.from_unmasked_2d_grid_shape_and_regular_grid(
                unmasked_sparse_shape=(3, 4), regular_grid=grid
            )

            assert (
                sparse_to_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4, 5])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [
                        [1.0, -0.5],
                        [0.0, -1.5],
                        [0.0, -0.5],
                        [0.0, 0.5],
                        [0.0, 1.5],
                        [-1.0, -0.5],
                    ]
                )
            ).all()

        def test__mask_with_offset_centre__origin_of_sparse_to_grid_moves_to_give_same_pairings(
            self
        ):

            mask = msk.Mask(
                array=np.array(
                    [
                        [True, True, True, False, True],
                        [True, True, False, False, False],
                        [True, True, True, False, True],
                        [True, True, True, True, True],
                        [True, True, True, True, True],
                    ]
                ),
                pixel_scale=1.0,
            )

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

            # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
            # the central (3x3) pixels only.

            sparse_to_grid = grids.SparseToRegularGrid.from_unmasked_2d_grid_shape_and_regular_grid(
                unmasked_sparse_shape=(3, 3), regular_grid=grid
            )

            assert (sparse_to_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4])).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [[2.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [0.0, 1.0]]
                )
            ).all()

        def test__same_as_above_but_different_offset(self):

            mask = msk.Mask(
                array=np.array(
                    [
                        [True, True, True, True, True],
                        [True, True, True, False, True],
                        [True, True, False, False, False],
                        [True, True, True, False, True],
                        [True, True, True, True, True],
                    ]
                ),
                pixel_scale=2.0,
            )

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

            # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
            # the central (3x3) pixels only.

            sparse_to_grid = grids.SparseToRegularGrid.from_unmasked_2d_grid_shape_and_regular_grid(
                unmasked_sparse_shape=(3, 3), regular_grid=grid
            )

            assert (sparse_to_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4])).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [[2.0, 2.0], [0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [-2.0, 2.0]]
                )
            ).all()

        def test__from_shape_and_regular__sets_up_with_correct_shape_and_pixel_scales(
            self, mask_7x7
        ):

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask_7x7)

            sparse_to_grid = grids.SparseToRegularGrid.from_unmasked_2d_grid_shape_and_regular_grid(
                unmasked_sparse_shape=(3, 3), regular_grid=grid
            )

            assert (
                sparse_to_grid.regular_to_sparse
                == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [
                        [1.0, -1.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, -1.0],
                        [0.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, -1.0],
                        [-1.0, 0.0],
                        [-1.0, 1.0],
                    ]
                )
            ).all()
            assert sparse_to_grid.regular == pytest.approx(grid, 1e-4)

        def test__same_as_above__but_4x3_image(self):

            mask = msk.Mask(
                array=np.array(
                    [
                        [True, False, True],
                        [False, False, False],
                        [False, False, False],
                        [True, False, True],
                    ]
                ),
                pixel_scale=1.0,
            )

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

            sparse_to_grid = grids.SparseToRegularGrid.from_unmasked_2d_grid_shape_and_regular_grid(
                unmasked_sparse_shape=(4, 3), regular_grid=grid
            )

            assert (
                sparse_to_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [
                        [1.5, 0.0],
                        [0.5, -1.0],
                        [0.5, 0.0],
                        [0.5, 1.0],
                        [-0.5, -1.0],
                        [-0.5, 0.0],
                        [-0.5, 1.0],
                        [-1.5, 0.0],
                    ]
                )
            ).all()

        def test__same_as_above__but_3x4_image(self):

            mask = msk.Mask(
                array=np.array(
                    [
                        [True, False, True, True],
                        [False, False, False, False],
                        [True, False, True, True],
                    ]
                ),
                pixel_scale=1.0,
            )

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

            sparse_to_grid = grids.SparseToRegularGrid.from_unmasked_2d_grid_shape_and_regular_grid(
                unmasked_sparse_shape=(3, 4), regular_grid=grid
            )

            assert (
                sparse_to_grid.regular_to_sparse == np.array([0, 1, 2, 3, 4, 5])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [
                        [1.0, -0.5],
                        [0.0, -1.5],
                        [0.0, -0.5],
                        [0.0, 0.5],
                        [0.0, 1.5],
                        [-1.0, -0.5],
                    ]
                )
            ).all()

        def test__from_shape_and_regular__offset_mask__origin_shift_corrects(self):

            mask = msk.Mask(
                array=np.array(
                    [
                        [True, True, False, False, False],
                        [True, True, False, False, False],
                        [True, True, False, False, False],
                        [True, True, True, True, True],
                        [True, True, True, True, True],
                    ]
                ),
                pixel_scale=1.0,
            )

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

            sparse_to_grid = grids.SparseToRegularGrid.from_unmasked_2d_grid_shape_and_regular_grid(
                unmasked_sparse_shape=(3, 3), regular_grid=grid
            )

            assert (
                sparse_to_grid.regular_to_sparse
                == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [
                        [2.0, 0.0],
                        [2.0, 1.0],
                        [2.0, 2.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [1.0, 2.0],
                        [0.0, 0.0],
                        [0.0, 1.0],
                        [0.0, 2.0],
                    ]
                )
            ).all()
            assert sparse_to_grid.regular == pytest.approx(grid, 1e-4)

    class TestUnmaskeedShapeAndWeightImage:
        def test__cluster_weight_map_all_ones__kmenas_grid_is_grid_overlapping_image(
            self
        ):

            mask = msk.Mask(
                array=np.array(
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ),
                pixel_scale=0.5,
            )

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=1)

            cluster_grid = grids.ClusterGrid.from_mask_and_cluster_pixel_scale(
                mask=mask, cluster_pixel_scale=mask.pixel_scale
            )

            cluster_weight_map = np.ones(mask.pixels_in_mask)

            sparse_to_grid_weight = grids.SparseToRegularGrid.from_total_pixels_cluster_grid_and_cluster_weight_map(
                total_pixels=8,
                regular_grid=grid,
                cluster_grid=cluster_grid,
                cluster_weight_map=cluster_weight_map,
                n_iter=10,
                max_iter=20,
                seed=1,
            )

            assert (
                sparse_to_grid_weight.sparse
                == np.array(
                    [
                        [-0.25, 0.25],
                        [0.5, -0.5],
                        [0.75, 0.5],
                        [0.25, 0.5],
                        [-0.5, -0.25],
                        [-0.5, -0.75],
                        [-0.75, 0.5],
                        [-0.25, 0.75],
                    ]
                )
            ).all()

            assert (
                sparse_to_grid_weight.regular_to_sparse
                == np.array([1, 1, 2, 2, 1, 1, 3, 3, 5, 4, 0, 7, 5, 4, 6, 6])
            ).all()

        def test__cluster_weight_map_changes_grid_from_above(self):

            mask = msk.Mask(
                array=np.array(
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ),
                pixel_scale=0.5,
            )

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=1)

            cluster_grid = grids.ClusterGrid.from_mask_and_cluster_pixel_scale(
                mask=mask, cluster_pixel_scale=mask.pixel_scale
            )

            cluster_weight_map = np.ones(mask.pixels_in_mask)
            cluster_weight_map[0:15] = 0.00000001

            sparse_to_grid_weight = grids.SparseToRegularGrid.from_total_pixels_cluster_grid_and_cluster_weight_map(
                total_pixels=8,
                regular_grid=grid,
                cluster_grid=cluster_grid,
                cluster_weight_map=cluster_weight_map,
                n_iter=10,
                max_iter=30,
                seed=1,
            )

            assert sparse_to_grid_weight.sparse[1] == pytest.approx(
                np.array([0.4166666, -0.0833333]), 1.0e-4
            )

            assert (
                sparse_to_grid_weight.regular_to_sparse
                == np.array([5, 1, 0, 0, 5, 1, 1, 4, 3, 6, 7, 4, 3, 6, 2, 2])
            ).all()

        def test__cluster_weight_map_all_ones__cluster_pixel_scale_leads_to_binning_up_by_factor_2(
            self
        ):

            mask = msk.Mask(
                array=np.full(fill_value=False, shape=(8, 8)), pixel_scale=0.5
            )

            grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=1)

            cluster_grid = grids.ClusterGrid.from_mask_and_cluster_pixel_scale(
                mask=mask, cluster_pixel_scale=2.0 * mask.pixel_scale
            )

            cluster_weight_map = np.ones(cluster_grid.shape[0])

            sparse_to_grid_weight = grids.SparseToRegularGrid.from_total_pixels_cluster_grid_and_cluster_weight_map(
                total_pixels=8,
                regular_grid=grid,
                cluster_grid=cluster_grid,
                cluster_weight_map=cluster_weight_map,
                n_iter=10,
                max_iter=30,
                seed=1,
            )

            assert (
                sparse_to_grid_weight.sparse
                == np.array(
                    [
                        [-0.5, 0.5],
                        [1.0, -1.0],
                        [1.5, 1.0],
                        [0.5, 1.0],
                        [-1.0, -0.5],
                        [-1.0, -1.5],
                        [-1.5, 1.0],
                        [-0.5, 1.5],
                    ]
                )
            ).all()

            assert (
                sparse_to_grid_weight.regular_to_sparse
                == np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        1,
                        1,
                        1,
                        1,
                        3,
                        3,
                        3,
                        3,
                        1,
                        1,
                        1,
                        1,
                        3,
                        3,
                        3,
                        3,
                        5,
                        5,
                        4,
                        4,
                        0,
                        0,
                        7,
                        7,
                        5,
                        5,
                        4,
                        4,
                        0,
                        0,
                        7,
                        7,
                        5,
                        5,
                        4,
                        4,
                        6,
                        6,
                        6,
                        6,
                        5,
                        5,
                        4,
                        4,
                        6,
                        6,
                        6,
                        6,
                    ]
                )
            ).all()


class TestGridStack(object):
    def test__grids(self, grid_stack):

        assert (grid_stack.regular == np.array([[0.0, 0.0]])).all()
        np.testing.assert_almost_equal(
            grid_stack.sub,
            np.array([[0.25, -0.25], [0.25, 0.25], [-0.25, -0.25], [-0.25, 0.25]]),
        )
        assert (
            grid_stack.blurring
            == np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            )
        ).all()
        assert (grid_stack.pixelization == np.array([[0.0, 0.0]])).all()

    def test__from_shape_and_pixel_scale(self):

        mask = msk.Mask(
            np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
            pixel_scale=2.0,
        )

        grid_stack_mask = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=mask, sub_grid_size=2, psf_shape=(1, 1)
        )

        grid_stack_shape = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(
            shape=(3, 3), pixel_scale=2.0, sub_grid_size=2
        )

        assert (grid_stack_mask.regular == grid_stack_shape.regular).all()
        assert (grid_stack_mask.sub == grid_stack_shape.sub).all()
        assert (grid_stack_mask.pixelization == np.array([[0.0, 0.0]])).all()

    def test__from_unmasked_grid_2d(self):

        grid_2d = np.array(
            [
                [[2.0, -1.0], [2.0, 0.0], [2.0, 1.0]],
                [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]],
                [[-2.0, -1.0], [-2.0, 0.0], [-2.0, 1.0]],
            ]
        )

        grid = grids.Grid.from_unmasked_grid_2d(grid_2d=grid_2d)
        grid = grids.Grid.from_unmasked_grid_2d(grid_2d=grid_2d)
        grid_stack = grids.GridStack.from_unmasked_grid_2d(grid_2d=grid_2d)

        assert (grid == grid_stack.regular).all()
        assert (grid.mask == grid_stack.regular.mask).all()
        assert (grid == grid_stack.sub).all()
        assert (grid.mask == grid_stack.sub.mask).all()
        assert grid.sub_grid_size == grid_stack.sub.sub_grid_size

    def test__padded_grid_stack_from_psf_shape(self):

        mask = np.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )

        mask = msk.Mask(array=mask, pixel_scale=1.0)

        grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=mask, sub_grid_size=2, psf_shape=(3, 3)
        )

        padded_grid_stack = grid_stack.padded_grid_stack_from_psf_shape(
            psf_shape=(3, 3)
        )

        regular_padded_grid_util = grid_util.grid_1d_from_shape_pixel_scales_sub_grid_size_and_origin(
            shape=(6, 6), pixel_scales=(1.0, 1.0), sub_grid_size=1
        )

        sub_padded_grid_util = grid_util.grid_1d_from_mask_pixel_scales_sub_grid_size_and_origin(
            mask=np.full((6, 6), False), pixel_scales=(1.0, 1.0), sub_grid_size=2
        )

        assert padded_grid_stack.regular == pytest.approx(
            regular_padded_grid_util, 1e-4
        )
        assert padded_grid_stack.sub == pytest.approx(sub_padded_grid_util, 1e-4)
        assert (padded_grid_stack.blurring == np.array([0.0, 0.0])).all()
        assert (padded_grid_stack.pixelization == np.array([[0.0, 0.0]])).all()
        assert padded_grid_stack.regular.interpolator is None
        assert padded_grid_stack.sub.interpolator is None

        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(
            shape=(6, 6), pixel_scale=1.0
        )

        regular_interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=padded_grid_stack.regular, interp_pixel_scale=1.0
        )

        sub_interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=padded_grid_stack.sub, interp_pixel_scale=1.0
        )

        grid_stack.regular.interpolator = regular_interpolator
        grid_stack.sub.interpolator = sub_interpolator

        padded_grid_stack = grid_stack.padded_grid_stack_from_psf_shape(
            psf_shape=(3, 3)
        )

        assert (
            padded_grid_stack.regular.interpolator.vtx == regular_interpolator.vtx
        ).all()
        assert (padded_grid_stack.sub.interpolator.vtx == sub_interpolator.vtx).all()

    def test__scaled_array_2d_from_array_1d(self, grid_stack):

        scaled_array_from_grid_stack = grid_stack.scaled_array_2d_from_array_1d(
            array_1d=np.ones(5)
        )
        scaled_array_from_regular = grid_stack.regular.scaled_array_2d_from_array_1d(
            array_1d=np.ones(5)
        )

        assert (scaled_array_from_grid_stack == scaled_array_from_regular).all()

    def test__apply_function_retains_attributes(self, grid_stack):

        grid_stack.pixelization = grids.PixelizationGrid(
            arr=np.array([[1.0, 1.0]]), regular_to_pixelization=1
        )

        def add_one(coords):
            return np.add(1, coords)

        assert isinstance(grid_stack.regular, grids.Grid)
        assert grid_stack.regular.mask is not None

        assert isinstance(grid_stack.sub, grids.Grid)
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

        assert isinstance(grid_stack.pixelization, grids.PixelizationGrid)
        assert grid_stack.pixelization.regular_to_pixelization == 1
        assert grid_stack.regular.mask is not None

    def test__apply_function(self, grid_stack):
        grid_stack.pixelization = grid_stack.regular

        def add_one(coords):
            return np.add(1, coords)

        new_collection = grid_stack.apply_function(add_one)
        assert isinstance(new_collection, grids.GridStack)
        assert (new_collection.regular == np.add(1, np.array([[0.0, 0.0]]))).all()
        np.testing.assert_almost_equal(
            new_collection.sub,
            np.add(
                1,
                np.array([[0.25, -0.25], [0.25, 0.25], [-0.25, -0.25], [-0.25, 0.25]]),
            ),
        )
        assert (
            new_collection.blurring
            == np.add(
                1,
                np.array(
                    [
                        [1.0, -1.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, -1.0],
                        [0.0, 1.0],
                        [-1.0, -1.0],
                        [-1.0, 0.0],
                        [-1.0, 1.0],
                    ]
                ),
            )
        ).all()
        assert (new_collection.pixelization == np.add(1, np.array([[0.0, 0.0]]))).all()

    def test__map_function(self, grid_stack):

        grid_stack.pixelization = grids.PixelizationGrid(
            arr=np.array([[1.0, 1.0]]), regular_to_pixelization=1
        )

        def add_number(coords, number):
            return np.add(coords, number)

        new_collection = grid_stack.map_function(add_number, [1, 2, 3, 1])

        assert isinstance(new_collection, grids.GridStack)
        assert (new_collection.regular == np.add(1, np.array([[0.0, 0.0]]))).all()
        np.testing.assert_almost_equal(
            new_collection.sub,
            np.add(
                2,
                np.array([[0.25, -0.25], [0.25, 0.25], [-0.25, -0.25], [-0.25, 0.25]]),
            ),
        )
        assert (
            new_collection.blurring
            == np.add(
                3,
                np.array(
                    [
                        [1.0, -1.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, -1.0],
                        [0.0, 1.0],
                        [-1.0, -1.0],
                        [-1.0, 0.0],
                        [-1.0, 1.0],
                    ]
                ),
            )
        ).all()

        assert (new_collection.pixelization == np.add(1, np.array([[1.0, 1.0]]))).all()
        assert new_collection.pixelization.regular_to_pixelization == 1

    def test__new_grid_stack_with_grids_added(self, grid_stack):

        grid_stack = grid_stack.new_grid_stack_with_grids_added(pixelization=1)

        assert (grid_stack.regular == np.array([[0.0, 0.0]])).all()
        np.testing.assert_almost_equal(
            grid_stack.sub,
            np.array([[0.25, -0.25], [0.25, 0.25], [-0.25, -0.25], [-0.25, 0.25]]),
        )
        assert (
            grid_stack.blurring
            == np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            )
        ).all()
        assert grid_stack.pixelization == 1
        # assert grid_stack.cluster == None
        #
        # grid_stack = grid_stack.new_grid_stack_with_grids_added(cluster=2)
        #
        # assert (grid_stack.regular == np.array([[0., 0.]])).all()
        # np.testing.assert_almost_equal(grid_stack.sub, np.array([[0.25, -0.25],
        #                                                          [0.25, 0.25],
        #                                                          [-0.25, -0.25],
        #                                                          [-0.25, 0.25]]))
        # assert (grid_stack.blurring == np.array([[1., -1.],
        #                                          [1., 0.],
        #                                          [1., 1.],
        #                                          [0., -1.],
        #                                          [0., 1.],
        #                                          [-1., -1.],
        #                                          [-1., 0.],
        #                                          [-1., 1.]])).all()
        # assert grid_stack.pixelization == 1
        # assert grid_stack.cluster == 2

    def test__new_grid_stack_with_interpolator_added_to_each_grid(self):

        mask = np.array(
            [
                [True, True, True, True, True, True],
                [True, True, True, False, False, True],
                [True, False, True, True, True, True],
                [True, True, True, False, False, True],
                [True, True, True, True, True, True],
            ]
        )
        mask = msk.Mask(array=mask, pixel_scale=2.0)

        grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=mask, sub_grid_size=2, psf_shape=(3, 3)
        )

        new_grid_stack = grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(
            interp_pixel_scale=1.0
        )

        grid_manual = grids.Grid.from_mask_and_sub_grid_size(mask=mask)
        sub_grid_manual = grids.Grid.from_mask_and_sub_grid_size(
            mask=mask, sub_grid_size=2
        )
        blurring_grid_manual = grids.Grid.blurring_grid_from_mask_and_psf_shape(
            mask=mask, psf_shape=(3, 3)
        )

        assert (new_grid_stack.regular == grid_manual).all()
        np.testing.assert_almost_equal(new_grid_stack.sub, sub_grid_manual)

        assert (new_grid_stack.blurring == blurring_grid_manual).all()

        regular_interpolator_manual = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=grid_manual, interp_pixel_scale=1.0
        )
        sub_interpolator_manual = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=sub_grid_manual, interp_pixel_scale=1.0
        )
        blurring_interpolator_manual = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=blurring_grid_manual.mask,
            grid=blurring_grid_manual,
            interp_pixel_scale=1.0,
        )

        assert (
            new_grid_stack.regular.interpolator.vtx == regular_interpolator_manual.vtx
        ).all()
        assert (
            new_grid_stack.regular.interpolator.wts == regular_interpolator_manual.wts
        ).all()

        assert (
            new_grid_stack.sub.interpolator.vtx == sub_interpolator_manual.vtx
        ).all()
        assert (
            new_grid_stack.sub.interpolator.wts == sub_interpolator_manual.wts
        ).all()

        assert (
            new_grid_stack.blurring.interpolator.vtx == blurring_interpolator_manual.vtx
        ).all()
        assert (
            new_grid_stack.blurring.interpolator.wts == blurring_interpolator_manual.wts
        ).all()


class TestImageGridBorder(object):
    class TestFromMask:
        def test__simple_mask_border_pixels_is_border(self):

            mask = np.array(
                [
                    [True, True, True, True, True, True, True, True, True, True],
                    [True, False, False, False, False, False, False, False, True, True],
                    [True, False, True, True, True, True, True, False, True, True],
                    [True, False, True, False, False, False, True, False, True, True],
                    [True, False, True, False, True, False, True, False, True, True],
                    [True, False, True, False, False, False, True, False, True, True],
                    [True, False, True, True, True, True, True, False, True, True],
                    [True, False, False, False, False, False, False, False, True, True],
                    [True, True, True, True, True, True, True, True, True, True],
                ]
            )

            mask = msk.Mask(mask, pixel_scale=3.0)

            border = grids.GridBorder.from_mask(mask)

            assert (
                border
                == np.array(
                    [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        13,
                        14,
                        17,
                        18,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                        31,
                    ]
                )
            ).all()

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
            grid_stack = grids.GridStack(
                regular=regular_grid, sub=sub_grid, blurring=None
            )

            border = grids.GridBorder(arr=np.arange(32))
            relocated_grids = border.relocated_grid_stack_from_grid_stack(grid_stack)

            assert relocated_grids.regular[0:32] == pytest.approx(
                np.asarray(grid_circle)[0:32], 1e-3
            )
            assert relocated_grids.regular[32] == pytest.approx(
                np.array([0.1, 0.0]), 1e-3
            )
            assert relocated_grids.regular[33] == pytest.approx(
                np.array([-0.2, -0.3]), 1e-3
            )
            assert relocated_grids.regular[34] == pytest.approx(
                np.array([0.5, 0.4]), 1e-3
            )
            assert relocated_grids.regular[35] == pytest.approx(
                np.array([0.7, -0.1]), 1e-3
            )

            assert relocated_grids.sub[0:32] == pytest.approx(
                np.asarray(grid_circle)[0:32], 1e-3
            )
            assert relocated_grids.sub[32] == pytest.approx(np.array([0.1, 0.0]), 1e-3)
            assert relocated_grids.sub[33] == pytest.approx(
                np.array([-0.2, -0.3]), 1e-3
            )
            assert relocated_grids.sub[34] == pytest.approx(np.array([0.5, 0.4]), 1e-3)
            assert relocated_grids.sub[35] == pytest.approx(np.array([0.5, 0.3]), 1e-3)

        def test__8_points_with_border_as_circle__points_go_to_circle_edge(self):

            grid = np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                    [0.7071, 0.7071],
                    [0.7071, -0.7071],
                    [-0.7071, 0.7071],
                    [-0.7071, -0.7071],
                    [10.0, 10.0],
                    [10.0, -10.0],
                    [-10.0, 10],
                    [-10.0, -10.0],
                ]
            )

            grid_stack = grids.GridStack(regular=grid, sub=grid, blurring=None)

            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            border = grids.GridBorder(border_pixels)

            relocated_grids = border.relocated_grid_stack_from_grid_stack(grid_stack)

            assert relocated_grids.regular[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.regular[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.regular[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.regular[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.regular[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.regular[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.regular[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.regular[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.regular[8] == pytest.approx(
                np.array([0.7071, 0.7071]), 1e-3
            )
            assert relocated_grids.regular[9] == pytest.approx(
                np.array([0.7071, -0.7071]), 1e-3
            )
            assert relocated_grids.regular[10] == pytest.approx(
                np.array([-0.7071, 0.7071]), 1e-3
            )
            assert relocated_grids.regular[11] == pytest.approx(
                np.array([-0.7071, -0.7071]), 1e-3
            )

            assert relocated_grids.sub[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.sub[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.sub[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.sub[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.sub[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.sub[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.sub[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.sub[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.sub[8] == pytest.approx(
                np.array([0.7071, 0.7071]), 1e-3
            )
            assert relocated_grids.sub[9] == pytest.approx(
                np.array([0.7071, -0.7071]), 1e-3
            )
            assert relocated_grids.sub[10] == pytest.approx(
                np.array([-0.7071, 0.7071]), 1e-3
            )
            assert relocated_grids.sub[11] == pytest.approx(
                np.array([-0.7071, -0.7071]), 1e-3
            )

        def test__same_as_above_but_ensure_positive_origin_moves_points(self):
            grid = np.array(
                [
                    [2.0, 1.0],
                    [1.0, 2.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0 + 0.7071, 1.0 + 0.7071],
                    [1.0 + 0.7071, 1.0 - 0.7071],
                    [1.0 - 0.7071, 1.0 + 0.7071],
                    [1.0 - 0.7071, 1.0 - 0.7071],
                    [11.0, 11.0],
                    [11.0, -9.0],
                    [-9.0, 11],
                    [-9.0, -9.0],
                ]
            )
            grid_stack = grids.GridStack(regular=grid, sub=grid, blurring=None)

            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            border = grids.GridBorder(border_pixels)

            relocated_grids = border.relocated_grid_stack_from_grid_stack(grid_stack)

            assert relocated_grids.regular[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.regular[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.regular[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.regular[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.regular[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.regular[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.regular[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.regular[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.regular[8] == pytest.approx(
                np.array([1.0 + 0.7071, 1.0 + 0.7071]), 1e-3
            )
            assert relocated_grids.regular[9] == pytest.approx(
                np.array([1.0 + 0.7071, 1.0 - 0.7071]), 1e-3
            )
            assert relocated_grids.regular[10] == pytest.approx(
                np.array([1.0 - 0.7071, 1.0 + 0.7071]), 1e-3
            )
            assert relocated_grids.regular[11] == pytest.approx(
                np.array([1.0 - 0.7071, 1.0 - 0.7071]), 1e-3
            )

            assert relocated_grids.sub[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.sub[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.sub[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.sub[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.sub[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.sub[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.sub[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.sub[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.sub[8] == pytest.approx(
                np.array([1.0 + 0.7071, 1.0 + 0.7071]), 1e-3
            )
            assert relocated_grids.sub[9] == pytest.approx(
                np.array([1.0 + 0.7071, 1.0 - 0.7071]), 1e-3
            )
            assert relocated_grids.sub[10] == pytest.approx(
                np.array([1.0 - 0.7071, 1.0 + 0.7071]), 1e-3
            )
            assert relocated_grids.sub[11] == pytest.approx(
                np.array([1.0 - 0.7071, 1.0 - 0.7071]), 1e-3
            )

        def test__same_as_above_but_ensure_negative_origin_moves_points(self):
            grid = np.array(
                [
                    [0.0, -1.0],
                    [-1.0, 0.0],
                    [-2.0, -1.0],
                    [-1.0, -2.0],
                    [-1.0 + 0.7071, -1.0 + 0.7071],
                    [-1.0 + 0.7071, -1.0 - 0.7071],
                    [-1.0 - 0.7071, -1.0 + 0.7071],
                    [-1.0 - 0.7071, -1.0 - 0.7071],
                    [9.0, 9.0],
                    [9.0, -11.0],
                    [-11.0, 9],
                    [-11.0, -11.0],
                ]
            )
            grid_stack = grids.GridStack(regular=grid, sub=grid, blurring=None)
            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            border = grids.GridBorder(border_pixels)

            relocated_grids = border.relocated_grid_stack_from_grid_stack(grid_stack)

            assert relocated_grids.regular[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.regular[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.regular[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.regular[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.regular[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.regular[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.regular[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.regular[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.regular[8] == pytest.approx(
                np.array([-1.0 + 0.7071, -1.0 + 0.7071]), 1e-3
            )
            assert relocated_grids.regular[9] == pytest.approx(
                np.array([-1.0 + 0.7071, -1.0 - 0.7071]), 1e-3
            )
            assert relocated_grids.regular[10] == pytest.approx(
                np.array([-1.0 - 0.7071, -1.0 + 0.7071]), 1e-3
            )
            assert relocated_grids.regular[11] == pytest.approx(
                np.array([-1.0 - 0.7071, -1.0 - 0.7071]), 1e-3
            )

            assert relocated_grids.sub[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grids.sub[1] == pytest.approx(grid[1], 1e-3)
            assert relocated_grids.sub[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grids.sub[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grids.sub[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grids.sub[5] == pytest.approx(grid[5], 1e-3)
            assert relocated_grids.sub[6] == pytest.approx(grid[6], 1e-3)
            assert relocated_grids.sub[7] == pytest.approx(grid[7], 1e-3)
            assert relocated_grids.sub[8] == pytest.approx(
                np.array([-1.0 + 0.7071, -1.0 + 0.7071]), 1e-3
            )
            assert relocated_grids.sub[9] == pytest.approx(
                np.array([-1.0 + 0.7071, -1.0 - 0.7071]), 1e-3
            )
            assert relocated_grids.sub[10] == pytest.approx(
                np.array([-1.0 - 0.7071, -1.0 + 0.7071]), 1e-3
            )
            assert relocated_grids.sub[11] == pytest.approx(
                np.array([-1.0 - 0.7071, -1.0 - 0.7071]), 1e-3
            )

        def test__point_is_inside_border_but_further_than_minimum_border_point_radii__does_not_relocate(
            self
        ):
            grid = np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -0.9],
                    [0.7071, 0.7071],
                    [0.7071, -0.7071],
                    [-0.7071, 0.7071],
                    [-0.7071, -0.7071],
                    [0.02, 0.95],
                ]
            )

            grid_stack = grids.GridStack(regular=grid, sub=grid, blurring=None)

            border_pixels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

            border = grids.GridBorder(border_pixels)

            relocated_grids = border.relocated_grid_stack_from_grid_stack(grid_stack)

            assert relocated_grids.regular[8] == pytest.approx(
                np.array([0.02, 0.95]), 1e-4
            )
            assert relocated_grids.sub[8] == pytest.approx(np.array([0.02, 0.95]), 1e-4)

        def test__inside_border_no_relocations__also_include_sparse_grid(self):

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
            grid_stack = grids.GridStack(
                regular=regular_grid,
                sub=sub_grid,
                blurring=None,
                pixelization=regular_grid,
            )

            border = grids.GridBorder(arr=np.arange(32))
            relocated_grids = border.relocated_grid_stack_from_grid_stack(grid_stack)

            assert relocated_grids.regular[0:32] == pytest.approx(
                np.asarray(grid_circle)[0:32], 1e-3
            )
            assert relocated_grids.regular[32] == pytest.approx(
                np.array([0.1, 0.0]), 1e-3
            )
            assert relocated_grids.regular[33] == pytest.approx(
                np.array([-0.2, -0.3]), 1e-3
            )
            assert relocated_grids.regular[34] == pytest.approx(
                np.array([0.5, 0.4]), 1e-3
            )
            assert relocated_grids.regular[35] == pytest.approx(
                np.array([0.7, -0.1]), 1e-3
            )

            assert relocated_grids.sub[0:32] == pytest.approx(
                np.asarray(grid_circle)[0:32], 1e-3
            )
            assert relocated_grids.sub[32] == pytest.approx(np.array([0.1, 0.0]), 1e-3)
            assert relocated_grids.sub[33] == pytest.approx(
                np.array([-0.2, -0.3]), 1e-3
            )
            assert relocated_grids.sub[34] == pytest.approx(np.array([0.5, 0.4]), 1e-3)
            assert relocated_grids.sub[35] == pytest.approx(np.array([0.5, 0.3]), 1e-3)

            assert relocated_grids.pixelization[0:32] == pytest.approx(
                np.asarray(grid_circle)[0:32], 1e-3
            )
            assert relocated_grids.pixelization[32] == pytest.approx(
                np.array([0.1, 0.0]), 1e-3
            )
            assert relocated_grids.pixelization[33] == pytest.approx(
                np.array([-0.2, -0.3]), 1e-3
            )
            assert relocated_grids.pixelization[34] == pytest.approx(
                np.array([0.5, 0.4]), 1e-3
            )
            assert relocated_grids.pixelization[35] == pytest.approx(
                np.array([0.7, -0.1]), 1e-3
            )


class TestInterpolator:
    def test_decorated_function__values_from_function_has_1_dimensions__returns_1d_result(
        self
    ):

        # noinspection PyUnusedLocal
        @grids.grid_interpolate
        def func(
            profile,
            grid,
            return_in_2d=False,
            return_binned=False,
            grid_radial_minimum=None,
        ):
            result = np.zeros(grid.shape[0])
            result[0] = 1
            return result

        regular = grids.Grid.from_mask_and_sub_grid_size(
            mask=msk.Mask.unmasked_for_shape_and_pixel_scale((3, 3), 1)
        )

        values = func(None, regular)

        assert values.ndim == 1
        assert values.shape == (9,)
        assert (values == np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])).all()

        regular = grids.Grid.from_mask_and_sub_grid_size(
            mask=msk.Mask.unmasked_for_shape_and_pixel_scale((3, 3), 1)
        )
        regular.interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            regular.mask, regular, interp_pixel_scale=0.5
        )
        interp_values = func(None, regular)
        assert interp_values.ndim == 1
        assert interp_values.shape == (9,)
        assert (interp_values != np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])).any()

    def test_decorated_function__values_from_function_has_2_dimensions__returns_2d_result(
        self
    ):

        # noinspection PyUnusedLocal
        @grids.grid_interpolate
        def func(
            profile,
            grid,
            return_in_2d=False,
            return_binned=False,
            grid_radial_minimum=None,
        ):
            result = np.zeros((grid.shape[0], 2))
            result[0, :] = 1
            return result

        regular = grids.Grid.from_mask_and_sub_grid_size(
            mask=msk.Mask.unmasked_for_shape_and_pixel_scale((3, 3), 1)
        )

        values = func(None, regular)

        assert values.ndim == 2
        assert values.shape == (9, 2)
        assert (
            values
            == np.array(
                [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            )
        ).all()

        regular = grids.Grid.from_mask_and_sub_grid_size(
            mask=msk.Mask.unmasked_for_shape_and_pixel_scale((3, 3), 1)
        )
        regular.interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            regular.mask, regular, interp_pixel_scale=0.5
        )

        interp_values = func(None, regular)
        assert interp_values.ndim == 2
        assert interp_values.shape == (9, 2)
        assert (
            interp_values
            != np.array(
                np.array(
                    [
                        [1, 1],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                    ]
                )
            )
        ).any()

    def test__20x20_deflection_angles_no_central_pixels__interpolated_accurately(self):

        shape = (20, 20)
        pixel_scale = 1.0

        mask = msk.Mask.circular_annular(
            shape=shape,
            pixel_scale=pixel_scale,
            inner_radius_arcsec=3.0,
            outer_radius_arcsec=8.0,
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        isothermal = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        true_deflections = isothermal.deflections_from_grid(grid=grid)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=grid, interp_pixel_scale=1.0
        )

        interp_deflections_values = isothermal.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.001
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.001

    def test__move_centre_of_galaxy__interpolated_accurately(self):
        shape = (24, 24)
        pixel_scale = 1.0

        mask = msk.Mask.circular_annular(
            shape=shape,
            pixel_scale=pixel_scale,
            inner_radius_arcsec=3.0,
            outer_radius_arcsec=8.0,
            centre=(3.0, 3.0),
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        isothermal = mp.SphericalIsothermal(centre=(3.0, 3.0), einstein_radius=1.0)

        true_deflections = isothermal.deflections_from_grid(grid=grid)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=grid, interp_pixel_scale=1.0
        )

        interp_deflections_values = isothermal.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.001
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.001

    def test__different_interpolation_pixel_scales_still_works(self):
        shape = (28, 28)
        pixel_scale = 1.0

        mask = msk.Mask.circular_annular(
            shape=shape,
            pixel_scale=pixel_scale,
            inner_radius_arcsec=3.0,
            outer_radius_arcsec=8.0,
            centre=(3.0, 3.0),
        )

        grid = grids.Grid.from_mask_and_sub_grid_size(mask=mask)

        isothermal = mp.SphericalIsothermal(centre=(3.0, 3.0), einstein_radius=1.0)

        true_deflections = isothermal.deflections_from_grid(grid=grid)

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=grid, interp_pixel_scale=0.2
        )

        interp_deflections_values = isothermal.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.001
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.001

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=grid, interp_pixel_scale=0.5
        )

        interp_deflections_values = isothermal.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.01
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.01

        interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
            mask=mask, grid=grid, interp_pixel_scale=1.1
        )

        interp_deflections_values = isothermal.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.1
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.1


class MockProfile(object):
    def __init__(self, values):

        self.values = values

    @grids.reshape_returned_array
    def array_from_grid(self, grid, return_in_2d=True, return_binned=True):
        return self.values

    @grids.reshape_returned_grid
    def grid_from_grid(self, grid, return_in_2d=True, return_binned=True):
        return self.values


class TestReturnArrayFormat(object):
    def test__array_1d_from_function__decorator_changes_array_dimensions_depending_on_inputs(
        self
    ):

        profile = MockProfile(values=np.ones(4))

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=0.1, sub_grid_size=1
        )

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (array_from_grid == np.ones(4)).all()

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (array_from_grid == np.ones((2, 2))).all()

        profile = MockProfile(values=np.ones(16))

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=0.1, sub_grid_size=2
        )

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (array_from_grid == np.ones(16)).all()

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (array_from_grid == np.ones((4, 4))).all()

    def test__array_1d_from_function__array_is_binned_if_used__maintains_correct_dimensions(
        self
    ):

        sub_array = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                3.0,
                4.0,
                4.0,
                4.0,
                4.0,
            ]
        )

        profile = MockProfile(values=sub_array)

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=0.1, sub_grid_size=2
        )

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (array_from_grid == sub_array).all()

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (
            array_from_grid
            == np.array(
                [
                    [1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 4.0, 4.0],
                    [3.0, 3.0, 4.0, 4.0],
                ]
            )
        ).all()

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        assert (array_from_grid == np.array([1.0, 2.0, 3.0, 4.0])).all()

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=True, return_binned=True
        )

        assert (array_from_grid == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

    def test__only_needs_input_grid_as_type_grid_if_there_is_a_mapping_to_2d_or_bin_up(
        self
    ):

        profile = MockProfile(values=np.ones(4))

        grid = np.ones(shape=(4, 2))

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (array_from_grid == np.array(np.ones(4))).all()

        with pytest.raises(exc.GridException):

            profile.array_from_grid(grid=grid, return_in_2d=True, return_binned=False)

            profile.array_from_grid(grid=grid, return_in_2d=False, return_binned=True)

            profile.array_from_grid(grid=grid, return_in_2d=True, return_binned=True)

    def test__returned_array_from_function_is_2d__grid_in__decorator_convert_dimensions_to_1d_first(
        self
    ):

        profile = MockProfile(values=np.ones((4, 4)))

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(4, 4), pixel_scale=0.1
        )

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (array_from_grid == np.array(np.ones(16))).all()

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (array_from_grid == np.array(np.ones((4, 4)))).all()

    def test__returned_array_from_function_is_2d__sub_grid_in__decorator_converts_dimensions_to_1_first(
        self
    ):

        profile = MockProfile(values=np.ones((8, 8)))

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(4, 4), pixel_scale=0.1, sub_grid_size=2
        )

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (array_from_grid == np.ones(64)).all()

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (array_from_grid == np.ones((8, 8))).all()

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        assert (array_from_grid == np.ones(16)).all()

        array_from_grid = profile.array_from_grid(
            grid=grid, return_in_2d=True, return_binned=True
        )

        assert (array_from_grid == np.ones((4, 4))).all()


class TestReturnGridFormat(object):
    def test__grid_1d_from_function__decorator_changes_grid_dimensions_to_2d(self):

        profile = MockProfile(values=np.ones(shape=(4, 2)))

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=0.1, sub_grid_size=1
        )

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (grid_from_grid == np.ones((4, 2))).all()

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (grid_from_grid == np.ones((2, 2, 2))).all()

        profile = MockProfile(values=np.ones(shape=(16, 2)))

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=0.1, sub_grid_size=2
        )

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (grid_from_grid == np.ones(shape=(16, 2))).all()

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (grid_from_grid == np.ones(shape=(4, 4, 2))).all()

    def test__grid_1d_from_function__binned_up_function_is_used_if_true__maintains_correct_dimensions(
        self
    ):

        sub_grid_values = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [4.0, 4.0],
                [4.0, 4.0],
                [4.0, 4.0],
            ]
        )

        profile = MockProfile(values=sub_grid_values)

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(2, 2), pixel_scale=0.1, sub_grid_size=2
        )

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (grid_from_grid == sub_grid_values).all()

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (
            grid_from_grid
            == np.array(
                [
                    [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]],
                    [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]],
                    [[3.0, 3.0], [3.0, 3.0], [4.0, 4.0], [4.0, 4.0]],
                    [[3.0, 3.0], [3.0, 3.0], [4.0, 4.0], [4.0, 4.0]],
                ]
            )
        ).all()

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        assert (
            grid_from_grid == np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        ).all()

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=True
        )

        assert (
            grid_from_grid
            == np.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])
        ).all()

    def test__only_needs_input_grid_as_type_grid_if_there_is_a_mapping_to_2d_or_bin_up(
        self
    ):

        profile = MockProfile(values=np.ones(shape=(4, 2)))

        grid = np.ones(shape=(4, 2))

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (grid_from_grid == np.ones(shape=(4, 2))).all()

        with pytest.raises(exc.GridException):

            profile.grid_from_grid(grid=grid, return_in_2d=True, return_binned=False)

            profile.grid_from_grid(grid=grid, return_in_2d=False, return_binned=True)

            profile.grid_from_grid(grid=grid, return_in_2d=True, return_binned=True)

    def test__grid_of_function_output_in_2d__decorator_converts_to_1d_first(self):

        profile = MockProfile(values=np.ones(shape=(4, 4, 2)))

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(4, 4), pixel_scale=0.1
        )

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (grid_from_grid == np.array(np.ones(shape=(16, 2)))).all()

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (grid_from_grid == np.array(np.ones(shape=(4, 4, 2)))).all()

    def test__sub_grid_of_function_returned_in_2d__decorator_changes_grid_dimensions_to_2d(
        self
    ):

        profile = MockProfile(values=np.ones(shape=(8, 8, 2)))

        grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
            shape=(4, 4), pixel_scale=0.1, sub_grid_size=2
        )

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (grid_from_grid == np.ones((64, 2))).all()

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (grid_from_grid == np.ones((8, 8, 2))).all()

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        assert (grid_from_grid == np.ones((16, 2))).all()

        grid_from_grid = profile.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=True
        )

        assert (grid_from_grid == np.ones((4, 4, 2))).all()
