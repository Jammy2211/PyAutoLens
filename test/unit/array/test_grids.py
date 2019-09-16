import numpy as np
import pytest

import autolens as al


@pytest.fixture(name="grid")
def make_grid():
    mask = al.Mask(
        np.array([[True, False, True], [False, False, False], [True, False, True]]),
        pixel_scale=1.0,
        sub_size=1,
    )

    return al.Grid.from_mask(mask=mask)


class TestGrid:
    def test__from_mask__compare_to_array_util(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=1)

        grid_via_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, sub_size=1, pixel_scales=(2.0, 2.0)
        )

        grid = al.Grid.from_mask(mask=mask)

        assert type(grid) == al.Grid
        assert grid == pytest.approx(grid_via_util, 1e-4)
        assert grid.pixel_scale == 2.0
        assert (
            grid.mask.mapping.mask_1d_index_tomask_index
            == mask.mapping.mask_1d_index_tomask_index
        ).all()
        assert grid.interpolator == None

        grid_2d = mask.mapping.grid_2d_from_grid_1d(grid_1d=grid)

        assert (grid.in_2d == grid_2d).all()

        mask = np.array([[True, True, True], [True, False, False], [True, True, False]])

        mask = al.Mask(mask, pixel_scale=3.0, sub_size=2)

        grid_via_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, pixel_scales=(3.0, 3.0), sub_size=2
        )

        grid = al.Grid.from_mask(mask=mask)

        assert grid == pytest.approx(grid_via_util, 1e-4)

    def test__unlensed_unsubbed_grid_property_is_grid_with_sub_size_1(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=2)

        grid = al.Grid.from_mask(mask=mask)

        assert (
            grid.unlensed_unsubbed_1d
            == np.array([[2.0, 1.0], [2.0, 3.0], [0.0, -1.0], [-2.0, 1.0], [-2.0, 3.0]])
        ).all()

    def test__grid_unlensed_1d_property__compare_to_grid_util(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )

        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=1)

        grid = al.Grid(arr=np.array([[1.0, 1.0], [1.0, 1.0]]), mask=mask)

        grid_via_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, sub_size=1, pixel_scales=(2.0, 2.0)
        )

        assert grid.unlensed_1d == pytest.approx(grid_via_util, 1e-4)

        grid_via_util = al.grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
            shape=(3, 4), sub_size=1, pixel_scales=(2.0, 2.0)
        )

        assert grid.unlensed_unmasked_1d == pytest.approx(grid_via_util, 1e-4)

        mask = al.Mask(
            np.array([[True, False, True], [False, False, False], [True, False, True]]),
            pixel_scale=1.0,
            sub_size=1,
        )

        grid = al.Grid.from_mask(mask=mask)

        assert grid.unlensed_1d == pytest.approx(
            np.array([[1, 0], [0, -1], [0, 0], [0, 1], [-1, 0]]), 1e-4
        )

        grid_via_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((3, 3), False), pixel_scales=(1.0, 1.0), sub_size=1
        )

        assert grid.unlensed_unmasked_1d == pytest.approx(grid_via_util, 1e-4)

        mask = al.Mask(
            np.array([[True, False, True], [False, False, False], [True, False, True]]),
            pixel_scale=1.0,
            sub_size=2,
        )

        grid = al.Grid.from_mask(mask=mask)

        grid_via_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((3, 3), False), pixel_scales=(1.0, 1.0), sub_size=2
        )

        assert grid.unlensed_unmasked_1d == pytest.approx(grid_via_util, 1e-4)

    def test__grid_unlensed_2d_property__compare_to_grid_util(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )

        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=1)

        grid = al.Grid(arr=np.array([[1.0, 1.0], [1.0, 1.0]]), mask=mask)

        grid_via_util = al.grid_util.grid_2d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, sub_size=1, pixel_scales=(2.0, 2.0)
        )

        assert grid.unlensed_2d == pytest.approx(grid_via_util, 1e-4)

        grid_via_util = al.grid_util.grid_2d_from_shape_pixel_scales_sub_size_and_origin(
            shape=(3, 4), sub_size=1, pixel_scales=(2.0, 2.0)
        )

        assert grid.unlensed_unmasked_2d == pytest.approx(grid_via_util, 1e-4)

        mask = al.Mask(
            np.array([[True, False, True], [False, False, False], [True, False, True]]),
            pixel_scale=1.0,
            sub_size=1,
        )

        grid = al.Grid.from_mask(mask=mask)

        assert grid.unlensed_2d == pytest.approx(
            np.array(
                [
                    [[0, 0], [1, 0], [0, 0]],
                    [[0, -1], [0, 0], [0, 1]],
                    [[0, 0], [-1, 0], [0, 0]],
                ]
            ),
            1e-4,
        )

        grid_via_util = al.grid_util.grid_2d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((3, 3), False), pixel_scales=(1.0, 1.0), sub_size=1
        )

        assert grid.unlensed_unmasked_2d == pytest.approx(grid_via_util, 1e-4)

        mask = al.Mask(
            np.array([[True, False, True], [False, False, False], [True, False, True]]),
            pixel_scale=1.0,
            sub_size=2,
        )

        grid = al.Grid.from_mask(mask)

        grid_via_util = al.grid_util.grid_2d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((3, 3), False), pixel_scales=(1.0, 1.0), sub_size=2
        )

        assert grid.unlensed_unmasked_2d == pytest.approx(grid_via_util, 1e-4)

    def test__from_shape_and_pixel_scale__compare_to_grid_util(self):
        mask = np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        )
        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=1)

        grid_via_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(3, 4), pixel_scale=2.0, sub_size=1
        )

        assert type(grid) == al.Grid
        assert grid == pytest.approx(grid_via_util, 1e-4)
        assert grid.pixel_scale == 2.0
        assert (
            grid.mapping.mask_1d_index_tomask_index
            == mask.mapping.mask_1d_index_tomask_index
        ).all()

        mask = np.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )

        grid_via_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, pixel_scales=(3.0, 3.0), sub_size=2
        )

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(3, 3), pixel_scale=3.0, sub_size=2
        )

        assert grid == pytest.approx(grid_via_util, 1e-4)

    def test__blurring_grid_from_mask__compare_to_array_util(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, False, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=2)

        blurring_mask_util = al.mask_util.blurring_mask_from_mask_and_psf_shape(
            mask=mask, psf_shape=(3, 5)
        )

        blurring_grid_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid = al.Grid.from_mask(mask=mask)

        blurring_grid = grid.blurring_grid_from_psf_shape(psf_shape=(3, 5))

        blurring_mask = mask.blurring_mask_from_psf_shape(psf_shape=(3, 5))

        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scale == 2.0
        assert (
            blurring_grid.mask.mapping.mask_1d_index_tomask_index
            == blurring_mask.mapping.mask_1d_index_tomask_index
        ).all()
        assert blurring_grid.sub_size == 1

    def test__blurring_grid_from_psf_shape__compare_to_array_util(self):
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

        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=2)

        blurring_mask_util = al.mask_util.blurring_mask_from_mask_and_psf_shape(
            mask=mask, psf_shape=(3, 5)
        )

        blurring_grid_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=2)
        blurring_grid = al.Grid.blurring_grid_from_mask_and_psf_shape(
            mask=mask, psf_shape=(3, 5)
        )

        blurring_mask = mask.blurring_mask_from_psf_shape(psf_shape=(3, 5))

        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scale == 2.0
        assert (
            blurring_grid.mask.mapping.mask_1d_index_tomask_index
            == blurring_mask.mapping.mask_1d_index_tomask_index
        ).all()
        assert blurring_grid.sub_size == 1

    def test__padded_grid_from_psf_shape__matches_grid_2d_after_padding(self):

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(4, 4), pixel_scale=3.0, sub_size=1
        )

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(3, 3))

        padded_grid_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((6, 6), False), pixel_scales=(3.0, 3.0), sub_size=1
        )

        assert padded_grid.shape == (36, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 6))).all()
        assert (padded_grid == padded_grid_util).all()
        assert padded_grid.interpolator is None

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(4, 5), pixel_scale=2.0, sub_size=1
        )

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(3, 3))

        padded_grid_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((6, 7), False), pixel_scales=(2.0, 2.0), sub_size=1
        )

        assert padded_grid.shape == (42, 2)
        assert (padded_grid == padded_grid_util).all()

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(5, 4), pixel_scale=1.0, sub_size=1
        )

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(3, 3))

        padded_grid_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((7, 6), False), pixel_scales=(1.0, 1.0), sub_size=1
        )

        assert padded_grid.shape == (42, 2)
        assert (padded_grid == padded_grid_util).all()

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(5, 5), pixel_scale=8.0, sub_size=1
        )

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(2, 5))

        padded_grid_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_size=1
        )

        assert padded_grid.shape == (54, 2)
        assert (padded_grid == padded_grid_util).all()

        mask = al.Mask(array=np.full((5, 4), False), pixel_scale=2.0, sub_size=2)

        grid = al.Grid.from_mask(mask=mask)

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(3, 3))

        padded_grid_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((7, 6), False), pixel_scales=(2.0, 2.0), sub_size=2
        )

        assert padded_grid.shape == (168, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(7, 6))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)
        assert padded_grid.interpolator is None

        mask = al.Mask(array=np.full((2, 5), False), pixel_scale=8.0, sub_size=4)

        grid = al.Grid.from_mask(mask=mask)

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(5, 5))

        padded_grid_util = al.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_size=4
        )

        assert padded_grid.shape == (864, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 9))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)

    def test__padded_grid_from_psf_shape__has_interpolator_grid_if_had_one_before(self):
        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(4, 4), pixel_scale=3.0, sub_size=1
        )

        grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=0.1)

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(3, 3))

        assert padded_grid.interpolator is not None
        assert padded_grid.interpolator.pixel_scale_interpolation_grid == 0.1

        mask = al.Mask.unmasked_from_shape_pixel_scale_and_sub_size(
            shape=(6, 6), pixel_scale=3.0, sub_size=1
        )

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=padded_grid, pixel_scale_interpolation_grid=0.1
        )

        assert (padded_grid.interpolator.vtx == interpolator.vtx).all()
        assert (padded_grid.interpolator.wts == interpolator.wts).all()

        mask = al.Mask(array=np.full((5, 4), False), pixel_scale=2.0, sub_size=2)

        grid = al.Grid.from_mask(mask=mask)

        grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=0.1)

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=(3, 3))

        assert padded_grid.interpolator is not None
        assert padded_grid.interpolator.pixel_scale_interpolation_grid == 0.1

        mask = al.Mask.unmasked_from_shape_pixel_scale_and_sub_size(
            shape=(7, 6), pixel_scale=2.0, sub_size=2
        )

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=padded_grid, pixel_scale_interpolation_grid=0.1
        )

        assert (padded_grid.interpolator.vtx == interpolator.vtx).all()
        assert (padded_grid.interpolator.wts == interpolator.wts).all()

    def test__trimmed_array_2d_from_padded_array_1d_and_image_shape(self):
        mask = al.Mask(array=np.full((4, 4), False), pixel_scale=1.0, sub_size=1)

        grid = al.Grid(arr=np.empty((0)), mask=mask)

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

        mask = al.Mask(array=np.full((5, 3), False), pixel_scale=1.0, sub_size=1)

        grid = al.Grid(arr=np.empty((0)), mask=mask)

        array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

        array_2d = grid.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(3, 1)
        )

        assert (array_2d == np.array([[5.0], [8.0], [2.0]])).all()

        mask = al.Mask(array=np.full((3, 5), False), pixel_scale=1.0, sub_size=1)

        grid = al.Grid(arr=np.empty((0)), mask=mask)

        array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

        array_2d = grid.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(1, 3)
        )

        assert (array_2d == np.array([[7.0, 8.0, 9.0]])).all()

    def test_sub_mask_1d_index_to_mask_1d_index(self, grid):
        assert (
            grid.mapping.sub_mask_1d_index_to_mask_1d_index == np.array(range(5))
        ).all()

    def test__sub_border_1d_indexes__compare_to_array_util(self):
        mask = np.array(
            [
                [False, False, False, False, False, False, False, True],
                [False, True, True, True, True, True, False, True],
                [False, True, False, False, False, True, False, True],
                [False, True, False, True, False, True, False, True],
                [False, True, False, False, False, True, False, True],
                [False, True, True, True, True, True, False, True],
                [False, False, False, False, False, False, False, True],
            ]
        )

        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=2)

        sub_border_1d_indexes_util = al.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        grid = al.Grid.from_mask(mask=mask)

        assert grid.sub_border_1d_indexes == pytest.approx(
            sub_border_1d_indexes_util, 1e-4
        )

    def test__masked_shape_arcsec(self):
        mask = al.Mask.circular(
            shape=(3, 3), radius_arcsec=1.0, pixel_scale=1.0, sub_size=1
        )

        grid = al.Grid(arr=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=mask)
        assert grid.masked_shape_arcsec == (3.0, 2.0)

        grid = al.Grid(arr=np.array([[1.5, 1.0], [-1.5, -1.0], [0.1, 0.1]]), mask=mask)
        assert grid.masked_shape_arcsec == (3.0, 2.0)

        grid = al.Grid(arr=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0]]), mask=mask)
        assert grid.masked_shape_arcsec == (4.5, 4.0)

        grid = al.Grid(
            arr=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0], [7.0, -5.0]]), mask=mask
        )
        assert grid.masked_shape_arcsec == (8.5, 8.0)

    def test__in_radians(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=1)

        grid = al.Grid.from_mask(mask=mask)

        assert grid.in_radians[0, 0] == pytest.approx(0.00000969627362, 1.0e-8)
        assert grid.in_radians[0, 1] == pytest.approx(0.00000484813681, 1.0e-8)

        assert grid.in_radians[0, 0] == pytest.approx(
            2.0 * np.pi / (180 * 3600), 1.0e-8
        )
        assert grid.in_radians[0, 1] == pytest.approx(
            1.0 * np.pi / (180 * 3600), 1.0e-8
        )

    def test__yticks(self):
        mask = al.Mask.circular(
            shape=(3, 3), radius_arcsec=1.0, pixel_scale=1.0, sub_size=1
        )

        grid = al.Grid(arr=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=mask)
        assert grid.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        grid = al.Grid(arr=np.array([[3.0, 1.0], [-3.0, -1.0]]), mask=mask)
        assert grid.yticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        grid = al.Grid(arr=np.array([[5.0, 3.5], [2.0, -1.0]]), mask=mask)
        assert grid.yticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__xticks(self):
        mask = al.Mask.circular(
            shape=(3, 3), radius_arcsec=1.0, pixel_scale=1.0, sub_size=1
        )

        grid = al.Grid(arr=np.array([[1.0, 1.5], [-1.0, -1.5]]), mask=mask)
        assert grid.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        grid = al.Grid(arr=np.array([[1.0, 3.0], [-1.0, -3.0]]), mask=mask)
        assert grid.xticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        grid = al.Grid(arr=np.array([[3.5, 2.0], [-1.0, 5.0]]), mask=mask)
        assert grid.xticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__new_grid__with_interpolator__returns_grid_with_interpolator(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=1)

        grid = al.Grid.from_mask(mask=mask)

        grid_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=1.0
        )

        assert (grid[:, :] == grid_with_interp[:, :]).all()
        assert grid.mask == grid_with_interp.mask

        interpolator_manual = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=1.0
        )

        assert (grid.interpolator.vtx == interpolator_manual.vtx).all()
        assert (grid.interpolator.wts == interpolator_manual.wts).all()

    def test__new_grid__with_binned__returns_grid_with_binned(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=1)

        grid = al.Grid.from_mask(mask=mask)

        grid_binned = grid.new_grid_with_binned_grid(binned_grid=1)

        assert grid.binned == 1


class TestGridBorder(object):
    def test__border_grid_for_simple_mask(self):
        mask = np.array(
            [
                [False, False, False, False, False, False, False, True],
                [False, True, True, True, True, True, False, True],
                [False, True, False, False, False, True, False, True],
                [False, True, False, True, False, True, False, True],
                [False, True, False, False, False, True, False, True],
                [False, True, True, True, True, True, False, True],
                [False, False, False, False, False, False, False, True],
            ]
        )

        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=2)

        grid = al.Grid.from_mask(mask=mask)

        assert (
            grid.sub_border_grid
            == np.array(
                [
                    [6.5, -7.5],
                    [6.5, -5.5],
                    [6.5, -3.5],
                    [6.5, -0.5],
                    [6.5, 1.5],
                    [6.5, 3.5],
                    [6.5, 5.5],
                    [4.5, -7.5],
                    [4.5, 5.5],
                    [2.5, -7.5],
                ]
            )
        ).all()

    def test__inside_border_no_relocations(self):
        mask = al.Mask.circular(
            shape=(30, 30), radius_arcsec=1.0, pixel_scale=0.1, sub_size=1
        )

        grid = al.Grid.from_mask(mask=mask)

        grid_to_relocate = al.Grid(
            arr=np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert (
            relocated_grid == np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]])
        ).all()
        assert relocated_grid.mask == mask
        assert relocated_grid.sub_size == 1

        mask = al.Mask.circular(
            shape=(30, 30), radius_arcsec=1.0, pixel_scale=0.1, sub_size=2
        )

        grid = al.Grid.from_mask(mask=mask)

        grid_to_relocate = al.Grid(
            arr=np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert (
            relocated_grid == np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]])
        ).all()
        assert relocated_grid.mask == mask
        assert relocated_grid.sub_size == 2

    def test__outside_border_are_relocations(self):
        mask = al.Mask.circular(
            shape=(30, 30), radius_arcsec=1.0, pixel_scale=0.1, sub_size=1
        )

        grid = al.Grid.from_mask(mask=mask)

        grid_to_relocate = al.Grid(
            arr=np.array([[10.1, 0.0], [0.0, 10.1], [-10.1, -10.1]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array([[0.95, 0.0], [0.0, 0.95], [-0.7017, -0.7017]]), 0.1
        )
        assert relocated_grid.mask == mask
        assert relocated_grid.sub_size == 1

        mask = al.Mask.circular(
            shape=(30, 30), radius_arcsec=1.0, pixel_scale=0.1, sub_size=2
        )

        grid = al.Grid.from_mask(mask=mask)

        grid_to_relocate = al.Grid(
            arr=np.array([[10.1, 0.0], [0.0, 10.1], [-10.1, -10.1]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array([[0.9778, 0.0], [0.0, 0.97788], [-0.7267, -0.7267]]), 0.1
        )
        assert relocated_grid.mask == mask
        assert relocated_grid.sub_size == 2

    def test__outside_border_are_relocations__positive_origin_included_in_relocate(
        self
    ):
        mask = al.Mask.circular(
            shape=(60, 60),
            radius_arcsec=1.0,
            pixel_scale=0.1,
            centre=(1.0, 1.0),
            sub_size=1,
        )

        grid = al.Grid.from_mask(mask=mask)

        grid_to_relocate = al.Grid(
            arr=np.array([[11.1, 1.0], [1.0, 11.1], [-11.1, -11.1]]),
            sub_size=1,
            mask=mask,
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array(
                [[2.0, 1.0], [1.0, 2.0], [1.0 - np.sqrt(2) / 2, 1.0 - np.sqrt(2) / 2]]
            ),
            0.1,
        )
        assert relocated_grid.mask == mask
        assert relocated_grid.sub_size == 1

        mask = al.Mask.circular(
            shape=(60, 60),
            radius_arcsec=1.0,
            pixel_scale=0.1,
            centre=(1.0, 1.0),
            sub_size=2,
        )

        grid = al.Grid.from_mask(mask=mask)

        grid_to_relocate = al.Grid(
            arr=np.array([[11.1, 1.0], [1.0, 11.1], [-11.1, -11.1]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array(
                [
                    [1.9263, 1.0 - 0.0226],
                    [1.0 - 0.0226, 1.9263],
                    [1.0 - 0.7267, 1.0 - 0.7267],
                ]
            ),
            0.1,
        )
        assert relocated_grid.mask == mask
        assert relocated_grid.sub_size == 2


class TestBinnedGrid:
    def test__from_mask_and_pixel_scale_binned_grid__correct_binned_bin_up_calculated(
        self, mask_7x7, grid_7x7
    ):
        mask_7x7.pixel_scale = 1.0
        binned_grid = al.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
            mask=mask_7x7, pixel_scale_binned_grid=1.0
        )

        assert (binned_grid == grid_7x7).all()
        assert (binned_grid.mask == mask_7x7).all()
        assert binned_grid.bin_up_factor == 1
        assert (
            binned_grid.binned_mask_1d_index_to_mask_1d_indexes
            == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
        ).all()

        mask_7x7.pixel_scale = 1.0
        binned_grid = al.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
            mask=mask_7x7, pixel_scale_binned_grid=1.9
        )

        assert binned_grid.bin_up_factor == 1
        assert (binned_grid.mask == mask_7x7).all()
        assert (
            binned_grid.binned_mask_1d_index_to_mask_1d_indexes
            == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
        ).all()

        mask_7x7.pixel_scale = 1.0
        binned_grid = al.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
            mask=mask_7x7, pixel_scale_binned_grid=2.0
        )

        assert binned_grid.bin_up_factor == 2
        assert (
            binned_grid.mask
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
            binned_grid
            == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert (
            binned_grid.binned_mask_1d_index_to_mask_1d_indexes
            == np.array([[0, -1, -1, -1], [1, 2, -1, -1], [3, 6, -1, -1], [4, 5, 7, 8]])
        ).all()

        mask_7x7.pixel_scale = 2.0
        binned_grid = al.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
            mask=mask_7x7, pixel_scale_binned_grid=1.0
        )

        assert binned_grid.bin_up_factor == 1


class TestPixelizationGrid:
    def test_pix_grid__attributes(self):
        pix_grid = al.PixelizationGrid(
            arr=np.array([[1.0, 1.0], [2.0, 2.0]]),
            mask_1d_index_to_nearest_pixelization_1d_index=np.array([0, 1]),
        )

        assert type(pix_grid) == al.PixelizationGrid
        assert (pix_grid == np.array([[1.0, 1.0], [2.0, 2.0]])).all()
        assert (
            pix_grid.mask_1d_index_to_nearest_pixelization_1d_index == np.array([0, 1])
        ).all()

    def test__from_unmasked_sparse_shape_and_grid(self):
        mask = al.Mask(
            array=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scale=0.5,
            sub_size=1,
        )

        grid = al.Grid.from_mask(mask=mask)

        sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=(10, 10), grid=grid
        )

        pixelization_grid = al.PixelizationGrid.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=(10, 10), grid=grid
        )

        assert (sparse_to_grid.sparse == pixelization_grid).all()
        assert (
            sparse_to_grid.mask_1d_index_to_sparse_1d_index
            == pixelization_grid.mask_1d_index_to_nearest_pixelization_1d_index
        ).all()


class TestSparseToGrid:
    class TestUnmaskedShape:
        def test__properties_consistent_with_mapping_util(self):
            mask = al.Mask(
                array=np.array(
                    [[True, False, True], [False, False, False], [True, False, True]]
                ),
                pixel_scale=0.5,
                sub_size=1,
            )

            grid = al.Grid.from_mask(mask=mask)

            sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(10, 10), grid=grid
            )

            unmasked_sparse_grid_util = al.grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
                shape=(10, 10), pixel_scales=(0.15, 0.15), sub_size=1, origin=(0.0, 0.0)
            )

            unmasked_sparse_grid_pixel_centres = grid.mask.grid_arcsec_to_grid_pixel_centres(
                grid_arcsec=unmasked_sparse_grid_util
            )

            total_sparse_pixels = al.mask_util.total_sparse_pixels_from_mask(
                mask=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            )

            regular_to_unmasked_sparse_util = al.grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(
                grid_arcsec_1d=grid,
                shape=(10, 10),
                pixel_scales=(0.15, 0.15),
                origin=(0.0, 0.0),
            ).astype(
                "int"
            )

            sparse_to_unmasked_sparse_util = al.sparse_mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
                total_sparse_pixels=total_sparse_pixels,
                mask=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            ).astype(
                "int"
            )

            unmasked_sparse_to_sparse_util = al.sparse_mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(
                mask=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
                total_sparse_pixels=total_sparse_pixels,
            ).astype(
                "int"
            )

            mask_1d_index_to_sparse_1d_index_util = al.sparse_mapping_util.mask_1d_index_to_sparse_1d_index_from_sparse_mappings(
                regular_to_unmasked_sparse=regular_to_unmasked_sparse_util,
                unmasked_sparse_to_sparse=unmasked_sparse_to_sparse_util,
            )

            sparse_grid_util = al.sparse_mapping_util.sparse_grid_from_unmasked_sparse_grid(
                unmasked_sparse_grid=unmasked_sparse_grid_util,
                sparse_to_unmasked_sparse=sparse_to_unmasked_sparse_util,
            )

            assert (
                sparse_to_grid.mask_1d_index_to_sparse_1d_index
                == mask_1d_index_to_sparse_1d_index_util
            ).all()
            assert (sparse_to_grid.sparse == sparse_grid_util).all()

        def test__sparse_grid_overlaps_mask_perfectly__masked_pixels_in_masked_sparse_grid(
            self
        ):
            mask = al.Mask(
                array=np.array(
                    [[True, False, True], [False, False, False], [True, False, True]]
                ),
                pixel_scale=1.0,
                sub_size=1,
            )

            grid = al.Grid.from_mask(mask=mask)

            sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 3), grid=grid
            )

            assert (
                sparse_to_grid.mask_1d_index_to_sparse_1d_index
                == np.array([0, 1, 2, 3, 4])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
                )
            ).all()

        def test__same_as_above_but_4x3_grid_and_mask(self):
            mask = al.Mask(
                array=np.array(
                    [
                        [True, False, True],
                        [False, False, False],
                        [False, False, False],
                        [True, False, True],
                    ]
                ),
                pixel_scale=1.0,
                sub_size=1,
            )

            grid = al.Grid.from_mask(mask=mask)

            sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(4, 3), grid=grid
            )

            assert (
                sparse_to_grid.mask_1d_index_to_sparse_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7])
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
            mask = al.Mask(
                array=np.array(
                    [
                        [True, False, True, True],
                        [False, False, False, False],
                        [True, False, True, True],
                    ]
                ),
                pixel_scale=1.0,
                sub_size=1,
            )

            grid = al.Grid.from_mask(mask=mask)

            sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 4), grid=grid
            )

            assert (
                sparse_to_grid.mask_1d_index_to_sparse_1d_index
                == np.array([0, 1, 2, 3, 4, 5])
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
            mask = al.Mask(
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
                sub_size=1,
            )

            grid = al.Grid.from_mask(mask=mask)

            # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
            # the central (3x3) pixels only.

            sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 3), grid=grid
            )

            assert (
                sparse_to_grid.mask_1d_index_to_sparse_1d_index
                == np.array([0, 1, 2, 3, 4])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [[2.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [0.0, 1.0]]
                )
            ).all()

        def test__same_as_above_but_different_offset(self):
            mask = al.Mask(
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
                sub_size=1,
            )

            grid = al.Grid.from_mask(mask=mask)

            # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
            # the central (3x3) pixels only.

            sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 3), grid=grid
            )

            assert (
                sparse_to_grid.mask_1d_index_to_sparse_1d_index
                == np.array([0, 1, 2, 3, 4])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [[2.0, 2.0], [0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [-2.0, 2.0]]
                )
            ).all()

        def test__from_grid_and_unmasked_shape__sets_up_with_correct_shape_and_pixel_scales(
            self, mask_7x7
        ):
            grid = al.Grid.from_mask(mask=mask_7x7)

            sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                grid=grid, unmasked_sparse_shape=(3, 3)
            )

            assert (
                sparse_to_grid.mask_1d_index_to_sparse_1d_index
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

        def test__same_as_above__but_4x3_image(self):
            mask = al.Mask(
                array=np.array(
                    [
                        [True, False, True],
                        [False, False, False],
                        [False, False, False],
                        [True, False, True],
                    ]
                ),
                pixel_scale=1.0,
                sub_size=1,
            )

            grid = al.Grid.from_mask(mask=mask)

            sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(4, 3), grid=grid
            )

            assert (
                sparse_to_grid.mask_1d_index_to_sparse_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7])
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
            mask = al.Mask(
                array=np.array(
                    [
                        [True, False, True, True],
                        [False, False, False, False],
                        [True, False, True, True],
                    ]
                ),
                pixel_scale=1.0,
                sub_size=1,
            )

            grid = al.Grid.from_mask(mask=mask)

            sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 4), grid=grid
            )

            assert (
                sparse_to_grid.mask_1d_index_to_sparse_1d_index
                == np.array([0, 1, 2, 3, 4, 5])
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

        def test__from_grid_and_shape__offset_mask__origin_shift_corrects(self):
            mask = al.Mask(
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
                sub_size=1,
            )

            grid = al.Grid.from_mask(mask=mask)

            sparse_to_grid = al.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 3), grid=grid
            )

            assert (
                sparse_to_grid.mask_1d_index_to_sparse_1d_index
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

    class TestUnmaskeedShapeAndWeightImage:
        def test__binned_weight_map_all_ones__kmenas_grid_is_grid_overlapping_image(
            self
        ):
            mask = al.Mask(
                array=np.array(
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ),
                pixel_scale=0.5,
                sub_size=1,
            )

            binned_grid = al.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
                mask=mask, pixel_scale_binned_grid=mask.pixel_scale
            )

            binned_weight_map = np.ones(mask.pixels_in_mask)

            sparse_to_grid_weight = al.SparseToGrid.from_total_pixels_binned_grid_and_weight_map(
                total_pixels=8,
                binned_grid=binned_grid,
                binned_weight_map=binned_weight_map,
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
                sparse_to_grid_weight.mask_1d_index_to_sparse_1d_index
                == np.array([1, 1, 2, 2, 1, 1, 3, 3, 5, 4, 0, 7, 5, 4, 6, 6])
            ).all()

        def test__binned_weight_map_changes_grid_from_above(self):
            mask = al.Mask(
                array=np.array(
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ),
                pixel_scale=0.5,
                sub_size=2,
            )

            binned_grid = al.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
                mask=mask, pixel_scale_binned_grid=mask.pixel_scale
            )

            binned_weight_map = np.ones(mask.pixels_in_mask)
            binned_weight_map[0:15] = 0.00000001

            sparse_to_grid_weight = al.SparseToGrid.from_total_pixels_binned_grid_and_weight_map(
                total_pixels=8,
                binned_grid=binned_grid,
                binned_weight_map=binned_weight_map,
                n_iter=10,
                max_iter=30,
                seed=1,
            )

            assert sparse_to_grid_weight.sparse[1] == pytest.approx(
                np.array([0.4166666, -0.0833333]), 1.0e-4
            )

            assert (
                sparse_to_grid_weight.mask_1d_index_to_sparse_1d_index
                == np.array([5, 1, 0, 0, 5, 1, 1, 4, 3, 6, 7, 4, 3, 6, 2, 2])
            ).all()

        def test__binned_weight_map_all_ones__pixel_scale_binned_grid_leads_to_binning_up_by_factor_2(
            self
        ):
            mask = al.Mask(
                array=np.full(fill_value=False, shape=(8, 8)),
                pixel_scale=0.5,
                sub_size=2,
            )

            binned_grid = al.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
                mask=mask, pixel_scale_binned_grid=2.0 * mask.pixel_scale
            )

            binned_weight_map = np.ones(binned_grid.shape[0])

            sparse_to_grid_weight = al.SparseToGrid.from_total_pixels_binned_grid_and_weight_map(
                total_pixels=8,
                binned_grid=binned_grid,
                binned_weight_map=binned_weight_map,
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
                sparse_to_grid_weight.mask_1d_index_to_sparse_1d_index
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


class TestInterpolator:
    def test_decorated_function__values_from_function_has_1_dimensions__returns_1d_result(
        self
    ):
        # noinspection PyUnusedLocal
        @al.grids.grid_interpolate
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

        grid = al.Grid.from_mask(
            mask=al.Mask.unmasked_from_shape_pixel_scale_and_sub_size(
                shape=(3, 3), pixel_scale=1, sub_size=1
            )
        )

        values = func(None, grid)

        assert values.ndim == 1
        assert values.shape == (9,)
        assert (values == np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])).all()

        grid = al.Grid.from_mask(
            mask=al.Mask.unmasked_from_shape_pixel_scale_and_sub_size(
                shape=(3, 3), pixel_scale=1, sub_size=1
            )
        )
        grid.interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            grid.mask, grid, pixel_scale_interpolation_grid=0.5
        )
        interp_values = func(None, grid)
        assert interp_values.ndim == 1
        assert interp_values.shape == (9,)
        assert (interp_values != np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])).any()

    def test_decorated_function__values_from_function_has_2_dimensions__returns_2d_result(
        self
    ):
        # noinspection PyUnusedLocal
        @al.grids.grid_interpolate
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

        grid = al.Grid.from_mask(
            mask=al.Mask.unmasked_from_shape_pixel_scale_and_sub_size(
                shape=(3, 3), pixel_scale=1, sub_size=1
            )
        )

        values = func(None, grid)

        assert values.ndim == 2
        assert values.shape == (9, 2)
        assert (
            values
            == np.array(
                [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            )
        ).all()

        grid = al.Grid.from_mask(
            mask=al.Mask.unmasked_from_shape_pixel_scale_and_sub_size(
                shape=(3, 3), pixel_scale=1, sub_size=1
            )
        )
        grid.interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            grid.mask, grid, pixel_scale_interpolation_grid=0.5
        )

        interp_values = func(None, grid)
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

        mask = al.Mask.circular_annular(
            shape=shape,
            pixel_scale=pixel_scale,
            sub_size=1,
            inner_radius_arcsec=3.0,
            outer_radius_arcsec=8.0,
        )

        grid = al.Grid.from_mask(mask=mask)

        isothermal = al.mass_profiles.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        true_deflections = isothermal.deflections_from_grid(grid=grid)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=1.0
        )

        interp_deflections_values = isothermal.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
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

        mask = al.Mask.circular_annular(
            shape=shape,
            pixel_scale=pixel_scale,
            sub_size=1,
            inner_radius_arcsec=3.0,
            outer_radius_arcsec=8.0,
            centre=(3.0, 3.0),
        )

        grid = al.Grid.from_mask(mask=mask)

        isothermal = al.mass_profiles.SphericalIsothermal(
            centre=(3.0, 3.0), einstein_radius=1.0
        )

        true_deflections = isothermal.deflections_from_grid(grid=grid)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=1.0
        )

        interp_deflections_values = isothermal.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
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

        mask = al.Mask.circular_annular(
            shape=shape,
            pixel_scale=pixel_scale,
            sub_size=1,
            inner_radius_arcsec=3.0,
            outer_radius_arcsec=8.0,
            centre=(3.0, 3.0),
        )

        grid = al.Grid.from_mask(mask=mask)

        isothermal = al.mass_profiles.SphericalIsothermal(
            centre=(3.0, 3.0), einstein_radius=1.0
        )

        true_deflections = isothermal.deflections_from_grid(grid=grid)

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.2
        )

        interp_deflections_values = isothermal.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
        )

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.001
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.001

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = isothermal.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
        )

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.01
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.01

        interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=1.1
        )

        interp_deflections_values = isothermal.deflections_from_grid(
            grid=interpolator.interp_grid, bypass_decorator=True
        )

        interpolated_deflections_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interpolated_deflections_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert np.max(true_deflections[:, 0] - interpolated_deflections_y) < 0.1
        assert np.max(true_deflections[:, 1] - interpolated_deflections_x) < 0.1
