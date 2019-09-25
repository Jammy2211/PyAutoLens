import autolens as al

import numpy as np
import pytest


class TestMappingScaledArray:

    def test__mask_1d_index_to_mask_2d_index__compare_to_array_util(self):

        mask = np.array([[True, True, True], [True, False, False], [True, True, False]])

        mask = al.Mask(array_2d=mask, pixel_scales=(7.0, 7.0), sub_size=1)

        mapping = al.Mapping(mask=mask)

        mask_1d_index_to_maskk_2d_index = al.mask_mapping_util.sub_mask_1d_index_to_sub_mask_2d_index_from_mask_and_sub_size(
            mask=mask, sub_size=1
        )

        assert mapping.mask_1d_index_to_mask_2d_index == pytest.approx(
            mask_1d_index_to_maskk_2d_index, 1e-4
        )

    def test__sub_mask_1d_index_to_mask_1d_index__compare_to_util(self):
        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, False]]
        )

        sub_mask_1d_index_to_mask_1d_index_util = al.mask_mapping_util.sub_mask_1d_index_to_mask_1d_index_from_mask(
            mask=mask, sub_size=2
        )
        mask = al.Mask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)
        mapping = al.Mapping(mask=mask)

        assert (
            mapping.sub_mask_1d_index_to_mask_1d_index
            == sub_mask_1d_index_to_mask_1d_index_util
        ).all()

    def test__sub_mask_1d_index_to_sub_mask_2d_index__compare_to_array_util(self):
        mask = np.array([[True, True, True], [True, False, False], [True, True, False]])

        mask = al.Mask(array_2d=mask, pixel_scales=(7.0, 7.0), sub_size=2)

        mapping = al.Mapping(mask=mask)

        sub_mask_1d_index_to_sub_mask_2d_index = al.mask_mapping_util.sub_mask_1d_index_to_sub_mask_2d_index_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert mapping.sub_mask_1d_index_to_sub_mask_2d_index == pytest.approx(
            sub_mask_1d_index_to_sub_mask_2d_index, 1e-4
        )

    def test__scaled_array_from_array_1d__compare_to_util(self):

        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )

        array_1d = np.array([1.0, 6.0, 4.0, 5.0, 2.0])

        array_2d_util = al.array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=array_1d, mask=mask, sub_size=1
        )

        masked_array_2d = array_2d_util * np.invert(mask)

        mask = al.Mask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=1)

        mapping = al.Mapping(mask=mask)

        scaled_array = mapping.scaled_array_from_array_1d(array_1d=array_1d)

        assert (scaled_array == array_1d).all()
        assert (scaled_array.in_1d == array_1d).all()
        assert (scaled_array.in_2d == masked_array_2d).all()
        assert (scaled_array.geometry.xticks == np.array([-6.0, -2.0, 2.0, 6.0])).all()
        assert (scaled_array.geometry.yticks == np.array([-4.5, -1.5, 1.5, 4.5])).all()
        assert scaled_array.geometry.shape_arcsec == (9.0, 12.0)
        assert scaled_array.geometry.pixel_scale == 3.0
        assert scaled_array.geometry.origin == (0.0, 0.0)

    def test__scaled_array_from_array_2d__compare_to_util(self):

        array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        mask = np.array(
            [
                [True, False, True],
                [False, False, False],
                [True, False, True],
                [True, True, True],
            ]
        )

        masked_array_2d = array_2d * np.invert(mask)

        array_1d_util = al.array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        mask = al.Mask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=1)

        mapping = al.Mapping(mask=mask)

        scaled_array = mapping.scaled_array_from_array_2d(array_2d=array_2d)

        assert (scaled_array == array_1d_util).all()
        assert (scaled_array.in_1d == array_1d_util).all()
        assert (scaled_array.in_2d == masked_array_2d).all()

    def test__scaled_array_from_sub_array_1d(self):

        mask = np.array([[False, True], [False, False]])
        mask = al.Mask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)
        mapping = al.Mapping(mask=mask)

        sub_array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        )

        scaled_array = mapping.scaled_array_from_sub_array_1d(sub_array_1d=sub_array_1d)

        assert (
            scaled_array.in_1d
            == sub_array_1d
        ).all()

        assert (
            scaled_array.in_2d
            == np.array(
                [
                    [1.0, 2.0, 0.0, 0.0],
                    [3.0, 4.0, 0.0, 0.0],
                    [9.0, 10.0, 13.0, 14.0],
                    [11.0, 12.0, 15.0, 16.0],
                ]
            )
        ).all()

    def test__scaled_array_from_sub_array_2d(self):
        sub_array_2d = np.array(
            [
                [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
            ]
        )

        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)
        mapping = al.Mapping(mask=mask)

        scaled_array = mapping.scaled_array_from_sub_array_2d(
            sub_array_2d=sub_array_2d
        )

        assert (
            scaled_array.in_1d
            == np.array(
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
        ).all()

        assert (
            scaled_array.in_2d
            == sub_array_2d
        ).all()

    def test__scaled_array_binned_from_sub_array_1d_by_binning_up(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)
        mapping = al.Mapping(mask=mask)

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

        scaled_array = mapping.scaled_array_binned_from_sub_array_1d(
            sub_array_1d=sub_array_1d
        )

        assert (scaled_array.in_1d == np.array([3.5, 2.0, 3.0, 2.0])).all()
        assert (scaled_array.in_2d == np.array([[3.5, 2.0, 0.0], [3.0, 0.0, 2.0]])).all()
        assert scaled_array.geometry.pixel_scales == (3.0, 3.0)
        assert scaled_array.geometry.origin == (0.0, 0.0)

    def test__sub_array_2d_from_sub_array_1d__use_2x3_mask(self):

        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)
        mapping = al.Mapping(mask=mask)

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

        sub_array_2d = mapping.sub_array_2d_from_sub_array_1d(sub_array_1d=sub_array_1d)

        assert (
            sub_array_2d
            == np.array(
                [
                    [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                    [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                    [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                    [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                ]
            )
        ).all()

    def test__sub_array_2d_binned_from_sub_array_1d(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)
        mapping = al.Mapping(mask=mask)

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

        sub_array_2d = mapping.sub_array_2d_binned_from_sub_array_1d(
            sub_array_1d=sub_array_1d
        )

        assert (sub_array_2d == np.array([[3.5, 2.0, 0.0], [3.0, 0.0, 2.0]])).all()

    def test__sub_array_to_1d_and_2d_and_back__returns_original_array(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)
        mapping = al.Mapping(mask=mask)

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

        sub_array_2d = mapping.sub_array_2d_from_sub_array_1d(sub_array_1d=sub_array_1d)

        scaled_array = mapping.scaled_array_from_sub_array_2d(
            sub_array_2d=sub_array_2d
        )

        assert (sub_array_1d == scaled_array.in_1d).all()

class TestMappingGrids:

    def test__grid_from_grid_2d__compare_to_util(self):

        grid_2d = np.array(
            [
                [[1, 1], [2, 2], [3, 3], [4, 4]],
                [[5, 5], [6, 6], [7, 7], [8, 8]],
                [[9, 9], [10, 10], [11, 11], [12, 12]],
            ]
        )

        mask = np.array(
            [
                [True, False, True, True],
                [False, False, False, True],
                [True, False, True, False],
            ]
        )

        masked_grid_2d = grid_2d * np.invert(mask[:, :, None])

        grid_1d_util = al.grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=grid_2d, mask=mask, sub_size=1
        )

        mask = al.Mask(array_2d=mask, pixel_scales=(2.0, 2.0), sub_size=1)

        mapping = al.Mapping(mask=mask)
        grid = mapping.grid_from_grid_2d(grid_2d=grid_2d)

        assert (grid == grid_1d_util).all()
        assert (grid.in_1d == grid).all()
        assert (grid.in_2d == masked_grid_2d).all()

    def test__grid_from_grid_1d__compare_to_util(self):

        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )

        grid_1d = np.array([[1.0, 1.0], [6.0, 6.0], [4.0, 4.0], [5.0, 5.0], [2.0, 2.0]])

        grid_2d_util = al.grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=grid_1d, mask=mask, sub_size=1
        )

        masked_grid_2d = grid_2d_util * np.invert(mask[:, :, None])

        mask = al.Mask(array_2d=mask, pixel_scales=(2.0, 2.0), sub_size=1)

        mapping = al.Mapping(mask=mask)
        grid = mapping.grid_from_grid_1d(grid_1d=grid_1d)

        assert (grid == grid_1d).all()
        assert (grid.in_1d == grid_1d).all()
        assert (grid.in_2d == masked_grid_2d).all()

    def test__grid_from_sub_grid_1d(self):

        mask = np.array([[False, True], [False, False]])
        mask = al.Mask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)
        mapping = al.Mapping(mask=mask)

        sub_grid_1d = np.array(
            [
                [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],
                [5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0],
                [9.0, 9.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0],
            ]
        )

        grid = mapping.grid_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

        assert (
            grid.in_1d
            == sub_grid_1d
        ).all()

        assert (grid.in_2d == np.array(
                [[[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                 [[3.0, 3.0], [4.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
                    [[5.0, 5.0], [6.0, 6.0], [9.0, 9.0], [10.0, 10.0]],
                    [[7.0, 7.0], [8.0, 8.0], [11.0, 11.0], [12.0, 12.0]],
                ]
            )).all()

    def test__grid_from_sub_grid_2d(self):

        sub_grid_2d = np.array(
            [
                [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                [[3.0, 3.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0], [4.0, 4.0], [4.0, 4.0]],
                [[3.0, 3.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0], [4.0, 4.0], [4.0, 4.0]],
            ]
        )

        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)
        mapping = al.Mapping(mask=mask)

        grid = mapping.grid_from_sub_grid_2d(
            sub_grid_2d=sub_grid_2d
        )

        assert (
            grid.in_1d
            == np.array(
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
        ).all()

        assert (
            grid.in_2d
            == sub_grid_2d
        ).all()

    def test__grid_binned_from_sub_grid_1d(self):

        mask = np.array([[False, True], [False, False]])
        mask = al.Mask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)
        mapping = al.Mapping(mask=mask)

        grid_1d = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 6.0],
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

        grid = mapping.grid_binned_from_sub_grid_1d(sub_grid_1d=grid_1d)

        assert (
            grid.in_1d == np.array([[2.5, 3.0], [10.5, 10.5], [14.5, 14.5]])
        ).all()

        assert (
            grid.in_2d == np.array([[[2.5, 3.0], [0.0, 0.0]],
                                    [[10.5, 10.5], [14.5, 14.5]]])
        ).all()

    def test__sub_grid_2d_from_sub_grid_1d(self):

        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)
        mapping = al.Mapping(mask=mask)

        sub_grid_1d = np.array(
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

        sub_grid_2d = mapping.sub_grid_2d_from_sub_grid_1d(
            sub_grid_1d=sub_grid_1d
        )

        assert (
            sub_grid_2d
            == np.array(
                [
                    [
                        [1.0, 1.0],
                        [1.0, 1.0],
                        [2.0, 2.0],
                        [2.0, 2.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    [
                        [1.0, 1.0],
                        [1.0, 1.0],
                        [2.0, 2.0],
                        [2.0, 2.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    [
                        [3.0, 3.0],
                        [3.0, 3.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [4.0, 4.0],
                        [4.0, 4.0],
                    ],
                    [
                        [3.0, 3.0],
                        [3.0, 3.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [4.0, 4.0],
                        [4.0, 4.0],
                    ],
                ]
            )
        ).all()

    def test__sub_grid_2d_binned_from_sub_grid_1d(self):

        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)
        mapping = al.Mapping(mask=mask)

        sub_grid_1d = np.array(
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

        sub_grid_2d_binned = mapping.sub_grid_2d_binned_from_sub_grid_1d(
            sub_grid_1d=sub_grid_1d
        )

        assert (
                sub_grid_2d_binned
                == np.array(
            [
                [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
                [[3.0, 3.0], [0.0, 0.0], [4.0, 4.0]],
            ]
        )
        ).all()

class TestMappingPaddedTrimmedGrids:

    def test__trimmed_array_2d_from_padded_array_1d_and_image_shape(self):
        mask = al.Mask(array_2d=np.full((4, 4), False), pixel_scales=(1.0, 1.0), sub_size=1)

        mapping = al.Mapping(mask=mask)

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

        array_2d = mapping.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(2, 2)
        )

        assert (array_2d == np.array([[6.0, 7.0], [1.0, 2.0]])).all()

        mask = al.Mask(array_2d=np.full((5, 3), False), pixel_scales=(1.0, 1.0), sub_size=1)

        mapping = al.Mapping(mask=mask)

        array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

        array_2d = mapping.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(3, 1)
        )

        assert (array_2d == np.array([[5.0], [8.0], [2.0]])).all()

        mask = al.Mask(array_2d=np.full((3, 5), False), pixel_scales=(1.0, 1.0), sub_size=1)

        mapping = al.Mapping(mask=mask)

        array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

        array_2d = mapping.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(1, 3)
        )

        assert (array_2d == np.array([[7.0, 8.0, 9.0]])).all()