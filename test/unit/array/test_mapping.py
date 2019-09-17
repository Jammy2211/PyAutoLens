import autolens as al

import numpy as np
import pytest


class TestMapping:
    def test__mask_1d_index_tomask_index__compare_to_array_util(self):

        mask = np.array([[True, True, True], [True, False, False], [True, True, False]])

        mask = al.Mask(array=mask, pixel_scale=7.0, sub_size=1)

        mapping = al.Mapping(mask=mask)

        mask_1d_index_tomask_index = al.mask_mapping_util.sub_mask_1d_index_to_submask_index_from_mask_and_sub_size(
            mask=mask, sub_size=1
        )

        assert mapping.mask_1d_index_tomask_index == pytest.approx(
            mask_1d_index_tomask_index, 1e-4
        )

    def test__array_1d_from_array_2d__compare_to_array_util(self):
        array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        mask = np.array(
            [
                [True, False, True],
                [False, False, False],
                [True, False, True],
                [True, True, True],
            ]
        )

        array_1d_util = al.array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        mask = al.Mask(array=mask, pixel_scale=3.0, sub_size=1)

        mapping = al.Mapping(mask=mask)

        array_1d = mapping.array_1d_from_array_2d(array_2d=array_2d)

        assert (array_1d == array_1d_util).all()

    def test__scaled_array_2d_from_array_1d__compare_to_util(self):
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

        mask = al.Mask(array=mask, pixel_scale=3.0, sub_size=1)

        mapping = al.Mapping(mask=mask)

        array_2d = mapping.array_2d_from_array_1d(array_1d=array_1d)

        assert (array_2d == array_2d_util).all()

        scaled_array_2d = mapping.scaled_array_2d_from_array_1d(array_1d=array_1d)

        assert (scaled_array_2d == array_2d_util).all()
        assert (scaled_array_2d.xticks == np.array([-6.0, -2.0, 2.0, 6.0])).all()
        assert (scaled_array_2d.yticks == np.array([-4.5, -1.5, 1.5, 4.5])).all()
        assert scaled_array_2d.shape_arcsec == (9.0, 12.0)
        assert scaled_array_2d.pixel_scale == 3.0
        assert scaled_array_2d.origin == (0.0, 0.0)

    def test__grid_2d_from_grid_1d__compare_to_util(self):
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

        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=1)

        mapping = al.Mapping(mask=mask)
        grid_2d = mapping.grid_2d_from_grid_1d(grid_1d=grid_1d)

        assert (grid_2d_util == grid_2d).all()

    def test__grid_1d_from_grid_2d__compare_to_util(self):
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

        grid_1d_util = al.grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=grid_2d, mask=mask, sub_size=1
        )

        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=1)

        mapping = al.Mapping(mask=mask)
        grid_1d = mapping.grid_1d_from_grid_2d(grid_2d=grid_2d)

        assert (grid_1d_util == grid_1d).all()

    def test__sub_mask_1d_index_to_submask_index__compare_to_array_util(self):
        mask = np.array([[True, True, True], [True, False, False], [True, True, False]])

        mask = al.Mask(array=mask, pixel_scale=7.0, sub_size=2)

        mapping = al.Mapping(mask=mask)

        sub_mask_1d_index_to_submask_index = al.mask_mapping_util.sub_mask_1d_index_to_submask_index_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert mapping.sub_mask_1d_index_to_submask_index == pytest.approx(
            sub_mask_1d_index_to_submask_index, 1e-4
        )

    def test__sub_array_2d_from_sub_array_1d__use_real_mask_and_grid(self):
        mask = np.array([[False, True], [False, False]])
        mask = al.Mask(array=mask, pixel_scale=3.0, sub_size=2)
        mapping = al.Mapping(mask=mask)

        sub_array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        )

        sub_array_2d = mapping.sub_array_2d_from_sub_array_1d(sub_array_1d=sub_array_1d)

        assert (
            sub_array_2d
            == np.array(
                [
                    [1.0, 2.0, 0.0, 0.0],
                    [3.0, 4.0, 0.0, 0.0],
                    [9.0, 10.0, 13.0, 14.0],
                    [11.0, 12.0, 15.0, 16.0],
                ]
            )
        ).all()

    def test__sub_array_2d_from_sub_array_1d__use_2x3_mask(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array=mask, pixel_scale=3.0, sub_size=2)
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

    def test__scaled_sub_array_2d_from_sub_array_1d(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array=mask, pixel_scale=3.0, sub_size=2)
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

        scaled_sub_array_2d = mapping.scaled_array_2d_with_sub_dimensions_from_sub_array_1d(
            sub_array_1d=sub_array_1d
        )

        assert (
            scaled_sub_array_2d
            == np.array(
                [
                    [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                    [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                    [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                    [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                ]
            )
        ).all()

        assert scaled_sub_array_2d.pixel_scales == (1.5, 1.5)
        assert scaled_sub_array_2d.origin == (0.0, 0.0)

    def test__scaled_array_from_sub_array_1d_by_binning_up(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array=mask, pixel_scale=3.0, sub_size=2)
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

        scaled_array_2d = mapping.scaled_array_2d_binned_from_sub_array_1d(
            sub_array_1d=sub_array_1d
        )

        assert (scaled_array_2d == np.array([[3.5, 2.0, 0.0], [3.0, 0.0, 2.0]])).all()

        assert scaled_array_2d.pixel_scales == (3.0, 3.0)
        assert scaled_array_2d.origin == (0.0, 0.0)

    def test__grid_1d_binned_from_sub_grid_1d__use_real_mask_and_grid(self):
        mask = np.array([[False, True], [False, False]])
        mask = al.Mask(array=mask, pixel_scale=3.0, sub_size=2)
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

        grid_1d_binned = mapping.grid_1d_binned_from_sub_grid_1d(sub_grid_1d=grid_1d)

        assert (
            grid_1d_binned == np.array([[2.5, 3.0], [10.5, 10.5], [14.5, 14.5]])
        ).all()

    def test__grid_2d_binned_from_sub_grid_1d__use_real_mask_and_grid(self):
        mask = np.array([[False, True], [False, False]])
        mask = al.Mask(array=mask, pixel_scale=3.0, sub_size=2)
        mapping = al.Mapping(mask=mask)

        sub_grid_1d = np.array(
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

        grid_2d_binned = mapping.grid_2d_binned_from_sub_grid_1d(
            sub_grid_1d=sub_grid_1d
        )

        assert (
            grid_2d_binned
            == np.array([[[2.5, 3.0], [0.0, 0.0]], [[10.5, 10.5], [14.5, 14.5]]])
        ).all()

    def test__sub_grid_1d_with_sub_dimensions_from_sub_grid_2d__use_real_mask_and_grid(
        self
    ):
        mask = np.array([[False, True], [False, False]])
        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=2)
        mapping = al.Mapping(mask=mask)

        sub_grid_2d = np.array(
            [
                [[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-1.0, -1.0]],
                [[3.0, 3.0], [4.0, 4.0], [-1.0, -1.0], [-1.0, -1.0]],
                [[5.0, 5.0], [6.0, 6.0], [9.0, 9.0], [10.0, 10.0]],
                [[7.0, 7.0], [8.0, 8.0], [11.0, 11.0], [12.0, 12.0]],
            ]
        )

        sub_grid_1d = mapping.sub_grid_1d_with_sub_dimensions_from_sub_grid_2d(
            sub_grid_2d=sub_grid_2d
        )

        assert (
            sub_grid_1d
            == np.array(
                [
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                    [4.0, 4.0],
                    [5.0, 5.0],
                    [6.0, 6.0],
                    [7.0, 7.0],
                    [8.0, 8.0],
                    [9.0, 9.0],
                    [10.0, 10.0],
                    [11.0, 11.0],
                    [12.0, 12.0],
                ]
            )
        ).all()

    def test__sub_grid_2d_with_sub_dimensions_from_sub_grid_1d__use_real_mask_and_grid(
        self
    ):
        mask = np.array([[False, True], [False, False]])
        mask = al.Mask(array=mask, pixel_scale=3.0, sub_size=2)
        mapping = al.Mapping(mask=mask)

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

        sub_grid_2d = mapping.sub_grid_2d_with_sub_dimensions_from_sub_grid_1d(
            sub_grid_1d=sub_grid_1d
        )

        assert (
            sub_grid_2d
            == np.array(
                [
                    [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                    [[3.0, 3.0], [4.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
                    [[9.0, 9.0], [10.0, 10.0], [13.0, 13.0], [14.0, 14.0]],
                    [[11.0, 11.0], [12.0, 12.0], [15.0, 15.0], [16.0, 16.0]],
                ]
            )
        ).all()

    def test__sub_grid_2d_from_sub_grid_1d__use_2x3_mask(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array=mask, pixel_scale=3.0, sub_size=2)
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

        sub_grid_2d = mapping.sub_grid_2d_with_sub_dimensions_from_sub_grid_1d(
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

    def test__sub_array_1d_from_sub_array_2d__numerical_values(self):
        sub_array_2d = np.array(
            [
                [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
            ]
        )

        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=2)
        mapping = al.Mapping(mask=mask)

        sub_array_1d = mapping.sub_array_1d_with_sub_dimensions_from_sub_array_2d(
            sub_array_2d=sub_array_2d
        )

        assert (
            sub_array_1d
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

    def test__sub_array_1d_from_sub_array_2d__compare_to_util(self):
        sub_array_2d = np.array(
            [
                [1.0, 1.0, 2.0, 2.0, 3.0, 10.0],
                [1.0, 1.0, 2.0, 2.0, 3.0, 10.0],
                [3.0, 3.0, 8.0, 1.0, 4.0, 4.0],
                [3.0, 3.0, 7.0, 2.0, 4.0, 4.0],
            ]
        )

        mask = np.array([[False, False, False], [True, True, False]])
        mask = al.Mask(array=mask, pixel_scale=2.0, sub_size=2)
        mapping = al.Mapping(mask=mask)

        sub_array_1d = mapping.sub_array_1d_with_sub_dimensions_from_sub_array_2d(
            sub_array_2d=sub_array_2d
        )

        sub_array_1d_util = al.array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_size=2
        )

        assert (sub_array_1d == sub_array_1d_util).all()

    def test__sub_array_to_1d_and_2d_and_back__returns_original_array(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mask = al.Mask(array=mask, pixel_scale=3.0, sub_size=2)
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

        sub_array_1d_new = mapping.sub_array_1d_with_sub_dimensions_from_sub_array_2d(
            sub_array_2d=sub_array_2d
        )

        assert (sub_array_1d == sub_array_1d_new).all()

    def test__sub_mask_1d_index_to_mask_1d_index__compare_to_util(self):
        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, False]]
        )

        sub_mask_1d_index_to_mask_1d_index_util = al.mask_mapping_util.sub_mask_1d_index_to_mask_1d_index_from_mask(
            mask=mask, sub_size=2
        )
        mask = al.Mask(array=mask, pixel_scale=3.0, sub_size=2)
        mapping = al.Mapping(mask=mask)

        assert (
            mapping.sub_mask_1d_index_to_mask_1d_index
            == sub_mask_1d_index_to_mask_1d_index_util
        ).all()


class MockObject(object):
    def __init__(self, values):
        self.values = values

    @al.mapping.reshape_returned_sub_array
    def array_from_grid(
        self, grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    ):
        return self.values

    @al.mapping.reshape_returned_grid
    def grid_from_grid(
        self, grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    ):
        return self.values


class TestMappingArrayDecorator(object):
    def test__array_1d_from_function__decorator_changes_array_dimensions_depending_on_inputs(
        self
    ):

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=0.1, sub_size=1
        )

        obj = MockObject(values=np.ones(4))

        array_via_decorator = obj.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (array_via_decorator == np.ones(4)).all()

        array_via_decorator = obj.array_from_grid(
            grid=grid, values=np.ones(4), return_in_2d=True, return_binned=False
        )

        assert (array_via_decorator == np.ones((2, 2))).all()

        obj = MockObject(values=np.ones(16))

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=0.1, sub_size=2
        )

        array_via_decorator = obj.array_from_grid(
            grid=grid, values=np.ones(16), return_in_2d=False, return_binned=False
        )

        assert (array_via_decorator == np.ones(16)).all()

        array_via_decorator = obj.array_from_grid(
            grid=grid, values=np.ones(16), return_in_2d=True, return_binned=False
        )

        assert (array_via_decorator == np.ones((4, 4))).all()

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

        obj = MockObject(values=sub_array)

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=0.1, sub_size=2
        )

        array_via_decorator = obj.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (array_via_decorator == sub_array).all()

        array_via_decorator = obj.array_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (
            array_via_decorator
            == np.array(
                [
                    [1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 4.0, 4.0],
                    [3.0, 3.0, 4.0, 4.0],
                ]
            )
        ).all()

        array_via_decorator = obj.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        assert (array_via_decorator == np.array([1.0, 2.0, 3.0, 4.0])).all()

        array_via_decorator = obj.array_from_grid(
            grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
        )

        assert (array_via_decorator == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

    def test__returned_array_from_function_is_2d__grid_in__decorator_convert_dimensions_to_1d_first(
        self
    ):
        obj = MockObject(values=np.ones((4, 4)))

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(4, 4), pixel_scale=0.1, sub_size=1
        )

        array_via_decorator = obj.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (array_via_decorator == np.array(np.ones(16))).all()

        array_via_decorator = obj.array_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (array_via_decorator == np.array(np.ones((4, 4)))).all()

    def test__returned_array_from_function_is_2d__sub_grid_in__decorator_converts_dimensions_to_1_first(
        self
    ):
        obj = MockObject(values=np.ones((8, 8)))

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(4, 4), pixel_scale=0.1, sub_size=2
        )

        array_via_decorator = obj.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (array_via_decorator == np.ones(64)).all()

        array_via_decorator = obj.array_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (array_via_decorator == np.ones((8, 8))).all()

        array_via_decorator = obj.array_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        assert (array_via_decorator == np.ones(16)).all()

        array_via_decorator = obj.array_from_grid(
            grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
        )

        assert (array_via_decorator == np.ones((4, 4))).all()


class TestMappingGridDecorator(object):
    def test__grid_1d_from_function__decorator_changes_grid_dimensions_to_2d(self):
        obj = MockObject(values=np.ones(shape=(4, 2)))

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=0.1, sub_size=1
        )

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (grid_via_decorator == np.ones((4, 2))).all()

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (grid_via_decorator == np.ones((2, 2, 2))).all()

        obj = MockObject(values=np.ones(shape=(16, 2)))

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=0.1, sub_size=2
        )

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (grid_via_decorator == np.ones(shape=(16, 2))).all()

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (grid_via_decorator == np.ones(shape=(4, 4, 2))).all()

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

        obj = MockObject(values=sub_grid_values)

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(2, 2), pixel_scale=0.1, sub_size=2
        )

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (grid_via_decorator == sub_grid_values).all()

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (
            grid_via_decorator
            == np.array(
                [
                    [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]],
                    [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]],
                    [[3.0, 3.0], [3.0, 3.0], [4.0, 4.0], [4.0, 4.0]],
                    [[3.0, 3.0], [3.0, 3.0], [4.0, 4.0], [4.0, 4.0]],
                ]
            )
        ).all()

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        assert (
            grid_via_decorator
            == np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        ).all()

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
        )

        assert (
            grid_via_decorator
            == np.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])
        ).all()

    def test__grid_of_function_output_in_2d__decorator_converts_to_1d_first(self):
        obj = MockObject(values=np.ones(shape=(4, 4, 2)))

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(4, 4), pixel_scale=0.1, sub_size=1
        )

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (grid_via_decorator == np.array(np.ones(shape=(16, 2)))).all()

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (grid_via_decorator == np.array(np.ones(shape=(4, 4, 2)))).all()

    def test__sub_grid_of_function_returned_in_2d__decorator_changes_grid_dimensions_to_2d(
        self
    ):
        obj = MockObject(values=np.ones(shape=(8, 8, 2)))

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(4, 4), pixel_scale=0.1, sub_size=2
        )

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        assert (grid_via_decorator == np.ones((64, 2))).all()

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        assert (grid_via_decorator == np.ones((8, 8, 2))).all()

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        assert (grid_via_decorator == np.ones((16, 2))).all()

        grid_via_decorator = obj.grid_from_grid(
            grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
        )

        assert (grid_via_decorator == np.ones((4, 4, 2))).all()
