import autolens as al
import os

import numpy as np


test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestSubGrid1DFromSubGrid2D(object):
    def test__map_simple_grids__sub_grid_1(self):

        grid_2d = np.array(
            [
                [[1, 1], [2, 2], [3, 3]],
                [[4, 4], [5, 5], [6, 6]],
                [[7, 7], [8, 8], [9, 9]],
            ]
        )

        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        grid_1d = al.grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=grid_2d, mask=mask, sub_size=1
        )

        assert (grid_1d == np.array([[5, 5]])).all()

        grid_2d = np.array(
            [
                [[1, 1], [2, 2], [3, 3]],
                [[4, 4], [5, 5], [6, 6]],
                [[7, 7], [8, 8], [9, 9]],
            ]
        )

        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        grid_1d = al.grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=grid_2d, mask=mask, sub_size=1
        )

        assert (grid_1d == np.array([[2, 2], [4, 4], [5, 5], [6, 6], [8, 8]])).all()

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

        grid_1d = al.grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=grid_2d, mask=mask, sub_size=1
        )

        assert (
            grid_1d == np.array([[2, 2], [5, 5], [6, 6], [7, 7], [10, 10], [12, 12]])
        ).all()

        grid_2d = np.array(
            [
                [[1, 1], [2, 2], [3, 3]],
                [[4, 4], [5, 5], [6, 6]],
                [[7, 7], [8, 8], [9, 9]],
                [[10, 10], [11, 11], [12, 12]],
            ]
        )

        mask = np.array(
            [
                [True, False, True],
                [False, False, False],
                [True, False, True],
                [True, True, True],
            ]
        )

        grid_1d = al.grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=grid_2d, mask=mask, sub_size=1
        )

        assert (grid_1d == np.array([[2, 2], [4, 4], [5, 5], [6, 6], [8, 8]])).all()

    def test__map_simple_grids__sub_grid_2(self):

        sub_grid_2d = np.array(
            [
                [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                [[7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]],
                [[13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18]],
                [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                [[7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]],
                [[13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18]],
            ]
        )

        mask = np.array([[True, False, True], [True, False, True], [True, True, False]])

        sub_array_1d = al.grid_mapping_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=sub_grid_2d, mask=mask, sub_size=2
        )

        assert (
            sub_array_1d
            == np.array(
                [
                    [3, 3],
                    [4, 4],
                    [9, 9],
                    [10, 10],
                    [15, 15],
                    [16, 16],
                    [3, 3],
                    [4, 4],
                    [11, 11],
                    [12, 12],
                    [17, 17],
                    [18, 18],
                ]
            )
        ).all()


class TestSubGrid2DFromSubGrid1d(object):
    def test__simple_2d_array__is_masked_and_mapped__sub_size_1(self):

        grid_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        mask = np.full(fill_value=False, shape=(2, 2))

        grid_2d = al.grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=grid_1d, mask=mask, sub_size=1
        )

        assert (
            grid_2d == np.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])
        ).all()

        grid_1d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        mask = np.array([[False, False], [False, True]])

        grid_2d = al.grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=grid_1d, mask=mask, sub_size=1
        )

        assert (
            grid_2d == np.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [0.0, 0.0]]])
        ).all()

        grid_1d = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [-1.0, -1.0],
                [-2.0, -2.0],
                [-3.0, -3.0],
            ]
        )

        mask = np.array(
            [
                [False, False, True, True],
                [False, True, True, True],
                [False, False, True, False],
            ]
        )

        grid_2d = al.grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=grid_1d, mask=mask, sub_size=1
        )

        assert (
            grid_2d
            == np.array(
                [
                    [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                    [[3.0, 3.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[-1.0, -1.0], [-2.0, -2.0], [0.0, 0.0], [-3.0, -3.0]],
                ]
            )
        ).all()

    def test__simple_2d_grid__is_masked_and_mapped__sub_size_2(self):

        grid_1d = np.array(
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
                [4.0, 4.0],
            ]
        )

        mask = np.array([[False, False], [False, True]])

        grid_2d = al.grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=grid_1d, mask=mask, sub_size=2
        )

        assert (
            grid_2d
            == np.array(
                [
                    [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]],
                    [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]],
                    [[3.0, 3.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0]],
                    [[3.0, 3.0], [4.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()
