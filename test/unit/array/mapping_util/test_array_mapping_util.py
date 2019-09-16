import autolens as al
import os

import numpy as np


test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestMap1dIndexesTo2dIndex(object):
    def test__9_1d_indexes_from_0_to_8__map_to_shape_3x3(self):

        indexes_1d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

        indexes_2d = al.array_mapping_util.map_1d_indexes_to_2d_indexes_for_shape(
            indexes_1d=indexes_1d, shape=(3, 3)
        )

        assert (
            indexes_2d
            == np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
            )
        ).all()

    def test__6_1d_indexes_from_0_to_5__map_to_shape_2x3(self):

        indexes_1d = np.array([0, 1, 2, 3, 4, 5])

        indexes_2d = al.array_mapping_util.map_1d_indexes_to_2d_indexes_for_shape(
            indexes_1d=indexes_1d, shape=(2, 3)
        )

        assert (
            indexes_2d == np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
        ).all()

    def test__6_1d_indexes_from_0_to_5__map_to_shape_3x2(self):

        indexes_1d = np.array([0, 1, 2, 3, 4, 5])

        indexes_2d = al.array_mapping_util.map_1d_indexes_to_2d_indexes_for_shape(
            indexes_1d=indexes_1d, shape=(3, 2)
        )

        assert (
            indexes_2d == np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
        ).all()

    def test__9_1d_indexes_from_0_to_8_different_order__map_to_shape_3x3(self):

        indexes_1d = np.array([1, 4, 7, 8, 0, 2, 3, 5, 6])

        indexes_2d = al.array_mapping_util.map_1d_indexes_to_2d_indexes_for_shape(
            indexes_1d=indexes_1d, shape=(3, 3)
        )

        assert (
            indexes_2d
            == np.array(
                [[0, 1], [1, 1], [2, 1], [2, 2], [0, 0], [0, 2], [1, 0], [1, 2], [2, 0]]
            )
        ).all()


class TestMap2dIndexesTo1dIndex(object):
    def test__9_2d_indexes_from_0_0_to_2_2__map_to_shape_3x3(self):

        indexes_2d = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )

        indexes_1d = al.array_mapping_util.map_2d_indexes_to_1d_indexes_for_shape(
            indexes_2d=indexes_2d, shape=(3, 3)
        )

        assert (indexes_1d == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__6_1d_indexes_from_0_0_to_1_2__map_to_shape_2x3(self):

        indexes_2d = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])

        indexes_1d = al.array_mapping_util.map_2d_indexes_to_1d_indexes_for_shape(
            indexes_2d=indexes_2d, shape=(2, 3)
        )

        assert (indexes_1d == np.array([0, 1, 2, 3, 4, 5])).all()

    def test__6_1d_indexes_from_0_0_to_2_1__map_to_shape_3x2(self):

        indexes_2d = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])

        indexes_1d = al.array_mapping_util.map_2d_indexes_to_1d_indexes_for_shape(
            indexes_2d=indexes_2d, shape=(3, 2)
        )

        assert (indexes_1d == np.array([0, 1, 2, 3, 4, 5])).all()

    def test__9_1d_indexes_from_0_0_to_2_2_different_order__map_to_shape_3x3(self):

        indexes_2d = np.array(
            [[0, 1], [1, 1], [2, 1], [2, 2], [0, 0], [0, 2], [1, 0], [1, 2], [2, 0]]
        )

        indexes_1d = al.array_mapping_util.map_2d_indexes_to_1d_indexes_for_shape(
            indexes_2d=indexes_2d, shape=(3, 3)
        )

        assert (indexes_1d == np.array([1, 4, 7, 8, 0, 2, 3, 5, 6])).all()


class TestSubArray1DFromSubArray2d(object):
    def test__map_simple_data__sub_size_1(self):

        array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        array_1d = al.array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        assert (array_1d == np.array([5])).all()

        array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        array_1d = al.array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        assert (array_1d == np.array([2, 4, 5, 6, 8])).all()

        array_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

        mask = np.array(
            [
                [True, False, True, True],
                [False, False, False, True],
                [True, False, True, False],
            ]
        )

        array_1d = al.array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        assert (array_1d == np.array([2, 5, 6, 7, 10, 12])).all()

        array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        mask = np.array(
            [
                [True, False, True],
                [False, False, False],
                [True, False, True],
                [True, True, True],
            ]
        )

        array_1d = al.array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        assert (array_1d == np.array([2, 4, 5, 6, 8])).all()

    def test__map_simple_data__sub_size_2(self):

        sub_array_2d = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
            ]
        )

        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        sub_array_1d = al.array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_size=2
        )

        assert (sub_array_1d == np.array([15, 16, 3, 4])).all()

        mask = np.array([[True, False, True], [True, False, True], [True, True, False]])

        sub_array_1d = al.array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_size=2
        )

        assert (
            sub_array_1d == np.array([3, 4, 9, 10, 15, 16, 3, 4, 11, 12, 17, 18])
        ).all()

        sub_array_2d = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 7, 7],
                [7, 8, 9, 10, 11, 12, 7, 7, 7],
                [13, 14, 15, 16, 17, 18, 7, 7, 7],
                [1, 2, 3, 4, 5, 6, 7, 7, 7],
                [7, 8, 9, 10, 11, 12, 7, 7, 7],
                [13, 14, 15, 16, 17, 18, 7, 7, 7],
            ]
        )

        mask = np.array(
            [
                [True, False, True, True],
                [False, False, False, True],
                [True, False, True, False],
            ]
        )

        sub_array_1d = al.array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_size=2
        )

        assert (
            sub_array_1d
            == np.array(
                [
                    3,
                    4,
                    9,
                    10,
                    13,
                    14,
                    1,
                    2,
                    15,
                    16,
                    3,
                    4,
                    17,
                    18,
                    5,
                    6,
                    9,
                    10,
                    15,
                    16,
                    7,
                    7,
                    7,
                    7,
                ]
            )
        ).all()

        sub_array_2d = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [7, 7, 7, 7, 7, 7],
                [7, 7, 7, 7, 7, 7],
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

        sub_array_1d = al.array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_size=2
        )

        assert (
            sub_array_1d
            == np.array(
                [3, 4, 9, 10, 13, 14, 1, 2, 15, 16, 3, 4, 17, 18, 5, 6, 9, 10, 15, 16]
            )
        ).all()

    def test__setup_2x2_data__sub_size_3(self):

        sub_array_2d = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
            ]
        )

        mask = np.array([[False, True], [True, False]])

        sub_array_1d = al.array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=sub_array_2d, mask=mask, sub_size=3
        )

        assert (
            sub_array_1d
            == np.array([1, 2, 3, 7, 8, 9, 13, 14, 15, 4, 5, 6, 10, 11, 12, 16, 17, 18])
        ).all()


class TestSubArray2dFromSubArray1d(object):
    def test__simple_2d_array__is_masked_and_mapped__sub_size_1(self):

        array_1d = np.array([1.0, 2.0, 3.0, 4.0])

        mask = np.full(fill_value=False, shape=(2, 2))

        array_2d = al.array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=array_1d, mask=mask, sub_size=1
        )

        assert (array_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        array_1d = np.array([1.0, 2.0, 3.0])

        mask = np.array([[False, False], [False, True]])

        array_2d = al.array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=array_1d, mask=mask, sub_size=1
        )

        assert (array_2d == np.array([[1.0, 2.0], [3.0, 0.0]])).all()

        array_1d = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])

        mask = np.array(
            [
                [False, False, True, True],
                [False, True, True, True],
                [False, False, True, False],
            ]
        )

        array_2d = al.array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=array_1d, mask=mask, sub_size=1
        )

        assert (
            array_2d
            == np.array(
                [[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [-1.0, -2.0, 0.0, -3.0]]
            )
        ).all()

    def test__simple_2d_array__is_masked_and_mapped__sub_size_2(self):

        array_1d = np.array(
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0]
        )

        mask = np.array([[False, False], [False, True]])

        array_2d = al.array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=array_1d, mask=mask, sub_size=2
        )

        assert (
            array_2d
            == np.array(
                [
                    [1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 0.0, 0.0],
                    [3.0, 4.0, 0.0, 0.0],
                ]
            )
        ).all()
