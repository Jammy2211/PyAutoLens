import autolens as al
import os

import numpy as np


test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestSparseToUnmaskedSparse:
    def test__mask_full_false__image_mask_and_pixel_centres_fully_overlap__each_sparse_maps_to_unmaked_sparse(
        self
    ):

        ma = al.Mask(
            array=np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )

        total_masked_pixels = al.mask_util.total_sparse_pixels_from_mask(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        sparse_to_unmasked_sparse = al.sparse_mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_masked_pixels,
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        assert (
            sparse_to_unmasked_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        ).all()

    def test__same_as_above__but_remove_some_centre_pixels_and_change_order__order_does_not_change_mapping(
        self
    ):

        ma = al.Mask(
            array=np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [2, 2], [1, 1], [0, 2], [2, 0], [0, 2]]
        )

        total_masked_pixels = al.mask_util.total_sparse_pixels_from_mask(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        sparse_to_unmasked_sparse = al.sparse_mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_masked_pixels,
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        assert (sparse_to_unmasked_sparse == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    def test__mask_is_cross__some_pix_pixels_are_masked__omitted_from_mapping(self):

        ma = al.Mask(
            array=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )

        total_masked_pixels = al.mask_util.total_sparse_pixels_from_mask(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        sparse_to_unmasked_sparse = al.sparse_mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_masked_pixels,
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        assert (sparse_to_unmasked_sparse == np.array([1, 3, 4, 5, 7])).all()

    def test__same_as_above__different_mask_and_centres(self):

        ma = al.Mask(
            array=np.array(
                [[False, False, True], [False, False, False], [True, False, False]]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1]]
        )

        total_masked_pixels = al.mask_util.total_sparse_pixels_from_mask(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        sparse_to_unmasked_sparse = al.sparse_mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_masked_pixels,
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        assert (sparse_to_unmasked_sparse == np.array([0, 1, 5])).all()

    def test__same_as_above__but_3x4_mask(self):

        ma = al.Mask(
            array=np.array(
                [
                    [True, True, False, True],
                    [False, False, False, False],
                    [True, True, False, True],
                ]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1], [2, 3], [2, 2]]
        )

        total_masked_pixels = al.mask_util.total_sparse_pixels_from_mask(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        sparse_to_unmasked_sparse = al.sparse_mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_masked_pixels,
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        assert (sparse_to_unmasked_sparse == np.array([2, 3, 4, 5, 7])).all()

    def test__same_as_above__but_4x3_mask(self):

        ma = al.Mask(
            array=np.array(
                [
                    [True, False, True],
                    [True, False, True],
                    [False, False, False],
                    [True, False, True],
                ]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1], [2, 2], [3, 1]]
        )

        total_masked_pixels = al.mask_util.total_sparse_pixels_from_mask(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        sparse_to_unmasked_sparse = al.sparse_mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_masked_pixels,
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        assert (sparse_to_unmasked_sparse == np.array([1, 5, 6, 7])).all()


class TestUnmaskedSparseToSparse:
    def test__mask_full_false__image_mask_and_pixel_centres_fully_overlap__each_pix_maps_to_unmaked_pix(
        self
    ):

        ma = al.Mask(
            array=np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )

        unmasked_sparse_to_sparse = al.sparse_mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=9,
        )

        assert (
            unmasked_sparse_to_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        ).all()

    def test__same_as_above__but_remove_some_centre_pixels_and_change_order__order_does_not_change_mapping(
        self
    ):

        ma = al.Mask(
            array=np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [2, 2], [1, 1], [0, 2], [2, 0], [0, 2]]
        )

        unmasked_sparse_to_sparse = al.sparse_mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=9,
        )

        assert (unmasked_sparse_to_sparse == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    def test__mask_is_cross__some_pix_pixels_are_masked__omitted_from_mapping(self):

        ma = al.Mask(
            array=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )

        unmasked_sparse_to_sparse = al.sparse_mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=5,
        )

        assert (
            unmasked_sparse_to_sparse == np.array([0, 0, 1, 1, 2, 3, 4, 4, 4])
        ).all()

    def test__same_as_above__different_mask_and_centres(self):

        ma = al.Mask(
            array=np.array(
                [[False, False, True], [False, False, False], [True, False, False]]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1]]
        )

        unmasked_sparse_to_sparse = al.sparse_mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=4,
        )

        assert (unmasked_sparse_to_sparse == np.array([0, 1, 2, 2, 2, 2])).all()

    def test__same_as_above__but_3x4_mask(self):

        ma = al.Mask(
            array=np.array(
                [
                    [True, True, False, True],
                    [False, False, False, False],
                    [True, True, False, True],
                ]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1], [2, 3], [0, 2]]
        )

        unmasked_sparse_to_sparse = al.sparse_mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=5,
        )

        assert (unmasked_sparse_to_sparse == np.array([0, 0, 0, 1, 2, 3, 4, 4])).all()

    def test__same_as_above__but_4x3_mask(self):

        ma = al.Mask(
            array=np.array(
                [
                    [True, False, True],
                    [True, False, True],
                    [False, False, False],
                    [True, False, True],
                ]
            ),
            pixel_scale=1.0,
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1], [2, 2], [3, 1]]
        )

        unmasked_sparse_to_sparse = al.sparse_mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=6,
        )

        assert (unmasked_sparse_to_sparse == np.array([0, 0, 1, 1, 1, 1, 2, 3])).all()


class TestRegularToSparse:
    def test__simple_cases_for_regular_to_unmasked_sparse_and__unmasked_sparse_to_sparse(
        self
    ):

        regular_to_unmasked_sparse = np.array([0, 1, 2, 3, 4])
        unmasked_sparse_to_sparse = np.array([0, 1, 2, 3, 4])
        mask_1d_index_to_sparse_1d_index = al.sparse_mapping_util.mask_1d_index_to_sparse_1d_index_from_sparse_mappings(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            unmasked_sparse_to_sparse=unmasked_sparse_to_sparse,
        )

        assert (mask_1d_index_to_sparse_1d_index == np.array([0, 1, 2, 3, 4])).all()

        regular_to_unmasked_sparse = np.array([0, 1, 2, 3, 4])
        unmasked_sparse_to_sparse = np.array([0, 1, 5, 7, 18])
        mask_1d_index_to_sparse_1d_index = al.sparse_mapping_util.mask_1d_index_to_sparse_1d_index_from_sparse_mappings(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            unmasked_sparse_to_sparse=unmasked_sparse_to_sparse,
        )

        assert (mask_1d_index_to_sparse_1d_index == np.array([0, 1, 5, 7, 18])).all()

        regular_to_unmasked_sparse = np.array([1, 1, 1, 1, 2])
        unmasked_sparse_to_sparse = np.array([0, 10, 15, 3, 4])
        mask_1d_index_to_sparse_1d_index = al.sparse_mapping_util.mask_1d_index_to_sparse_1d_index_from_sparse_mappings(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            unmasked_sparse_to_sparse=unmasked_sparse_to_sparse,
        )

        assert (
            mask_1d_index_to_sparse_1d_index == np.array([10, 10, 10, 10, 15])
        ).all()

        regular_to_unmasked_sparse = np.array([5, 6, 7, 8, 9])
        unmasked_sparse_to_sparse = np.array([0, 1, 2, 3, 4, 19, 18, 17, 16, 15])
        mask_1d_index_to_sparse_1d_index = al.sparse_mapping_util.mask_1d_index_to_sparse_1d_index_from_sparse_mappings(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            unmasked_sparse_to_sparse=unmasked_sparse_to_sparse,
        )

        assert (
            mask_1d_index_to_sparse_1d_index == np.array([19, 18, 17, 16, 15])
        ).all()


class TestSparseGridFromUnmaskedSparseGrid:
    def test__simple_unmasked_sparse_grid__full_grid_pix_grid_same_size__straightforward_mappings(
        self
    ):

        unmasked_sparse_grid = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        )
        sparse_to_unmasked_sparse = np.array([0, 1, 2, 3])
        pix_grid = al.sparse_mapping_util.sparse_grid_from_unmasked_sparse_grid(
            unmasked_sparse_grid=unmasked_sparse_grid,
            sparse_to_unmasked_sparse=sparse_to_unmasked_sparse,
        )

        assert (
            pix_grid == np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        ).all()

        unmasked_sparse_grid = np.array(
            [[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]]
        )
        sparse_to_unmasked_sparse = np.array([0, 1, 2, 3])
        pix_grid = al.sparse_mapping_util.sparse_grid_from_unmasked_sparse_grid(
            unmasked_sparse_grid=unmasked_sparse_grid,
            sparse_to_unmasked_sparse=sparse_to_unmasked_sparse,
        )

        assert (
            pix_grid == np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])
        ).all()

        unmasked_sparse_grid = np.array(
            [[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]]
        )
        sparse_to_unmasked_sparse = np.array([1, 0, 3, 2])
        pix_grid = al.sparse_mapping_util.sparse_grid_from_unmasked_sparse_grid(
            unmasked_sparse_grid=unmasked_sparse_grid,
            sparse_to_unmasked_sparse=sparse_to_unmasked_sparse,
        )

        assert (
            pix_grid == np.array([[4.0, 5.0], [0.0, 0.0], [8.0, 7.0], [2.0, 2.0]])
        ).all()

    def test__simple_unmasked_sparse_grid__full_grid_pix_bigger_than_pix__straightforward_mappings(
        self
    ):

        unmasked_sparse_grid = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        )
        sparse_to_unmasked_sparse = np.array([1, 2])
        pix_grid = al.sparse_mapping_util.sparse_grid_from_unmasked_sparse_grid(
            unmasked_sparse_grid=unmasked_sparse_grid,
            sparse_to_unmasked_sparse=sparse_to_unmasked_sparse,
        )

        assert (pix_grid == np.array([[1.0, 1.0], [2.0, 2.0]])).all()

        unmasked_sparse_grid = np.array(
            [[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]]
        )
        sparse_to_unmasked_sparse = np.array([2, 2, 3])
        pix_grid = al.sparse_mapping_util.sparse_grid_from_unmasked_sparse_grid(
            unmasked_sparse_grid=unmasked_sparse_grid,
            sparse_to_unmasked_sparse=sparse_to_unmasked_sparse,
        )

        assert (pix_grid == np.array([[2.0, 2.0], [2.0, 2.0], [8.0, 7.0]])).all()

        unmasked_sparse_grid = np.array(
            [
                [0.0, 0.0],
                [4.0, 5.0],
                [2.0, 2.0],
                [8.0, 7.0],
                [11.0, 11.0],
                [-20.0, -15.0],
            ]
        )
        sparse_to_unmasked_sparse = np.array([1, 0, 5, 2])
        pix_grid = al.sparse_mapping_util.sparse_grid_from_unmasked_sparse_grid(
            unmasked_sparse_grid=unmasked_sparse_grid,
            sparse_to_unmasked_sparse=sparse_to_unmasked_sparse,
        )

        assert (
            pix_grid == np.array([[4.0, 5.0], [0.0, 0.0], [-20.0, -15.0], [2.0, 2.0]])
        ).all()
