import numpy as np
from auto_lens.imaging import mask
import pytest
from auto_lens import exc


class TestMask(object):
    class TestConstructor(object):

        def test__simple_array_in(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=1)

            assert (msk == np.array([[True, True, True],
                                     [True, False, True],
                                     [True, True, True]])).all()
            assert msk.pixel_scale == 1.0
            assert msk.central_pixel_coordinates == (1.0, 1.0)
            assert msk.shape == (3, 3)
            assert msk.shape_arc_seconds == (3.0, 3.0)

        def test__rectangular_array_in(self):
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

    class TestCircular(object):

        def test__input_big_mask__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(3, 3), pixel_scale=1.0, radius_mask=5)

            assert msk.shape == (3, 3)
            assert (msk == np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False]])).all()

        def test__odd_x_odd_mask_input_radius_small__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(3, 3), pixel_scale=1, radius_mask=0.5)
            assert (msk == np.array([[True, True, True],
                                     [True, False, True],
                                     [True, True, True]])).all()

        def test__odd_x_odd_mask_input_radius_medium__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(3, 3), pixel_scale=1, radius_mask=1)

            assert (msk == np.array([[True, False, True],
                                     [False, False, False],
                                     [True, False, True]])).all()

        def test__odd_x_odd_mask_input_radius_large__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(3, 3), pixel_scale=1, radius_mask=3)

            assert (msk == np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False]])).all()

        def test__even_x_odd_mask_input_radius_small__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(4, 3), pixel_scale=1, radius_mask=0.5)

            assert (msk == np.array([[True, True, True],
                                     [True, False, True],
                                     [True, False, True],
                                     [True, True, True]])).all()

        def test__even_x_odd_mask_input_radius_medium__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(4, 3), pixel_scale=1, radius_mask=1.50001)

            assert (msk == np.array([[True, False, True],
                                     [False, False, False],
                                     [False, False, False],
                                     [True, False, True]])).all()

        def test__even_x_odd_mask_input_radius_large__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(4, 3), pixel_scale=1, radius_mask=3)

            assert (msk == np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False],
                                     [False, False, False]])).all()

        def test__even_x_even_mask_input_radius_small__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(4, 4), pixel_scale=1, radius_mask=0.72)

            assert (msk == np.array([[True, True, True, True],
                                     [True, False, False, True],
                                     [True, False, False, True],
                                     [True, True, True, True]])).all()

        def test__even_x_even_mask_input_radius_medium__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(4, 4), pixel_scale=1, radius_mask=1.7)

            assert (msk == np.array([[True, False, False, True],
                                     [False, False, False, False],
                                     [False, False, False, False],
                                     [True, False, False, True]])).all()

        def test__even_x_even_mask_input_radius_large__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(4, 4), pixel_scale=1, radius_mask=3)

            assert (msk == np.array([[False, False, False, False],
                                     [False, False, False, False],
                                     [False, False, False, False],
                                     [False, False, False, False]])).all()

        def test__centre_shift__simple_shift_back(self):
            msk = mask.Mask.circular(shape_arc_seconds=(3, 3), pixel_scale=1, radius_mask=0.5, centre=(-1, 0))

            assert msk.shape == (3, 3)
            assert (msk == np.array([[True, False, True],
                                     [True, True, True],
                                     [True, True, True]])).all()

        def test__centre_shift__simple_shift_forward(self):
            msk = mask.Mask.circular(shape_arc_seconds=(3, 3), pixel_scale=1, radius_mask=0.5, centre=(0, 1))

            assert msk.shape == (3, 3)
            assert (msk == np.array([[True, True, True],
                                     [True, True, False],
                                     [True, True, True]])).all()

        def test__centre_shift__diagonal_shift(self):
            msk = mask.Mask.circular(shape_arc_seconds=(3, 3), pixel_scale=1, radius_mask=0.5, centre=(1, 1))

            assert (msk == np.array([[True, True, True],
                                     [True, True, True],
                                     [True, True, False]])).all()

    class TestAnnular(object):

        def test__odd_x_odd_mask_inner_radius_zero_outer_radius_small__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(3, 3), pixel_scale=1, inner_radius_mask=0,
                                    outer_radius_mask=0.5)

            assert (msk == np.array([[True, True, True],
                                     [True, False, True],
                                     [True, True, True]])).all()

        def test__odd_x_odd_mask_inner_radius_small_outer_radius_large__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(3, 3), pixel_scale=1, inner_radius_mask=0.5,
                                    outer_radius_mask=3)

            assert (msk == np.array([[False, False, False],
                                     [False, True, False],
                                     [False, False, False]])).all()

        def test__even_x_odd_mask_inner_radius_small_outer_radius_medium__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(4, 3), pixel_scale=1, inner_radius_mask=0.51,
                                    outer_radius_mask=1.51)

            assert (msk == np.array([[True, False, True],
                                     [False, True, False],
                                     [False, True, False],
                                     [True, False, True]])).all()

        def test__even_x_odd_mask_inner_radius_medium_outer_radius_large__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(4, 3), pixel_scale=1, inner_radius_mask=1.51,
                                    outer_radius_mask=3)

            assert (msk == np.array([[False, True, False],
                                     [True, True, True],
                                     [True, True, True],
                                     [False, True, False]])).all()

        def test__even_x_even_mask_inner_radius_small_outer_radius_medium__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(4, 4), pixel_scale=1, inner_radius_mask=0.81,
                                    outer_radius_mask=2)

            assert (msk == np.array([[True, False, False, True],
                                     [False, True, True, False],
                                     [False, True, True, False],
                                     [True, False, False, True]])).all()

        def test__even_x_even_mask_inner_radius_medium_outer_radius_large__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(4, 4), pixel_scale=1, inner_radius_mask=1.71,
                                    outer_radius_mask=3)

            assert (msk == np.array([[False, True, True, False],
                                     [True, True, True, True],
                                     [True, True, True, True],
                                     [False, True, True, False]])).all()

        def test__centre_shift__simple_shift_back(self):
            msk = mask.Mask.annular(shape_arc_seconds=(3, 3), pixel_scale=1, inner_radius_mask=0.5,
                                    outer_radius_mask=3, centre=(-1.0, 0.0))

            assert msk.shape == (3, 3)
            assert (msk == np.array([[False, True, False],
                                     [False, False, False],
                                     [False, False, False]])).all()

        def test__centre_shift__simple_shift_forward(self):
            msk = mask.Mask.annular(shape_arc_seconds=(3, 3), pixel_scale=1, inner_radius_mask=0.5,
                                    outer_radius_mask=3, centre=(0.0, 1.0))

            assert msk.shape == (3, 3)
            assert (msk == np.array([[False, False, False],
                                     [False, False, True],
                                     [False, False, False]])).all()

        def test__centre_shift__diagonal_shift(self):
            msk = mask.Mask.annular(shape_arc_seconds=(3, 3), pixel_scale=1, inner_radius_mask=0.5,
                                    outer_radius_mask=3, centre=(1.0, 1.0))

            assert msk.shape == (3, 3)
            assert (msk == np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, True]])).all()

    class TestUnmasked(object):

        def test__3x3__input__all_are_false(self):
            msk = mask.Mask.unmasked(shape_arc_seconds=(3, 3), pixel_scale=1)

            assert msk.shape == (3, 3)
            assert (msk == np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False]])).all()

        def test__3x2__input__all_are_false(self):
            msk = mask.Mask.unmasked(shape_arc_seconds=(1.5, 1.0), pixel_scale=0.5)

            assert msk.shape == (3, 2)
            assert (msk == np.array([[False, False],
                                     [False, False],
                                     [False, False]])).all()

        def test__5x5__input__all_are_false(self):
            msk = mask.Mask.unmasked(shape_arc_seconds=(5, 5), pixel_scale=1)

            assert msk.shape == (5, 5)
            assert (msk == np.array([[False, False, False, False, False],
                                     [False, False, False, False, False],
                                     [False, False, False, False, False],
                                     [False, False, False, False, False],
                                     [False, False, False, False, False]])).all()

    class TestComputeGridCoordsImage(object):

        def test__setup_3x3_image_one_coordinate(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_grid = msk.compute_grid_coords_image()

            assert (image_grid[0] == np.array([0.0, 0.0])).all()

        def test__setup_3x3_image__five_coordinates(self):
            msk = np.array([[True, False, True],
                            [False, False, False],
                            [True, False, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_grid = msk.compute_grid_coords_image()

            assert (image_grid == np.array([[-3., 0.], [0., -3.], [0., 0.], [0., 3.], [3., 0.]])).all()

        def test__setup_4x4_image__ten_coordinates__new_pixel_scale(self):
            msk = np.array([[True, False, False, True],
                            [False, False, False, True],
                            [True, False, False, True],
                            [False, False, False, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            image_grid = msk.compute_grid_coords_image()

            assert (image_grid == np.array(
                [[-1.5, -0.5], [-1.5, 0.5], [-0.5, -1.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5],
                 [1.5, -1.5], [1.5, -0.5], [1.5, 0.5]])).all()

        def test__setup_3x4_image__six_coordinates(self):
            msk = np.array([[True, False, True, True],
                            [False, False, False, True],
                            [True, False, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_grid = msk.compute_grid_coords_image()

            assert (image_grid == np.array(
                [[-3., -1.5], [0., -4.5], [0., -1.5], [0., 1.5], [3., -1.5], [3., 4.5]])).all()

    class TestComputeGridCoordsImageSub(object):

        def test__3x3_mask_with_one_pixel__2x2_sub_grid__coordinates(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = msk.compute_grid_coords_image_sub(grid_size_sub=2)

            assert (image_sub_grid == np.array([[[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]])).all()

        def test__3x3_mask_with_row_of_pixels__2x2_sub_grid__coordinates(self):
            msk = np.array([[True, True, True],
                            [False, False, False],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = msk.compute_grid_coords_image_sub(grid_size_sub=2)

            assert (image_sub_grid == np.array([[[-0.5, -3.5], [-0.5, -2.5], [0.5, -3.5], [0.5, -2.5]],
                                                [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]],
                                                [[-0.5, 2.5], [-0.5, 3.5], [0.5, 2.5], [0.5, 3.5]]])).all()

        def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__coordinates(self):
            msk = np.array([[True, True, False],
                            [False, False, False],
                            [True, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = msk.compute_grid_coords_image_sub(grid_size_sub=2)

            assert (image_sub_grid == np.array([[[-3.5, 2.5], [-3.5, 3.5], [-2.5, 2.5], [-2.5, 3.5]],
                                                [[-0.5, -3.5], [-0.5, -2.5], [0.5, -3.5], [0.5, -2.5]],
                                                [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]],
                                                [[-0.5, 2.5], [-0.5, 3.5], [0.5, 2.5], [0.5, 3.5]],
                                                [[2.5, 2.5], [2.5, 3.5], [3.5, 2.5], [3.5, 3.5]]])).all()

        def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__different_pixel_scale(self):
            msk = np.array([[True, True, False],
                            [False, False, False],
                            [True, True, False]])

            msk = mask.Mask(msk, pixel_scale=0.3)

            image_sub_grid = msk.compute_grid_coords_image_sub(grid_size_sub=2)

            image_sub_grid = np.round(image_sub_grid, decimals=2)

            assert (image_sub_grid == np.array([[[-0.35, 0.25], [-0.35, 0.35], [-0.25, 0.25], [-0.25, 0.35]],
                                                [[-0.05, -0.35], [-0.05, -0.25], [0.05, -0.35], [0.05, -0.25]],
                                                [[-0.05, -0.05], [-0.05, 0.05], [0.05, -0.05], [0.05, 0.05]],
                                                [[-0.05, 0.25], [-0.05, 0.35], [0.05, 0.25], [0.05, 0.35]],
                                                [[0.25, 0.25], [0.25, 0.35], [0.35, 0.25], [0.35, 0.35]]])).all()

        def test__3x3_mask_with_one_pixel__3x3_sub_grid__coordinates(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = msk.compute_grid_coords_image_sub(grid_size_sub=3)

            assert (image_sub_grid == np.array([[[-0.75, -0.75], [-0.75, 0.], [-0.75, 0.75], [0., -0.75], [0., 0.],
                                                 [0., 0.75], [0.75, -0.75], [0.75, 0.], [0.75, 0.75]]])).all()

        def test__3x3_mask_with_one_row__3x3_sub_grid__coordinates(self):
            msk = np.array([[True, True, False],
                            [True, False, True],
                            [True, True, False]])

            msk = mask.Mask(msk, pixel_scale=2.0)

            image_sub_grid = msk.compute_grid_coords_image_sub(grid_size_sub=3)

            assert (image_sub_grid == np.array([[[-2.5, 1.5], [-2.5, 2.], [-2.5, 2.5], [-2., 1.5], [-2., 2.],
                                                 [-2., 2.5], [-1.5, 1.5], [-1.5, 2.], [-1.5, 2.5]],
                                                [[-0.5, -0.5], [-0.5, 0.], [-0.5, 0.5], [0., -0.5], [0., 0.], [0., 0.5],
                                                 [0.5, -0.5], [0.5, 0.], [0.5, 0.5]],
                                                [[1.5, 1.5], [1.5, 2.], [1.5, 2.5], [2., 1.5], [2., 2.], [2., 2.5],
                                                 [2.5, 1.5], [2.5, 2.], [2.5, 2.5]]])).all()

        def test__4x4_mask_with_one_pixel__4x4_sub_grid__coordinates(self):
            msk = np.array([[True, True, True, True],
                            [True, False, False, True],
                            [True, False, False, True],
                            [True, True, True, False]])

            msk = mask.Mask(msk, pixel_scale=2.0)

            image_sub_grid = msk.compute_grid_coords_image_sub(grid_size_sub=4)

            image_sub_grid = np.round(image_sub_grid, decimals=1)

            assert (image_sub_grid == np.array([[[-1.6, -1.6], [-1.6, -1.2], [-1.6, -0.8], [-1.6, -0.4], [-1.2, -1.6],
                                                 [-1.2, -1.2], [-1.2, -0.8], [-1.2, -0.4], [-0.8, -1.6], [-0.8, -1.2],
                                                 [-0.8, -0.8], [-0.8, -0.4], [-0.4, -1.6], [-0.4, -1.2], [-0.4, -0.8],
                                                 [-0.4, -0.4]],
                                                [[-1.6, 0.4], [-1.6, 0.8], [-1.6, 1.2], [-1.6, 1.6], [-1.2, 0.4],
                                                 [-1.2, 0.8], [-1.2, 1.2], [-1.2, 1.6], [-0.8, 0.4], [-0.8, 0.8],
                                                 [-0.8, 1.2], [-0.8, 1.6], [-0.4, 0.4], [-0.4, 0.8], [-0.4, 1.2],
                                                 [-0.4, 1.6]],
                                                [[0.4, -1.6], [0.4, -1.2], [0.4, -0.8], [0.4, -0.4], [0.8, -1.6],
                                                 [0.8, -1.2], [0.8, -0.8], [0.8, -0.4], [1.2, -1.6], [1.2, -1.2],
                                                 [1.2, -0.8], [1.2, -0.4], [1.6, -1.6], [1.6, -1.2], [1.6, -0.8],
                                                 [1.6, -0.4]],
                                                [[0.4, 0.4], [0.4, 0.8], [0.4, 1.2], [0.4, 1.6], [0.8, 0.4], [0.8, 0.8],
                                                 [0.8, 1.2], [0.8, 1.6], [1.2, 0.4], [1.2, 0.8], [1.2, 1.2], [1.2, 1.6],
                                                 [1.6, 0.4], [1.6, 0.8], [1.6, 1.2], [1.6, 1.6]],
                                                [[2.4, 2.4], [2.4, 2.8], [2.4, 3.2], [2.4, 3.6], [2.8, 2.4], [2.8, 2.8],
                                                 [2.8, 3.2], [2.8, 3.6], [3.2, 2.4], [3.2, 2.8], [3.2, 3.2], [3.2, 3.6],
                                                 [3.6, 2.4], [3.6, 2.8], [3.6, 3.2], [3.6, 3.6]]])).all()

        def test__4x3_mask_with_one_pixel__2x2_sub_grid__coordinates(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, False, False],
                            [False, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = msk.compute_grid_coords_image_sub(grid_size_sub=2)

            assert (image_sub_grid == np.array(
                [[[-2., -0.5], [-2., 0.5], [-1., -0.5], [-1., 0.5]], [[1., -0.5], [1., 0.5], [2., -0.5], [2., 0.5]],
                 [[1., 2.5], [1., 3.5], [2., 2.5], [2., 3.5]], [[4., -3.5], [4., -2.5], [5., -3.5], [5., -2.5]]])).all()

        def test__3x4_mask_with_one_pixel__2x2_sub_grid__coordinates(self):
            msk = np.array([[True, True, True, False],
                            [True, False, False, True],
                            [False, True, False, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = msk.compute_grid_coords_image_sub(grid_size_sub=2)

            assert (image_sub_grid == np.array(
                [[[-3.5, 4.], [-3.5, 5.], [-2.5, 4.], [-2.5, 5.]], [[-0.5, -2.], [-0.5, -1.], [0.5, -2.], [0.5, -1.]],
                 [[-0.5, 1.], [-0.5, 2.], [0.5, 1.], [0.5, 2.]], [[2.5, -5.], [2.5, -4.], [3.5, -5.], [3.5, -4.]],
                 [[2.5, 1.], [2.5, 2.], [3.5, 1.], [3.5, 2.]]])).all()

    class TestComputeGridCoordsBlurring(object):

        def test__3x3_blurring_mask_correct_coordinates(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_grid = msk.compute_grid_coords_blurring(psf_size=(3, 3))

            assert (blurring_grid == np.array(
                [[-3., -3.], [-3., 0.], [-3., 3.], [0., -3.], [0., 3.], [3., -3.], [3., 0.], [3., 3.]])).all()

        def test__3x5_blurring_mask_correct_coordinates(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, False, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            # Blurring mask

            # [[True, True, True, True, True, True, True],
            # [True, True, False, False, False, True, True],
            # [True, True, False, False, False, True, True],
            # [True, True, False, True, False, True, True],
            # [True, True, False, False, False, True, True],
            # [True, True, False, False, False, True, True],
            # [True, True, True, True, True, True, True]])

            blurring_grid = msk.compute_grid_coords_blurring(psf_size=(3, 5))

            assert (blurring_grid == np.array(
                [[-3., -6.], [-3., -3.], [-3., 0.], [-3., 3.], [-3., 6.], [0., -6.], [0., -3.], [0., 3.], [0., 6.],
                 [3., -6.], [3., -3.], [3., 0.], [3., 3.], [3., 6.]])).all()

        def test__5x3_blurring_mask_correct_coordinates(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, False, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            # Blurring mask

            # [[True, True, True, True, True, True, True],
            #  [True, True, True, True, True, True, True],
            #  [True, False, False, False, False, False, True],
            #  [True, False, False, True, False, False, True],
            #  [True, False, False, False, False, False, True],
            #  [True, True, True, True, True, True, True],
            #  [True, True, True, True, True, True, True]]

            blurring_grid = msk.compute_grid_coords_blurring(psf_size=(5, 3))

            assert (blurring_grid == np.array(
                [[-6., -3.], [-6., 0.], [-6., 3.], [-3., -3.], [-3., 0.], [-3., 3.], [0., -3.], [0., 3.], [3., -3.],
                 [3., 0.], [3., 3.], [6., -3.], [6., 0.], [6., 3.]])).all()

    class TestComputeGridData(object):

        def test__setup_3x3_data(self):
            data = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_data = msk.compute_grid_data(data)

            assert (grid_data[0] == np.array([5])).all()

        def test__setup_3x3_data__five_now_in_mask(self):
            data = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

            msk = np.array([[True, False, True],
                            [False, False, False],
                            [True, False, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_data = msk.compute_grid_data(data)

            assert (grid_data[0] == np.array([2])).all()
            assert (grid_data[1] == np.array([4])).all()
            assert (grid_data[2] == np.array([5])).all()
            assert (grid_data[3] == np.array([6])).all()
            assert (grid_data[4] == np.array([8])).all()

        def test__setup_3x4_data(self):
            data = np.array([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12]])

            msk = np.array([[True, False, True, True],
                            [False, False, False, True],
                            [True, False, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_data = msk.compute_grid_data(data)

            assert (grid_data[0] == np.array([2])).all()
            assert (grid_data[1] == np.array([5])).all()
            assert (grid_data[2] == np.array([6])).all()
            assert (grid_data[3] == np.array([7])).all()
            assert (grid_data[4] == np.array([10])).all()

        def test__setup_4x3_data__five_now_in_mask(self):
            data = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9],
                             [10, 11, 12]])

            msk = np.array([[True, False, True],
                            [False, False, False],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_data = msk.compute_grid_data(data)

            assert (grid_data[0] == np.array([2])).all()
            assert (grid_data[1] == np.array([4])).all()
            assert (grid_data[2] == np.array([5])).all()
            assert (grid_data[3] == np.array([6])).all()
            assert (grid_data[4] == np.array([8])).all()

    class TestComputeBlurringMask(object):

        def test__size__3x3_small_mask(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(3, 3))

            assert (blurring_mask == np.array([[False, False, False],
                                               [False, True, False],
                                               [False, False, False]])).all()

        def test__size__3x3__large_mask(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, False, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(3, 3))

            assert (blurring_mask == np.array([[True, True, True, True, True, True, True],
                                               [True, True, True, True, True, True, True],
                                               [True, True, False, False, False, True, True],
                                               [True, True, False, True, False, True, True],
                                               [True, True, False, False, False, True, True],
                                               [True, True, True, True, True, True, True],
                                               [True, True, True, True, True, True, True]])).all()

        def test__size__5x5__large_mask(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, False, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(5, 5))

            assert (blurring_mask == np.array([[True, True, True, True, True, True, True],
                                               [True, False, False, False, False, False, True],
                                               [True, False, False, False, False, False, True],
                                               [True, False, False, True, False, False, True],
                                               [True, False, False, False, False, False, True],
                                               [True, False, False, False, False, False, True],
                                               [True, True, True, True, True, True, True]])).all()

        def test__size__5x3__large_mask(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, False, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(5, 3))

            assert (blurring_mask == np.rot90(np.array([[True, True, True, True, True, True, True],
                                                        [True, True, True, True, True, True, True],
                                                        [True, False, False, False, False, False, True],
                                                        [True, False, False, True, False, False, True],
                                                        [True, False, False, False, False, False, True],
                                                        [True, True, True, True, True, True, True],
                                                        [True, True, True, True, True, True, True]]))).all()

        def test__size__3x5__large_mask(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, False, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(3, 5))

            assert (blurring_mask == np.rot90(np.array([[True, True, True, True, True, True, True],
                                                        [True, True, False, False, False, True, True],
                                                        [True, True, False, False, False, True, True],
                                                        [True, True, False, True, False, True, True],
                                                        [True, True, False, False, False, True, True],
                                                        [True, True, False, False, False, True, True],
                                                        [True, True, True, True, True, True, True]]))).all()

        def test__size__3x3__multiple_points(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, False, True, True, True, False, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, False, True, True, True, False, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(3, 3))

            assert (blurring_mask == np.array([[False, False, False, True, False, False, False],
                                               [False, True, False, True, False, True, False],
                                               [False, False, False, True, False, False, False],
                                               [True, True, True, True, True, True, True],
                                               [False, False, False, True, False, False, False],
                                               [False, True, False, True, False, True, False],
                                               [False, False, False, True, False, False, False]])).all()

        def test__size__5x5__multiple_points(self):
            msk = np.array([[True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, False, True, True, True, False, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, False, True, True, True, False, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(5, 5))

            assert (blurring_mask == np.array([[False, False, False, False, False, False, False, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, True, False, False, False, True, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, True, False, False, False, True, False, False],
                                               [False, False, False, False, False, False, False, False, False],
                                               [False, False, False, False, False, False, False, False,
                                                False]])).all()

        def test__size__5x3__multiple_points(self):
            msk = np.array([[True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, False, True, True, True, False, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, False, True, True, True, False, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(5, 3))

            assert (blurring_mask == np.rot90(np.array([[True, True, True, True, True, True, True, True, True],
                                                        [False, False, False, False, False, False, False, False, False],
                                                        [False, False, True, False, False, False, True, False, False],
                                                        [False, False, False, False, False, False, False, False, False],
                                                        [True, True, True, True, True, True, True, True, True],
                                                        [False, False, False, False, False, False, False, False, False],
                                                        [False, False, True, False, False, False, True, False, False],
                                                        [False, False, False, False, False, False, False, False, False],
                                                        [True, True, True, True, True, True, True, True, True]]))).all()

        def test__size__3x5__multiple_points(self):
            msk = np.array([[True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, False, True, True, True, False, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, False, True, True, True, False, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(3, 5))

            assert (blurring_mask == np.rot90(np.array([[True, False, False, False, True, False, False, False, True],
                                                        [True, False, False, False, True, False, False, False, True],
                                                        [True, False, True, False, True, False, True, False, True],
                                                        [True, False, False, False, True, False, False, False, True],
                                                        [True, False, False, False, True, False, False, False, True],
                                                        [True, False, False, False, True, False, False, False, True],
                                                        [True, False, True, False, True, False, True, False, True],
                                                        [True, False, False, False, True, False, False, False, True],
                                                        [True, False, False, False, True, False, False, False,
                                                         True]]))).all()

        def test__size__3x3__even_sized_image(self):
            msk = np.array([[True, True, True, True, True, True, True, True],
                            [True, False, True, True, True, False, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, False, True, True, True, False, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(3, 3))

            assert (blurring_mask == np.array([[False, False, False, True, False, False, False, True],
                                               [False, True, False, True, False, True, False, True],
                                               [False, False, False, True, False, False, False, True],
                                               [True, True, True, True, True, True, True, True],
                                               [False, False, False, True, False, False, False, True],
                                               [False, True, False, True, False, True, False, True],
                                               [False, False, False, True, False, False, False, True],
                                               [True, True, True, True, True, True, True, True]])).all()

        def test__size__5x5__even_sized_image(self):
            msk = np.array([[True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, False, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(5, 5))

            assert (blurring_mask == np.array([[True, True, True, True, True, True, True, True],
                                               [True, True, True, True, True, True, True, True],
                                               [True, True, True, True, True, True, True, True],
                                               [True, True, True, False, False, False, False, False],
                                               [True, True, True, False, False, False, False, False],
                                               [True, True, True, False, False, True, False, False],
                                               [True, True, True, False, False, False, False, False],
                                               [True, True, True, False, False, False, False, False]])).all()

        def test__size__3x3__rectangular_8x9_image(self):
            msk = np.array([[True, True, True, True, True, True, True, True, True],
                            [True, False, True, True, True, False, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, False, True, True, True, False, True, True, True],
                            [True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(3, 3))

            assert (blurring_mask == np.array([[False, False, False, True, False, False, False, True, True],
                                               [False, True, False, True, False, True, False, True, True],
                                               [False, False, False, True, False, False, False, True, True],
                                               [True, True, True, True, True, True, True, True, True],
                                               [False, False, False, True, False, False, False, True, True],
                                               [False, True, False, True, False, True, False, True, True],
                                               [False, False, False, True, False, False, False, True, True],
                                               [True, True, True, True, True, True, True, True, True]])).all()

        def test__size__3x3__rectangular_9x8_image(self):
            msk = np.array([[True, True, True, True, True, True, True, True],
                            [True, False, True, True, True, False, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, False, True, True, True, False, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_mask = msk.compute_blurring_mask(kernal_shape=(3, 3))

            assert (blurring_mask == np.array([[False, False, False, True, False, False, False, True],
                                               [False, True, False, True, False, True, False, True],
                                               [False, False, False, True, False, False, False, True],
                                               [True, True, True, True, True, True, True, True],
                                               [False, False, False, True, False, False, False, True],
                                               [False, True, False, True, False, True, False, True],
                                               [False, False, False, True, False, False, False, True],
                                               [True, True, True, True, True, True, True, True],
                                               [True, True, True, True, True, True, True, True]])).all()

        def test__size__5x5__multiple_points__mask_extends_beyond_border_so_raises_mask_exception(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, False, True, True, True, False, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, False, True, True, True, False, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            with pytest.raises(exc.MaskException):
                msk.compute_blurring_mask(kernal_shape=(5, 5))

    class TestComputeBorderPixels(object):

        def test__7x7_mask_one_central_pixel__is_entire_border(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, False, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border_pixels = msk.compute_grid_border()

            assert (border_pixels == np.array([0])).all()

        def test__7x7_mask_nine_central_pixels__is_border(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border_pixels = msk.compute_grid_border()

            assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

        def test__7x7_mask_rectangle_of_fifteen_central_pixels__is_border(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border_pixels = msk.compute_grid_border()

            assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14])).all()

        def test__8x7_mask_add_edge_pixels__also_in_border(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, False, True, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, False, False, False, False, False, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border_pixels = msk.compute_grid_border()

            assert (border_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17])).all()

        def test__8x7_mask_big_square(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border_pixels = msk.compute_grid_border()

            assert (border_pixels == np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 26, 27, 28, 29])).all()

        def test__7x8_mask_add_edge_pixels__also_in_border(self):
            msk = np.array([[True, True, True, True, True, True, True, True],
                            [True, True, True, False, True, True, True, True],
                            [True, True, False, False, False, True, True, True],
                            [True, True, False, False, False, True, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, True, False, False, False, True, True, True],
                            [True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border_pixels = msk.compute_grid_border()

            assert (border_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14])).all()

        def test__7x8_mask_big_square(self):
            msk = np.array([[True, True, True, True, True, True, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border_pixels = msk.compute_grid_border()

            assert (border_pixels == np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24])).all()

    class TestComputeGridMapperTo2D(object):

        def test__setup_3x3_image_one_pixel(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            mapper_data_to_2d = msk.compute_grid_mapper_data_to_pixel()

            assert (mapper_data_to_2d[0] == np.array([1, 1])).all()

        def test__setup_3x3_image__five_pixels(self):
            msk = np.array([[True, False, True],
                            [False, False, False],
                            [True, False, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            mapper_data_to_2d = msk.compute_grid_mapper_data_to_pixel()

            assert (mapper_data_to_2d[0] == np.array([0, 1])).all()
            assert (mapper_data_to_2d[1] == np.array([1, 0])).all()
            assert (mapper_data_to_2d[2] == np.array([1, 1])).all()
            assert (mapper_data_to_2d[3] == np.array([1, 2])).all()
            assert (mapper_data_to_2d[4] == np.array([2, 1])).all()

        def test__setup_3x4_image__six_pixels(self):
            msk = np.array([[True, False, True, True],
                            [False, False, False, True],
                            [True, False, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            mapper_data_to_2d = msk.compute_grid_mapper_data_to_pixel()

            assert (mapper_data_to_2d[0] == np.array([0, 1])).all()
            assert (mapper_data_to_2d[1] == np.array([1, 0])).all()
            assert (mapper_data_to_2d[2] == np.array([1, 1])).all()
            assert (mapper_data_to_2d[3] == np.array([1, 2])).all()
            assert (mapper_data_to_2d[4] == np.array([2, 1])).all()
            assert (mapper_data_to_2d[5] == np.array([2, 3])).all()

        def test__setup_4x3_image__six_pixels(self):
            msk = np.array([[True, False, True],
                            [False, False, False],
                            [True, False, True],
                            [True, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            mapper_data_to_2d = msk.compute_grid_mapper_data_to_pixel()

            assert (mapper_data_to_2d[0] == np.array([0, 1])).all()
            assert (mapper_data_to_2d[1] == np.array([1, 0])).all()
            assert (mapper_data_to_2d[2] == np.array([1, 1])).all()
            assert (mapper_data_to_2d[3] == np.array([1, 2])).all()
            assert (mapper_data_to_2d[4] == np.array([2, 1])).all()
            assert (mapper_data_to_2d[5] == np.array([3, 2])).all()

    class TestMapperSparsePixels(object):

        # TODO : These tests are over crowded, should break up into more self contained things.

        def test__7x7_circle_mask__five_central_pixels__sparse_grid_size_1(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, False, False, False, False, False, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=1)

            assert (sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__7x7_circle_mask__sparse_grid_size_1(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=1)

            assert (sparse_to_image == np.arange(21)).all()
            assert (image_to_sparse == np.arange(21)).all()

        def test__7x7_rectangle_mask__sparse_grid_size_1(self):
            msk = np.array([[False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=1)

            assert (sparse_to_image == np.arange(49)).all()
            assert (image_to_sparse == np.arange(49)).all()

        def test__7x7_circle_mask__sparse_grid_size_2(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=2)

            assert (sparse_to_image == np.array([4, 6, 14, 16])).all()
            assert (image_to_sparse == np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
                                                 1, 2, 2, 2, 3, 3, 2, 2, 3])).all()

        def test__8x8_sporadic_mask__sparse_grid_size_2(self):
            msk = np.array([[True, True, True, True, True, True, False, False],
                            [True, True, False, False, False, True, False, False],
                            [True, False, False, False, False, False, False, False],
                            [True, False, False, False, False, False, False, False],
                            [True, False, False, False, False, False, False, False],
                            [True, True, False, False, False, True, False, False],
                            [True, True, True, True, True, True, False, False],
                            [True, True, False, False, False, True, False, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=2)

            assert (sparse_to_image == np.array([0, 8, 10, 12, 22, 24, 26, 33])).all()
            assert (image_to_sparse == np.array([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 2, 2, 3, 3,
                                                 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 4, 4, 5, 6, 6,
                                                 7, 7, 4, 4, 7, 7, 7])).all()

        def test__7x7_circle_mask_trues_on_even_values__sparse_grid_size_2(self):
            msk = np.array([[False, True, False, True, False, True, False],
                            [True, True, True, True, True, True, True],
                            [False, True, False, True, False, True, False],
                            [True, True, True, True, True, True, True],
                            [False, True, False, True, False, True, False],
                            [True, True, True, True, True, True, True],
                            [False, True, False, True, False, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=2)

            assert (sparse_to_image == np.arange(16)).all()
            assert (image_to_sparse == np.arange(16)).all()

        def test__7x7_circle_mask__sparse_grid_size_3(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=3)

            assert (sparse_to_image == np.array([10])).all()
            assert (image_to_sparse == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all()

        def test__7x7_circle_mask_more_points_added__sparse_grid_size_3(self):
            msk = np.array([[False, True, True, False, True, False, False],
                            [True, True, False, False, False, True, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, False],
                            [True, False, False, False, False, False, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=3)

            assert (sparse_to_image == np.array([0, 1, 3, 14, 17, 26])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 2, 1, 1, 1, 0, 3, 3, 3, 4, 3, 3, 3, 3, 4, 4, 3, 3, 3,
                                                 3, 4, 3, 3, 3, 5])).all()

        def test__7x7_mask_trues_on_values_which_divide_by_3__sparse_grid_size_3(self):
            msk = np.array([[False, True, True, False, True, True, False],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [False, True, True, False, True, True, False],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [False, True, True, False, True, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=3)

            assert (sparse_to_image == np.arange(9)).all()
            assert (image_to_sparse == np.arange(9)).all()

        def test__8x8_mask_trues_on_values_which_divide_by_3_and_other_values__sparse_grid_size_3(self):
            msk = np.array([[False, True, False, False, True, True, False],
                            [True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [False, True, True, False, True, True, False],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [False, False, False, False, False, False, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=3)

            assert (sparse_to_image == np.array([0, 2, 3, 7, 8, 9, 10, 13, 16])).all()
            assert (image_to_sparse == np.array([0, 1, 1, 2, 4, 4, 4, 3, 4, 5, 6, 6, 7, 7, 7, 8, 8])).all()

        def test__8x7__five_central_pixels__sparse_grid_size_1(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, False, False, False, False, False, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=1)

            assert (sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__8x7__five_central_pixels_2__sparse_grid_size_1(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, False, False, False, False, False, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=1)

            assert (sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__8x7__five_central_pixels__sparse_grid_size_2(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=2)

            assert (sparse_to_image == np.array([1, 3])).all()
            assert (image_to_sparse == np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])).all()

        def test__7x8__five_central_pixels__sparse_grid_size_1(self):
            msk = np.array([[True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=1)

            assert (sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__7x8__five_central_pixels__sparse_grid_size_2(self):
            msk = np.array([[True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=2)

            assert (sparse_to_image == np.array([1, 3])).all()
            assert (image_to_sparse == np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])).all()

        def test__7x8__more_central_pixels__sparse_grid_size_2(self):
            msk = np.array([[True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = msk.compute_grid_mapper_sparse(sparse_grid_size=2)

            assert (sparse_to_image == np.array([1, 3, 11, 13])).all()
            assert (image_to_sparse == np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3])).all()
