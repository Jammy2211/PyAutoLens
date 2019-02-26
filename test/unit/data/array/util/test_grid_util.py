import os

import numpy as np
import pytest

from autolens.data.array.util import grid_util


test_data_dir = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestGrid2d:

    def test__array_3x3__sets_up_arcsecond_grid(self):

        grid_2d = grid_util.regular_grid_2d_from_shape_pixel_scales_and_origin(shape=(3, 3), pixel_scales=(2.0, 1.0))

        assert (grid_2d == np.array([[[2., -1.], [2., 0.], [2., 1.]],
                                     [[0., -1.], [0., 0.], [0., 1.]],
                                     [[-2., -1.], [-2., 0.], [-2., 1.]]])).all()

    def test__array_4x4_and_different_pixel_scale__sets_up_arcsecond_grid(self):

        grid_2d = grid_util.regular_grid_2d_from_shape_pixel_scales_and_origin(shape=(4, 4), pixel_scales=(0.5, 0.5))

        assert (grid_2d == np.array([[[0.75, -0.75], [0.75, -0.25], [0.75, 0.25], [0.75, 0.75]],
                                     [[0.25, -0.75], [0.25, -0.25], [0.25, 0.25], [0.25, 0.75]],
                                     [[-0.25, -0.75], [-0.25, -0.25], [-0.25, 0.25], [-0.25, 0.75]],
                                     [[-0.75, -0.75], [-0.75, -0.25], [-0.75, 0.25], [-0.75, 0.75]]])).all()

    def test__array_2x3__sets_up_arcsecond_grid(self):
        grid_2d = grid_util.regular_grid_2d_from_shape_pixel_scales_and_origin(shape=(2, 3), pixel_scales=(1.0, 1.0))

        assert (grid_2d == np.array([[[0.5, -1.], [0.5, 0.], [0.5, 1.]],
                                     [[-0.5, -1.], [-0.5, 0.], [-0.5, 1.]]])).all()

    def test__array_3x2__sets_up_arcsecond_grid(self):
        grid_2d = grid_util.regular_grid_2d_from_shape_pixel_scales_and_origin(shape=(3, 2), pixel_scales=(1.0, 1.0))

        assert (grid_2d == np.array([[[1., -0.5], [1., 0.5]],
                                     [[0., -0.5], [0., 0.5]],
                                     [[-1., -0.5], [-1., 0.5]]])).all()

    def test__array_3x3___input_origin__shifts_grid_by_origin(self):

        grid_2d = grid_util.regular_grid_2d_from_shape_pixel_scales_and_origin(shape=(3, 3), pixel_scales=(2.0, 1.0),
                                                                               origin=(1.0, 1.0))

        assert (grid_2d == np.array([[[3., 0.], [3., 1.], [3., 2.]],
                                     [[1.,  0.], [1., 1.], [1., 2.]],
                                     [[-1., 0.], [-1., 1.], [-1., 2.]]])).all()

    def test__array_3x2__different_origin(self):
        grid_2d = grid_util.regular_grid_2d_from_shape_pixel_scales_and_origin(shape=(3, 2), pixel_scales=(1.0, 1.0),
                                                                               origin=(3.0, -2.0))

        assert (grid_2d == np.array([[[4., -2.5], [4., -1.5]],
                                     [[3., -2.5], [3., -1.5]],
                                     [[2., -2.5], [2., -1.5]]])).all()


class TestGrid1d:

    def test__array_3x3__sets_up_arcsecond_grid(self):

        grid_2d = grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=(3, 3), pixel_scales=(2.0, 1.0))

        assert (grid_2d == np.array([[2., -1.], [2., 0.], [2., 1.],
                                     [0., -1.], [0., 0.], [0., 1.],
                                     [-2., -1.], [-2., 0.], [-2., 1.]])).all()

    def test__array_4x4_and_different_pixel_scale__sets_up_arcsecond_grid(self):

        grid_2d = grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=(4, 4), pixel_scales=(0.5, 0.5))

        assert (grid_2d == np.array([[0.75, -0.75], [0.75, -0.25], [0.75, 0.25], [0.75, 0.75],
                                     [0.25, -0.75], [0.25, -0.25], [0.25, 0.25], [0.25, 0.75],
                                     [-0.25, -0.75], [-0.25, -0.25], [-0.25, 0.25], [-0.25, 0.75],
                                     [-0.75, -0.75], [-0.75, -0.25], [-0.75, 0.25], [-0.75, 0.75]])).all()

    def test__array_2x3__sets_up_arcsecond_grid(self):
        grid_2d = grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=(2, 3), pixel_scales=(1.0, 1.0))

        assert (grid_2d == np.array([[0.5, -1.], [0.5, 0.], [0.5, 1.],
                                     [-0.5, -1.], [-0.5, 0.], [-0.5, 1.]])).all()

    def test__array_3x2__sets_up_arcsecond_grid(self):
        grid_2d = grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=(3, 2), pixel_scales=(1.0, 1.0))

        assert (grid_2d == np.array([[1., -0.5], [1., 0.5],
                                     [0., -0.5], [0., 0.5],
                                     [-1., -0.5], [-1., 0.5]])).all()

    def test__array_3x3__input_origin__shifts_grid_by_origin(self):

        grid_2d = grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=(3, 3), pixel_scales=(2.0, 1.0),
                                                                               origin=(1.0, 1.0))

        assert (grid_2d == np.array([[3., 0.], [3., 1.], [3., 2.],
                                     [1., 0.], [1., 1.], [1., 2.],
                                     [-1., 0.], [-1., 1.], [-1., 2.]])).all()

    def test__array_3x2__different_origin(self):

        grid_2d = grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=(3, 2), pixel_scales=(1.0, 1.0),
                                                                               origin=(3.0, -2.0))

        assert (grid_2d == np.array([[4., -2.5], [4., -1.5],
                                     [3., -2.5], [3., -1.5],
                                     [2., -2.5], [2., -1.5]])).all()


class TestRegularGridMasked(object):

    def test__setup_3x3_image_1_coordinate_in_mask(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        image_grid = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(3.0, 6.0))

        assert (image_grid[0] == np.array([0.0, 0.0])).all()

    def test__setup_3x3_image__five_coordinates_in_mask(self):
        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        image_grid = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(6.0, 3.0))

        assert (image_grid == np.array([[6., 0.],
                                        [0., -3.], [0., 0.], [0., 3.],
                                        [-6., 0.]])).all()

    def test__setup_4x4_image__ten_coordinates_in_grid__new_pixel_scale(self):
        mask = np.array([[True, False, False, True],
                         [False, False, False, True],
                         [True, False, False, True],
                         [False, False, False, True]])

        image_grid = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(1.0, 1.0))

        assert (image_grid == np.array([[1.5, -0.5], [1.5, 0.5],
                                        [0.5, -1.5], [0.5, -0.5], [0.5, 0.5],
                                        [-0.5, -0.5], [-0.5, 0.5],
                                        [-1.5, -1.5], [-1.5, -0.5], [-1.5, 0.5]])).all()

    def test__setup_3x4_image__six_grid(self):
        mask = np.array([[True, False, True, True],
                         [False, False, False, True],
                         [True, False, True, False]])

        image_grid = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(3.0, 3.0))

        assert (image_grid == np.array([[3., -1.5],
                                        [0., -4.5], [0., -1.5], [0., 1.5],
                                        [-3., -1.5], [-3., 4.5]])).all()

    def test__setup_3x3_image__five_coordinates_in_mask__include_nonzero_origin(self):
        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        image_grid = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(6.0, 3.0),
                                                                                        origin=(1.0, 1.0))

        assert image_grid == pytest.approx(np.array([[7., 1.],
                                                     [1., -2.], [1., 1.], [1., 4.],
                                                     [-5., 1.]]), 1e-4)

    def test__setup_3x4_image__six_grid__include_nonzero_origin(self):
        mask = np.array([[True, False, True, True],
                         [False, False, False, True],
                         [True, False, True, False]])

        image_grid = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=(3.0, 3.0),
                                                                                        origin=(1.0, 2.0))

        assert image_grid == pytest.approx(np.array([[4., 0.5],
                                                     [1., -2.5], [1., 0.5], [1., 3.5],
                                                     [-2., 0.5], [-2., 6.5]]), 1e-4)


class TestSubGridMasked(object):

    def test__3x3_mask_with_one_pixel__2x2_sub_grid(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 6.0),
                                                                                         sub_grid_size=2)

        assert (sub_grid[0:4] == np.array([[0.5, -1.0], [0.5, 1.0],
                                           [-0.5, -1.0], [-0.5, 1.0]])).all()

    def test__3x3_mask_with_row_of_pixels__2x2_sub_grid(self):
        mask = np.array([[True, True, True],
                         [False, False, False],
                         [True, True, True]])

        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 3.0),
                                                                                         sub_grid_size=2)

        assert (sub_grid[0:4] == np.array([[0.5, -3.5], [0.5, -2.5],
                                           [-0.5, -3.5], [-0.5, -2.5]])).all()

        assert (sub_grid[4:8] == np.array([[0.5, -0.5], [0.5, 0.5],
                                           [-0.5, -0.5], [-0.5, 0.5]])).all()

        assert (sub_grid[8:12] == np.array([[0.5, 2.5], [0.5, 3.5],
                                            [-0.5, 2.5], [-0.5, 3.5]])).all()

    def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid(self):
        mask = np.array([[True, True, False],
                         [False, False, False],
                         [True, True, False]])

        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 3.0),
                                                                                         sub_grid_size=2)

        assert (sub_grid == np.array([[3.5, 2.5], [3.5, 3.5], [2.5, 2.5], [2.5, 3.5],
                                      [0.5, -3.5], [0.5, -2.5], [-0.5, -3.5], [-0.5, -2.5],
                                      [0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5],
                                      [0.5, 2.5], [0.5, 3.5], [-0.5, 2.5], [-0.5, 3.5],
                                      [-2.5, 2.5], [-2.5, 3.5], [-3.5, 2.5], [-3.5, 3.5]])).all()

    def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__different_pixel_scale(self):
        mask = np.array([[True, True, False],
                         [False, False, False],
                         [True, True, False]])

        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(0.3, 0.3),
                                                                                         sub_grid_size=2)

        sub_grid = np.round(sub_grid, decimals=2)

        np.testing.assert_almost_equal(sub_grid,
                                       np.array([[0.35, 0.25], [0.35, 0.35], [0.25, 0.25], [0.25, 0.35],
                                                 [0.05, -0.35], [0.05, -0.25], [-0.05, -0.35], [-0.05, -0.25],
                                                 [0.05, -0.05], [0.05, 0.05], [-0.05, -0.05], [-0.05, 0.05],
                                                 [0.05, 0.25], [0.05, 0.35], [-0.05, 0.25], [-0.05, 0.35],
                                                 [-0.25, 0.25], [-0.25, 0.35], [-0.35, 0.25], [-0.35, 0.35]]))

    def test__3x3_mask_with_one_pixel__3x3_sub_grid(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 3.0),
                                                                                         sub_grid_size=3)

        assert (sub_grid == np.array([[[0.75, -0.75], [0.75, 0.], [0.75, 0.75], [0., -0.75], [0., 0.],
                                       [0., 0.75], [-0.75, -0.75], [-0.75, 0.], [-0.75, 0.75]]])).all()

    def test__3x3_mask_with_one_row__3x3_sub_grid(self):
        mask = np.array([[True, True, False],
                         [True, False, True],
                         [True, True, False]])

        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(2.0, 2.0),
                                                                                         sub_grid_size=3)

        assert (sub_grid == np.array([[2.5, 1.5], [2.5, 2.], [2.5, 2.5],
                                      [2., 1.5], [2., 2.], [2., 2.5],
                                      [1.5, 1.5], [1.5, 2.], [1.5, 2.5],
                                      [0.5, -0.5], [0.5, 0.], [0.5, 0.5],
                                      [0., -0.5], [0., 0.], [0., 0.5],
                                      [-0.5, -0.5], [-0.5, 0.], [-0.5, 0.5],
                                      [-1.5, 1.5], [-1.5, 2.], [-1.5, 2.5],
                                      [-2., 1.5], [-2., 2.], [-2., 2.5],
                                      [-2.5, 1.5], [-2.5, 2.], [-2.5, 2.5]])).all()

    def test__4x4_mask_with_one_pixel__4x4_sub_grid(self):
        mask = np.array([[True, True, True, True],
                         [True, False, False, True],
                         [True, False, False, True],
                         [True, True, True, False]])

        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(2.0, 2.0),
                                                                                         sub_grid_size=4)

        sub_grid = np.round(sub_grid, decimals=1)

        assert (sub_grid == np.array([[1.6, -1.6], [1.6, -1.2], [1.6, -0.8], [1.6, -0.4],
                                      [1.2, -1.6], [1.2, -1.2], [1.2, -0.8], [1.2, -0.4],
                                      [0.8, -1.6], [0.8, -1.2], [0.8, -0.8], [0.8, -0.4],
                                      [0.4, -1.6], [0.4, -1.2], [0.4, -0.8], [0.4, -0.4],
                                      [1.6, 0.4], [1.6, 0.8], [1.6, 1.2], [1.6, 1.6],
                                      [1.2, 0.4], [1.2, 0.8], [1.2, 1.2], [1.2, 1.6],
                                      [0.8, 0.4], [0.8, 0.8], [0.8, 1.2], [0.8, 1.6],
                                      [0.4, 0.4], [0.4, 0.8], [0.4, 1.2], [0.4, 1.6],
                                      [-0.4, -1.6], [-0.4, -1.2], [-0.4, -0.8], [-0.4, -0.4],
                                      [-0.8, -1.6], [-0.8, -1.2], [-0.8, -0.8], [-0.8, -0.4],
                                      [-1.2, -1.6], [-1.2, -1.2], [-1.2, -0.8], [-1.2, -0.4],
                                      [-1.6, -1.6], [-1.6, -1.2], [-1.6, -0.8], [-1.6, -0.4],
                                      [-0.4, 0.4], [-0.4, 0.8], [-0.4, 1.2], [-0.4, 1.6],
                                      [-0.8, 0.4], [-0.8, 0.8], [-0.8, 1.2], [-0.8, 1.6],
                                      [-1.2, 0.4], [-1.2, 0.8], [-1.2, 1.2], [-1.2, 1.6],
                                      [-1.6, 0.4], [-1.6, 0.8], [-1.6, 1.2], [-1.6, 1.6],
                                      [-2.4, 2.4], [-2.4, 2.8], [-2.4, 3.2], [-2.4, 3.6],
                                      [-2.8, 2.4], [-2.8, 2.8], [-2.8, 3.2], [-2.8, 3.6],
                                      [-3.2, 2.4], [-3.2, 2.8], [-3.2, 3.2], [-3.2, 3.6],
                                      [-3.6, 2.4], [-3.6, 2.8], [-3.6, 3.2], [-3.6, 3.6]])).all()

    def test__4x3_mask_with_one_pixel__2x2_sub_grid(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, False, False],
                         [False, True, True]])

        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 3.0),
                                                                                         sub_grid_size=2)

        assert (sub_grid == np.array([[2., -0.5], [2., 0.5], [1., -0.5], [1., 0.5],
                                      [-1., -0.5], [-1., 0.5], [-2., -0.5], [-2., 0.5],
                                      [-1., 2.5], [-1., 3.5], [-2., 2.5], [-2., 3.5],
                                      [-4., -3.5], [-4., -2.5], [-5., -3.5], [-5., -2.5]])).all()

    def test__3x4_mask_with_one_pixel__2x2_sub_grid(self):
        mask = np.array([[True, True, True, False],
                         [True, False, False, True],
                         [False, True, False, True]])

        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 3.0),
                                                                                         sub_grid_size=2)

        assert (sub_grid == np.array([[3.5, 4.], [3.5, 5.], [2.5, 4.], [2.5, 5.],
                                      [0.5, -2.], [0.5, -1.], [-0.5, -2.], [-0.5, -1.],
                                      [0.5, 1.], [0.5, 2.], [-0.5, 1.], [-0.5, 2.],
                                      [-2.5, -5.], [-2.5, -4.], [-3.5, -5.], [-3.5, -4.],
                                      [-2.5, 1.], [-2.5, 2.], [-3.5, 1.], [-3.5, 2.]])).all()

    def test__3x3_mask_with_one_pixel__2x2_sub_grid__include_nonzero_origin(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 6.0),
                                                                                         sub_grid_size=2, origin=(1.0, 1.0))

        assert sub_grid[0:4] == pytest.approx(np.array([[1.5, 0.0], [1.5, 2.0],
                                                        [0.5, 0.0], [0.5, 2.0]]), 1e-4)

    def test__3x3_mask_with_one_row__3x3_sub_grid__include_nonzero_origin(self):
        mask = np.array([[True, True, False],
                         [True, False, True],
                         [True, True, False]])

        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(2.0, 2.0),
                                                                                         sub_grid_size=3, origin=(1.0, -1.0))

        assert sub_grid == pytest.approx(np.array([[3.5, 0.5], [3.5, 1.], [3.5, 1.5],
                                                   [3., 0.5], [3., 1.], [3., 1.5],
                                                   [2.5, 0.5], [2.5, 1.], [2.5, 1.5],
                                                   [1.5, -1.5], [1.5, -1.], [1.5, -0.5],
                                                   [1., -1.5], [1., -1.], [1., -0.5],
                                                   [0.5, -1.5], [0.5, -1.], [0.5, -0.5],
                                                   [-0.5, 0.5], [-0.5, 1.], [-0.5, 1.5],
                                                   [-1., 0.5], [-1., 1.], [-1., 1.5],
                                                   [-1.5, 0.5], [-1.5, 1.], [-1.5, 1.5]]), 1e-4)


class TestGridConversions(object):

    def test__1d_arc_second_grid_to_1d_pixel_grid__coordinates_in_origins_of_pixels(self):

        grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0],
                                     [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=grid_arcsec, shape=(2, 2),
                                                                      pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[0.5, 0.5], [0.5, 1.5],
                                         [1.5, 0.5], [1.5, 1.5]])).all()

        grid_arcsec = np.array([[3.0, -6.0], [3.0, 0.0], [3.0, 6.0],
                                         [0.0, -6.0], [0.0, 0.0], [0.0, 6.0],
                                         [-3.0, -6.0], [-3.0, 0.0], [-3.0, 6.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=grid_arcsec, shape=(3, 3),
                                                                      pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[0.5, 0.5], [0.5, 1.5], [0.5, 2.5],
                                         [1.5, 0.5], [1.5, 1.5], [1.5, 2.5],
                                         [2.5, 0.5], [2.5, 1.5], [2.5, 2.5]])).all()

    def test__same_as_above__pixels__but_coordinates_are_top_left_of_each_pixel(self):

        grid_arcsec = np.array([[2.0, -4], [2.0, 0.0],
                                     [0.0, -4], [0.0, 0.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=grid_arcsec, shape=(2, 2),
                                                                      pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1],
                                         [1, 0], [1, 1]])).all()

        grid_arcsec = np.array([[4.5, -9.0], [4.5, -3.0], [4.5, 3.0],
                                     [1.5, -9.0], [1.5, -3.0], [1.5, 3.0],
                                     [-1.5, -9.0], [-1.5, -3.0], [-1.5, 3.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=grid_arcsec, shape=(3, 3),
                                                                      pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1], [0, 2],
                                         [1, 0], [1, 1], [1, 2],
                                         [2, 0], [2, 1], [2, 2]])).all()

    def test__same_as_above___pixels__but_coordinates_are_bottom_right_of_each_pixel(self):

        grid_arcsec = np.array([[0.0, 0.0], [0.0, 4.0],
                                     [-2.0, 0.0], [-2.0, 4.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=grid_arcsec, shape=(2, 2),
                                                                      pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[1, 1], [1, 2],
                                         [2, 1], [2, 2]])).all()

        grid_arcsec = np.array([[1.5, -3.0], [1.5, 3.0], [1.5, 9.0],
                                     [-1.5, -3.0], [-1.5, 3.0], [-1.5, 9.0],
                                     [-4.5, -3.0], [-4.5, 3.0], [-4.5, 9.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=grid_arcsec, shape=(3, 3),
                                                                      pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[1, 1], [1, 2], [1, 3],
                                         [2, 1], [2, 2], [2, 3],
                                         [3, 1], [3, 2], [3, 3]])).all()

    def test__same_as_above___arcsec_to_pixel__but_nonzero_origin(self):

        # -1.0 from all entries for a origin of (-1.0, -1.0)
        grid_arcsec = np.array([[-1.0, -1.0], [-1.0, 3.0],
                                     [-3.0, -1.0], [-3.0, 3.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=grid_arcsec, shape=(2, 2),
                                                                      pixel_scales=(2.0, 4.0), origin=(-1.0, -1.0))

        assert (grid_pixels == np.array([[1, 1], [1, 2],
                                         [2, 1], [2, 2]])).all()

        # -1.0, +2.0, for origin of (-1.0, +2.0)
        grid_arcsec = np.array([[0.5, -1.0], [0.5, 5.0], [0.5, 11.0],
                                     [-2.5, -1.0], [-2.5, 5.0], [-2.5, 11.0],
                                     [-5.5, -1.0], [-5.5, 5.0], [-5.5, 11.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=grid_arcsec, shape=(3, 3),
                                                                      pixel_scales=(3.0, 6.0), origin=(-1.0, 2.0))

        assert (grid_pixels == np.array([[1, 1], [1, 2], [1, 3],
                                         [2, 1], [2, 2], [2, 3],
                                         [3, 1], [3, 2], [3, 3]])).all()

    def test__1d_arc_second_grid_to_1d_pixel_centre_grid__coordinates_in_origins_of_pixels(self):

        grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0],
                                     [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec, shape=(2, 2),
                                                                             pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1],
                                         [1, 0], [1, 1]])).all()

        grid_arcsec = np.array([[3.0, -6.0], [3.0, 0.0], [3.0, 6.0],
                                         [0.0, -6.0], [0.0, 0.0], [0.0, 6.0],
                                         [-3.0, -6.0], [-3.0, 0.0], [-3.0, 6.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec, shape=(3, 3),
                                                                             pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1], [0, 2],
                                         [1, 0], [1, 1], [1, 2],
                                         [2, 0], [2, 1], [2, 2]])).all()

    def test__same_as_above_but_coordinates_are_top_left_of_each_pixel(self):

        grid_arcsec = np.array([[1.99, -3.99], [1.99, 0.01],
                                     [-0.01, -3.99], [-0.01, 0.01]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec, shape=(2, 2),
                                                                             pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1],
                                         [1, 0], [1, 1]])).all()

        grid_arcsec = np.array([[4.49, -8.99], [4.49, -2.99], [4.49, 3.01],
                                     [1.49, -8.99], [1.49, -2.99], [1.49, 3.01],
                                     [-1.51, -8.99], [-1.51, -2.99], [-1.51, 3.01]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec, shape=(3, 3),
                                                                             pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1], [0, 2],
                                         [1, 0], [1, 1], [1, 2],
                                         [2, 0], [2, 1], [2, 2]])).all()

    def test__same_as_above_but_coordinates_are_bottom_right_of_each_pixel(self):

        grid_arcsec = np.array([[0.01, -0.01], [0.01, 3.99],
                                     [-1.99, -0.01], [-1.99, 3.99]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec, shape=(2, 2),
                                                                             pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1],
                                         [1, 0], [1, 1]])).all()

        grid_arcsec = np.array([[1.51, -3.01], [1.51, 2.99], [1.51, 8.99],
                                     [-1.49, -3.01], [-1.49, 2.99], [-1.49, 8.99],
                                     [-4.49, -3.01], [-4.49, 2.99], [-4.49, 8.99]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec, shape=(3, 3),
                                                                             pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1], [0, 2],
                                         [1, 0], [1, 1], [1, 2],
                                         [2, 0], [2, 1], [2, 2]])).all()

    def test__same_as_above__arcsec_to_pixel_origin__but_nonzero_origin(self):

        # +1.0 for all entries for a origin of (1.0, 1.0)
        grid_arcsec = np.array([[2.0, -1.0], [2.0, 3.0],
                                     [0.0, -1.0], [0.0, 3.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec, shape=(2, 2),
                                                                             pixel_scales=(2.0, 4.0), origin=(1.0, 1.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1],
                                         [1, 0], [1, 1]])).all()

        # +1.0, -2.0, for origin of (1.0, -2.0)
        grid_arcsec = np.array([[4.0, -8.0], [4.0, -2.0], [4.0, 4.0],
                                     [1.0, -8.0], [1.0, -2.0], [1.0, 4.0],
                                     [-2.0, -8.0], [-2.0, -2.0], [-2.0, 4.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_centres_1d(grid_arcsec_1d=grid_arcsec, shape=(3, 3),
                                                                             pixel_scales=(3.0, 6.0), origin=(1.0, -2.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1], [0, 2],
                                         [1, 0], [1, 1], [1, 2],
                                         [2, 0], [2, 1], [2, 2]])).all()

    def test__1d_arc_second_grid_to_1d_pixel_1d_index_grid__coordinates_in_origins_of_pixels(self):

        grid_arcsec = np.array([[1.0, -2.0], [1.0, 2.0],
                                     [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(grid_arcsec_1d=grid_arcsec, shape=(2, 2),
                                                                             pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([0, 1, 2, 3])).all()

        grid_arcsec = np.array([[3.0, -6.0], [3.0, 0.0], [3.0, 6.0],
                                         [0.0, -6.0], [0.0, 0.0], [0.0, 6.0],
                                         [-3.0, -6.0], [-3.0, 0.0], [-3.0, 6.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(grid_arcsec_1d=grid_arcsec, shape=(3, 3),
                                                                             pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__same_as_above_1d_index__but_coordinates_are_top_left_of_each_pixel(self):

        grid_arcsec = np.array([[1.99, -3.99], [1.99, 0.01],
                                     [-0.01, -3.99], [-0.01, 0.01]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(grid_arcsec_1d=grid_arcsec, shape=(2, 2),
                                                                             pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([0, 1, 2, 3])).all()


        grid_arcsec = np.array([[4.49, -8.99], [4.49, -2.99], [4.49, 3.01],
                                     [1.49, -8.99], [1.49, -2.99], [1.49, 3.01],
                                     [-1.51, -8.99], [-1.51, -2.99], [-1.51, 3.01]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(grid_arcsec_1d=grid_arcsec, shape=(3, 3),
                                                                             pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__same_as_above_1d_index__but_coordinates_are_bottom_right_of_each_pixel(self):

        grid_arcsec = np.array([[0.01, -0.01], [0.01, 3.99],
                                     [-1.99, -0.01], [-1.99, 3.99]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(grid_arcsec_1d=grid_arcsec, shape=(2, 2),
                                                                             pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([0, 1, 2, 3])).all()

        grid_arcsec = np.array([[1.51, -3.01], [1.51, 2.99], [1.51, 8.99],
                                     [-1.49, -3.01], [-1.49, 2.99], [-1.49, 8.99],
                                     [-4.49, -3.01], [-4.49, 2.99], [-4.49, 8.99]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(grid_arcsec_1d=grid_arcsec, shape=(3, 3),
                                                                             pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__same_as_above__1d_index__arcsec_to_pixel_origin__but_nonzero_origin(self):

        # +1.0 for all entries for a origin of (1.0, 1.0)
        grid_arcsec = np.array([[2.0, -1.0], [2.0, 3.0],
                                     [0.0, -1.0], [0.0, 3.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(grid_arcsec_1d=grid_arcsec, shape=(2, 2),
                                                                             pixel_scales=(2.0, 4.0), origin=(1.0, 1.0))

        assert (grid_pixels == np.array([0, 1, 2, 3])).all()

        # +1.0, -2.0, for origin of (1.0, -2.0)
        grid_arcsec = np.array([[4.0, -8.0], [4.0, -2.0], [4.0, 4.0],
                                     [1.0, -8.0], [1.0, -2.0], [1.0, 4.0],
                                     [-2.0, -8.0], [-2.0, -2.0], [-2.0, 4.0]])

        grid_pixels = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(grid_arcsec_1d=grid_arcsec, shape=(3, 3),
                                                                             pixel_scales=(3.0, 6.0), origin=(1.0, -2.0))

        assert (grid_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__1d_pixel_origin_grid_to_1d_arc_second_grid__coordinates_in_origins_of_pixels(self):

        grid_pixels = np.array([[0.5, 0.5], [0.5, 1.5],
                                 [1.5, 0.5], [1.5, 1.5]])

        grid_arcsec = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels, shape=(2, 2),
                                                                           pixel_scales=(2.0, 4.0))

        assert (grid_arcsec == np.array([[1.0, -2.0], [1.0, 2.0],
                                              [-1.0, -2.0], [-1.0, 2.0]])).all()

        grid_pixels = np.array([[0.5, 0.5], [0.5, 1.5], [0.5, 2.5],
                                 [1.5, 0.5], [1.5, 1.5], [1.5, 2.5],
                                 [2.5, 0.5], [2.5, 1.5], [2.5, 2.5]])

        grid_arcsec = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels, shape=(3, 3),
                                                                           pixel_scales=(3.0, 6.0))

        assert (grid_arcsec == np.array([[3.0, -6.0], [3.0, 0.0], [3.0, 6.0],
                                              [0.0, -6.0], [0.0, 0.0], [0.0, 6.0],
                                              [-3.0, -6.0], [-3.0, 0.0], [-3.0, 6.0]])).all()

    def test__same_as_above__pixel_to_arcsec__but_coordinates_are_top_left_of_each_pixel(self):

        grid_pixels = np.array([[0, 0], [0, 1],
                                 [1, 0], [1, 1]])

        grid_arcsec = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels, shape=(2, 2),
                                                                           pixel_scales=(2.0, 4.0))

        assert (grid_arcsec == np.array([[2.0, -4], [2.0, 0.0],
                                             [0.0, -4], [0.0, 0.0]])).all()



        grid_pixels = np.array([[0, 0], [0, 1], [0, 2],
                                 [1, 0], [1, 1], [1, 2],
                                 [2, 0], [2, 1], [2, 2]])

        grid_arcsec = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels, shape=(3, 3),
                                                                           pixel_scales=(3.0, 6.0))

        assert (grid_arcsec == np.array([[4.5, -9.0], [4.5, -3.0], [4.5, 3.0],
                                             [1.5, -9.0], [1.5, -3.0], [1.5, 3.0],
                                             [-1.5, -9.0], [-1.5, -3.0], [-1.5, 3.0]])).all()

    def test__same_as_above__pixel_to_arcsec_but_coordinates_are_bottom_right_of_each_pixel(self):

        grid_pixels = np.array([[1, 1], [1, 2],
                                [2, 1], [2, 2]])

        grid_arcsec = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels, shape=(2, 2),
                                                                           pixel_scales=(2.0, 4.0))

        assert (grid_arcsec == np.array([[0.0, 0.0], [0.0, 4.0],
                                     [-2.0, 0.0], [-2.0, 4.0]])).all()

        grid_pixels = np.array([[1, 1], [1, 2], [1, 3],
                                [2, 1], [2, 2], [2, 3],
                                [3, 1], [3, 2], [3, 3]])

        grid_arcsec = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels, shape=(3, 3),
                                                                           pixel_scales=(3.0, 6.0))

        assert (grid_arcsec == np.array([[1.5, -3.0], [1.5, 3.0], [1.5, 9.0],
                                     [-1.5, -3.0], [-1.5, 3.0], [-1.5, 9.0],
                                     [-4.5, -3.0], [-4.5, 3.0], [-4.5, 9.0]])).all()

    def test__same_as_above__pixel_to_arcsec__nonzero_origin(self):

        grid_pixels = np.array([[0.5, 0.5], [0.5, 1.5],
                                 [1.5, 0.5], [1.5, 1.5]])

        grid_arcsec = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels, shape=(2, 2),
                                                                           pixel_scales=(2.0, 4.0), origin=(-1.0, -1.0))

        # -1.0 from all entries for a origin of (-1.0, -1.0)
        assert (grid_arcsec == np.array([[0.0, -3.0], [0.0, 1.0],
                                              [-2.0, -3.0], [-2.0, 1.0]])).all()

        grid_pixels = np.array([[0.5, 0.5], [0.5, 1.5], [0.5, 2.5],
                                 [1.5, 0.5], [1.5, 1.5], [1.5, 2.5],
                                 [2.5, 0.5], [2.5, 1.5], [2.5, 2.5]])

        grid_arcsec = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=grid_pixels, shape=(3, 3),
                                                                           pixel_scales=(3.0, 6.0), origin=(-1.0, 2.0))

        # -1.0, +2.0, for origin of (-1.0, 2.0)
        assert grid_arcsec == pytest.approx(np.array([[2.0, -4.0], [2.0, 2.0], [2.0, 8.0],
                                                           [-1.0, -4.0], [-1.0, 2.0], [-1.0, 8.0],
                                                           [-4.0, -4.0], [-4.0, 2.0], [-4.0, 8.0]]), 1e-4)
        
    def test__2d_arc_second_grid_to_2d_pixel_centre_grid__coordinates_in_origins_of_pixels(self):

        grid_arcsec = np.array([[[1.0, -2.0], [1.0, 2.0]],
                                [[-1.0, -2.0], [-1.0, 2.0]]])

        grid_pixels = grid_util.grid_arcsec_2d_to_grid_pixel_centres_2d(grid_arcsec_2d=grid_arcsec, shape=(2, 2),
                                                                        pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[[0, 0], [0, 1]],
                                         [[1, 0], [1, 1]]])).all()

        grid_arcsec = np.array([[[3.0, -6.0], [3.0, 0.0], [3.0, 6.0]],
                                [[0.0, -6.0], [0.0, 0.0], [0.0, 6.0]],
                                [[-3.0, -6.0], [-3.0, 0.0], [-3.0, 6.0]]])

        grid_pixels = grid_util.grid_arcsec_2d_to_grid_pixel_centres_2d(grid_arcsec_2d=grid_arcsec, shape=(3, 3),
                                                                             pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[[0, 0], [0, 1], [0, 2]],
                                         [[1, 0], [1, 1], [1, 2]],
                                         [[2, 0], [2, 1], [2, 2]]])).all()

    def test__2d_same_as_above_but_coordinates_are_top_left_of_each_pixel(self):

        grid_arcsec = np.array([[[1.99, -3.99], [1.99, 0.01]],
                                [[-0.01, -3.99], [-0.01, 0.01]]])

        grid_pixels = grid_util.grid_arcsec_2d_to_grid_pixel_centres_2d(grid_arcsec_2d=grid_arcsec, shape=(2, 2),
                                                                             pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[[0, 0], [0, 1]],
                                         [[1, 0], [1, 1]]])).all()

        grid_arcsec = np.array([[[4.49, -8.99], [4.49, -2.99], [4.49, 3.01]],
                                     [[1.49, -8.99], [1.49, -2.99], [1.49, 3.01]],
                                     [[-1.51, -8.99], [-1.51, -2.99], [-1.51, 3.01]]])

        grid_pixels = grid_util.grid_arcsec_2d_to_grid_pixel_centres_2d(grid_arcsec_2d=grid_arcsec, shape=(3, 3),
                                                                             pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[[0, 0], [0, 1], [0, 2]],
                                         [[1, 0], [1, 1], [1, 2]],
                                         [[2, 0], [2, 1], [2, 2]]])).all()

    def test__2d_same_as_above_but_coordinates_are_bottom_right_of_each_pixel(self):

        grid_arcsec = np.array([[[0.01, -0.01], [0.01, 3.99]],
                                [[-1.99, -0.01], [-1.99, 3.99]]])

        grid_pixels = grid_util.grid_arcsec_2d_to_grid_pixel_centres_2d(grid_arcsec_2d=grid_arcsec, shape=(2, 2),
                                                                             pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[[0, 0], [0, 1]],
                                         [[1, 0], [1, 1]]])).all()

        grid_arcsec = np.array([[[1.51, -3.01], [1.51, 2.99], [1.51, 8.99]],
                                 [[-1.49, -3.01], [-1.49, 2.99], [-1.49, 8.99]],
                                 [[-4.49, -3.01], [-4.49, 2.99], [-4.49, 8.99]]])

        grid_pixels = grid_util.grid_arcsec_2d_to_grid_pixel_centres_2d(grid_arcsec_2d=grid_arcsec, shape=(3, 3),
                                                                             pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[[0, 0], [0, 1], [0, 2]],
                                         [[1, 0], [1, 1], [1, 2]],
                                         [[2, 0], [2, 1], [2, 2]]])).all()

    def test__2d_same_as_above__arcsec_to_pixel_origin__but_nonzero_origin(self):

        # +1.0 for all entries for a origin of (1.0, 1.0)
        grid_arcsec = np.array([[[2.0, -1.0], [2.0, 3.0]],
                                [[0.0, -1.0], [0.0, 3.0]]])

        grid_pixels = grid_util.grid_arcsec_2d_to_grid_pixel_centres_2d(grid_arcsec_2d=grid_arcsec, shape=(2, 2),
                                                                             pixel_scales=(2.0, 4.0), origin=(1.0, 1.0))

        assert (grid_pixels == np.array([[[0, 0], [0, 1]],
                                         [[1, 0], [1, 1]]])).all()

        # +1.0, -2.0, for origin of (1.0, -2.0)
        grid_arcsec = np.array([[[4.0, -8.0], [4.0, -2.0], [4.0, 4.0]],
                                 [[1.0, -8.0], [1.0, -2.0], [1.0, 4.0]],
                                 [[-2.0, -8.0], [-2.0, -2.0], [-2.0, 4.0]]])

        grid_pixels = grid_util.grid_arcsec_2d_to_grid_pixel_centres_2d(grid_arcsec_2d=grid_arcsec, shape=(3, 3),
                                                                             pixel_scales=(3.0, 6.0), origin=(1.0, -2.0))

        assert (grid_pixels == np.array([[[0, 0], [0, 1], [0, 2]],
                                         [[1, 0], [1, 1], [1, 2]],
                                         [[2, 0], [2, 1], [2, 2]]])).all()