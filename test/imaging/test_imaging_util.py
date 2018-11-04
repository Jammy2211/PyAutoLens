import os

import numpy as np
import pytest

from autolens import exc
from autolens.imaging import imaging_util as util

test_data_dir = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name="memoizer")
def make_memoizer():
    return util.Memoizer()


class TestMemoizer(object):
    def test_storing(self, memoizer):
        @memoizer
        def func(arg):
            return "result for {}".format(arg)

        func(1)
        func(2)
        func(1)

        assert memoizer.results == {"('arg', 1)": "result for 1", "('arg', 2)": "result for 2"}
        assert memoizer.calls == 2

    def test_multiple_arguments(self, memoizer):
        @memoizer
        def func(arg1, arg2):
            return arg1 * arg2

        func(1, 2)
        func(2, 1)
        func(1, 2)

        assert memoizer.results == {"('arg1', 1), ('arg2', 2)": 2, "('arg1', 2), ('arg2', 1)": 2}
        assert memoizer.calls == 2

    def test_key_word_arguments(self, memoizer):
        @memoizer
        def func(arg1=0, arg2=0):
            return arg1 * arg2

        func(arg1=1)
        func(arg2=1)
        func(arg1=1)
        func(arg1=1, arg2=1)

        assert memoizer.results == {"('arg1', 1)": 0, "('arg2', 1)": 0, "('arg1', 1), ('arg2', 1)": 1}
        assert memoizer.calls == 3

    def test_key_word_for_positional(self, memoizer):
        @memoizer
        def func(arg):
            return "result for {}".format(arg)

        func(1)
        func(arg=2)
        func(arg=1)

        assert memoizer.calls == 2

    def test_methods(self, memoizer):
        class Class(object):
            def __init__(self, value):
                self.value = value

            @memoizer
            def method(self):
                return self.value

        one = Class(1)
        two = Class(2)

        assert one.method() == 1
        assert two.method() == 2


class TestTotalPixels:

    def test__total_image_pixels_from_mask(self):
        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        assert util.total_image_pixels_from_mask(mask) == 5

    def test__total_sub_pixels_from_mask(self):
        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        assert util.total_sub_pixels_from_mask_and_sub_grid_size(mask, sub_grid_size=2) == 20

    def test__total_border_pixels_from_mask(self):
        mask = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, False, False, False, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])

        assert util.total_border_pixels_from_mask(mask) == 8


class TestGrid2d:

    def test__array_3x3__sets_up_arcsecond_grid(self):

        grid_2d = util.image_grid_2d_from_shape_and_pixel_scales(shape=(3, 3), pixel_scales=(2.0, 1.0))

        assert (grid_2d == np.array([[[2., -1.], [2., 0.], [2., 1.]],
                                     [[0., -1.], [0., 0.], [0., 1.]],
                                     [[-2., -1.], [-2., 0.], [-2., 1.]]])).all()

    def test__array_4x4_and_different_pixel_scale__sets_up_arcsecond_grid(self):

        grid_2d = util.image_grid_2d_from_shape_and_pixel_scales(shape=(4, 4), pixel_scales=(0.5, 0.5))

        assert (grid_2d == np.array([[[0.75, -0.75], [0.75, -0.25], [0.75, 0.25], [0.75, 0.75]],
                                     [[0.25, -0.75], [0.25, -0.25], [0.25, 0.25], [0.25, 0.75]],
                                     [[-0.25, -0.75], [-0.25, -0.25], [-0.25, 0.25], [-0.25, 0.75]],
                                     [[-0.75, -0.75], [-0.75, -0.25], [-0.75, 0.25], [-0.75, 0.75]]])).all()

    def test__array_2x3__sets_up_arcsecond_grid(self):
        grid_2d = util.image_grid_2d_from_shape_and_pixel_scales(shape=(2, 3), pixel_scales=(1.0, 1.0))

        assert (grid_2d == np.array([[[0.5, -1.], [0.5, 0.], [0.5, 1.]],
                                     [[-0.5, -1.], [-0.5, 0.], [-0.5, 1.]]])).all()

    def test__array_3x2__sets_up_arcsecond_grid(self):
        grid_2d = util.image_grid_2d_from_shape_and_pixel_scales(shape=(3, 2), pixel_scales=(1.0, 1.0))

        assert (grid_2d == np.array([[[1., -0.5], [1., 0.5]],
                                     [[0., -0.5], [0., 0.5]],
                                     [[-1., -0.5], [-1., 0.5]]])).all()


class TestGrid1d:

    def test__array_3x3__sets_up_arcsecond_grid(self):

        grid_2d = util.image_grid_1d_from_shape_and_pixel_scales(shape=(3, 3), pixel_scales=(2.0, 1.0))

        assert (grid_2d == np.array([[2., -1.], [2., 0.], [2., 1.],
                                     [0., -1.], [0., 0.], [0., 1.],
                                     [-2., -1.], [-2., 0.], [-2., 1.]])).all()

    def test__array_4x4_and_different_pixel_scale__sets_up_arcsecond_grid(self):

        grid_2d = util.image_grid_1d_from_shape_and_pixel_scales(shape=(4, 4), pixel_scales=(0.5, 0.5))

        assert (grid_2d == np.array([[0.75, -0.75], [0.75, -0.25], [0.75, 0.25], [0.75, 0.75],
                                     [0.25, -0.75], [0.25, -0.25], [0.25, 0.25], [0.25, 0.75],
                                     [-0.25, -0.75], [-0.25, -0.25], [-0.25, 0.25], [-0.25, 0.75],
                                     [-0.75, -0.75], [-0.75, -0.25], [-0.75, 0.25], [-0.75, 0.75]])).all()

    def test__array_2x3__sets_up_arcsecond_grid(self):
        grid_2d = util.image_grid_1d_from_shape_and_pixel_scales(shape=(2, 3), pixel_scales=(1.0, 1.0))

        assert (grid_2d == np.array([[0.5, -1.], [0.5, 0.], [0.5, 1.],
                                     [-0.5, -1.], [-0.5, 0.], [-0.5, 1.]])).all()

    def test__array_3x2__sets_up_arcsecond_grid(self):
        grid_2d = util.image_grid_1d_from_shape_and_pixel_scales(shape=(3, 2), pixel_scales=(1.0, 1.0))

        assert (grid_2d == np.array([[1., -0.5], [1., 0.5],
                                     [0., -0.5], [0., 0.5],
                                     [-1., -0.5], [-1., 0.5]])).all()


class TestImageGridMasked(object):

    def test__setup_3x3_image_1_coordinate_in_mask(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        image_grid = util.image_grid_1d_masked_from_mask_and_pixel_scales(mask=mask, pixel_scales=(3.0, 6.0))

        assert (image_grid[0] == np.array([0.0, 0.0])).all()

    def test__setup_3x3_image__five_coordinates_in_mask(self):
        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        image_grid = util.image_grid_1d_masked_from_mask_and_pixel_scales(mask=mask, pixel_scales=(6.0, 3.0))

        assert (image_grid == np.array([           [6., 0.],
                                        [0., -3.], [0., 0.], [0., 3.],
                                                   [-6., 0.]])).all()

    def test__setup_4x4_image__ten_coordinates_in_grid__new_pixel_scale(self):
        mask = np.array([[True, False, False, True],
                         [False, False, False, True],
                         [True, False, False, True],
                         [False, False, False, True]])

        image_grid = util.image_grid_1d_masked_from_mask_and_pixel_scales(mask=mask, pixel_scales=(1.0, 1.0))

        assert (image_grid == np.array(             [[1.5, -0.5], [1.5, 0.5],
                                       [0.5, -1.5], [0.5, -0.5], [0.5, 0.5],
                                                     [-0.5, -0.5], [-0.5, 0.5],
                                        [-1.5, -1.5], [-1.5, -0.5], [-1.5, 0.5]])).all()

    def test__setup_3x4_image__six_grid(self):
        mask = np.array([[True, False, True, True],
                         [False, False, False, True],
                         [True, False, True, False]])

        image_grid = util.image_grid_1d_masked_from_mask_and_pixel_scales(mask=mask, pixel_scales=(3.0, 3.0))

        assert (image_grid == np.array([             [3., -1.5],
                                       [0., -4.5], [0., -1.5], [0., 1.5],
                                                    [-3., -1.5],           [-3., 4.5]])).all()


class TestSubGridMasked(object):

    def test__3x3_mask_with_one_pixel__2x2_sub_grid__grid(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        sub_grid = util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 6.0),
                                                                                    sub_grid_size=2)

        assert (sub_grid[0:4] == np.array([[0.5, -1.0], [0.5, 1.0],
                                          [-0.5, -1.0], [-0.5, 1.0]])).all()

    def test__3x3_mask_with_row_of_pixels__2x2_sub_grid__grid(self):
        mask = np.array([[True, True, True],
                         [False, False, False],
                         [True, True, True]])

        sub_grid = util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 3.0),
                                                                                    sub_grid_size=2)

        assert (sub_grid[0:4] == np.array([[0.5, -3.5], [0.5, -2.5],
                                           [-0.5, -3.5], [-0.5, -2.5]])).all()

        assert (sub_grid[4:8] == np.array([[0.5, -0.5], [0.5, 0.5],
                                           [-0.5, -0.5], [-0.5, 0.5]])).all()

        assert (sub_grid[8:12] == np.array([[0.5,  2.5], [0.5, 3.5],
                                            [-0.5, 2.5], [-0.5, 3.5]])).all()

    def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__grid(self):
        mask = np.array([[True, True, False],
                         [False, False, False],
                         [True, True, False]])

        sub_grid = util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 3.0),
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

        sub_grid = util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(0.3, 0.3),
                                                                                    sub_grid_size=2)

        sub_grid = np.round(sub_grid, decimals=2)

        np.testing.assert_almost_equal(sub_grid,
                                       np.array([[0.35, 0.25], [0.35, 0.35], [0.25, 0.25], [0.25, 0.35],
                                                 [0.05, -0.35], [0.05, -0.25], [-0.05, -0.35], [-0.05, -0.25],
                                                 [0.05, -0.05], [0.05, 0.05], [-0.05, -0.05], [-0.05, 0.05],
                                                 [0.05, 0.25], [0.05, 0.35], [-0.05, 0.25], [-0.05, 0.35],
                                                 [-0.25, 0.25], [-0.25, 0.35], [-0.35, 0.25], [-0.35, 0.35]]))

    def test__3x3_mask_with_one_pixel__3x3_sub_grid__grid(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        sub_grid = util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 3.0),
                                                                                    sub_grid_size=3)

        assert (sub_grid == np.array([[[0.75, -0.75], [0.75, 0.], [0.75, 0.75], [0., -0.75], [0., 0.],
                                       [0., 0.75], [-0.75, -0.75], [-0.75, 0.], [-0.75, 0.75]]])).all()

    def test__3x3_mask_with_one_row__3x3_sub_grid__grid(self):
        mask = np.array([[True, True, False],
                         [True, False, True],
                         [True, True, False]])

        sub_grid = util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(2.0, 2.0),
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

    def test__4x4_mask_with_one_pixel__4x4_sub_grid__grid(self):
        mask = np.array([[True, True, True, True],
                         [True, False, False, True],
                         [True, False, False, True],
                         [True, True, True, False]])

        sub_grid = util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(2.0, 2.0),
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

    def test__4x3_mask_with_one_pixel__2x2_sub_grid__grid(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, False, False],
                         [False, True, True]])

        sub_grid = util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 3.0),
                                                                                    sub_grid_size=2)

        assert (sub_grid == np.array([[2., -0.5], [2., 0.5], [1., -0.5], [1., 0.5],
                                      [-1., -0.5], [-1., 0.5], [-2., -0.5], [-2., 0.5],
                                       [-1., 2.5], [-1., 3.5], [-2., 2.5], [-2., 3.5],
                                      [-4., -3.5], [-4., -2.5], [-5., -3.5], [-5., -2.5]])).all()

    def test__3x4_mask_with_one_pixel__2x2_sub_grid__grid(self):

        mask = np.array([[True, True, True, False],
                         [True, False, False, True],
                         [False, True, False, True]])

        sub_grid = util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask, pixel_scales=(3.0, 3.0),
                                                                                    sub_grid_size=2)

        assert (sub_grid == np.array([[3.5, 4.], [3.5, 5.], [2.5, 4.], [2.5, 5.],
                                      [0.5, -2.], [0.5, -1.], [-0.5, -2.], [-0.5, -1.],
                                      [0.5, 1.], [0.5, 2.], [-0.5, 1.], [-0.5, 2.],
                                      [-2.5, -5.], [-2.5, -4.], [-3.5, -5.], [-3.5, -4.],
                                      [-2.5, 1.], [-2.5, 2.], [-3.5, 1.], [-3.5, 2.]])).all()


class TestGridPixelArcSecondConversion(object):

    def test__1d_arc_second_grid_to_1d_pixel_centred_grid__coordinates_in_centres_of_pixels(self):

        grid_arc_seconds = np.array([[1.0, -2.0], [1.0, 2.0],
                                     [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels = util.grid_arc_seconds_1d_to_grid_pixel_centres_1d(grid_arc_seconds=grid_arc_seconds, shape=(2, 2),
                                                                        pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1],
                                         [1, 0], [1, 1]])).all()

        grid_arc_seconds = np.array([[3.0, -6.0], [3.0, 0.0], [3.0, 6.0],
                                         [0.0, -6.0], [0.0, 0.0], [0.0, 6.0],
                                         [-3.0, -6.0], [-3.0, 0.0], [-3.0, 6.0]])

        grid_pixels = util.grid_arc_seconds_1d_to_grid_pixel_centres_1d(grid_arc_seconds=grid_arc_seconds, shape=(3, 3),
                                                                        pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1], [0, 2],
                                         [1, 0], [1, 1], [1, 2],
                                         [2, 0], [2, 1], [2, 2]])).all()

    def test__same_as_above_but_coordinates_are_top_left_of_each_pixel(self):

        grid_arc_seconds = np.array([[1.99, -3.99], [1.99, 0.01],
                                     [-0.01, -3.99], [-0.01, 0.01]])

        grid_pixels = util.grid_arc_seconds_1d_to_grid_pixel_centres_1d(grid_arc_seconds=grid_arc_seconds, shape=(2, 2),
                                                                        pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1],
                                         [1, 0], [1, 1]])).all()

        grid_arc_seconds = np.array([[4.49, -8.99], [4.49, -2.99], [4.49, 3.01],
                                     [1.49, -8.99], [1.49, -2.99], [1.49, 3.01],
                                     [-1.51, -8.99], [-1.51, -2.99], [-1.51, 3.01]])

        grid_pixels = util.grid_arc_seconds_1d_to_grid_pixel_centres_1d(grid_arc_seconds=grid_arc_seconds, shape=(3, 3),
                                                                        pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1], [0, 2],
                                         [1, 0], [1, 1], [1, 2],
                                         [2, 0], [2, 1], [2, 2]])).all()

    def test__same_as_above_but_coordinates_are_bottom_right_of_each_pixel(self):

        grid_arc_seconds = np.array([[0.01, -0.01], [0.01, 3.99],
                                     [-1.99, -0.01], [-1.99, 3.99]])

        grid_pixels = util.grid_arc_seconds_1d_to_grid_pixel_centres_1d(grid_arc_seconds=grid_arc_seconds, shape=(2, 2),
                                                                        pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1],
                                         [1, 0], [1, 1]])).all()

        grid_arc_seconds = np.array([[1.51, -3.01], [1.51, 2.99], [1.51, 8.99],
                                     [-1.49, -3.01], [-1.49, 2.99], [-1.49, 8.99],
                                     [-4.49, -3.01], [-4.49, 2.99], [-4.49, 8.99]])

        grid_pixels = util.grid_arc_seconds_1d_to_grid_pixel_centres_1d(grid_arc_seconds=grid_arc_seconds, shape=(3, 3),
                                                                        pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1], [0, 2],
                                         [1, 0], [1, 1], [1, 2],
                                         [2, 0], [2, 1], [2, 2]])).all()

    def test__1d_arc_second_grid_to_1d_pixel_grid__coordinates_in_centres_of_pixels(self):

        grid_arc_seconds = np.array([[1.0, -2.0], [1.0, 2.0],
                                     [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels = util.grid_arc_seconds_1d_to_grid_pixels_1d(grid_arc_seconds=grid_arc_seconds, shape=(2, 2),
                                                                        pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[0.5, 0.5], [0.5, 1.5],
                                         [1.5, 0.5], [1.5, 1.5]])).all()

        grid_arc_seconds = np.array([[3.0, -6.0], [3.0, 0.0], [3.0, 6.0],
                                         [0.0, -6.0], [0.0, 0.0], [0.0, 6.0],
                                         [-3.0, -6.0], [-3.0, 0.0], [-3.0, 6.0]])

        grid_pixels = util.grid_arc_seconds_1d_to_grid_pixels_1d(grid_arc_seconds=grid_arc_seconds, shape=(3, 3),
                                                                        pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[0.5, 0.5], [0.5, 1.5], [0.5, 2.5],
                                         [1.5, 0.5], [1.5, 1.5], [1.5, 2.5],
                                         [2.5, 0.5], [2.5, 1.5], [2.5, 2.5]])).all()

    def test__same_as_above__pixels__but_coordinates_are_top_left_of_each_pixel(self):

        grid_arc_seconds = np.array([[2.0, -4], [2.0, 0.0],
                                     [0.0, -4], [0.0, 0.0]])

        grid_pixels = util.grid_arc_seconds_1d_to_grid_pixels_1d(grid_arc_seconds=grid_arc_seconds, shape=(2, 2),
                                                                        pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1],
                                         [1, 0], [1, 1]])).all()

        grid_arc_seconds = np.array([[4.5, -9.0], [4.5, -3.0], [4.5, 3.0],
                                     [1.5, -9.0], [1.5, -3.0], [1.5, 3.0],
                                     [-1.5, -9.0], [-1.5, -3.0], [-1.5, 3.0]])

        grid_pixels = util.grid_arc_seconds_1d_to_grid_pixels_1d(grid_arc_seconds=grid_arc_seconds, shape=(3, 3),
                                                                        pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[0, 0], [0, 1], [0, 2],
                                         [1, 0], [1, 1], [1, 2],
                                         [2, 0], [2, 1], [2, 2]])).all()

    def test__same_as_above___pixels__but_coordinates_are_bottom_right_of_each_pixel(self):

        grid_arc_seconds = np.array([[0.0, 0.0], [0.0, 4.0],
                                     [-2.0, 0.0], [-2.0, 4.0]])

        grid_pixels = util.grid_arc_seconds_1d_to_grid_pixels_1d(grid_arc_seconds=grid_arc_seconds, shape=(2, 2),
                                                                        pixel_scales=(2.0, 4.0))

        assert (grid_pixels == np.array([[1, 1], [1, 2],
                                         [2, 1], [2, 2]])).all()

        grid_arc_seconds = np.array([[1.5, -3.0], [1.5, 3.0], [1.5, 9.0],
                                     [-1.5, -3.0], [-1.5, 3.0], [-1.5, 9.0],
                                     [-4.5, -3.0], [-4.5, 3.0], [-4.5, 9.0]])

        grid_pixels = util.grid_arc_seconds_1d_to_grid_pixels_1d(grid_arc_seconds=grid_arc_seconds, shape=(3, 3),
                                                                        pixel_scales=(3.0, 6.0))

        assert (grid_pixels == np.array([[1, 1], [1, 2], [1, 3],
                                         [2, 1], [2, 2], [2, 3],
                                         [3, 1], [3, 2], [3, 3]])).all()

    def test__1d_pixel_centre_grid_to_1d_arc_second_grid__coordinates_in_centres_of_pixels(self):

        grid_pixels = np.array([[0.5, 0.5], [0.5, 1.5],
                                 [1.5, 0.5], [1.5, 1.5]])

        grid_arc_seconds = util.grid_pixels_1d_to_grid_arc_seconds_1d(grid_pixels=grid_pixels, shape=(2, 2),
                                                                      pixel_scales=(2.0, 4.0))

        assert (grid_arc_seconds == np.array([[1.0, -2.0], [1.0, 2.0],
                                              [-1.0, -2.0], [-1.0, 2.0]])).all()

        grid_pixels = np.array([[0.5, 0.5], [0.5, 1.5], [0.5, 2.5],
                                 [1.5, 0.5], [1.5, 1.5], [1.5, 2.5],
                                 [2.5, 0.5], [2.5, 1.5], [2.5, 2.5]])

        grid_arc_seconds = util.grid_pixels_1d_to_grid_arc_seconds_1d(grid_pixels=grid_pixels, shape=(3, 3),
                                                                      pixel_scales=(3.0, 6.0))

        assert (grid_arc_seconds == np.array([[3.0, -6.0], [3.0, 0.0], [3.0, 6.0],
                                              [0.0, -6.0], [0.0, 0.0], [0.0, 6.0],
                                              [-3.0, -6.0], [-3.0, 0.0], [-3.0, 6.0]])).all()

    def test__same_as_above__pixel_to_arcsec__but_coordinates_are_top_left_of_each_pixel(self):

        grid_pixels = np.array([[0, 0], [0, 1],
                                 [1, 0], [1, 1]])

        grid_arc_seconds = util.grid_pixels_1d_to_grid_arc_seconds_1d(grid_pixels=grid_pixels, shape=(2, 2),
                                                                      pixel_scales=(2.0, 4.0))

        assert (grid_arc_seconds == np.array([[2.0, -4], [2.0, 0.0],
                                             [0.0, -4], [0.0, 0.0]])).all()



        grid_pixels = np.array([[0, 0], [0, 1], [0, 2],
                                 [1, 0], [1, 1], [1, 2],
                                 [2, 0], [2, 1], [2, 2]])

        grid_arc_seconds = util.grid_pixels_1d_to_grid_arc_seconds_1d(grid_pixels=grid_pixels, shape=(3, 3),
                                                                      pixel_scales=(3.0, 6.0))

        assert (grid_arc_seconds == np.array([[4.5, -9.0], [4.5, -3.0], [4.5, 3.0],
                                             [1.5, -9.0], [1.5, -3.0], [1.5, 3.0],
                                             [-1.5, -9.0], [-1.5, -3.0], [-1.5, 3.0]])).all()

    def test__same_as_above__pixel_to_arcsec_but_coordinates_are_bottom_right_of_each_pixel(self):

        grid_pixels = np.array([[1, 1], [1, 2],
                                [2, 1], [2, 2]])

        grid_arc_seconds = util.grid_pixels_1d_to_grid_arc_seconds_1d(grid_pixels=grid_pixels, shape=(2, 2),
                                                                      pixel_scales=(2.0, 4.0))

        assert (grid_arc_seconds == np.array([[0.0, 0.0], [0.0, 4.0],
                                     [-2.0, 0.0], [-2.0, 4.0]])).all()

        grid_pixels = np.array([[1, 1], [1, 2], [1, 3],
                                [2, 1], [2, 2], [2, 3],
                                [3, 1], [3, 2], [3, 3]])

        grid_arc_seconds = util.grid_pixels_1d_to_grid_arc_seconds_1d(grid_pixels=grid_pixels, shape=(3, 3),
                                                                      pixel_scales=(3.0, 6.0))

        assert (grid_arc_seconds == np.array([[1.5, -3.0], [1.5, 3.0], [1.5, 9.0],
                                     [-1.5, -3.0], [-1.5, 3.0], [-1.5, 9.0],
                                     [-4.5, -3.0], [-4.5, 3.0], [-4.5, 9.0]])).all()


class TestGridToPixel(object):

    def test__setup_3x3_image_one_pixel(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        grid_to_pixel = util.grid_to_pixel_from_mask(mask)

        assert (grid_to_pixel == np.array([[1, 1]])).all()

    def test__setup_3x3_image__five_pixels(self):
        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        grid_to_pixel = util.grid_to_pixel_from_mask(mask)

        assert (grid_to_pixel == np.array([        [0, 1],
                                           [1, 0], [1, 1], [1, 2],
                                                   [2, 1]])).all()

    def test__setup_3x4_image__six_pixels(self):

        mask = np.array([[True, False, True, True],
                         [False, False, False, True],
                         [True, False, True, False]])

        grid_to_pixel = util.grid_to_pixel_from_mask(mask)

        assert (grid_to_pixel == np.array([        [0, 1],
                                           [1, 0], [1, 1], [1, 2],
                                                   [2, 1],         [2, 3]])).all()

    def test__setup_4x3_image__six_pixels(self):
        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True],
                         [True, True, False]])

        grid_to_pixel = util.grid_to_pixel_from_mask(mask)

        assert (grid_to_pixel == np.array([        [0, 1],
                                           [1, 0], [1, 1], [1, 2],
                                                   [2, 1],
                                                           [3, 2]])).all()


class TestSubToImage(object):

    def test__3x3_mask_with_1_pixel__2x2_sub_grid__correct_sub_to_image(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        sub_to_image = util.sub_to_image_from_mask(mask, sub_grid_size=2)

        assert (sub_to_image == np.array([0, 0, 0, 0])).all()

    def test__3x3_mask_with_row_of_pixels_pixel__2x2_sub_grid__correct_sub_to_image(self):
        mask = np.array([[True, True, True],
                         [False, False, False],
                         [True, True, True]])

        sub_to_image = util.sub_to_image_from_mask(mask, sub_grid_size=2)

        assert (sub_to_image == np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])).all()

    def test__3x3_mask_with_row_of_pixels_pixel__3x3_sub_grid__correct_sub_to_image(self):
        mask = np.array([[True, True, True],
                         [False, False, False],
                         [True, True, True]])

        sub_to_image = util.sub_to_image_from_mask(mask, sub_grid_size=3)

        assert (sub_to_image == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1,
                                          2, 2, 2, 2, 2, 2, 2, 2, 2])).all()


class TestImageBorderPixels(object):

    def test__7x7_mask_one_central_pixel__is_entire_border(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        border_pixels = util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0])).all()

    def test__7x7_mask_nine_central_pixels__is_border(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        border_pixels = util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

    def test__7x7_mask_rectangle_of_fifteen_central_pixels__is_border(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True]])

        border_pixels = util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14])).all()

    def test__8x7_mask_add_edge_pixels__also_in_border(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, False, False, False, False, False, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True]])

        border_pixels = util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17])).all()

    def test__8x7_mask_big_square(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, False, False, False, False, False, True],
                         [True, True, True, True, True, True, True]])

        border_pixels = util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 26, 27, 28, 29])).all()

    def test__7x8_mask_add_edge_pixels__also_in_border(self):
        mask = np.array([[True, True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True, True],
                         [True, True, False, False, False, True, True, True],
                         [True, True, False, False, False, True, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, True, False, False, False, True, True, True],
                         [True, True, True, True, True, True, True, True]])

        border_pixels = util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14])).all()

    def test__7x8_mask_big_square(self):
        mask = np.array([[True, True, True, True, True, True, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, True, True, True, True, True, True, True]])

        border_pixels = util.border_pixels_from_mask(mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24])).all()


class TestSubBorderPixels(object):

    def test__7x7_mask__2x2_sub_grid__nine_central_pixels__is_border(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        border_sub_pixels = util.border_sub_pixels_from_mask_pixel_scales_and_sub_grid_size(mask=mask,
                                                                                            pixel_scales=(3.0, 3.0),
                                                                                            sub_grid_size=2)

        assert (border_sub_pixels == np.array([0, 4, 9, 12, 21, 26, 30, 35])).all()

    def test__7x7_mask__4x4_sub_grid_nine_central_pixels__is_border(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        border_sub_pixels = util.border_sub_pixels_from_mask_pixel_scales_and_sub_grid_size(mask=mask,
                                                                                            pixel_scales=(3.0, 3.0),
                                                                                            sub_grid_size=4)

        assert (border_sub_pixels == np.array([0, 16, 35, 48, 83, 108, 124, 143])).all()

    def test__7x7_mask_rectangle_of_fifteen_central_pixels__is_border(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True]])

        border_sub_pixels = util.border_sub_pixels_from_mask_pixel_scales_and_sub_grid_size(mask=mask,
                                                                                            pixel_scales=(3.0, 3.0),
                                                                                            sub_grid_size=2)
        assert (border_sub_pixels == np.array([0, 4, 9, 12, 21, 24, 33, 38, 47, 50, 54, 59])).all()

    def test__8x7_mask_add_edge_pixels__also_in_border(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, False, False, False, False, False, True],
                         [True, True, False, False, False, True, True],
                         [True, True, False, False, False, True, True],
                         [True, True, True, True, True, True, True]])

        border_sub_pixels = util.border_sub_pixels_from_mask_pixel_scales_and_sub_grid_size(mask=mask,
                                                                                            pixel_scales=(3.0, 3.0),
                                                                                            sub_grid_size=2)
        assert (border_sub_pixels == np.array([0, 4, 8, 13, 16, 25, 30, 34, 43, 47, 50, 59, 62, 66, 71])).all()

    def test__7x8_mask_add_edge_pixels__also_in_border(self):
        mask = np.array([[True, True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True, True],
                         [True, True, False, False, False, True, True, True],
                         [True, True, False, False, False, True, True, True],
                         [True, False, False, False, False, False, True, True],
                         [True, True, False, False, False, True, True, True],
                         [True, True, True, True, True, True, True, True]])

        border_sub_pixels = util.border_sub_pixels_from_mask_pixel_scales_and_sub_grid_size(mask=mask,
                                                                                            pixel_scales=(3.0, 3.0),
                                                                                            sub_grid_size=2)
        assert (border_sub_pixels == np.array([0, 4, 8, 13, 16, 25, 30, 34, 43, 47, 50, 54, 59])).all()


class TestMaskCircular(object):

    def test__input_big_mask__mask(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=1.0,
                                                                    radius_arcsec=5.0)

        assert mask.shape == (3, 3)
        assert (mask == np.array([[False, False, False],
                                  [False, False, False],
                                  [False, False, False]])).all()

    def test__3x3_mask_input_radius_small__mask(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=1.0,
                                                                    radius_arcsec=0.5)

        assert (mask == np.array([[True, True, True],
                                  [True, False, True],
                                  [True, True, True]])).all()

    def test__3x3_mask_input_radius_medium__mask(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=1.0,
                                                                    radius_arcsec=1.3)

        assert (mask == np.array([[True,  False, True],
                                  [False, False, False],
                                  [True,  False, True]])).all()

    def test__3x3_mask_input_radius_large__mask(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=1.0,
                                                                    radius_arcsec=3.0)

        assert (mask == np.array([[False, False, False],
                                  [False, False, False],
                                  [False, False, False]])).all()

    def test__4x3_mask_input_radius_small__mask(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(4, 3), pixel_scale=1.0,
                                                                    radius_arcsec=0.5)

        assert (mask == np.array([[True, True, True],
                                  [True, False, True],
                                  [True, False, True],
                                  [True, True, True]])).all()

    def test__4x3_mask_input_radius_medium__mask(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(4, 3), pixel_scale=1.0,
                                                                    radius_arcsec=1.5001)

        assert (mask == np.array([[True, False, True],
                                  [False, False, False],
                                  [False, False, False],
                                  [True, False, True]])).all()

    def test__4x3_mask_input_radius_large__mask(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(4, 3), pixel_scale=1.0,
                                                                    radius_arcsec=3.0)

        assert (mask == np.array([[False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [False, False, False]])).all()

    def test__4x4_mask_input_radius_small__mask(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(4, 4), pixel_scale=1.0,
                                                                    radius_arcsec=0.72)

        assert (mask == np.array([[True, True, True, True],
                                  [True, False, False, True],
                                  [True, False, False, True],
                                  [True, True, True, True]])).all()

    def test__4x4_mask_input_radius_medium__mask(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(4, 4), pixel_scale=1.0,
                                                                    radius_arcsec=1.7)

        assert (mask == np.array([[True, False, False, True],
                                  [False, False, False, False],
                                  [False, False, False, False],
                                  [True, False, False, True]])).all()

    def test__4x4_mask_input_radius_large__mask(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(4, 4), pixel_scale=1.0,
                                                                    radius_arcsec=3.0)

        assert (mask == np.array([[False, False, False, False],
                                  [False, False, False, False],
                                  [False, False, False, False],
                                  [False, False, False, False]])).all()

    def test__centre_shift__simple_shift_downwards(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=3.0,
                                                                    radius_arcsec=0.5, centre=(-3, 0))

        assert mask.shape == (3, 3)
        assert (mask == np.array([[True, True, True],
                                  [True, True, True],
                                  [True, False, True]])).all()

    def test__centre_shift__simple_shift_right(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=3.0,
                                                                    radius_arcsec=0.5, centre=(0.0, 3.0))

        assert mask.shape == (3, 3)
        assert (mask == np.array([[True, True, True],
                                  [True, True, False],
                                  [True, True, True]])).all()

    def test__centre_shift__diagonal_shift(self):
        mask = util.mask_circular_from_shape_pixel_scale_and_radius(shape=(3, 3), pixel_scale=3.0,
                                                                    radius_arcsec=0.5, centre=(3, 3))

        assert (mask == np.array([[True, True, False],
                                  [True, True, True],
                                  [True, True, True]])).all()


class TestMaskAnnular(object):

    def test__3x3_mask_inner_radius_zero_outer_radius_small__mask(self):
        mask = util.mask_annular_from_shape_pixel_scale_and_radii(shape=(3, 3), pixel_scale=1.0,
                                                                  inner_radius_arcsec=0.0, outer_radius_arcsec=0.5)

        assert (mask == np.array([[True, True, True],
                                  [True, False, True],
                                  [True, True, True]])).all()

    def test__3x3_mask_inner_radius_small_outer_radius_large__mask(self):
        mask = util.mask_annular_from_shape_pixel_scale_and_radii(shape=(3, 3), pixel_scale=1.0,
                                                                  inner_radius_arcsec=0.5, outer_radius_arcsec=3.0)

        assert (mask == np.array([[False, False, False],
                                  [False, True, False],
                                  [False, False, False]])).all()

    def test__4x3_mask_inner_radius_small_outer_radius_medium__mask(self):
        mask = util.mask_annular_from_shape_pixel_scale_and_radii(shape=(4, 3), pixel_scale=1.0,
                                                                  inner_radius_arcsec=0.51, outer_radius_arcsec=1.51)

        assert (mask == np.array([[True, False, True],
                                  [False, True, False],
                                  [False, True, False],
                                  [True, False, True]])).all()

    def test__4x3_mask_inner_radius_medium_outer_radius_large__mask(self):
        mask = util.mask_annular_from_shape_pixel_scale_and_radii(shape=(4, 3), pixel_scale=1.0,
                                                                  inner_radius_arcsec=1.51, outer_radius_arcsec=3.0)

        assert (mask == np.array([[False, True, False],
                                  [True, True, True],
                                  [True, True, True],
                                  [False, True, False]])).all()

    def test__3x3_mask_inner_radius_small_outer_radius_medium__mask(self):
        mask = util.mask_annular_from_shape_pixel_scale_and_radii(shape=(4, 4), pixel_scale=1.0,
                                                                  inner_radius_arcsec=0.81, outer_radius_arcsec=2.0)

        assert (mask == np.array([[True, False, False, True],
                                  [False, True, True, False],
                                  [False, True, True, False],
                                  [True, False, False, True]])).all()

    def test__4x4_mask_inner_radius_medium_outer_radius_large__mask(self):
        mask = util.mask_annular_from_shape_pixel_scale_and_radii(shape=(4, 4), pixel_scale=1.0,
                                                                  inner_radius_arcsec=1.71, outer_radius_arcsec=3.0)

        assert (mask == np.array([[False, True, True, False],
                                  [True, True, True, True],
                                  [True, True, True, True],
                                  [False, True, True, False]])).all()

    def test__centre_shift__simple_shift_upwards(self):
        mask = util.mask_annular_from_shape_pixel_scale_and_radii(shape=(3, 3), pixel_scale=3.0,
                                                                  inner_radius_arcsec=0.5,
                                                                  outer_radius_arcsec=9.0, centre=(3.0, 0.0))

        assert mask.shape == (3, 3)
        assert (mask == np.array([[False, True, False],
                                  [False, False, False],
                                  [False, False, False]])).all()

    def test__centre_shift__simple_shift_forward(self):
        mask = util.mask_annular_from_shape_pixel_scale_and_radii(shape=(3, 3), pixel_scale=3.0,
                                                                  inner_radius_arcsec=0.5,
                                                                  outer_radius_arcsec=9.0, centre=(0.0, 3.0))

        assert mask.shape == (3, 3)
        assert (mask == np.array([[False, False, False],
                                  [False, False, True],
                                  [False, False, False]])).all()

    def test__centre_shift__diagonal_shift(self):
        mask = util.mask_annular_from_shape_pixel_scale_and_radii(shape=(3, 3), pixel_scale=3.0,
                                                                  inner_radius_arcsec=0.5,
                                                                  outer_radius_arcsec=9.0, centre=(-3.0, 3.0))

        assert mask.shape == (3, 3)
        assert (mask == np.array([[False, False, False],
                                  [False, False, False],
                                  [False, False, True]])).all()


class TestMaskAntiAnnular(object):

    def test__5x5_mask_inner_radius_includes_central_pixel__outer_extended_beyond_radius(self):

        mask = util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape=(5, 5), pixel_scale=1.0,
                                                                       inner_radius_arcsec=0.5, outer_radius_arcsec=10.0,
                                                                       outer_radius_2_arcsec=20.0)

        assert (mask == np.array([[True, True, True, True, True],
                                  [True, True, True, True, True],
                                  [True, True, False, True, True],
                                  [True, True, True, True, True],
                                  [True, True, True, True, True]])).all()

    def test__5x5_mask_inner_radius_includes_3x3_central_pixels__outer_extended_beyond_radius(self):

        mask = util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape=(5, 5), pixel_scale=1.0,
                                                                       inner_radius_arcsec=1.5, outer_radius_arcsec=10.0,
                                                                       outer_radius_2_arcsec=20.0)

        assert (mask == np.array([[True,  True,  True,  True, True],
                                  [True, False, False, False, True],
                                  [True, False, False, False, True],
                                  [True, False, False, False, True],
                                  [True,  True,  True,  True, True]])).all()

    def test__5x5_mask_inner_radius_includes_central_pixel__outer_radius_includes_outer_pixels(self):

        mask = util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape=(5, 5), pixel_scale=1.0,
                                                                       inner_radius_arcsec=0.5, outer_radius_arcsec=1.5,
                                                                       outer_radius_2_arcsec=20.0)

        assert (mask == np.array([[False, False, False, False, False],
                                  [False, True,  True,  True,  False],
                                  [False, True, False,  True,  False],
                                  [False, True,  True,  True,  False],
                                  [False, False, False, False, False]])).all()

    def test__7x7_second_outer_radius_mask_works_too(self):

        mask = util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape=(7, 7), pixel_scale=1.0,
                                                                       inner_radius_arcsec=0.5, outer_radius_arcsec=1.5,
                                                                       outer_radius_2_arcsec=2.9)

        assert (mask == np.array([[True,  True,  True,  True,  True,  True, True],
                                  [True, False, False, False, False, False, True],
                                  [True, False, True,   True,  True, False, True],
                                  [True, False, True,  False,  True, False, True],
                                  [True, False, True,   True,  True, False, True],
                                  [True, False, False, False, False, False, True],
                                  [True,  True,  True,  True,  True,  True, True]])).all()

    def test__centre_shift__diagonal_shift(self):

        mask = util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape=(7, 7), pixel_scale=3.0,
                                                                       inner_radius_arcsec=1.5, outer_radius_arcsec=4.5,
                                                                       outer_radius_2_arcsec=8.7, centre=(3.0, -3.0))

        assert (mask == np.array([[True,  True,  True,  True,  True,  True,  True],
                                  [True,  True,  True,  True,  True,  True,  True],
                                  [True,  True, False, False, False, False, False],
                                  [True,  True, False, True,   True,  True, False],
                                  [True,  True, False, True,  False,  True, False],
                                  [True,  True, False, True,   True,  True, False],
                                  [True,  True, False, False, False, False, False]])).all()


class TestMaskBlurring(object):

    def test__size__3x3_small_mask(self):
        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 3))

        assert (blurring_mask == np.array([[False, False, False],
                                           [False, True, False],
                                           [False, False, False]])).all()

    def test__size__3x3__large_mask(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 3))

        assert (blurring_mask == np.array([[True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True],
                                           [True, True, False, False, False, True, True],
                                           [True, True, False, True, False, True, True],
                                           [True, True, False, False, False, True, True],
                                           [True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True]])).all()

    def test__size__5x5__large_mask(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(5, 5))

        assert (blurring_mask == np.array([[True, True, True, True, True, True, True],
                                           [True, False, False, False, False, False, True],
                                           [True, False, False, False, False, False, True],
                                           [True, False, False, True, False, False, True],
                                           [True, False, False, False, False, False, True],
                                           [True, False, False, False, False, False, True],
                                           [True, True, True, True, True, True, True]])).all()

    def test__size__5x3__large_mask(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(5, 3))

        assert (blurring_mask == np.rot90(np.array([[True, True, True, True, True, True, True],
                                                    [True, True, True, True, True, True, True],
                                                    [True, False, False, False, False, False, True],
                                                    [True, False, False, True, False, False, True],
                                                    [True, False, False, False, False, False, True],
                                                    [True, True, True, True, True, True, True],
                                                    [True, True, True, True, True, True, True]]))).all()

    def test__size__3x5__large_mask(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 5))

        assert (blurring_mask == np.rot90(np.array([[True, True, True, True, True, True, True],
                                                    [True, True, False, False, False, True, True],
                                                    [True, True, False, False, False, True, True],
                                                    [True, True, False, True, False, True, True],
                                                    [True, True, False, False, False, True, True],
                                                    [True, True, False, False, False, True, True],
                                                    [True, True, True, True, True, True, True]]))).all()

    def test__size__3x3__multiple_points(self):
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True],
                         [True, True, True, True, True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 3))

        assert (blurring_mask == np.array([[False, False, False, True, False, False, False],
                                           [False, True, False, True, False, True, False],
                                           [False, False, False, True, False, False, False],
                                           [True, True, True, True, True, True, True],
                                           [False, False, False, True, False, False, False],
                                           [False, True, False, True, False, True, False],
                                           [False, False, False, True, False, False, False]])).all()

    def test__size__5x5__multiple_points(self):
        mask = np.array([[True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(5, 5))

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
        mask = np.array([[True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(5, 3))

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
        mask = np.array([[True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 5))

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
        mask = np.array([[True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 3))

        assert (blurring_mask == np.array([[False, False, False, True, False, False, False, True],
                                           [False, True, False, True, False, True, False, True],
                                           [False, False, False, True, False, False, False, True],
                                           [True, True, True, True, True, True, True, True],
                                           [False, False, False, True, False, False, False, True],
                                           [False, True, False, True, False, True, False, True],
                                           [False, False, False, True, False, False, False, True],
                                           [True, True, True, True, True, True, True, True]])).all()

    def test__size__5x5__even_sized_image(self):
        mask = np.array([[True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(5, 5))

        assert (blurring_mask == np.array([[True, True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True, True],
                                           [True, True, True, True, True, True, True, True],
                                           [True, True, True, False, False, False, False, False],
                                           [True, True, True, False, False, False, False, False],
                                           [True, True, True, False, False, True, False, False],
                                           [True, True, True, False, False, False, False, False],
                                           [True, True, True, False, False, False, False, False]])).all()

    def test__size__3x3__rectangular_8x9_image(self):
        mask = np.array([[True, True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True, True],
                         [True, True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 3))

        assert (blurring_mask == np.array([[False, False, False, True, False, False, False, True, True],
                                           [False, True, False, True, False, True, False, True, True],
                                           [False, False, False, True, False, False, False, True, True],
                                           [True, True, True, True, True, True, True, True, True],
                                           [False, False, False, True, False, False, False, True, True],
                                           [False, True, False, True, False, True, False, True, True],
                                           [False, False, False, True, False, False, False, True, True],
                                           [True, True, True, True, True, True, True, True, True]])).all()

    def test__size__3x3__rectangular_9x8_image(self):
        mask = np.array([[True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True, True]])

        blurring_mask = util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(3, 3))

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
        mask = np.array([[True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, True, True, True, True, True, True],
                         [True, False, True, True, True, False, True],
                         [True, True, True, True, True, True, True]])

        with pytest.raises(exc.MaskException):
            util.mask_blurring_from_mask_and_psf_shape(mask, psf_shape=(5, 5))


class TestMap2DArrayTo1d(object):

    def test__setup_3x3_data(self):
        array_2d = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

        mask = np.array([[True, True, True],
                         [True, False, True],
                         [True, True, True]])

        array_1d = util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(mask, array_2d)

        assert (array_1d == np.array([5])).all()

    def test__setup_3x3_array__five_now_in_mask(self):
        array_2d = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True]])

        array_1d = util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(mask, array_2d)

        assert (array_1d == np.array([2, 4, 5, 6, 8])).all()

    def test__setup_3x4_array(self):
        array_2d = np.array([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12]])

        mask = np.array([[True, False, True, True],
                         [False, False, False, True],
                         [True, False, True, False]])

        array_1d = util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(mask, array_2d)

        assert (array_1d == np.array([2, 5, 6, 7, 10, 12])).all()

    def test__setup_4x3_array__five_now_in_mask(self):
        array_2d = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9],
                             [10, 11, 12]])

        mask = np.array([[True, False, True],
                         [False, False, False],
                         [True, False, True],
                         [True, True, True]])

        array_1d = util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(mask, array_2d)

        assert (array_1d == np.array([2, 4, 5, 6, 8])).all()


class TestMapMasked1DArrayTo2d(object):

    def test__2d_array_is_2x2__is_not_masked__maps_correctly(self):
        array_1d = np.array([1.0, 2.0, 3.0, 4.0])

        one_to_two = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        shape = (2, 2)

        array_2d = util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(array_1d, shape, one_to_two)

        assert (array_2d == np.array([[1.0, 2.0],
                                      [3.0, 4.0]])).all()

    def test__2d_array_is_2x2__is_masked__maps_correctly(self):
        array_1d = np.array([1.0, 2.0, 3.0])

        one_to_two = np.array([[0, 0], [0, 1], [1, 0]])
        shape = (2, 2)

        array_2d = util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(array_1d, shape, one_to_two)

        assert (array_2d == np.array([[1.0, 2.0],
                                      [3.0, 0.0]])).all()

    def test__different_shape_and_mappings(self):
        array_1d = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])

        one_to_two = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [2, 3]])
        shape = (3, 4)

        array_2d = util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(array_1d, shape, one_to_two)

        assert (array_2d == np.array([[1.0, 2.0, 0.0, 0.0],
                                      [3.0, 0.0, 0.0, 0.0],
                                      [-1.0, -2.0, 0.0, -3.0]])).all()


class TestMapUnmasked1dArrayTo2d(object):

    def test__1d_array_in__maps_it_to_4x4_2d_array(self):
        array_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        array_2d = util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d, shape=(4, 4))

        assert (array_2d == np.array([[1.0, 2.0, 3.0, 4.0],
                                      [5.0, 6.0, 7.0, 8.0],
                                      [9.0, 10.0, 11.0, 12.0],
                                      [13.0, 14.0, 15.0, 16.0]])).all()

    def test__1d_array_in__can_map_it_to_2x3_2d_array(self):
        array_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        array_2d = util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d, shape=(2, 3))

        assert (array_2d == np.array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]])).all()

    def test__1d_array_in__can_map_it_to_3x2_2d_array(self):
        array_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        array_2d = util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d, shape=(3, 2))

        assert (array_2d == np.array([[1.0, 2.0],
                                      [3.0, 4.0],
                                      [5.0, 6.0]])).all()


class TestResize:

    def test__trim__from_7x7_to_3x3(self):
        array = np.ones((7, 7))
        array[3, 3] = 2.0

        modified = util.resize_array_2d(array_2d=array, new_shape=(3, 3))

        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

    def test__trim__from_7x7_to_4x4(self):
        array = np.ones((7, 7))
        array[3, 3] = 2.0

        modified = util.resize_array_2d(array_2d=array, new_shape=(4, 4))


        assert (modified == np.array([[1.0, 1.0, 1.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0],
                                      [1.0, 1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0]])).all()

    def test__trim__from_6x6_to_4x4(self):

        array = np.ones((6, 6))
        array[2:4, 2:4] = 2.0

        modified = util.resize_array_2d(array_2d=array, new_shape=(4, 4))

        assert (modified == np.array([[1.0, 1.0, 1.0, 1.0],
                                      [1.0, 2.0, 2.0, 1.0],
                                      [1.0, 2.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0]])).all()

    def test__trim__from_6x6_to_3x3(self):

        array = np.ones((6, 6))
        array[2:4, 2:4] = 2.0

        modified = util.resize_array_2d(array_2d=array, new_shape=(3, 3))

        assert (modified == np.array([[2.0, 2.0, 1.0],
                                      [2.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

    def test__trim__from_5x4_to_3x2(self):
        array = np.ones((5, 4))
        array[2, 1:3] = 2.0

        modified = util.resize_array_2d(array_2d=array, new_shape=(3, 2))

        assert (modified == np.array([[1.0, 1.0],
                                      [2.0, 2.0],
                                      [1.0, 1.0]])).all()

    def test__trim__from_4x5_to_2x3(self):
        array = np.ones((4, 5))
        array[1:3, 2] = 2.0

        modified = util.resize_array_2d(array_2d=array, new_shape=(2, 3))

        assert (modified == np.array([[1.0, 2.0, 1.0],
                                      [1.0, 2.0, 1.0]])).all()

    def test__trim_with_new_centre_as_input(self):

        array = np.ones((7, 7))
        array[4, 4] = 2.0
        modified = util.resize_array_2d(array_2d=array, new_shape=(3, 3), new_centre=(4,4))
        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

        array = np.ones((6, 6))
        array[3, 4] = 2.0
        modified = util.resize_array_2d(array_2d=array, new_shape=(3, 3), new_centre=(3,4))
        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

        array = np.ones((9, 8))
        array[4, 3] = 2.0
        modified = util.resize_array_2d(array_2d=array, new_shape=(3, 3), new_centre=(4,3))
        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

        array = np.ones((8, 9))
        array[3, 5] = 2.0
        modified = util.resize_array_2d(array_2d=array, new_shape=(3, 3), new_centre=(3,5))
        assert (modified == np.array([[1.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 1.0]])).all()

    def test__pad__from_3x3_to_5x5(self):

        array = np.ones((3, 3))
        array[1, 1] = 2.0

        modified = util.resize_array_2d(array_2d=array, new_shape=(5, 5))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 1.0, 2.0, 1.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

    def test__pad__from_3x3_to_4x4(self):

        array = np.ones((3, 3))
        array[1, 1] = 2.0

        modified = util.resize_array_2d(array_2d=array, new_shape=(4, 4))


        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0],
                                      [0.0, 1.0, 2.0, 1.0],
                                      [0.0, 1.0, 1.0, 1.0]])).all()

    def test__pad__from_4x4_to_6x6(self):

        array = np.ones((4, 4))
        array[1:3, 1:3] = 2.0

        modified = util.resize_array_2d(array_2d=array, new_shape=(6, 6))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                      [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],])).all()

    def test__pad__from_4x4_to_5x5(self):

        array = np.ones((4, 4))
        array[1:3, 1:3] = 2.0

        modified = util.resize_array_2d(array_2d=array, new_shape=(5, 5))

        assert (modified == np.array([[1.0, 1.0, 1.0, 1.0, 0.0],
                                      [1.0, 2.0, 2.0, 1.0, 0.0],
                                      [1.0, 2.0, 2.0, 1.0, 0.0],
                                      [1.0, 1.0, 1.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

    def test__pad__from_3x2_to_5x4(self):
        array = np.ones((3, 2))
        array[1, 0:2] = 2.0

        modified = util.resize_array_2d(array_2d=array, new_shape=(5, 4))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 0.0],
                                      [0.0, 2.0, 2.0, 0.0],
                                      [0.0, 1.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0]])).all()

    def test__pad__from_2x3_to_4x5(self):
        array = np.ones((2, 3))
        array[0:2, 1] = 2.0

        modified = util.resize_array_2d(array_2d=array, new_shape=(4, 5))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 2.0, 1.0, 0.0],
                                      [0.0, 1.0, 2.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0],])).all()

    def test__pad__with_input_new_centre(self):

        array = np.ones((3, 3))
        array[2, 2] = 2.0
        modified = util.resize_array_2d(array_2d=array, new_shape=(5, 5), new_centre=(2, 2))

        assert (modified == np.array([[1.0, 1.0, 1.0, 0.0, 0.0],
                                      [1.0, 1.0, 1.0, 0.0, 0.0],
                                      [1.0, 1.0, 2.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        array = np.ones((2, 3))
        array[0, 0] = 2.0
        modified = util.resize_array_2d(array_2d=array, new_shape=(4, 5), new_centre=(0, 1))

        assert (modified == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 2.0, 1.0, 1.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 0.0]])).all()


class TestFits:

    def test__numpy_array_from_fits__3x3_all_ones(self):
        arr = util.numpy_array_from_fits(path=test_data_dir + '3x3_ones.fits', hdu=0)

        assert (arr == np.ones((3, 3))).all()

    def test__numpy_array_from_fits__4x3_all_ones(self):
        arr = util.numpy_array_from_fits(path=test_data_dir + '4x3_ones.fits', hdu=0)

        assert (arr == np.ones((4, 3))).all()

    def test__numpy_array_to_fits__output_and_load(self):
        if os.path.exists(test_data_dir + 'test.fits'):
            os.remove(test_data_dir + 'test.fits')

        arr = np.array([[10., 30., 40.],
                        [92., 19., 20.]])

        util.numpy_array_to_fits(arr, path=test_data_dir + 'test.fits')

        array_load = util.numpy_array_from_fits(path=test_data_dir + 'test.fits', hdu=0)

        assert (arr == array_load).all()


class TestVariancesFromNoise:

    def test__noise_all_1s__variances_all_1s(self):
        noise = np.array([[1.0, 1.0],
                          [1.0, 1.0]])

        assert (util.compute_variances_from_noise(noise) == np.array([[1.0, 1.0],
                                                                      [1.0, 1.0]])).all()

    def test__noise_all_2s__variances_all_4s(self):
        noise = np.array([[2.0, 2.0],
                          [2.0, 2.0]])

        assert (util.compute_variances_from_noise(noise) == np.array([[4.0, 4.0],
                                                                      [4.0, 4.0]])).all()

    def test__noise_all_05s__variances_all_025s(self):
        noise = np.array([[0.5, 0.5],
                          [0.5, 0.5]])

        assert (util.compute_variances_from_noise(noise) == np.array([[0.25, 0.25],
                                                                      [0.25, 0.25]])).all()
