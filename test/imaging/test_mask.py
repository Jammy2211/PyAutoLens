import numpy as np
from src.imaging import mask
import pytest
from src import exc


@pytest.fixture(name="msk")
def make_mask():
    return mask.Mask(np.array([[True, False, True],
                               [False, False, False],
                               [True, False, True]]))


@pytest.fixture(name="centre_mask")
def make_centre_mask():
    return mask.Mask(np.array([[True, True, True],
                               [True, False, True],
                               [True, True, True]]))


@pytest.fixture(name="sub_coordinate_grid")
def make_sub_coordinate_grid(msk):
    return mask.SubCoordinateGrid.from_mask(msk)


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

        def test__3x3_mask_input_radius_small__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(3, 3), pixel_scale=1, radius_mask=0.5)
            assert (msk == np.array([[True, True, True],
                                     [True, False, True],
                                     [True, True, True]])).all()

        def test__3x3_mask_input_radius_medium__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(3, 3), pixel_scale=1, radius_mask=1)

            assert (msk == np.array([[True, False, True],
                                     [False, False, False],
                                     [True, False, True]])).all()

        def test__3x3_mask_input_radius_large__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(3, 3), pixel_scale=1, radius_mask=3)

            assert (msk == np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False]])).all()

        def test__4x3_mask_input_radius_small__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(4, 3), pixel_scale=1, radius_mask=0.5)

            assert (msk == np.array([[True, True, True],
                                     [True, False, True],
                                     [True, False, True],
                                     [True, True, True]])).all()

        def test__4x3_mask_input_radius_medium__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(4, 3), pixel_scale=1, radius_mask=1.50001)

            assert (msk == np.array([[True, False, True],
                                     [False, False, False],
                                     [False, False, False],
                                     [True, False, True]])).all()

        def test__4x3_mask_input_radius_large__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(4, 3), pixel_scale=1, radius_mask=3)

            assert (msk == np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False],
                                     [False, False, False]])).all()

        def test__4x4_mask_input_radius_small__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(4, 4), pixel_scale=1, radius_mask=0.72)

            assert (msk == np.array([[True, True, True, True],
                                     [True, False, False, True],
                                     [True, False, False, True],
                                     [True, True, True, True]])).all()

        def test__4x4_mask_input_radius_medium__mask(self):
            msk = mask.Mask.circular(shape_arc_seconds=(4, 4), pixel_scale=1, radius_mask=1.7)

            assert (msk == np.array([[True, False, False, True],
                                     [False, False, False, False],
                                     [False, False, False, False],
                                     [True, False, False, True]])).all()

        def test__4x4_mask_input_radius_large__mask(self):
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

        def test__3x3_mask_inner_radius_zero_outer_radius_small__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(3, 3), pixel_scale=1, inner_radius_mask=0,
                                    outer_radius_mask=0.5)

            assert (msk == np.array([[True, True, True],
                                     [True, False, True],
                                     [True, True, True]])).all()

        def test__3x3_mask_inner_radius_small_outer_radius_large__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(3, 3), pixel_scale=1, inner_radius_mask=0.5,
                                    outer_radius_mask=3)

            assert (msk == np.array([[False, False, False],
                                     [False, True, False],
                                     [False, False, False]])).all()

        def test__4x4_mask_inner_radius_small_outer_radius_medium__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(4, 3), pixel_scale=1, inner_radius_mask=0.51,
                                    outer_radius_mask=1.51)

            assert (msk == np.array([[True, False, True],
                                     [False, True, False],
                                     [False, True, False],
                                     [True, False, True]])).all()

        def test__4x3_mask_inner_radius_medium_outer_radius_large__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(4, 3), pixel_scale=1, inner_radius_mask=1.51,
                                    outer_radius_mask=3)

            assert (msk == np.array([[False, True, False],
                                     [True, True, True],
                                     [True, True, True],
                                     [False, True, False]])).all()

        def test__3x3_mask_inner_radius_small_outer_radius_medium__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(4, 4), pixel_scale=1, inner_radius_mask=0.81,
                                    outer_radius_mask=2)

            assert (msk == np.array([[True, False, False, True],
                                     [False, True, True, False],
                                     [False, True, True, False],
                                     [True, False, False, True]])).all()

        def test__4x4_mask_inner_radius_medium_outer_radius_large__mask(self):
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

    class TestForSimulate(object):

        def test__3x3_image__3x3_psf_size__5x5_image_made_border_all_masked(self):
            msk = mask.Mask.for_simulate(shape_arc_seconds=(3, 3), pixel_scale=1, psf_size=(3, 3))

            assert (msk == np.array([[True, True, True, True, True],
                                     [True, False, False, False, True],
                                     [True, False, False, False, True],
                                     [True, False, False, False, True],
                                     [True, True, True, True, True]])).all()

        def test__3x3_image__5x5_psf_size__7x7_image_made_border_all_masked(self):
            msk = mask.Mask.for_simulate(shape_arc_seconds=(3, 3), pixel_scale=1, psf_size=(5, 5))

            assert (msk == np.array([[True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True],
                                     [True, True, False, False, False, True, True],
                                     [True, True, False, False, False, True, True],
                                     [True, True, False, False, False, True, True],
                                     [True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True]])).all()

        def test__4x3_image__3x3_psf_size__6x5_image_made_border_all_masked(self):
            msk = mask.Mask.for_simulate(shape_arc_seconds=(4, 3), pixel_scale=1, psf_size=(3, 3))

            assert (msk == np.array([[True, True, True, True, True],
                                     [True, False, False, False, True],
                                     [True, False, False, False, True],
                                     [True, False, False, False, True],
                                     [True, False, False, False, True],
                                     [True, True, True, True, True]])).all()

        def test__4x3_image__5x5_psf_size__8x7_image_made_border_all_masked(self):
            msk = mask.Mask.for_simulate(shape_arc_seconds=(4, 3), pixel_scale=1, psf_size=(5, 5))

            assert (msk == np.array([[True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True],
                                     [True, True, False, False, False, True, True],
                                     [True, True, False, False, False, True, True],
                                     [True, True, False, False, False, True, True],
                                     [True, True, False, False, False, True, True],
                                     [True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True]])).all()

        def test__3x4_image__3x3_psf_size__5x6_image_made_border_all_masked(self):
            msk = mask.Mask.for_simulate(shape_arc_seconds=(3, 4), pixel_scale=1, psf_size=(3, 3))

            assert (msk == np.array([[True, True, True, True, True, True],
                                     [True, False, False, False, False, True],
                                     [True, False, False, False, False, True],
                                     [True, False, False, False, False, True],
                                     [True, True, True, True, True, True]])).all()

        def test__3x4_image__5x5_psf_size__7x8_image_made_border_all_masked(self):
            msk = mask.Mask.for_simulate(shape_arc_seconds=(3, 4), pixel_scale=1, psf_size=(5, 5))

            assert (msk == np.array([[True, True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True, True],
                                     [True, True, False, False, False, False, True, True],
                                     [True, True, False, False, False, False, True, True],
                                     [True, True, False, False, False, False, True, True],
                                     [True, True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True, True]])).all()

        def test__4x4_image__3x3_psf_size__6x6_image_made_border_all_masked(self):
            msk = mask.Mask.for_simulate(shape_arc_seconds=(4, 4), pixel_scale=1, psf_size=(3, 3))

            assert (msk == np.array([[True, True, True, True, True, True],
                                     [True, False, False, False, False, True],
                                     [True, False, False, False, False, True],
                                     [True, False, False, False, False, True],
                                     [True, False, False, False, False, True],
                                     [True, True, True, True, True, True]])).all()

        def test__4x4_image__5x5_psf_size__8x8_image_made_border_all_masked(self):
            msk = mask.Mask.for_simulate(shape_arc_seconds=(4, 4), pixel_scale=1, psf_size=(5, 5))

            assert (msk == np.array([[True, True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True, True],
                                     [True, True, False, False, False, False, True, True],
                                     [True, True, False, False, False, False, True, True],
                                     [True, True, False, False, False, False, True, True],
                                     [True, True, False, False, False, False, True, True],
                                     [True, True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True, True]])).all()

        def test__non_square_psf_shape__raises_error(self):
            with pytest.raises(exc.KernelException):
                mask.Mask.for_simulate(shape_arc_seconds=(4, 4), pixel_scale=1, psf_size=(3, 5))

    class TestComputeGridCoordsImage(object):

        def test__setup_3x3_image_one_coordinate(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_grid = msk.coordinate_grid

            assert (image_grid[0] == np.array([0.0, 0.0])).all()

        def test__setup_3x3_image__five_coordinates(self):
            msk = np.array([[True, False, True],
                            [False, False, False],
                            [True, False, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_grid = msk.coordinate_grid

            assert (image_grid == np.array([[-3., 0.], [0., -3.], [0., 0.], [0., 3.], [3., 0.]])).all()

        def test__setup_4x4_image__ten_coordinates__new_pixel_scale(self):
            msk = np.array([[True, False, False, True],
                            [False, False, False, True],
                            [True, False, False, True],
                            [False, False, False, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            image_grid = msk.coordinate_grid

            assert (image_grid == np.array(
                [[-1.5, -0.5], [-1.5, 0.5], [-0.5, -1.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5],
                 [1.5, -1.5], [1.5, -0.5], [1.5, 0.5]])).all()

        def test__setup_3x4_image__six_coordinates(self):
            msk = np.array([[True, False, True, True],
                            [False, False, False, True],
                            [True, False, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_grid = msk.coordinate_grid

            assert (image_grid == np.array(
                [[-3., -1.5], [0., -4.5], [0., -1.5], [0., 1.5], [3., -1.5], [3., 4.5]])).all()

    class TestComputeGridCoordsImageSub(object):

        def test__3x3_mask_with_one_pixel__2x2_sub_grid__coordinates(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = mask.SubCoordinateGrid.from_mask(msk, 2)

            assert (image_sub_grid == np.array([[[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]])).all()

        def test__3x3_mask_with_row_of_pixels__2x2_sub_grid__coordinates(self):
            msk = np.array([[True, True, True],
                            [False, False, False],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = mask.SubCoordinateGrid.from_mask(msk, 2)

            assert (image_sub_grid == np.array([[-0.5, -3.5], [-0.5, -2.5], [0.5, -3.5], [0.5, -2.5],
                                                [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5],
                                                [-0.5, 2.5], [-0.5, 3.5], [0.5, 2.5], [0.5, 3.5]])).all()

        def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__coordinates(self):
            msk = np.array([[True, True, False],
                            [False, False, False],
                            [True, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = mask.SubCoordinateGrid.from_mask(msk, 2)

            assert (image_sub_grid == np.array([[-3.5, 2.5], [-3.5, 3.5], [-2.5, 2.5], [-2.5, 3.5],
                                                [-0.5, -3.5], [-0.5, -2.5], [0.5, -3.5], [0.5, -2.5],
                                                [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5],
                                                [-0.5, 2.5], [-0.5, 3.5], [0.5, 2.5], [0.5, 3.5],
                                                [2.5, 2.5], [2.5, 3.5], [3.5, 2.5], [3.5, 3.5]])).all()

        def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__different_pixel_scale(self):
            msk = np.array([[True, True, False],
                            [False, False, False],
                            [True, True, False]])

            msk = mask.Mask(msk, pixel_scale=0.3)

            image_sub_grid = mask.SubCoordinateGrid.from_mask(msk, 2)

            image_sub_grid = np.round(image_sub_grid, decimals=2)

            np.testing.assert_almost_equal(image_sub_grid,
                                           np.array([[-0.35, 0.25], [-0.35, 0.35], [-0.25, 0.25], [-0.25, 0.35],
                                                     [-0.05, -0.35], [-0.05, -0.25], [0.05, -0.35], [0.05, -0.25],
                                                     [-0.05, -0.05], [-0.05, 0.05], [0.05, -0.05], [0.05, 0.05],
                                                     [-0.05, 0.25], [-0.05, 0.35], [0.05, 0.25], [0.05, 0.35],
                                                     [0.25, 0.25], [0.25, 0.35], [0.35, 0.25], [0.35, 0.35]]))

        def test__3x3_mask_with_one_pixel__3x3_sub_grid__coordinates(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = mask.SubCoordinateGrid.from_mask(msk, 3)

            assert (image_sub_grid == np.array([[[-0.75, -0.75], [-0.75, 0.], [-0.75, 0.75], [0., -0.75], [0., 0.],
                                                 [0., 0.75], [0.75, -0.75], [0.75, 0.], [0.75, 0.75]]])).all()

        def test__3x3_mask_with_one_row__3x3_sub_grid__coordinates(self):
            msk = np.array([[True, True, False],
                            [True, False, True],
                            [True, True, False]])

            msk = mask.Mask(msk, pixel_scale=2.0)

            image_sub_grid = mask.SubCoordinateGrid.from_mask(msk, 3)

            assert (image_sub_grid == np.array([[-2.5, 1.5], [-2.5, 2.], [-2.5, 2.5], [-2., 1.5], [-2., 2.],
                                                [-2., 2.5], [-1.5, 1.5], [-1.5, 2.], [-1.5, 2.5],
                                                [-0.5, -0.5], [-0.5, 0.], [-0.5, 0.5], [0., -0.5], [0., 0.], [0., 0.5],
                                                [0.5, -0.5], [0.5, 0.], [0.5, 0.5],
                                                [1.5, 1.5], [1.5, 2.], [1.5, 2.5], [2., 1.5], [2., 2.], [2., 2.5],
                                                [2.5, 1.5], [2.5, 2.], [2.5, 2.5]])).all()

        def test__4x4_mask_with_one_pixel__4x4_sub_grid__coordinates(self):
            msk = np.array([[True, True, True, True],
                            [True, False, False, True],
                            [True, False, False, True],
                            [True, True, True, False]])

            msk = mask.Mask(msk, pixel_scale=2.0)

            image_sub_grid = mask.SubCoordinateGrid.from_mask(msk, 4)

            image_sub_grid = np.round(image_sub_grid, decimals=1)

            assert (image_sub_grid == np.array([[-1.6, -1.6], [-1.6, -1.2], [-1.6, -0.8], [-1.6, -0.4], [-1.2, -1.6],
                                                [-1.2, -1.2], [-1.2, -0.8], [-1.2, -0.4], [-0.8, -1.6], [-0.8, -1.2],
                                                [-0.8, -0.8], [-0.8, -0.4], [-0.4, -1.6], [-0.4, -1.2], [-0.4, -0.8],
                                                [-0.4, -0.4],
                                                [-1.6, 0.4], [-1.6, 0.8], [-1.6, 1.2], [-1.6, 1.6], [-1.2, 0.4],
                                                [-1.2, 0.8], [-1.2, 1.2], [-1.2, 1.6], [-0.8, 0.4], [-0.8, 0.8],
                                                [-0.8, 1.2], [-0.8, 1.6], [-0.4, 0.4], [-0.4, 0.8], [-0.4, 1.2],
                                                [-0.4, 1.6],
                                                [0.4, -1.6], [0.4, -1.2], [0.4, -0.8], [0.4, -0.4], [0.8, -1.6],
                                                [0.8, -1.2], [0.8, -0.8], [0.8, -0.4], [1.2, -1.6], [1.2, -1.2],
                                                [1.2, -0.8], [1.2, -0.4], [1.6, -1.6], [1.6, -1.2], [1.6, -0.8],
                                                [1.6, -0.4],
                                                [0.4, 0.4], [0.4, 0.8], [0.4, 1.2], [0.4, 1.6], [0.8, 0.4], [0.8, 0.8],
                                                [0.8, 1.2], [0.8, 1.6], [1.2, 0.4], [1.2, 0.8], [1.2, 1.2], [1.2, 1.6],
                                                [1.6, 0.4], [1.6, 0.8], [1.6, 1.2], [1.6, 1.6],
                                                [2.4, 2.4], [2.4, 2.8], [2.4, 3.2], [2.4, 3.6], [2.8, 2.4], [2.8, 2.8],
                                                [2.8, 3.2], [2.8, 3.6], [3.2, 2.4], [3.2, 2.8], [3.2, 3.2], [3.2, 3.6],
                                                [3.6, 2.4], [3.6, 2.8], [3.6, 3.2], [3.6, 3.6]])).all()

        def test__4x3_mask_with_one_pixel__2x2_sub_grid__coordinates(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, False, False],
                            [False, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = mask.SubCoordinateGrid.from_mask(msk, 2)

            assert (image_sub_grid == np.array(
                [[-2., -0.5], [-2., 0.5], [-1., -0.5], [-1., 0.5], [1., -0.5], [1., 0.5], [2., -0.5], [2., 0.5],
                 [1., 2.5], [1., 3.5], [2., 2.5], [2., 3.5], [4., -3.5], [4., -2.5], [5., -3.5], [5., -2.5]])).all()

        def test__3x4_mask_with_one_pixel__2x2_sub_grid__coordinates(self):
            msk = np.array([[True, True, True, False],
                            [True, False, False, True],
                            [False, True, False, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = mask.SubCoordinateGrid.from_mask(msk, 2)

            assert (image_sub_grid == np.array(
                [[-3.5, 4.], [-3.5, 5.], [-2.5, 4.], [-2.5, 5.], [-0.5, -2.], [-0.5, -1.], [0.5, -2.], [0.5, -1.],
                 [-0.5, 1.], [-0.5, 2.], [0.5, 1.], [0.5, 2.], [2.5, -5.], [2.5, -4.], [3.5, -5.], [3.5, -4.],
                 [2.5, 1.], [2.5, 2.], [3.5, 1.], [3.5, 2.]])).all()

    class TestComputeGridCoordsBlurring(object):

        def test__3x3_blurring_mask_correct_coordinates(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_grid = msk.blurring_mask_for_kernel_shape(kernel_shape=(3, 3)).coordinate_grid

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

            blurring_grid = msk.blurring_mask_for_kernel_shape(kernel_shape=(3, 5)).coordinate_grid

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

            blurring_grid = msk.blurring_mask_for_kernel_shape(kernel_shape=(5, 3)).coordinate_grid

            assert (blurring_grid == np.array(
                [[-6., -3.], [-6., 0.], [-6., 3.], [-3., -3.], [-3., 0.], [-3., 3.], [0., -3.], [0., 3.], [3., -3.],
                 [3., 0.], [3., 3.], [6., -3.], [6., 0.], [6., 3.]])).all()

    class TestComputeGridSubtoImage(object):

        def test__3x3_mask_with_1_pixel__2x2_sub_grid__correct_sub_to_image(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sub_to_image = mask.SubCoordinateGrid.from_mask(msk, 2).sub_to_image

            assert (sub_to_image == np.array([0, 0, 0, 0])).all()

        def test__3x3_mask_with_row_of_pixels_pixel__2x2_sub_grid__correct_sub_to_image(self):
            msk = np.array([[True, True, True],
                            [False, False, False],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sub_to_image = mask.SubCoordinateGrid.from_mask(msk, 2).sub_to_image

            assert (sub_to_image == np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])).all()

        def test__3x3_mask_with_row_of_pixels_pixel__3x3_sub_grid__correct_sub_to_image(self):
            msk = np.array([[True, True, True],
                            [False, False, False],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sub_to_image = mask.SubCoordinateGrid.from_mask(msk, 3).sub_to_image

            assert (sub_to_image == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              1, 1, 1, 1, 1, 1, 1, 1, 1,
                                              2, 2, 2, 2, 2, 2, 2, 2, 2])).all()

    class TestComputeGridData(object):

        def test__setup_3x3_data(self):
            data = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_data = msk.masked_1d_array_from_2d_array(data)

            assert (grid_data[0] == np.array([5])).all()

        def test__setup_3x3_data__five_now_in_mask(self):
            data = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

            msk = np.array([[True, False, True],
                            [False, False, False],
                            [True, False, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_data = msk.masked_1d_array_from_2d_array(data)

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

            grid_data = msk.masked_1d_array_from_2d_array(data)

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

            grid_data = msk.masked_1d_array_from_2d_array(data)

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(3, 3))

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(3, 3))

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(5, 5))

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(5, 3))

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(3, 5))

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(3, 3))

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(5, 5))

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(5, 3))

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(3, 5))

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(3, 3))

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(5, 5))

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(3, 3))

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

            blurring_mask = msk.blurring_mask_for_kernel_shape(kernel_shape=(3, 3))

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
                msk.blurring_mask_for_kernel_shape(kernel_shape=(5, 5))

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

            border_pixels = msk.border_pixel_indices

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

            border_pixels = msk.border_pixel_indices

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

            border_pixels = msk.border_pixel_indices

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

            border_pixels = msk.border_pixel_indices

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

            border_pixels = msk.border_pixel_indices

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

            border_pixels = msk.border_pixel_indices

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

            border_pixels = msk.border_pixel_indices

            assert (border_pixels == np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24])).all()

    class TestComputeGridMapperTo2D(object):

        def test__setup_3x3_image_one_pixel(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            mapper_data_to_2d = msk.grid_to_pixel()

            assert (mapper_data_to_2d[0] == np.array([1, 1])).all()

        def test__setup_3x3_image__five_pixels(self):
            msk = np.array([[True, False, True],
                            [False, False, False],
                            [True, False, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            mapper_data_to_2d = msk.grid_to_pixel()

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

            mapper_data_to_2d = msk.grid_to_pixel()

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

            mapper_data_to_2d = msk.grid_to_pixel()

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

            grid_cluster_pixelization = mask.SparseMask(msk, 1)

            assert (grid_cluster_pixelization.sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__7x7_circle_mask__sparse_grid_size_1(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_cluster_pixelization = mask.SparseMask(msk, 1)

            assert (grid_cluster_pixelization.sparse_to_image == np.arange(21)).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.arange(21)).all()

        def test__7x7_rectangle_mask__sparse_grid_size_1(self):
            msk = np.array([[False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False],
                            [False, False, False, False, False, False, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_cluster_pixelization = mask.SparseMask(msk, 1)

            assert (grid_cluster_pixelization.sparse_to_image == np.arange(49)).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.arange(49)).all()

        def test__7x7_circle_mask__sparse_grid_size_2(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_cluster_pixelization = mask.SparseMask(msk, 2)

            assert (grid_cluster_pixelization.sparse_to_image == np.array([4, 6, 14, 16])).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
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

            grid_cluster_pixelization = mask.SparseMask(msk, 2)

            assert (grid_cluster_pixelization.sparse_to_image == np.array([0, 8, 10, 12, 22, 24, 26, 33])).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.array([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 2, 2, 3, 3,
                                                                           1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 4,
                                                                           4, 5, 6, 6,
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

            grid_cluster_pixelization = mask.SparseMask(msk, 2)

            assert (grid_cluster_pixelization.sparse_to_image == np.arange(16)).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.arange(16)).all()

        def test__7x7_circle_mask__sparse_grid_size_3(self):
            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_cluster_pixelization = mask.SparseMask(msk, 3)

            assert (grid_cluster_pixelization.sparse_to_image == np.array([10])).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all()

        def test__7x7_circle_mask_more_points_added__sparse_grid_size_3(self):
            msk = np.array([[False, True, True, False, True, False, False],
                            [True, True, False, False, False, True, True],
                            [True, False, False, False, False, False, True],
                            [True, False, False, False, False, False, False],
                            [True, False, False, False, False, False, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_cluster_pixelization = mask.SparseMask(msk, 3)

            assert (grid_cluster_pixelization.sparse_to_image == np.array([0, 1, 3, 14, 17, 26])).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.array(
                [0, 1, 2, 2, 1, 1, 1, 0, 3, 3, 3, 4, 3, 3, 3, 3, 4, 4, 3, 3, 3,
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

            grid_cluster_pixelization = mask.SparseMask(msk, 3)

            assert (grid_cluster_pixelization.sparse_to_image == np.arange(9)).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.arange(9)).all()

        def test__8x8_mask_trues_on_values_which_divide_by_3_and_other_values__sparse_grid_size_3(self):
            msk = np.array([[False, True, False, False, True, True, False],
                            [True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [False, True, True, False, True, True, False],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [False, False, False, False, False, False, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_cluster_pixelization = mask.SparseMask(msk, 3)

            assert (grid_cluster_pixelization.sparse_to_image == np.array([0, 2, 3, 7, 8, 9, 10, 13, 16])).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.array(
                [0, 1, 1, 2, 4, 4, 4, 3, 4, 5, 6, 6, 7, 7, 7, 8, 8])).all()

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

            grid_cluster_pixelization = mask.SparseMask(msk, 1)

            assert (grid_cluster_pixelization.sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

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

            grid_cluster_pixelization = mask.SparseMask(msk, 1)

            assert (grid_cluster_pixelization.sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

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

            grid_cluster_pixelization = mask.SparseMask(msk, 2)

            assert (grid_cluster_pixelization.sparse_to_image == np.array([1, 3])).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])).all()

        def test__7x8__five_central_pixels__sparse_grid_size_1(self):
            msk = np.array([[True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_cluster_pixelization = mask.SparseMask(msk, 1)

            assert (grid_cluster_pixelization.sparse_to_image == np.array([0, 1, 2, 3, 4])).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.array([0, 1, 2, 3, 4])).all()

        def test__7x8__five_central_pixels__sparse_grid_size_2(self):
            msk = np.array([[True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_cluster_pixelization = mask.SparseMask(msk, 2)

            assert (grid_cluster_pixelization.sparse_to_image == np.array([1, 3])).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])).all()

        def test__7x8__more_central_pixels__sparse_grid_size_2(self):
            msk = np.array([[True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_cluster_pixelization = mask.SparseMask(msk, 2)

            assert (grid_cluster_pixelization.sparse_to_image == np.array([1, 3, 11, 13])).all()
            assert (grid_cluster_pixelization.image_to_sparse == np.array(
                [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3])).all()


class TestSubCoordinateGrid(object):
    def test_sub_coordinate_grid(self, sub_coordinate_grid):
        assert sub_coordinate_grid.shape == (5, 2)
        assert (sub_coordinate_grid == np.array([[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]])).all()

    def test_sub_to_pixel(self, sub_coordinate_grid):
        assert (sub_coordinate_grid.sub_to_image == np.array(range(5))).all()

    def test_sub_data_to_image(self, sub_coordinate_grid):
        assert (sub_coordinate_grid.sub_data_to_image(np.array(range(5))) == np.array(range(5))).all()

    def test_setup_mappings_using_mask(self):
        msk = np.array([[True, False, True],
                        [False, False, False],
                        [True, False, True]])

        msk = mask.Mask(msk, pixel_scale=3.0)

        sub_coordinate_grid = mask.SubCoordinateGrid.from_mask(msk, 2)

        assert sub_coordinate_grid.sub_grid_size == 2
        assert sub_coordinate_grid.sub_grid_fraction == (1.0 / 4.0)

        assert (sub_coordinate_grid.sub_to_image == np.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])).all()


@pytest.fixture(name="memoizer")
def make_memoizer():
    return mask.Memoizer()


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


@pytest.fixture(name="coordinate_collection")
def make_coordinate_collection(centre_mask):
    return mask.CoordinateCollection.from_mask_subgrid_size_and_blurring_shape(centre_mask, 2, (3, 3))


class TestCoordinateCollection(object):
    def test_coordinate_collection(self, coordinate_collection):
        assert (coordinate_collection.image_coords == np.array([[0., 0.]])).all()
        np.testing.assert_almost_equal(coordinate_collection.sub_grid_coords, np.array([[-0.16666667, -0.16666667],
                                                                                        [-0.16666667, 0.16666667],
                                                                                        [0.16666667, -0.16666667],
                                                                                        [0.16666667, 0.16666667]]))
        assert (coordinate_collection.blurring_coords == np.array([[-1., -1.],
                                                                   [-1., 0.],
                                                                   [-1., 1.],
                                                                   [0., -1.],
                                                                   [0., 1.],
                                                                   [1., -1.],
                                                                   [1., 0.],
                                                                   [1., 1.]])).all()

    def test_apply_function(self, coordinate_collection):
        def add_one(coords):
            return np.add(1, coords)

        new_collection = coordinate_collection.apply_function(add_one)
        assert isinstance(new_collection, mask.CoordinateCollection)
        assert (new_collection.image_coords == np.add(1, np.array([[0., 0.]]))).all()
        np.testing.assert_almost_equal(new_collection.sub_grid_coords, np.add(1, np.array([[-0.16666667, -0.16666667],
                                                                                           [-0.16666667, 0.16666667],
                                                                                           [0.16666667, -0.16666667],
                                                                                           [0.16666667, 0.16666667]])))
        assert (new_collection.blurring_coords == np.add(1, np.array([[-1., -1.],
                                                                      [-1., 0.],
                                                                      [-1., 1.],
                                                                      [0., -1.],
                                                                      [0., 1.],
                                                                      [1., -1.],
                                                                      [1., 0.],
                                                                      [1., 1.]]))).all()

    def test_map_function(self, coordinate_collection):
        def add_number(coords, number):
            return np.add(coords, number)

        new_collection = coordinate_collection.map_function(add_number, [1, 2, 3])

        assert isinstance(new_collection, mask.CoordinateCollection)
        assert (new_collection.image_coords == np.add(1, np.array([[0., 0.]]))).all()
        np.testing.assert_almost_equal(new_collection.sub_grid_coords, np.add(2, np.array([[-0.16666667, -0.16666667],
                                                                                           [-0.16666667, 0.16666667],
                                                                                           [0.16666667, -0.16666667],
                                                                                           [0.16666667, 0.16666667]])))
        assert (new_collection.blurring_coords == np.add(3, np.array([[-1., -1.],
                                                                      [-1., 0.],
                                                                      [-1., 1.],
                                                                      [0., -1.],
                                                                      [0., 1.],
                                                                      [1., -1.],
                                                                      [1., 0.],
                                                                      [1., 1.]]))).all()


class GridBorder(object):

    def __init__(self, border_pixels, polynomial_degree=3, centre=(0.0, 0.0)):
        """ The border of a set of grid coordinates, which relocates coordinates outside of the border to its edge.

        This is required to ensure highly demagnified data_to_image in the centre of an image do not bias a source
        pixelization.

        Parameters
        ----------
        border_pixels : np.ndarray
            The the border source data_to_image, specified by their 1D index in *image_grid*.
        polynomial_degree : int
            The degree of the polynomial used to fit the source-plane border edge.
        """

        self.centre = centre

        self.border_pixels = border_pixels
        self.polynomial_degree = polynomial_degree
        self.centre = centre

        self.thetas = None
        self.radii = None
        self.polynomial = None

    def coordinates_to_centre(self, coordinates):
        """ Converts coordinates to the profiles's centre.

        This is performed via a translation, which subtracts the profile centre from the coordinates.

        Parameters
        ----------
        coordinates
            The (x, y) coordinates of the profile.

        Returns
        ----------
        The coordinates at the profile's centre.
        """
        return np.subtract(coordinates, self.centre)

    def relocate_coordinates_outside_border(self, coordinates):
        """For an input set of coordinates, return a new set of coordinates where every coordinate outside the border
        is relocated to its edge.

        Parameters
        ----------
        coordinates : ndarray
            The coordinates which are to have border relocations take place.
        """

        self.polynomial_fit_to_border(coordinates)

        relocated_coordinates = np.zeros(coordinates.shape)

        for (i, coordinate) in enumerate(coordinates):
            relocated_coordinates[i] = self.relocated_coordinate(coordinate)

        return relocated_coordinates

    def relocate_sub_coordinates_outside_border(self, coordinates, sub_coordinates):
        """For an input sub-coordinates, return a coordinates where all sub-coordinates outside the border are relocated
        to its edge.
        """

        # TODO : integrate these as functions into GridCoords and SubGrid, or pass in a GridCoords / SubGrid?

        self.polynomial_fit_to_border(coordinates)

        relocated_sub_coordinates = np.zeros(sub_coordinates.shape)

        for image_pixel in range(len(coordinates)):
            for (sub_pixel, sub_coordinate) in enumerate(sub_coordinates[image_pixel]):
                relocated_sub_coordinates[image_pixel, sub_pixel] = self.relocated_coordinate(sub_coordinate)

        return relocated_sub_coordinates

    def coordinates_angle_from_x(self, coordinates):
        """
        Compute the angle in degrees between the image_grid and plane positive x-axis, defined counter-clockwise.

        Parameters
        ----------
        coordinates : Union((float, float), ndarray)
            The x and y image_grid of the plane.

        Returns
        ----------
        The angle between the image_grid and the x-axis.
        """
        shifted_coordinates = self.coordinates_to_centre(coordinates)
        theta_from_x = np.degrees(np.arctan2(shifted_coordinates[1], shifted_coordinates[0]))
        if theta_from_x < 0.0:
            theta_from_x += 360.
        return theta_from_x

    def polynomial_fit_to_border(self, coordinates):

        border_coordinates = coordinates[self.border_pixels]

        self.thetas = list(map(lambda r: self.coordinates_angle_from_x(r), border_coordinates))
        self.radii = list(map(lambda r: self.coordinates_to_radius(r), border_coordinates))
        self.polynomial = np.polyfit(self.thetas, self.radii, self.polynomial_degree)

    def radius_at_theta(self, theta):
        """For a an angle theta from the x-axis, return the setup_border_pixels radius via the polynomial fit"""
        return np.polyval(self.polynomial, theta)

    def move_factor(self, coordinate):
        """Get the move factor of a coordinate.
         A move-factor defines how far a coordinate outside the source-plane setup_border_pixels must be moved in order
         to lie on it. PlaneCoordinates already within the setup_border_pixels return a move-factor of 1.0, signifying
         they are already within the setup_border_pixels.

        Parameters
        ----------
        coordinate : (float, float)
            The x and y image_grid of the pixel to have its move-factor computed.
        """
        theta = self.coordinates_angle_from_x(coordinate)
        radius = self.coordinates_to_radius(coordinate)

        border_radius = self.radius_at_theta(theta)

        if radius > border_radius:
            return border_radius / radius
        else:
            return 1.0

    def relocated_coordinate(self, coordinate):
        """Get a coordinate relocated to the source-plane setup_border_pixels if initially outside of it.

        Parameters
        ----------
        coordinate : ndarray[float, float]
            The x and y image_grid of the pixel to have its move-factor computed.
        """
        move_factor = self.move_factor(coordinate)
        return coordinate[0] * move_factor, coordinate[1] * move_factor
