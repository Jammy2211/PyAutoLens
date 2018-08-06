import numpy as np
from autolens.imaging import mask
import pytest
from autolens import exc


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
    return mask.SubGrid.from_mask(msk)


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
            msk = mask.Mask.annular(shape_arc_seconds=(3, 3), pixel_scale=1, inner_radius=0,
                                    outer_radius=0.5)

            assert (msk == np.array([[True, True, True],
                                     [True, False, True],
                                     [True, True, True]])).all()

        def test__3x3_mask_inner_radius_small_outer_radius_large__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(3, 3), pixel_scale=1, inner_radius=0.5,
                                    outer_radius=3)

            assert (msk == np.array([[False, False, False],
                                     [False, True, False],
                                     [False, False, False]])).all()

        def test__4x4_mask_inner_radius_small_outer_radius_medium__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(4, 3), pixel_scale=1, inner_radius=0.51,
                                    outer_radius=1.51)

            assert (msk == np.array([[True, False, True],
                                     [False, True, False],
                                     [False, True, False],
                                     [True, False, True]])).all()

        def test__4x3_mask_inner_radius_medium_outer_radius_large__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(4, 3), pixel_scale=1, inner_radius=1.51,
                                    outer_radius=3)

            assert (msk == np.array([[False, True, False],
                                     [True, True, True],
                                     [True, True, True],
                                     [False, True, False]])).all()

        def test__3x3_mask_inner_radius_small_outer_radius_medium__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(4, 4), pixel_scale=1, inner_radius=0.81,
                                    outer_radius=2)

            assert (msk == np.array([[True, False, False, True],
                                     [False, True, True, False],
                                     [False, True, True, False],
                                     [True, False, False, True]])).all()

        def test__4x4_mask_inner_radius_medium_outer_radius_large__mask(self):
            msk = mask.Mask.annular(shape_arc_seconds=(4, 4), pixel_scale=1, inner_radius=1.71,
                                    outer_radius=3)

            assert (msk == np.array([[False, True, True, False],
                                     [True, True, True, True],
                                     [True, True, True, True],
                                     [False, True, True, False]])).all()

        def test__centre_shift__simple_shift_back(self):
            msk = mask.Mask.annular(shape_arc_seconds=(3, 3), pixel_scale=1, inner_radius=0.5,
                                    outer_radius=3, centre=(-1.0, 0.0))

            assert msk.shape == (3, 3)
            assert (msk == np.array([[False, True, False],
                                     [False, False, False],
                                     [False, False, False]])).all()

        def test__centre_shift__simple_shift_forward(self):
            msk = mask.Mask.annular(shape_arc_seconds=(3, 3), pixel_scale=1, inner_radius=0.5,
                                    outer_radius=3, centre=(0.0, 1.0))

            assert msk.shape == (3, 3)
            assert (msk == np.array([[False, False, False],
                                     [False, False, True],
                                     [False, False, False]])).all()

        def test__centre_shift__diagonal_shift(self):
            msk = mask.Mask.annular(shape_arc_seconds=(3, 3), pixel_scale=1, inner_radius=0.5,
                                    outer_radius=3, centre=(1.0, 1.0))

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

        def test__3x3_image__7x7_psf_size__7x7_image_made_border_all_masked(self):

            msk = mask.Mask.for_simulate(shape_arc_seconds=(3, 3), pixel_scale=1, psf_size=(7, 7))

            assert (msk == np.array([[True, True, True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True, True, True],
                                     [True, True, True, False, False, False, True, True, True],
                                     [True, True, True, False, False, False, True, True, True],
                                     [True, True, True, False, False, False, True, True, True],
                                     [True, True, True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True, True, True],
                                     [True, True, True, True, True, True, True, True, True]])).all()

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

        def test__setup_3x3_image__five_grid(self):
            msk = np.array([[True, False, True],
                            [False, False, False],
                            [True, False, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_grid = msk.coordinate_grid

            assert (image_grid == np.array([[-3., 0.], [0., -3.], [0., 0.], [0., 3.], [3., 0.]])).all()

        def test__setup_4x4_image__ten_grid__new_pixel_scale(self):
            msk = np.array([[True, False, False, True],
                            [False, False, False, True],
                            [True, False, False, True],
                            [False, False, False, True]])

            msk = mask.Mask(msk, pixel_scale=1.0)

            image_grid = msk.coordinate_grid

            assert (image_grid == np.array(
                [[-1.5, -0.5], [-1.5, 0.5], [-0.5, -1.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5],
                 [1.5, -1.5], [1.5, -0.5], [1.5, 0.5]])).all()

        def test__setup_3x4_image__six_grid(self):
            msk = np.array([[True, False, True, True],
                            [False, False, False, True],
                            [True, False, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_grid = msk.coordinate_grid

            assert (image_grid == np.array(
                [[-3., -1.5], [0., -4.5], [0., -1.5], [0., 1.5], [3., -1.5], [3., 4.5]])).all()

    class TestComputeGridCoordsImageSub(object):

        def test__3x3_mask_with_one_pixel__2x2_sub_grid__grid(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = mask.SubGrid.from_mask(msk, 2)

            assert (image_sub_grid == np.array([[[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]])).all()

        def test__3x3_mask_with_row_of_pixels__2x2_sub_grid__grid(self):
            msk = np.array([[True, True, True],
                            [False, False, False],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = mask.SubGrid.from_mask(msk, 2)

            assert (image_sub_grid == np.array([[-0.5, -3.5], [-0.5, -2.5], [0.5, -3.5], [0.5, -2.5],
                                                [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5],
                                                [-0.5, 2.5], [-0.5, 3.5], [0.5, 2.5], [0.5, 3.5]])).all()

        def test__3x3_mask_with_row_and_column_of_pixels__2x2_sub_grid__grid(self):
            msk = np.array([[True, True, False],
                            [False, False, False],
                            [True, True, False]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = mask.SubGrid.from_mask(msk, 2)

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

            image_sub_grid = mask.SubGrid.from_mask(msk, 2)

            image_sub_grid = np.round(image_sub_grid, decimals=2)

            np.testing.assert_almost_equal(image_sub_grid,
                                           np.array([[-0.35, 0.25], [-0.35, 0.35], [-0.25, 0.25], [-0.25, 0.35],
                                                     [-0.05, -0.35], [-0.05, -0.25], [0.05, -0.35], [0.05, -0.25],
                                                     [-0.05, -0.05], [-0.05, 0.05], [0.05, -0.05], [0.05, 0.05],
                                                     [-0.05, 0.25], [-0.05, 0.35], [0.05, 0.25], [0.05, 0.35],
                                                     [0.25, 0.25], [0.25, 0.35], [0.35, 0.25], [0.35, 0.35]]))

        def test__3x3_mask_with_one_pixel__3x3_sub_grid__grid(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = mask.SubGrid.from_mask(msk, 3)

            assert (image_sub_grid == np.array([[[-0.75, -0.75], [-0.75, 0.], [-0.75, 0.75], [0., -0.75], [0., 0.],
                                                 [0., 0.75], [0.75, -0.75], [0.75, 0.], [0.75, 0.75]]])).all()

        def test__3x3_mask_with_one_row__3x3_sub_grid__grid(self):
            msk = np.array([[True, True, False],
                            [True, False, True],
                            [True, True, False]])

            msk = mask.Mask(msk, pixel_scale=2.0)

            image_sub_grid = mask.SubGrid.from_mask(msk, 3)

            assert (image_sub_grid == np.array([[-2.5, 1.5], [-2.5, 2.], [-2.5, 2.5], [-2., 1.5], [-2., 2.],
                                                [-2., 2.5], [-1.5, 1.5], [-1.5, 2.], [-1.5, 2.5],
                                                [-0.5, -0.5], [-0.5, 0.], [-0.5, 0.5], [0., -0.5], [0., 0.], [0., 0.5],
                                                [0.5, -0.5], [0.5, 0.], [0.5, 0.5],
                                                [1.5, 1.5], [1.5, 2.], [1.5, 2.5], [2., 1.5], [2., 2.], [2., 2.5],
                                                [2.5, 1.5], [2.5, 2.], [2.5, 2.5]])).all()

        def test__4x4_mask_with_one_pixel__4x4_sub_grid__grid(self):
            msk = np.array([[True, True, True, True],
                            [True, False, False, True],
                            [True, False, False, True],
                            [True, True, True, False]])

            msk = mask.Mask(msk, pixel_scale=2.0)

            image_sub_grid = mask.SubGrid.from_mask(msk, 4)

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

        def test__4x3_mask_with_one_pixel__2x2_sub_grid__grid(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, False, False],
                            [False, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = mask.SubGrid.from_mask(msk, 2)

            assert (image_sub_grid == np.array(
                [[-2., -0.5], [-2., 0.5], [-1., -0.5], [-1., 0.5], [1., -0.5], [1., 0.5], [2., -0.5], [2., 0.5],
                 [1., 2.5], [1., 3.5], [2., 2.5], [2., 3.5], [4., -3.5], [4., -2.5], [5., -3.5], [5., -2.5]])).all()

        def test__3x4_mask_with_one_pixel__2x2_sub_grid__grid(self):
            msk = np.array([[True, True, True, False],
                            [True, False, False, True],
                            [False, True, False, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            image_sub_grid = mask.SubGrid.from_mask(msk, 2)

            assert (image_sub_grid == np.array(
                [[-3.5, 4.], [-3.5, 5.], [-2.5, 4.], [-2.5, 5.], [-0.5, -2.], [-0.5, -1.], [0.5, -2.], [0.5, -1.],
                 [-0.5, 1.], [-0.5, 2.], [0.5, 1.], [0.5, 2.], [2.5, -5.], [2.5, -4.], [3.5, -5.], [3.5, -4.],
                 [2.5, 1.], [2.5, 2.], [3.5, 1.], [3.5, 2.]])).all()

    class TestComputeGridCoordsBlurring(object):

        def test__3x3_blurring_mask_correct_grid(self):
            msk = np.array([[True, True, True],
                            [True, False, True],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            blurring_grid = msk.blurring_mask_for_kernel_shape(kernel_shape=(3, 3)).coordinate_grid

            assert (blurring_grid == np.array(
                [[-3., -3.], [-3., 0.], [-3., 3.], [0., -3.], [0., 3.], [3., -3.], [3., 0.], [3., 3.]])).all()

        def test__3x5_blurring_mask_correct_grid(self):
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

        def test__5x3_blurring_mask_correct_grid(self):
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

            sub_to_image = mask.SubGrid.from_mask(msk, 2).sub_to_image

            assert (sub_to_image == np.array([0, 0, 0, 0])).all()

        def test__3x3_mask_with_row_of_pixels_pixel__2x2_sub_grid__correct_sub_to_image(self):
            msk = np.array([[True, True, True],
                            [False, False, False],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sub_to_image = mask.SubGrid.from_mask(msk, 2).sub_to_image

            assert (sub_to_image == np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])).all()

        def test__3x3_mask_with_row_of_pixels_pixel__3x3_sub_grid__correct_sub_to_image(self):
            msk = np.array([[True, True, True],
                            [False, False, False],
                            [True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            sub_to_image = mask.SubGrid.from_mask(msk, 3).sub_to_image

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

            grid_data = msk.map_to_1d(data)

            assert (grid_data[0] == np.array([5])).all()

        def test__setup_3x3_data__five_now_in_mask(self):
            data = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

            msk = np.array([[True, False, True],
                            [False, False, False],
                            [True, False, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            grid_data = msk.map_to_1d(data)

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

            grid_data = msk.map_to_1d(data)

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

            grid_data = msk.map_to_1d(data)

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

    class TestComputeBorderSubPixels(object):

        def test__7x7_mask__2x2_sub_grid__nine_central_pixels__is_border(self):

            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border_sub_pixels = msk.border_sub_pixel_indices(sub_grid_size=2)

            assert (border_sub_pixels == np.array([0, 4, 9, 12, 21, 26, 30, 35])).all()

        def test__7x7_mask__4x4_sub_grid_nine_central_pixels__is_border(self):

            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border_sub_pixels = msk.border_sub_pixel_indices(sub_grid_size=4)

            assert (border_sub_pixels == np.array([0, 16, 35, 48, 83, 108, 124, 143])).all()

        def test__7x7_mask_rectangle_of_fifteen_central_pixels__is_border(self):

            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border_sub_pixels = msk.border_sub_pixel_indices(sub_grid_size=2)

            assert (border_sub_pixels == np.array([0, 4, 9, 12, 21, 24, 33, 38, 47, 50, 54, 59])).all()

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

            border_sub_pixels = msk.border_sub_pixel_indices(sub_grid_size=2)
            assert (border_sub_pixels == np.array([0, 4, 8, 13, 16, 25, 30, 34, 43, 47, 50, 59, 62, 66, 71])).all()

        def test__7x8_mask_add_edge_pixels__also_in_border(self):

            msk = np.array([[True, True, True, True, True, True, True, True],
                            [True, True, True, False, True, True, True, True],
                            [True, True, False, False, False, True, True, True],
                            [True, True, False, False, False, True, True, True],
                            [True, False, False, False, False, False, True, True],
                            [True, True, False, False, False, True, True, True],
                            [True, True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border_sub_pixels = msk.border_sub_pixel_indices(sub_grid_size=2)
            assert (border_sub_pixels == np.array([0, 4, 8, 13, 16, 25, 30, 34, 43, 47, 50, 54, 59])).all()

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

        sub_coordinate_grid = mask.SubGrid.from_mask(msk, 2)

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


@pytest.fixture(name="grids")
def make_grids(centre_mask):
    return mask.GridCollection.from_mask_sub_grid_size_and_blurring_shape(centre_mask, 2, (3, 3))


class TestGridCollection(object):
    def test_grids(self, grids):
        assert (grids.image == np.array([[0., 0.]])).all()
        np.testing.assert_almost_equal(grids.sub, np.array([[-0.16666667, -0.16666667],
                                                                                        [-0.16666667, 0.16666667],
                                                                                        [0.16666667, -0.16666667],
                                                                                        [0.16666667, 0.16666667]]))
        assert (grids.blurring == np.array([[-1., -1.],
                                                                   [-1., 0.],
                                                                   [-1., 1.],
                                                                   [0., -1.],
                                                                   [0., 1.],
                                                                   [1., -1.],
                                                                   [1., 0.],
                                                                   [1., 1.]])).all()

    def test_apply_function(self, grids):
        def add_one(coords):
            return np.add(1, coords)

        new_collection = grids.apply_function(add_one)
        assert isinstance(new_collection, mask.GridCollection)
        assert (new_collection.image == np.add(1, np.array([[0., 0.]]))).all()
        np.testing.assert_almost_equal(new_collection.sub, np.add(1, np.array([[-0.16666667, -0.16666667],
                                                                               [-0.16666667, 0.16666667],
                                                                               [0.16666667, -0.16666667],
                                                                               [0.16666667, 0.16666667]])))
        assert (new_collection.blurring == np.add(1, np.array([[-1., -1.],
                                                               [-1., 0.],
                                                               [-1., 1.],
                                                               [0., -1.],
                                                               [0., 1.],
                                                               [1., -1.],
                                                               [1., 0.],
                                                               [1., 1.]]))).all()

    def test_map_function(self, grids):
        def add_number(coords, number):
            return np.add(coords, number)

        new_collection = grids.map_function(add_number, [1, 2, 3])

        assert isinstance(new_collection, mask.GridCollection)
        assert (new_collection.image == np.add(1, np.array([[0., 0.]]))).all()
        np.testing.assert_almost_equal(new_collection.sub, np.add(2, np.array([[-0.16666667, -0.16666667],
                                                                               [-0.16666667, 0.16666667],
                                                                               [0.16666667, -0.16666667],
                                                                               [0.16666667, 0.16666667]])))
        assert (new_collection.blurring == np.add(3, np.array([[-1., -1.],
                                                               [-1., 0.],
                                                               [-1., 1.],
                                                               [0., -1.],
                                                               [0., 1.],
                                                               [1., -1.],
                                                               [1., 0.],
                                                               [1., 1.]]))).all()


class TestBorderCollection(object):

    class TestSetup:

        def test__simple_setup_using_constructor(self):

            image_border = mask.ImageGridBorder(arr=np.array([1, 2, 5]), polynomial_degree=4, centre=(1.0, 1.0))
            sub_border = mask.SubGridBorder(arr=np.array([1, 2, 3]), polynomial_degree=2, centre=(0.0, 1.0))

            border_collection = mask.BorderCollection(image=image_border, sub=sub_border)

            assert (border_collection.image == np.array([1, 2, 5])).all()
            assert border_collection.image.polynomial_degree == 4
            assert border_collection.image.centre == (1.0, 1.0)

            assert (border_collection.sub == np.array([1, 2, 3])).all()
            assert border_collection.sub.polynomial_degree == 2
            assert border_collection.sub.centre == (0.0, 1.0)

        def test__setup_from_mask(self):

            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, False, False, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border_collection = mask.BorderCollection.from_mask_and_sub_grid_size(mask=msk, sub_grid_size=2)

            assert (border_collection.image == np.array([0, 1])).all()
            assert (border_collection.sub == np.array([0, 5])).all()

    class TestRelocatedGridsFromGrids:
        
        def test__simple_case__new_grids_have_relocates(self):
            
            thetas = np.linspace(0.0, 2.0 * np.pi, 32)
            image_grid_circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))
            image_grid = image_grid_circle
            image_grid.append(np.array([0.1, 0.0]))
            image_grid.append(np.array([-0.2, -0.3]))
            image_grid.append(np.array([0.5, 0.4]))
            image_grid.append(np.array([0.7, -0.1]))
            image_grid = np.asarray(image_grid)

            image_border = mask.ImageGridBorder(arr=np.arange(32), polynomial_degree=3)

            thetas = np.linspace(0.0, 2.0 * np.pi, 32)
            sub_grid_circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))
            sub_grid = sub_grid_circle
            sub_grid.append(np.array([2.5, 0.0]))
            sub_grid.append(np.array([0.0, 3.0]))
            sub_grid.append(np.array([-2.5, 0.0]))
            sub_grid.append(np.array([-5.0, 5.0]))
            sub_grid = np.asarray(sub_grid)

            sub_border = mask.SubGridBorder(arr=np.arange(32), polynomial_degree=3)

            borders = mask.BorderCollection(image=image_border, sub=sub_border)

            grids = mask.GridCollection(image=image_grid, sub=sub_grid, blurring=None)

            relocated_grids = borders.relocated_grids_from_grids(grids)

            assert relocated_grids.image[0:32] == pytest.approx(np.asarray(image_grid_circle)[0:32], 1e-3)
            assert relocated_grids.image[32] == pytest.approx(np.array([0.1, 0.0]), 1e-3)
            assert relocated_grids.image[33] == pytest.approx(np.array([-0.2, -0.3]), 1e-3)
            assert relocated_grids.image[34] == pytest.approx(np.array([0.5, 0.4]), 1e-3)
            assert relocated_grids.image[35] == pytest.approx(np.array([0.7, -0.1]), 1e-3)

            assert relocated_grids.sub[0:32] == pytest.approx(np.asarray(sub_grid_circle)[0:32], 1e-3)
            assert relocated_grids.sub[32] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert relocated_grids.sub[33] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
            assert relocated_grids.sub[34] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            assert relocated_grids.sub[35] == pytest.approx(np.array([-0.707, 0.707]), 1e-3)


class TestImageGridBorder(object):

    class TestFromMask:

        def test__simple_mask_border_pixel_is_pixel(self):

            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, False, False, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border = mask.ImageGridBorder.from_mask(msk)

            assert (border == np.array([0, 1])).all()

    class TestThetasAndRadii:

        def test__four_grid_in_circle__all_in_border__correct_radii_and_thetas(self):

            grid = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            border = mask.ImageGridBorder(arr=np.arange(4), polynomial_degree=3)
            radii = border.grid_to_radii(grid)
            thetas = border.grid_to_thetas(grid)

            assert (radii == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert (thetas == np.array([0.0, 90.0, 180.0, 270.0])).all()

        def test__other_thetas_radii(self):
            grid = np.array([[2.0, 0.0], [2.0, 2.0], [-1.0, -1.0], [0.0, -3.0]])

            border = mask.ImageGridBorder(arr=np.arange(4), polynomial_degree=3)
            radii = border.grid_to_radii(grid)
            thetas = border.grid_to_thetas(grid)

            assert (radii == np.array([2.0, 2.0 * np.sqrt(2), np.sqrt(2.0), 3.0])).all()
            assert (thetas == np.array([0.0, 45.0, 225.0, 270.0])).all()

        def test__border_centre_offset__grid_same_r_and_theta_shifted(self):

            grid = np.array([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])

            border = mask.ImageGridBorder(arr=np.arange(4), polynomial_degree=3, centre=(1.0, 1.0))
            radii = border.grid_to_radii(grid)
            thetas = border.grid_to_thetas(grid)

            assert (radii == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert (thetas == np.array([0.0, 90.0, 180.0, 270.0])).all()

    class TestBorderPolynomialFit(object):

        def test__four_grid_in_circle__thetas_at_radius_are_each_grid_radius(self):

            grid = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            border = mask.ImageGridBorder(arr=np.arange(4), polynomial_degree=3)
            poly = border.polynomial_fit_to_border(grid)

            assert np.polyval(poly, 0.0) == pytest.approx(1.0, 1e-3)
            assert np.polyval(poly, 90.0) == pytest.approx(1.0, 1e-3)
            assert np.polyval(poly, 180.0) == pytest.approx(1.0, 1e-3)
            assert np.polyval(poly, 270.0) == pytest.approx(1.0, 1e-3)

        def test__eight_grid_in_circle__thetas_at_each_grid_are_the_radius(self):

            grid = np.array([[1.0, 0.0], [0.5 * np.sqrt(2), 0.5 * np.sqrt(2)],
                                    [0.0, 1.0], [-0.5 * np.sqrt(2), 0.5 * np.sqrt(2)],
                                    [-1.0, 0.0], [-0.5 * np.sqrt(2), -0.5 * np.sqrt(2)],
                                    [0.0, -1.0], [0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]])

            border = mask.ImageGridBorder(arr=
                                      np.arange(8), polynomial_degree=3)
            poly = border.polynomial_fit_to_border(grid)

            assert np.polyval(poly, 0.0) == pytest.approx(1.0, 1e-3)
            assert np.polyval(poly, 45.0) == pytest.approx(1.0, 1e-3)
            assert np.polyval(poly, 90.0) == pytest.approx(1.0, 1e-3)
            assert np.polyval(poly, 135.0) == pytest.approx(1.0, 1e-3)
            assert np.polyval(poly, 180.0) == pytest.approx(1.0, 1e-3)
            assert np.polyval(poly, 225.0) == pytest.approx(1.0, 1e-3)
            assert np.polyval(poly, 270.0) == pytest.approx(1.0, 1e-3)
            assert np.polyval(poly, 315.0) == pytest.approx(1.0, 1e-3)

    class TestMoveFactors(object):

        def test__inside_border__move_factor_is_1(self):
            
            grid = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            border = mask.ImageGridBorder(arr=np.arange(4), polynomial_degree=3)
            move_factors = border.move_factors_from_grid(grid)

            assert move_factors[0] == pytest.approx(1.0, 1e-4)
            assert move_factors[1] == pytest.approx(1.0, 1e-4)
            assert move_factors[2] == pytest.approx(1.0, 1e-4)
            assert move_factors[3] == pytest.approx(1.0, 1e-4)

        def test__outside_border_double_its_radius__move_factor_is_05(self):

            grid = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
                             [2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])

            border = mask.ImageGridBorder(arr=np.arange(4), polynomial_degree=3)
            move_factors = border.move_factors_from_grid(grid)

            assert move_factors[0] == pytest.approx(1.0, 1e-4)
            assert move_factors[1] == pytest.approx(1.0, 1e-4)
            assert move_factors[2] == pytest.approx(1.0, 1e-4)
            assert move_factors[3] == pytest.approx(1.0, 1e-4)
            assert move_factors[4] == pytest.approx(0.5, 1e-4)
            assert move_factors[5] == pytest.approx(0.5, 1e-4)
            assert move_factors[6] == pytest.approx(0.5, 1e-4)
            assert move_factors[7] == pytest.approx(0.5, 1e-4)

        def test__outside_border_as_above__but_shift_for_source_plane_centre(self):

            grid = np.array([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 0.0],
                             [3.0, 1.0], [1.0, 3.0], [1.0, 3.0], [3.0, 1.0]])

            border = mask.ImageGridBorder(arr=np.arange(4), polynomial_degree=3, centre=(1.0, 1.0))
            move_factors = border.move_factors_from_grid(grid)

            assert move_factors[0] == pytest.approx(1.0, 1e-4)
            assert move_factors[1] == pytest.approx(1.0, 1e-4)
            assert move_factors[2] == pytest.approx(1.0, 1e-4)
            assert move_factors[3] == pytest.approx(1.0, 1e-4)
            assert move_factors[4] == pytest.approx(0.5, 1e-4)
            assert move_factors[5] == pytest.approx(0.5, 1e-4)
            assert move_factors[6] == pytest.approx(0.5, 1e-4)
            assert move_factors[7] == pytest.approx(0.5, 1e-4)

    class TestRelocateCoordinates(object):

        def test__inside_border_no_relocations(self):

            thetas = np.linspace(0.0, 2.0 * np.pi, 32)
            grid_circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))
            grid = grid_circle
            grid.append(np.array([0.1, 0.0]))
            grid.append(np.array([-0.2, -0.3]))
            grid.append(np.array([0.5, 0.4]))
            grid.append(np.array([0.7, -0.1]))
            grid = np.asarray(grid)

            border = mask.ImageGridBorder(arr=np.arange(32), polynomial_degree=3)
            relocated_grid = border.relocated_grid_from_grid(grid)

            assert relocated_grid[0:32] == pytest.approx(np.asarray(grid_circle)[0:32], 1e-3)
            assert relocated_grid[32] == pytest.approx(np.array([0.1, 0.0]), 1e-3)
            assert relocated_grid[33] == pytest.approx(np.array([-0.2, -0.3]), 1e-3)
            assert relocated_grid[34] == pytest.approx(np.array([0.5, 0.4]), 1e-3)
            assert relocated_grid[35] == pytest.approx(np.array([0.7, -0.1]), 1e-3)

        def test__outside_border_simple_cases__relocates_to_source_border(self):

            thetas = np.linspace(0.0, 2.0 * np.pi, 32)
            grid_circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))
            grid = grid_circle
            grid.append(np.array([2.5, 0.0]))
            grid.append(np.array([0.0, 3.0]))
            grid.append(np.array([-2.5, 0.0]))
            grid.append(np.array([-5.0, 5.0]))
            grid = np.asarray(grid)

            border = mask.ImageGridBorder(arr=np.arange(32), polynomial_degree=3)
            relocated_grid = border.relocated_grid_from_grid(grid)

            assert relocated_grid[0:32] == pytest.approx(np.asarray(grid_circle)[0:32], 1e-3)
            assert relocated_grid[32] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert relocated_grid[33] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
            assert relocated_grid[34] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            assert relocated_grid[35] == pytest.approx(np.array([-0.707, 0.707]), 1e-3)

        def test__6_grid_total__2_outside_border__different_border__relocate_to_source_border(self):

            grid = np.array([[1.0, 0.0], [20., 20.], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
            border_pixels = np.array([0, 2, 3, 4])

            border = mask.ImageGridBorder(border_pixels, polynomial_degree=3)

            relocated_grid = border.relocated_grid_from_grid(grid)

            assert relocated_grid[0] == pytest.approx(grid[0], 1e-3)
            assert relocated_grid[1] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)
            assert relocated_grid[2] == pytest.approx(grid[2], 1e-3)
            assert relocated_grid[3] == pytest.approx(grid[3], 1e-3)
            assert relocated_grid[4] == pytest.approx(grid[4], 1e-3)
            assert relocated_grid[5] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)


class TestSubGridBorder(object):

    class TestFromMask:

        def test__simple_mask_border_pixel_is_pixel(self):

            msk = np.array([[True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, False, False, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True]])

            msk = mask.Mask(msk, pixel_scale=3.0)

            border = mask.SubGridBorder.from_mask(msk, sub_grid_size=2)

            assert (border == np.array([0, 5])).all()



