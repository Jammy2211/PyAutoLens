import numpy as np
import pytest

from autolens.data.array import mask
from autolens.data.array.util import interp_util

class TestInterpGridFromGrid:

    def test__interp_grid_same_shape_as_grid__origin_is_y0_x0__no_buffer__interp_grid_is_grid(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid = interp_util.interp_grid_arcsec_1d_from_grid_and_interp_shape_and_origin(grid_arcsec_1d=grid_1d, interp_shape=(3, 3),
                                                                                                   interp_origin=(0.0, 0.0), buffer=0.0)

        assert (interp_grid == grid_1d).all()

    def test__interp_grid_same_shape_as_grid__origin_is_y0_x0__buffer_is_1__interp_grid_is_x2_grid(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid = interp_util.interp_grid_arcsec_1d_from_grid_and_interp_shape_and_origin(grid_arcsec_1d=grid_1d, interp_shape=(3, 3),
                                                                                                   interp_origin=(0.0, 0.0), buffer=1.0)

        assert interp_grid == pytest.approx(np.array([[2., -2.], [2., 0.], [2., 2.],
                                                      [0., -2.], [0., 0.], [0., 2.],
                                                      [-2., -2.], [-2., 0.], [-2., 2.]]), 1.0e-4)

    def test__interp_grid_shape_is_4x4__origin_is_y0_x0__buffer_is_0__interp_grid_is_grid(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid = interp_util.interp_grid_arcsec_1d_from_grid_and_interp_shape_and_origin(grid_arcsec_1d=grid_1d, interp_shape=(4, 4),
                                                                                                   interp_origin=(0.0, 0.0), buffer=0.0)

        f = (1.0/3.0)

        assert interp_grid == pytest.approx(np.array([[1., -1.], [1., -f], [1., f], [1., 1.],
                                                      [ f, -1.], [ f, -f], [ f, f], [f, 1.],
                                                      [-f, -1.], [-f, -f], [-f, f], [-f, 1.],
                                                      [-1., -1.], [-1., -f], [-1., f], [-1., 1.]]), 1.0e-4)

    def test__interp_grid_shape_is_3x4__origin_is_y0_x0__buffer_is_0__interp_grid_uses_grid_bounds(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid = interp_util.interp_grid_arcsec_1d_from_grid_and_interp_shape_and_origin(grid_arcsec_1d=grid_1d, interp_shape=(3, 4),
                                                                                                   interp_origin=(0.0, 0.0), buffer=0.0)

        f = (1.0/3.0)

        assert interp_grid == pytest.approx(np.array([[1., -1.], [1., -f], [1., f], [1., 1.],
                                                      [0., -1.], [0., -f], [0., f], [0., 1.],
                                                      [-1., -1.], [-1., -f], [-1., f], [-1., 1.]]), 1.0e-4)

    def test__interp_grid_shape_is_4x3__origin_is_y0_x0__buffer_is_0__interp_grid_uses_grid_bounds(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid = interp_util.interp_grid_arcsec_1d_from_grid_and_interp_shape_and_origin(grid_arcsec_1d=grid_1d, interp_shape=(4, 3),
                                                                                                   interp_origin=(0.0, 0.0), buffer=0.0)

        f = (1.0/3.0)

        assert interp_grid == pytest.approx(np.array([[1., -1.], [1., 0], [1., 1.],
                                                      [ f, -1.], [ f, 0], [f, 1.],
                                                      [-f, -1.], [-f, 0], [-f, 1.],
                                                      [-1., -1.], [-1., 0], [-1., 1.]]), 1.0e-4)

    def test__move_origin_of_grid__coordinates_shift_to_around_origin(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid = interp_util.interp_grid_arcsec_1d_from_grid_and_interp_shape_and_origin(grid_arcsec_1d=grid_1d, interp_shape=(3, 3),
                                                                                                   interp_origin=(1.0, 0.0), buffer=0.0)

        assert (interp_grid == np.array([[0., -1.], [0., 0.], [0., 1.],
                                         [-1., -1.], [-1., 0.], [-1., 1.],
                                         [-2., -1.], [-2., 0.], [-2., 1.]])).all()


        interp_grid = interp_util.interp_grid_arcsec_1d_from_grid_and_interp_shape_and_origin(grid_arcsec_1d=grid_1d, interp_shape=(3, 3),
                                                                                                   interp_origin=(0.0, 1.0), buffer=0.0)

        assert (interp_grid == np.array([[1., -2.], [1., -1.], [1., 0.],
                                         [0., -2.], [0., -1.], [0., 0.],
                                         [-1., -2.], [-1., -1.], [-1., 0.]])).all()

        interp_grid = interp_util.interp_grid_arcsec_1d_from_grid_and_interp_shape_and_origin(grid_arcsec_1d=grid_1d, interp_shape=(3, 3),
                                                                                                   interp_origin=(1.0, 1.0), buffer=0.0)

        assert (interp_grid == np.array([[0., -2.], [0., -1.], [0., 0.],
                                         [-1., -2.], [-1., -1.], [-1., 0.],
                                         [-2., -2.], [-2., -1.], [-2., 0.]])).all()


class TestInterpPairsAndWeightsFromGrids:

    def test__interp_grid_is_2x2__put_coordinates_in_each_pixel__correct_pairs(self):

        # -1.0    0.0    1.0
        #   ---------------
        #  |       |       |
        #  |   0   |   1   |
        #  |-------x-------|
        #  |       |       |
        #  |   2   |   3   |
        #  |---------------|

        interp_grid_1d = np.array([[0.5, -0.5], [0.5, 0.5],
                                   [-0.5, -0.5], [-0.5, 0.5]])

        ### Put pixel in top left pixel (index 0), thus it is in the bottom right quadrant of that pixel.

        grid_arcsec_1d = np.array([[1.0e-4, -1.0e-4]])

        grid_to_interp_pixels = interp_util.grid_to_interp_pixels_from_grid_arcsec_1d_and_interp_grid(
            interp_grid_arcsec_1d=interp_grid_1d, interp_shape=(2, 2), interp_pixel_scales=(1.0, 1.0),
            interp_origin_arcsec=(0.0, 0.0), grid_arcsec_1d=grid_arcsec_1d)

        assert (grid_to_interp_pixels == np.array([[0, 1, 2, 3]])).all()

        ### Put pixel in top right pixel (index 1), thus it is in the bottom left quadrant of that pixel.

        grid_arcsec_1d = np.array([[1.0e-4, 1.0e-4]])

        grid_to_interp_pixels = interp_util.grid_to_interp_pixels_from_grid_arcsec_1d_and_interp_grid(
            interp_grid_arcsec_1d=interp_grid_1d, interp_shape=(2, 2), interp_pixel_scales=(1.0, 1.0),
            interp_origin_arcsec=(0.0, 0.0), grid_arcsec_1d=grid_arcsec_1d)

        assert (grid_to_interp_pixels == np.array([[1, 0, 2, 3]])).all()

        ### Put pixel in bottom left pixel (index 2), thus it is in the top right quadrant of that pixel.

        grid_arcsec_1d = np.array([[-1.0e-4, -1.0e-4]])

        grid_to_interp_pixels = interp_util.grid_to_interp_pixels_from_grid_arcsec_1d_and_interp_grid(
            interp_grid_arcsec_1d=interp_grid_1d, interp_shape=(2, 2), interp_pixel_scales=(1.0, 1.0),
            interp_origin_arcsec=(0.0, 0.0), grid_arcsec_1d=grid_arcsec_1d)

        assert (grid_to_interp_pixels == np.array([[2, 0, 1, 3]])).all()

        ### Put pixel in bottom right pixel (index 3), thus it is in the top left quadrant of that pixel.

        grid_arcsec_1d = np.array([[-1.0e-4, 1.0e-4]])

        grid_to_interp_pixels = interp_util.grid_to_interp_pixels_from_grid_arcsec_1d_and_interp_grid(
            interp_grid_arcsec_1d=interp_grid_1d, interp_shape=(2, 2), interp_pixel_scales=(1.0, 1.0),
            interp_origin_arcsec=(0.0, 0.0), grid_arcsec_1d=grid_arcsec_1d)

        assert (grid_to_interp_pixels == np.array([[3, 0, 1, 2]])).all()

    def test__same_grid_as_above__but_put_all_coordinates_in_grid_arcsec_1d_at_once(self):

        # -1.0    0.0    1.0
        #   ---------------
        #  |       |       |
        #  |   0   |   1   |
        #  |-------x-------|
        #  |       |       |
        #  |   2   |   3   |
        #  |---------------|

        interp_grid_1d = np.array([[0.5, -0.5], [0.5, 0.5],
                                   [-0.5, -0.5], [-0.5, 0.5]])

        grid_arcsec_1d = np.array([[1.0e-4, -1.0e-4], [1.0e-4, 1.0e-4], [-1.0e-4, -1.0e-4], [-1.0e-4, 1.0e-4]])

        grid_to_interp_pixels = interp_util.grid_to_interp_pixels_from_grid_arcsec_1d_and_interp_grid(
            interp_grid_arcsec_1d=interp_grid_1d, interp_shape=(2, 2), interp_pixel_scales=(1.0, 1.0),
            interp_origin_arcsec=(0.0, 0.0), grid_arcsec_1d=grid_arcsec_1d)

        assert (grid_to_interp_pixels[0,:] == np.array([[0, 1, 2, 3]])).all()
        assert (grid_to_interp_pixels[1,:] == np.array([[1, 0, 2, 3]])).all()
        assert (grid_to_interp_pixels[2,:] == np.array([[2, 0, 1, 3]])).all()
        assert (grid_to_interp_pixels[3,:] == np.array([[3, 0, 1, 2]])).all()

    def test__3x3_grid_put_coordinates_at_4_corners_of_top_right_pixel_and_bottom_left_pixel(self):

        # -0.75  -0.25    0.25    0.75
        #   -----------------------
        #  |       |       |       |
        #  |   0   |   1   |   2   |
        #  |---------------x-------|
        #  |       |       |       |
        #  |   3   |   4   |   5   |
        #  |-------x---------------|
        #  |       |       |       |
        #  |   6   |   7   |   8   |
        #  |-----------------------|

        interp_grid_1d = np.array([[ 0.5, -0.5], [ 0.5, 0.0], [ 0.5, 0.5],
                                   [ 0.0, -0.5], [ 0.0, 0.0], [ 0.0, 0.5],
                                   [-0.5, -0.5], [-0.5, 0.0], [-0.5, 0.5]])

        grid_arcsec_1d = np.array([[0.251, 0.249], [0.251, 0.251], [0.249, 0.249], [0.249, 0.251],
                                   [-0.249, -0.251], [-0.249, -0.249], [-0.251, -0.251], [-0.251, -0.249]])

        grid_to_interp_pixels = interp_util.grid_to_interp_pixels_from_grid_arcsec_1d_and_interp_grid(
            interp_grid_arcsec_1d=interp_grid_1d, interp_shape=(3, 3), interp_pixel_scales=(0.5, 0.5),
            interp_origin_arcsec=(0.0, 0.0), grid_arcsec_1d=grid_arcsec_1d)

        assert (grid_to_interp_pixels[0,:] == np.array([[1, 2, 4, 5]])).all()
        assert (grid_to_interp_pixels[1,:] == np.array([[2, 1, 4, 5]])).all()
        assert (grid_to_interp_pixels[2,:] == np.array([[4, 1, 2, 5]])).all()
        assert (grid_to_interp_pixels[3,:] == np.array([[5, 1, 2, 4]])).all()
        assert (grid_to_interp_pixels[4,:] == np.array([[3, 4, 6, 7]])).all()
        assert (grid_to_interp_pixels[5,:] == np.array([[4, 3, 6, 7]])).all()
        assert (grid_to_interp_pixels[6,:] == np.array([[6, 3, 4, 7]])).all()
        assert (grid_to_interp_pixels[7,:] == np.array([[7, 3, 4, 6]])).all()

    def test__2x3_grid_put_coordinates_at_two_central_pixels(self):

        #          -0.75  -0.25    0.25    0.75
        #     1.0   -----------------------
        #          |       |       |       |
        #          |   0   |   1   |   2   |
        #          |-------x-------x-------|
        #          |       |       |       |
        #          |   3   |   4   |   5   |
        #    -1.0  |-----------------------|


        interp_grid_1d = np.array([[ 0.5, -0.5], [ 0.5, 0.0], [ 0.5, 0.5],
                                   [-0.5, -0.5], [-0.5, 0.0], [-0.5, 0.5]])

        grid_arcsec_1d = np.array([[0.01, -0.251], [0.01, -0.249], [-0.01, -0.251], [-0.01, -0.249],
                                   [0.01,  0.249], [0.01,  0.251], [-0.01,  0.249], [-0.01,  0.251]])

        grid_to_interp_pixels = interp_util.grid_to_interp_pixels_from_grid_arcsec_1d_and_interp_grid(
            interp_grid_arcsec_1d=interp_grid_1d, interp_shape=(2, 3), interp_pixel_scales=(1.0, 0.5),
            interp_origin_arcsec=(0.0, 0.0), grid_arcsec_1d=grid_arcsec_1d)

        print(grid_to_interp_pixels[1,:])

        assert (grid_to_interp_pixels[0,:] == np.array([[0, 1, 3, 4]])).all()
        assert (grid_to_interp_pixels[1,:] == np.array([[1, 0, 3, 4]])).all()
        assert (grid_to_interp_pixels[2,:] == np.array([[3, 0, 1, 4]])).all()
        assert (grid_to_interp_pixels[3,:] == np.array([[4, 0, 1, 3]])).all()
        assert (grid_to_interp_pixels[4,:] == np.array([[1, 2, 4, 5]])).all()
        assert (grid_to_interp_pixels[5,:] == np.array([[2, 1, 4, 5]])).all()
        assert (grid_to_interp_pixels[6,:] == np.array([[4, 1, 2, 5]])).all()
        assert (grid_to_interp_pixels[7,:] == np.array([[5, 1, 2, 4]])).all()

    def test__3x2_grid_put_coordinates_at_two_central_pixels(self):

        #         -1.0            1.0
        #    0.75    ----------------
        #           |       |       |
        #           |   0   |   1   |
        #    0.25   |-------x-------|
        #           |       |       |
        #           |   2   |   3   |
        #   -0.25   |-------x-------|
        #           |       |       |
        #           |   4   |   5   |
        #   -0.75   |---------------|

        interp_grid_1d = np.array([[ 0.5, -0.5], [ 0.5, 0.5],
                                   [ 0.0, -0.5], [ 0.0, 0.5],
                                   [-0.5, -0.5], [-0.5, 0.5]])

        grid_arcsec_1d = np.array([[0.251, -0.01], [0.251, 0.01], [0.249, -0.01], [0.249, 0.01],
                                   [-0.249, -0.01], [-0.249, 0.01], [-0.251, -0.01], [-0.251, 0.01]])

        grid_to_interp_pixels = interp_util.grid_to_interp_pixels_from_grid_arcsec_1d_and_interp_grid(
            interp_grid_arcsec_1d=interp_grid_1d, interp_shape=(3, 2), interp_pixel_scales=(0.5, 1.0),
            interp_origin_arcsec=(0.0, 0.0), grid_arcsec_1d=grid_arcsec_1d)

        assert (grid_to_interp_pixels[0,:] == np.array([[0, 1, 2, 3]])).all()
        assert (grid_to_interp_pixels[1,:] == np.array([[1, 0, 2, 3]])).all()
        assert (grid_to_interp_pixels[2,:] == np.array([[2, 0, 1, 3]])).all()
        assert (grid_to_interp_pixels[3,:] == np.array([[3, 0, 1, 2]])).all()
        assert (grid_to_interp_pixels[4,:] == np.array([[2, 3, 4, 5]])).all()
        assert (grid_to_interp_pixels[5,:] == np.array([[3, 2, 4, 5]])).all()
        assert (grid_to_interp_pixels[6,:] == np.array([[4, 2, 3, 5]])).all()
        assert (grid_to_interp_pixels[7,:] == np.array([[5, 2, 3, 4]])).all()