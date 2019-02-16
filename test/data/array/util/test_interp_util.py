import numpy as np
import pytest

from autolens.data.array import mask
from autolens.data.array.util import interp_util

class TestInterpGridFromGrid:

    def test__interp_grid_has_pixel_scales_of_1__produces_4x4_grid__buffers_grid_by_half_a_pixel(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=grid_1d, interp_pixel_scales=(1.0, 1.0), interp_origin=(0.0, 0.0))

        assert (interp_grid_arcsec_1d == np.array([[ 1.5, -1.5], [ 1.5, -0.5], [ 1.5, 0.5], [ 1.5, 1.5],
                                                   [ 0.5, -1.5], [ 0.5, -0.5], [ 0.5, 0.5], [ 0.5, 1.5],
                                                   [-0.5, -1.5], [-0.5, -0.5], [-0.5, 0.5], [-0.5, 1.5],
                                                   [-1.5, -1.5], [-1.5, -0.5], [-1.5, 0.5], [-1.5, 1.5]])).all()

    def test__same_as_above__but_double_pixel_scale(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=grid_1d, interp_pixel_scales=(2.0, 2.0), interp_origin=(0.0, 0.0))

        assert (interp_grid_arcsec_1d == np.array([[2., -2.], [2., 0.], [2., 2.],
                                                   [0., -2.], [0., 0.], [0., 2.],
                                                   [-2., -2.], [-2., 0.], [-2., 2.]])).all()

    def test__2x2__grid_use_pixel_scale_of_05(self):

        grid_1d = np.array([[1., -1.],  [1., 1.],
                            [-1., -1.], [-1., 1.]])

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=grid_1d, interp_pixel_scales=(0.5, 0.5), interp_origin=(0.0, 0.0))

        assert interp_grid_arcsec_1d == pytest.approx(
            np.array([[ 1.25, -1.25], [ 1.25, -0.75], [ 1.25, -0.25], [ 1.25, 0.25], [ 1.25, 0.75], [ 1.25, 1.25],
                      [ 0.75, -1.25], [ 0.75, -0.75], [ 0.75, -0.25], [ 0.75, 0.25], [ 0.75, 0.75], [ 0.75, 1.25],
                      [ 0.25, -1.25], [ 0.25, -0.75], [ 0.25, -0.25], [ 0.25, 0.25], [ 0.25, 0.75], [ 0.25, 1.25],
                      [-0.25, -1.25], [-0.25, -0.75], [-0.25, -0.25], [-0.25, 0.25], [-0.25, 0.75], [-0.25, 1.25],
                      [-0.75, -1.25], [-0.75, -0.75], [-0.75, -0.25], [-0.75, 0.25], [-0.75, 0.75], [-0.75, 1.25],
                      [-1.25, -1.25], [-1.25, -0.75], [-1.25, -0.25], [-1.25, 0.25], [-1.25, 0.75], [-1.25, 1.25]]), 1e-4)

    def test__interp_grid_pixel_scales_are_1x2__works_correctly(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=grid_1d, interp_pixel_scales=(1.0, 2.0), interp_origin=(0.0, 0.0))

        assert (interp_grid_arcsec_1d == np.array([[ 1.5, -2.], [ 1.5, 0.], [ 1.5, 2.],
                                                   [ 0.5, -2.], [ 0.5, 0.], [ 0.5, 2.],
                                                   [-0.5, -2.], [-0.5, 0.], [-0.5, 2.],
                                                   [-1.5, -2.], [-1.5, 0.], [-1.5, 2.]])).all()

    def test__interp_grid_pixel_scales_are_2x1__works_correctly(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=grid_1d, interp_pixel_scales=(2.0, 1.0), interp_origin=(0.0, 0.0))

        assert (interp_grid_arcsec_1d == np.array([[ 2.0, -1.5], [ 2.0, -0.5], [ 2.0, 0.5], [ 2.0, 1.5],
                                                   [ 0.0, -1.5], [ 0.0, -0.5], [ 0.0, 0.5], [ 0.0, 1.5],
                                                   [-2.0, -1.5], [-2.0, -0.5], [-2.0, 0.5], [-2.0, 1.5]])).all()

    def test__move_origin_of_grid__coordinates_shift_to_around_origin(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=grid_1d, interp_pixel_scales=(1.0, 1.0), interp_origin=(1.0, 1.5))

        assert (interp_grid_arcsec_1d == np.array([[ 0.5, -3.0], [ 0.5, -2.0], [ 0.5, -1.0], [ 0.5, 0.0],
                                                   [-0.5, -3.0], [-0.5, -2.0], [-0.5, -1.0], [-0.5, 0.0],
                                                   [-1.5, -3.0], [-1.5, -2.0], [-1.5, -1.0], [-1.5, 0.0],
                                                   [-2.5, -3.0], [-2.5, -2.0], [-2.5, -1.0], [-2.5, 0.0]])).all()