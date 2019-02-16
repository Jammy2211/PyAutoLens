import numpy as np
import pytest

from autolens.data.array import mask as msk
from autolens.data.array import grids
from autolens.model.profiles import mass_profiles as mp
from autolens.data.array.util import interp_util

class TestInterpGridFromGrid:

    def test__interp_grid_has_pixel_scales_of_1__produces_4x4_grid__buffers_grid_by_half_a_pixel(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=grid_1d, interp_pixel_scales=(1.0, 1.0))

        assert (interp_grid_arcsec_1d == np.array([[ 1.5, -1.5], [ 1.5, -0.5], [ 1.5, 0.5], [ 1.5, 1.5],
                                                   [ 0.5, -1.5], [ 0.5, -0.5], [ 0.5, 0.5], [ 0.5, 1.5],
                                                   [-0.5, -1.5], [-0.5, -0.5], [-0.5, 0.5], [-0.5, 1.5],
                                                   [-1.5, -1.5], [-1.5, -0.5], [-1.5, 0.5], [-1.5, 1.5]])).all()

    def test__same_as_above__but_double_pixel_scale(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=grid_1d, interp_pixel_scales=(2.0, 2.0))

        assert (interp_grid_arcsec_1d == np.array([[2., -2.], [2., 0.], [2., 2.],
                                                   [0., -2.], [0., 0.], [0., 2.],
                                                   [-2., -2.], [-2., 0.], [-2., 2.]])).all()

    def test__2x2__grid_use_pixel_scale_of_05(self):

        grid_1d = np.array([[1., -1.],  [1., 1.],
                            [-1., -1.], [-1., 1.]])

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=grid_1d, interp_pixel_scales=(0.5, 0.5))

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
            grid_arcsec_1d=grid_1d, interp_pixel_scales=(1.0, 2.0))

        assert (interp_grid_arcsec_1d == np.array([[ 1.5, -2.], [ 1.5, 0.], [ 1.5, 2.],
                                                   [ 0.5, -2.], [ 0.5, 0.], [ 0.5, 2.],
                                                   [-0.5, -2.], [-0.5, 0.], [-0.5, 2.],
                                                   [-1.5, -2.], [-1.5, 0.], [-1.5, 2.]])).all()

    def test__interp_grid_pixel_scales_are_2x1__works_correctly(self):

        grid_1d = np.array([[1., -1.], [1., 0.], [1., 1.],
                            [0., -1.], [0., 0.], [0., 1.],
                            [-1., -1.], [-1., 0.], [-1., 1.]])

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=grid_1d, interp_pixel_scales=(2.0, 1.0))

        assert (interp_grid_arcsec_1d == np.array([[ 2.0, -1.5], [ 2.0, -0.5], [ 2.0, 0.5], [ 2.0, 1.5],
                                                   [ 0.0, -1.5], [ 0.0, -0.5], [ 0.0, 0.5], [ 0.0, 1.5],
                                                   [-2.0, -1.5], [-2.0, -0.5], [-2.0, 0.5], [-2.0, 1.5]])).all()

class TestIntegration:

    def test__20x20_deflection_angles_no_central_pixels__interpolated_accurately(self):

        shape = (20, 20)
        pixel_scale = 1.0

        mask = msk.Mask.circular_annular(shape=shape, pixel_scale=pixel_scale, inner_radius_arcsec=4.0,
                                         outer_radius_arcsec=8.0)

        regular_grid = grids.RegularGrid.from_mask(mask=mask)

        isothermal = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        deflections = isothermal.deflections_from_grid(grid=regular_grid)

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=regular_grid, interp_pixel_scales=(0.25, 0.25))

        interp_deflections = isothermal.deflections_from_grid(grid=interp_grid_arcsec_1d)

        interpolated_deflections = interp_util.interpolated_grid_from_values_grid_arcsec_1d_and_interp_grid_arcsec_2d(
            values=interp_deflections, grid_arcsec_1d=regular_grid, interp_grid_arcsec_1d=interp_grid_arcsec_1d)

        assert np.max(deflections - interpolated_deflections) < 0.001

    def test__move_centre_of_galaxy__interpolated_accurately(self):

        shape = (20, 20)
        pixel_scale = 1.0

        mask = msk.Mask.circular_annular(shape=shape, pixel_scale=pixel_scale, inner_radius_arcsec=4.0,
                                         outer_radius_arcsec=8.0, centre=(3.0, 3.0))

        regular_grid = grids.RegularGrid.from_mask(mask=mask)

        isothermal = mp.SphericalIsothermal(centre=(3.0, 3.0), einstein_radius=1.0)

        deflections = isothermal.deflections_from_grid(grid=regular_grid)

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=regular_grid, interp_pixel_scales=(0.25, 0.25))

        interp_deflections = isothermal.deflections_from_grid(grid=interp_grid_arcsec_1d)

        interpolated_deflections = interp_util.interpolated_grid_from_values_grid_arcsec_1d_and_interp_grid_arcsec_2d(
            values=interp_deflections, grid_arcsec_1d=regular_grid, interp_grid_arcsec_1d=interp_grid_arcsec_1d)

        assert np.max(deflections - interpolated_deflections) < 0.001

    def test__use_a_grid_of_lensed_coordinates__correct_interpolation(self):

        shape = (20, 20)
        pixel_scale = 1.0

        mask = msk.Mask.circular_annular(shape=shape, pixel_scale=pixel_scale, inner_radius_arcsec=4.0,
                                         outer_radius_arcsec=8.0, centre=(3.0, 3.0))

        regular_grid = grids.RegularGrid.from_mask(mask=mask)
        isothermal = mp.SphericalIsothermal(centre=(3.0, 3.0), einstein_radius=1.0)

        lensed_regular_grid = regular_grid - isothermal.deflections_from_grid(regular_grid)

        deflections = isothermal.deflections_from_grid(grid=lensed_regular_grid)

        interp_grid_arcsec_1d = interp_util.interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(
            grid_arcsec_1d=lensed_regular_grid, interp_pixel_scales=(0.2, 0.2))

        interp_deflections = isothermal.deflections_from_grid(grid=interp_grid_arcsec_1d)

        interpolated_deflections = interp_util.interpolated_grid_from_values_grid_arcsec_1d_and_interp_grid_arcsec_2d(
            values=interp_deflections, grid_arcsec_1d=lensed_regular_grid, interp_grid_arcsec_1d=interp_grid_arcsec_1d)

        assert np.max(deflections - interpolated_deflections) < 0.001