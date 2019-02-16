import numpy as np

from autolens.data.array.util import grid_util

def interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(grid_arcsec_1d, interp_pixel_scales,
                                                                                 interp_origin):

    y_min = np.min(grid_arcsec_1d[:, 0]) - (interp_pixel_scales[0] / 2.0)
    y_max = np.max(grid_arcsec_1d[:, 0]) + (interp_pixel_scales[0] / 2.0)
    x_min = np.min(grid_arcsec_1d[:, 1]) - (interp_pixel_scales[1] / 2.0)
    x_max = np.max(grid_arcsec_1d[:, 1]) + (interp_pixel_scales[1] / 2.0)

    interp_shape = (int((y_max - y_min) / interp_pixel_scales[0])+1, int((x_max - x_min) / interp_pixel_scales[1])+1)

    interp_origin = (-1.0*interp_origin[0], -1.0*interp_origin[1]) # Coordinate system means we have to flip the origin

    return grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(
        shape=interp_shape, pixel_scales=interp_pixel_scales, origin=interp_origin)