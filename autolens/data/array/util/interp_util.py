import numpy as np
from scipy.interpolate import griddata

from autolens.data.array.util import grid_util
from autolens.data.array.util import mapping_util

def interp_grid_arcsec_1d_from_grid_1d_arcsec_and_interp_pixel_scales_and_origin(grid_arcsec_1d, interp_pixel_scales):

    y_min = np.min(grid_arcsec_1d[:, 0]) - (interp_pixel_scales[0] / 2.0)
    y_max = np.max(grid_arcsec_1d[:, 0]) + (interp_pixel_scales[0] / 2.0)
    x_min = np.min(grid_arcsec_1d[:, 1]) - (interp_pixel_scales[1] / 2.0)
    x_max = np.max(grid_arcsec_1d[:, 1]) + (interp_pixel_scales[1] / 2.0)

    interp_shape = (int((y_max - y_min) / interp_pixel_scales[0])+1, int((x_max - x_min) / interp_pixel_scales[1])+1)

    interp_origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)

    return grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=interp_shape,
                                                                        pixel_scales=interp_pixel_scales,
                                                                        origin=interp_origin)

def interpolated_grid_from_values_grid_arcsec_1d_and_interp_grid_arcsec_2d(values, grid_arcsec_1d,
                                                                           interp_grid_arcsec_1d):

    return griddata(points=interp_grid_arcsec_1d, values=values, xi=grid_arcsec_1d, method='linear')