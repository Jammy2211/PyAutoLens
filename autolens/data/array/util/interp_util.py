import numpy as np

from autolens.data.array.util import grid_util

def interp_grid_arcsec_1d_from_grid_and_interp_shape_and_origin(grid_arcsec_1d, interp_shape, interp_origin, buffer=1.0e-8):

    y_min = np.min(grid_arcsec_1d[:, 0]) - buffer
    y_max = np.max(grid_arcsec_1d[:, 0]) + buffer
    x_min = np.min(grid_arcsec_1d[:, 1]) - buffer
    x_max = np.max(grid_arcsec_1d[:, 1]) + buffer

    pixel_scales = (float((y_max - y_min) / (interp_shape[0]-1)), float((x_max - x_min) / (interp_shape[1]-1)))

    interp_origin = (-1.0*interp_origin[0], -1.0*interp_origin[1]) # Coordinate system means we have to flip the origin

    return grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=interp_shape, pixel_scales=pixel_scales,
                                                                        origin=interp_origin)

def grid_to_interp_pixels_from_grid_arcsec_1d_and_interp_grid(grid_arcsec_1d, interp_grid_arcsec_1d, interp_shape,
                                                              interp_pixel_scales, interp_origin_arcsec):

    interp_centres_arcsec = grid_util.centres_from_shape_pixel_scales_and_origin(shape=interp_shape,
                                                                                 pixel_scales=interp_pixel_scales,
                                                                                 origin=interp_origin_arcsec)

    grid_to_interp_pixels = np.zeros(shape=(grid_arcsec_1d.shape[0], 4))

    for i in range(grid_arcsec_1d.shape[0]):

        interp_grid_pixel_y = int((-grid_arcsec_1d[i, 0] / interp_pixel_scales[0]) + interp_centres_arcsec[0] + 0.5)
        interp_grid_pixel_x = int((grid_arcsec_1d[i, 1] / interp_pixel_scales[1]) + interp_centres_arcsec[1] + 0.5)

        interp_grid_pixel_index = int(interp_grid_pixel_y * interp_shape[1] + interp_grid_pixel_x)

        grid_to_interp_pixels[i, 0] = interp_grid_pixel_index

        # If a coordinate is in the bottom-right of a pixel, we pair it with the interpolation pixels to the right,
        # bottom and bottom-right.

        if (grid_arcsec_1d[i,0] < interp_grid_arcsec_1d[interp_grid_pixel_index, 0] and
            grid_arcsec_1d[i,1] > interp_grid_arcsec_1d[interp_grid_pixel_index, 1]):

            grid_to_interp_pixels[i,1] = grid_to_interp_pixels[i,0] + 1 # to the right
            grid_to_interp_pixels[i,2] = grid_to_interp_pixels[i,0] + interp_shape[1] # to the bottom
            grid_to_interp_pixels[i,3] = grid_to_interp_pixels[i,0] + interp_shape[1] +1 # to the bottom right

        # If a coordinate is in the bottom-left of a pixel, we pair it with the interpolation pixels to the left,
        # bottom-left and bottom-right.

        if (grid_arcsec_1d[i,0] < interp_grid_arcsec_1d[interp_grid_pixel_index, 0] and
            grid_arcsec_1d[i,1] < interp_grid_arcsec_1d[interp_grid_pixel_index, 1]):

            grid_to_interp_pixels[i,1] = grid_to_interp_pixels[i,0] - 1 # to the left
            grid_to_interp_pixels[i,2] = grid_to_interp_pixels[i,0] + interp_shape[1] - 1 # to the bottom left
            grid_to_interp_pixels[i,3] = grid_to_interp_pixels[i,0] + interp_shape[1] # to the bottom

        # If a coordinate is in the top-right of a pixel, we pair it with the interpolation pixels to the top,
        # top-right and right.

        if (grid_arcsec_1d[i,0] > interp_grid_arcsec_1d[interp_grid_pixel_index, 0] and
            grid_arcsec_1d[i,1] > interp_grid_arcsec_1d[interp_grid_pixel_index, 1]):

            grid_to_interp_pixels[i,1] = grid_to_interp_pixels[i,0] - interp_shape[1] # to the top
            grid_to_interp_pixels[i,2] = grid_to_interp_pixels[i,0] - interp_shape[1] + 1 # to the top right
            grid_to_interp_pixels[i,3] = grid_to_interp_pixels[i,0] + 1 # to the right

        # If a coordinate is in the top-left of a pixel, we pair it with the interpolation pixels to the top-left,
        # top and left.

        if (grid_arcsec_1d[i,0] > interp_grid_arcsec_1d[interp_grid_pixel_index, 0] and
            grid_arcsec_1d[i,1] < interp_grid_arcsec_1d[interp_grid_pixel_index, 1]):

            grid_to_interp_pixels[i,1] = grid_to_interp_pixels[i,0] - interp_shape[1] - 1# to the top left
            grid_to_interp_pixels[i,2] = grid_to_interp_pixels[i,0] - interp_shape[1]  # to the top
            grid_to_interp_pixels[i,3] = grid_to_interp_pixels[i,0] - 1 # to the left


    return grid_to_interp_pixels