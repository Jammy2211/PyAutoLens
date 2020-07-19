from autoarray import decorator_util
import numpy as np
from autoarray import mask as msk
from autoarray.util import grid_util, mask_util
from autoarray.structures import grids
from autoarray.mask import mask as msk

import copy


class PositionsSolver:
    def __init__(self, grid, pixel_scale_precision, upscale_factor=2, maxiter=50):

        self.grid = grid.in_1d_binned
        self.pixel_scale_precision = pixel_scale_precision
        self.upscale_factor = upscale_factor
        self.maxiter = maxiter

    def solve(self, lensing_obj, source_plane_coordinate):

        grid = copy.copy(self.grid)
        pixel_scale = copy.copy(grid.pixel_scale)

        for i in range(self.maxiter):

            grid_trough = self.grid_trough_from(lensing_obj=lensing_obj, grid=grid, source_plane_coordinate=source_plane_coordinate,
                                                pixel_scale=pixel_scale)

            if pixel_scale < self.pixel_scale_precision or i == self.maxiter-1:

                return grids.GridCoordinates(coordinates=grid_trough)

            grid, y_shape, x_shape = grid_util.grid_buffed_from(
                grid_1d=grid_trough, pixel_scales=(pixel_scale, pixel_scale), buffer=1
            )

            grid = grids.GridCoordinates.from_grid_sparse_uniform_upscale(
                grid_sparse_uniform=grid,
                upscale_factor=self.upscale_factor,
                pixel_scale=pixel_scale,
            )

            pixel_scale /= self.upscale_factor

            grid, y_shape, x_shape = grid_util.grid_buffed_from(
                grid_1d=grid, pixel_scales=(pixel_scale, pixel_scale), buffer=1
            )

            grid = grids.Grid.manual_1d(grid=grid, pixel_scales=pixel_scale, shape_2d=(y_shape, x_shape))

    def grid_trough_from(self, lensing_obj, grid, source_plane_coordinate, pixel_scale):

        deflections = lensing_obj.deflections_from_grid(grid=grid)
        source_plane_grid = grid.grid_from_deflection_grid(deflection_grid=deflections)
        source_plane_distances = source_plane_grid.distances_from_coordinate(
            coordinate=source_plane_coordinate
        )

        grid_neighbors, grid_has_neighbors = grid_neighbors_1d_from(
            grid_1d=grid, pixel_scale=pixel_scale
        )
        return np.asarray(grid_trough_from(
            distance_1d=source_plane_distances,
            grid_1d=grid,
            neighbors=grid_neighbors.astype("int"),
            has_neighbors=grid_has_neighbors,
        ))


@decorator_util.jit()
def grid_neighbors_1d_from(grid_1d, pixel_scale):
    """From a (y,x) grid of coordinates, determine the 8 neighors of every coordinate on the grid which has 8
    neighboring (y,x) coordinates.

    Neighbor indexes use the 1D index of the pixel on the masked grid counting from the top-left right and down.

    For example:

         x x x  x x x x x x x
         x x x  x x x x x x x      Th s  s an example mask.Mask, where:
         x x x  x x x x x x x
         x x x  0 1 2 3 x x x      x = True (P xel  s masked and excluded from the gr d)
         x x x  4 5 6 7 x x x      o = False (P xel  s not masked and  ncluded  n the gr d)
         x x x  8 9 10 11 x x x
         x x x  x x x x x x x
         x x x  x x x x x x x
         x x x  x x x x x x x
         x x x  x x x x x x x

    On the grid above, the grid cells in 1D indxes 5 and 6 have 8 neighboring pixels and their entries in the
    grid_neighbors_1d array will be:

    grid_neighbors_1d[0,:] = [0, 1, 2, 4, 6, 8, 9, 10]
    grid_neighbors_1d[1,:] = [1, 2, 3, 5, 7, 9, 10, 11]

    The other pixels will be included in the grid_neighbors_1d array, but correspond to False entries in
    grid_has_neighbors and be omitted from calculations that use the neighbor array.

    Parameters
    ----------
    grid_1d : ndarray
        The irregular 1D grid of (y,x) coordinates over which a square uniform grid is overlaid.
    pixel_scale : float
        The pixel scale of the uniform grid that laid over the irregular grid of (y,x) coordinates.
    """

    grid_pixel_centres_1d, y_shape, x_shape = grid_util.grid_pixel_centres_1d_via_grid_1d_overlap(
        grid_1d=grid_1d, pixel_scales=(pixel_scale, pixel_scale)
    )

    mask_2d = np.full(shape=(y_shape+2, x_shape+2), fill_value=True)

    for grid_index in range(grid_1d.shape[0]):

        y_pixel = int(grid_pixel_centres_1d[grid_index, 0])
        x_pixel = int(grid_pixel_centres_1d[grid_index, 1])

        mask_2d[y_pixel, x_pixel] = False

    mask_1d_indexes = mask_util.sub_mask_1d_index_for_sub_mask_index_from_sub_mask_from(
        sub_mask=mask_2d
    )

    grid_has_neighbors = np.full(shape=(grid_1d.shape[0],), fill_value=False)
    grid_neighbors_1d = np.full(shape=(grid_1d.shape[0], 8), fill_value=-1.0)

    for grid_index in range(grid_1d.shape[0]):

        y_pixel = int(grid_pixel_centres_1d[grid_index, 0])
        x_pixel = int(grid_pixel_centres_1d[grid_index, 1])

        if not mask_util.check_if_edge_pixel(mask=mask_2d, y=y_pixel, x=x_pixel):

            grid_has_neighbors[grid_index] = True

            grid_neighbors_1d[grid_index, 0] = mask_1d_indexes[y_pixel - 1, x_pixel - 1]
            grid_neighbors_1d[grid_index, 1] = mask_1d_indexes[y_pixel - 1, x_pixel]
            grid_neighbors_1d[grid_index, 2] = mask_1d_indexes[y_pixel - 1, x_pixel + 1]
            grid_neighbors_1d[grid_index, 3] = mask_1d_indexes[y_pixel, x_pixel - 1]
            grid_neighbors_1d[grid_index, 4] = mask_1d_indexes[y_pixel, x_pixel + 1]
            grid_neighbors_1d[grid_index, 5] = mask_1d_indexes[y_pixel + 1, x_pixel - 1]
            grid_neighbors_1d[grid_index, 6] = mask_1d_indexes[y_pixel + 1, x_pixel]
            grid_neighbors_1d[grid_index, 7] = mask_1d_indexes[y_pixel + 1, x_pixel + 1]

    return grid_neighbors_1d, grid_has_neighbors


@decorator_util.jit()
def grid_trough_from(distance_1d, grid_1d, neighbors, has_neighbors):
    """ Given an input grid of (y,x) coordinates and a 1d array of their distances to the centre of the source,
    determine the coordinates which are closer to the source than their 8 neighboring pixels.

    These pixels are selected as the next closest set of pixels to the source and used to define the coordinates of
    the next higher resolution grid.

    Parameters
    ----------
    distance_1d : ndarray
        The distance of every (y,x) grid coordinate to the centre of the source in the source-plane.
    grid_1d : ndarray
        The irregular 1D grid of (y,x) coordinates whose distances to the source are compared.
    neighbors : ndarray
        A 2D array of shape [pixels, 8] giving the 1D index of every grid pixel to its 8 neighboring pixels.
    has_neighbors : ndarray
        An array of bools, where True means a pixel has 8 neighbors and False means it has less than 8 and is not
        compared to the source distance.
    """
    trough_list = []

    for grid_index in range(grid_1d.shape[0]):

        if has_neighbors[grid_index]:

            distance = distance_1d[grid_index]

            if (
                distance <= distance_1d[neighbors[grid_index, 0]]
                and distance <= distance_1d[neighbors[grid_index, 1]]
                and distance <= distance_1d[neighbors[grid_index, 2]]
                and distance <= distance_1d[neighbors[grid_index, 3]]
                and distance <= distance_1d[neighbors[grid_index, 4]]
                and distance <= distance_1d[neighbors[grid_index, 5]]
                and distance <= distance_1d[neighbors[grid_index, 6]]
                and distance <= distance_1d[neighbors[grid_index, 7]]
            ):

                trough_list.append(grid_1d[grid_index])

    return trough_list

