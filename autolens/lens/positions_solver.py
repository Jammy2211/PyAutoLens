from autoarray import decorator_util
import numpy as np
from autoarray import mask as msk
from autoarray.util import grid_util, mask_util
from autoarray.structures import grids
from autoarray.mask import mask as msk

import copy


class PositionsSolver:
    def __init__(
        self, grid, use_upscaling=True, pixel_scale_precision=None, upscale_factor=2
    ):

        self.grid = grid.in_1d_binned
        self.use_upscaling = use_upscaling
        self.pixel_scale_precision = pixel_scale_precision
        self.upscale_factor = upscale_factor

    def solve_from_tracer(self, tracer):
        """Needs work - idea is it solves for all image plane multiple image positions using the redshift distribution of
        the tracer."""
        return grids.GridCoordinates(
            coordinates=[
                self.solve(lensing_obj=tracer, source_plane_coordinate=centre)
                for centre in tracer.light_profile_centres.in_list[-1]
            ]
        )

    def solve(self, lensing_obj, source_plane_coordinate):

        grid = self.grid_peaks_from(
            lensing_obj=lensing_obj,
            grid=self.grid,
            source_plane_coordinate=source_plane_coordinate,
        )

        max_distance = None

        if not self.use_upscaling:
            return grids.GridCoordinates(coordinates=grid)

        while grid.pixel_scale > self.pixel_scale_precision:

            grid = self.grid_buffed_from(grid=grid, buffer=2)

            grid = self.grid_upscaled_from(grid=grid)

            grid = self.grid_buffed_from(grid=grid, buffer=1)

            grid = self.grid_peaks_from(
                lensing_obj=lensing_obj,
                grid=grid,
                source_plane_coordinate=source_plane_coordinate,
            )

            if len(grid) == 0:
                return None

            # grid, max_distance = self.grid_within_distance_of_centre(
            #     lensing_obj=lensing_obj,
            #     grid=grid,
            #     source_plane_coordinate=source_plane_coordinate,
            #     distance=max_distance,
            # )

            if not hasattr(grid, "pixel_scale"):
                return None

        return grids.GridCoordinates(coordinates=grid)

    def grid_upscaled_from(self, grid):
        return grids.GridCoordinatesUniform.from_grid_sparse_uniform_upscale(
            grid_sparse_uniform=grid,
            upscale_factor=self.upscale_factor,
            pixel_scales=grid.pixel_scales,
            shape_2d=grid.shape_2d,
        )

    def grid_buffed_from(self, grid, buffer):

        grid_buffed, y_shape, x_shape = grid_util.grid_buffed_from(
            grid_1d=grid, pixel_scales=grid.pixel_scales, buffer=buffer
        )

        return grids.GridCoordinatesUniform(
            coordinates=grid_buffed,
            pixel_scales=grid.pixel_scales,
            shape_2d=(y_shape, x_shape),
        )

    def grid_peaks_from(self, lensing_obj, grid, source_plane_coordinate):

        deflections = lensing_obj.deflections_from_grid(grid=grid)
        source_plane_grid = grid.grid_from_deflection_grid(deflection_grid=deflections)
        source_plane_distances = source_plane_grid.distances_from_coordinate(
            coordinate=source_plane_coordinate
        )

        grid_neighbors, grid_has_neighbors = grid_neighbors_1d_from(
            grid_1d=grid, pixel_scales=grid.pixel_scales
        )

        grid_peaks = grid_peaks_from(
            distance_1d=source_plane_distances,
            grid_1d=grid,
            neighbors=grid_neighbors.astype("int"),
            has_neighbors=grid_has_neighbors,
        )

        return grids.GridCoordinatesUniform(
            coordinates=grid_peaks, pixel_scales=grid.pixel_scales
        )

    def grid_within_distance_of_centre(
        self, lensing_obj, source_plane_coordinate, grid, distance
    ):

        deflections = lensing_obj.deflections_from_grid(grid=grid)
        source_plane_grid = grid.grid_from_deflection_grid(deflection_grid=deflections)
        source_plane_distances = source_plane_grid.distances_from_coordinate(
            coordinate=source_plane_coordinate
        )

        if distance is not None:
            grid_within_distance_of_centre = grid_within_distance(
                distances_1d=source_plane_distances,
                grid_1d=grid,
                within_distance=distance,
            )
        else:
            grid_within_distance_of_centre = grid

        if distance is None:
            distance = np.max(source_plane_distances)
        else:
            distance = min(distance, np.max(source_plane_distances))

        return (
            grids.GridCoordinatesUniform(
                coordinates=grid_within_distance_of_centre,
                pixel_scales=grid.pixel_scales,
            ),
            distance,
        )


@decorator_util.jit()
def grid_neighbors_1d_from(grid_1d, pixel_scales):
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
    pixel_scales : (float, float)
        The pixel scale of the uniform grid that laid over the irregular grid of (y,x) coordinates.
    """

    grid_pixel_centres_1d, y_shape, x_shape = grid_util.grid_pixel_centres_1d_via_grid_1d_overlap(
        grid_1d=grid_1d, pixel_scales=pixel_scales
    )

    mask_2d = np.full(shape=(y_shape + 2, x_shape + 2), fill_value=True)

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
def grid_neighbors_2_1d_from(grid_1d, pixel_scales):
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
    pixel_scales : (float, float)
        The pixel scale of the uniform grid that laid over the irregular grid of (y,x) coordinates.
    """

    grid_pixel_centres_1d, y_shape, x_shape = grid_util.grid_pixel_centres_1d_via_grid_1d_overlap(
        grid_1d=grid_1d, pixel_scales=pixel_scales
    )

    mask_2d = np.full(shape=(y_shape + 2, x_shape + 2), fill_value=True)

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
def grid_peaks_from(distance_1d, grid_1d, neighbors, has_neighbors):
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
    peaks_list = []

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

                peaks_list.append(grid_1d[grid_index])

    return peaks_list


@decorator_util.jit()
def grid_peaks_neighbor_total_from(distance_1d, grid_1d, neighbors, has_neighbors):
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

    grid_peaks_neighbor_total = np.zeros(shape=(grid_1d.shape[0],))

    for grid_index in range(grid_1d.shape[0]):

        if has_neighbors[grid_index]:

            distance = distance_1d[grid_index]

            grid_peaks_neighbor_total[grid_index] = np.sum(
                distance <= distance_1d[neighbors[grid_index, :]]
            )

    return grid_peaks_neighbor_total


@decorator_util.jit()
def grid_peaks_2_from(distance_1d, grid_1d, neighbors, has_neighbors):
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
    peaks_list = []

    grid_peaks_neighbor_total = grid_peaks_neighbor_total_from(
        distance_1d=distance_1d,
        grid_1d=grid_1d,
        neighbors=neighbors,
        has_neighbors=has_neighbors,
    )

    for grid_index in range(grid_1d.shape[0]):

        if grid_peaks_neighbor_total[grid_index] == 8:
            peaks_list.append(grid_1d[grid_index])
        elif grid_peaks_neighbor_total[grid_index] == 7:

            if (
                grid_peaks_neighbor_total[neighbors[grid_index, 0]] != 8
                and grid_peaks_neighbor_total[neighbors[grid_index, 1]] != 8
                and grid_peaks_neighbor_total[neighbors[grid_index, 2]] != 8
                and grid_peaks_neighbor_total[neighbors[grid_index, 3]] != 8
                and grid_peaks_neighbor_total[neighbors[grid_index, 4]] != 8
                and grid_peaks_neighbor_total[neighbors[grid_index, 5]] != 8
                and grid_peaks_neighbor_total[neighbors[grid_index, 6]] != 8
                and grid_peaks_neighbor_total[neighbors[grid_index, 7]] != 8
            ):

                if (
                    grid_peaks_neighbor_total[neighbors[grid_index, 0]] == 7
                    or grid_peaks_neighbor_total[neighbors[grid_index, 1]] == 7
                    or grid_peaks_neighbor_total[neighbors[grid_index, 2]] == 7
                    or grid_peaks_neighbor_total[neighbors[grid_index, 3]] == 7
                    or grid_peaks_neighbor_total[neighbors[grid_index, 4]] == 7
                    or grid_peaks_neighbor_total[neighbors[grid_index, 5]] == 7
                    or grid_peaks_neighbor_total[neighbors[grid_index, 6]] == 7
                    or grid_peaks_neighbor_total[neighbors[grid_index, 7]] == 7
                ):
                    peaks_list.append(grid_1d[grid_index])

    return peaks_list


@decorator_util.jit()
def grid_within_distance(distances_1d, grid_1d, within_distance):

    grid_new = []

    for grid_index in range(grid_1d.shape[0]):
        if distances_1d[grid_index] < within_distance:
            grid_new.append(grid_1d[grid_index])

    return grid_new
