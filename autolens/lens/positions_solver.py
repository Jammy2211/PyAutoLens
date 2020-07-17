from autoarray import decorator_util
import numpy as np
from autoarray import mask as msk
from autoarray.util import grid_util, mask_util
from autoarray.structures import grids
from autoarray.mask import mask as msk

import copy


class PositionsSolver:
    def __init__(self):

        pass

    def solve(self, lensing_obj, grid, source_plane_coordinate):

        grid = copy.copy(grid)
        pixel_scale = copy.copy(grid.pixel_scale)

        for i in range(3):

            grid, source_plane_distances = self.grid_within_circle_from(
                lensing_obj=lensing_obj,
                grid=grid,
                source_plane_coordinate=source_plane_coordinate,
                radius=2.0 * pixel_scale,
            )
            grid_neighbors, grid_has_neighbors = grid_neighbors_1d_from(
                grid_1d=grid, pixel_scale=pixel_scale
            )
            trough_coordinates = trough_coordinates_from(
                distance_1d=source_plane_distances,
                grid_1d=grid,
                neighbors=grid_neighbors.astype("int"),
                has_neighbors=grid_has_neighbors,
            )

            print(trough_coordinates)

            grid = grids.GridCoordinates.from_grid_sparse_uniform_upscale(
                grid_sparse_uniform=np.asarray(trough_coordinates),
                upscale_factor=3,
                pixel_scale=pixel_scale,
            )
            print(pixel_scale)
            print(grid[0, 0] - grid[4, 0])
            print(grid[0, 1] - grid[1, 1])
            stop
            pixel_scale /= 3

        #     grid = grids.GridCoordinates(coordinates=trough_coordinates)

        return grid

    def grid_within_circle_from(
        self, lensing_obj, grid, source_plane_coordinate, radius
    ):

        deflections = lensing_obj.deflections_from_grid(grid=grid)
        source_plane_grid = grid.grid_from_deflection_grid(deflection_grid=deflections)
        source_plane_squared_distances = source_plane_grid.squared_distances_from_coordinate(
            coordinate=source_plane_coordinate
        )

        mask_within_circle = source_plane_squared_distances < radius ** 2.0
        total_new_grid_pixels = sum(mask_within_circle)

        grid_new = np.zeros(shape=(total_new_grid_pixels, 2))
        source_plane_squared_distances_new = np.zeros(shape=(total_new_grid_pixels,))

        grid_new_index = 0

        for grid_index in range(grid.shape[0]):
            if mask_within_circle[grid_index]:
                grid_new[grid_new_index] = grid[grid_index]
                source_plane_squared_distances_new[
                    grid_new_index
                ] = source_plane_squared_distances[grid_index]
                grid_new_index += 1

        return (
            grids.GridCoordinates(coordinates=grid_new),
            source_plane_squared_distances_new,
        )


def grid_pixel_centres_1d_via_grid_1d_overlap(grid_1d, pixel_scale):
    """Take a 1D grid of irregular (y,x) coordinates over-lay a uniform square grid defined by an input pixel scale,
    where:

    1) The overlaid grid uses the extrema (y,x) coordinates of the irregular grid at the top-left, top-right,
    bottom-left and bottom-right.

    2) The over-laid grid is buffed by 1 pixel on every side of the grid.

    The (y,x) 2D pixel centres of the overlaid grid are then computed for every irregular (y,x) coordinate and returned
    along with the shape of the buffed overlaid grid.

    This is used to create small regular grids defined by a pixel scale around irregular (y,x) grid coordinates, for
    example when creating an upscaled subset of coordinates around the point.

    Parameters
    grid_1d : ndarray
        The irregular 1D grid of (y,x) coordinates over which a square uniform grid is overlaid.
    pixel_scale : float
        The pixel scale of the uniform grid that laid over the irregular grid of (y,x) coordinates.
    """

    y_size = np.max(grid_1d[:, 0]) - np.min(grid_1d[:, 0])
    x_size = np.max(grid_1d[:, 1]) - np.min(grid_1d[:, 1])

    y_shape = int(y_size / pixel_scale) + 3
    x_shape = int(x_size / pixel_scale) + 3

    y_origin = (np.max(grid_1d[:, 0]) + np.min(grid_1d[:, 0])) / 2.0
    x_origin = (np.max(grid_1d[:, 1]) + np.min(grid_1d[:, 1])) / 2.0

    return (
        grid_util.grid_pixel_centres_1d_from(
            grid_scaled_1d=grid_1d,
            shape_2d=(y_shape, x_shape),
            pixel_scales=(pixel_scale, pixel_scale),
            origin=(y_origin, x_origin),
        ),
        y_shape,
        x_shape,
    )


# @decorator_util.jit()
def grid_neighbors_1d_from(grid_1d, pixel_scale):

    grid_pixel_centres_1d, y_shape, x_shape = grid_pixel_centres_1d_via_grid_1d_overlap(
        grid_1d=grid_1d, pixel_scale=pixel_scale
    )

    mask_2d = np.full(shape=(y_shape, x_shape), fill_value=True)

    grid_pixel_centres_1d = grid_pixel_centres_1d_via_grid_1d_overlap(
        grid_1d=grid_1d, pixel_scale=pixel_scale
    )

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


# @decorator_util.jit()
def trough_coordinates_from(distance_1d, grid_1d, neighbors, has_neighbors):

    trough_coordinates = []

    for grid_index in range(grid_1d.shape[0]):

        if has_neighbors[grid_index]:

            distance = distance_1d[grid_index]

            if (
                distance < distance_1d[neighbors[grid_index, 0]]
                and distance < distance_1d[neighbors[grid_index, 1]]
                and distance < distance_1d[neighbors[grid_index, 2]]
                and distance < distance_1d[neighbors[grid_index, 3]]
                and distance < distance_1d[neighbors[grid_index, 4]]
                and distance < distance_1d[neighbors[grid_index, 5]]
                and distance < distance_1d[neighbors[grid_index, 6]]
                and distance < distance_1d[neighbors[grid_index, 7]]
            ):

                trough_coordinates.append(grid_1d[grid_index])

    return trough_coordinates


@decorator_util.jit()
def trough_pixels_from(array_2d, mask=None):

    if mask is None:
        mask = np.full(fill_value=False, shape=array_2d.shape)

    trough_pixels = []

    for y in range(1, array_2d.shape[0] - 1):
        for x in range(1, array_2d.shape[1] - 1):
            if not mask[y, x]:
                if (
                    array_2d[y, x] < array_2d[y + 1, x]
                    and array_2d[y, x] < array_2d[y + 1, x + 1]
                    and array_2d[y, x] < array_2d[y, x + 1]
                    and array_2d[y, x] < array_2d[y - 1, x + 1]
                    and array_2d[y, x] < array_2d[y - 1, x]
                    and array_2d[y, x] < array_2d[y - 1, x - 1]
                    and array_2d[y, x] < array_2d[y, x - 1]
                    and array_2d[y, x] < array_2d[y + 1, x - 1]
                ):

                    trough_pixels.append([y, x])

    return trough_pixels


@decorator_util.jit()
def quadrant_from_coordinate(coordinate):

    if coordinate[0] >= 0.0 and coordinate[1] <= 0.0:
        return 0
    elif coordinate[0] >= 0.0 and coordinate[1] >= 0.0:
        return 1
    elif coordinate[0] <= 0.0 and coordinate[1] <= 0.0:
        return 2
    elif coordinate[0] <= 0.0 and coordinate[1] >= 0.0:
        return 3


@decorator_util.jit()
def positions_at_coordinate_from(grid_2d, coordinate, mask=None):

    if mask is None:
        mask = np.full(fill_value=False, shape=(grid_2d.shape[0], grid_2d.shape[1]))

    grid_shifted = np.zeros(shape=grid_2d.shape)

    grid_shifted[:, :, 0] = grid_2d[:, :, 0] - coordinate[0]
    grid_shifted[:, :, 1] = grid_2d[:, :, 1] - coordinate[1]

    pixels_at_coordinate = []

    for y in range(1, grid_2d.shape[0] - 1):
        for x in range(1, grid_2d.shape[1] - 1):
            if not mask[y, x] and not mask_util.check_if_edge_pixel(
                mask=mask, y=y, x=x
            ):

                top_left_quadrant = quadrant_from_coordinate(
                    coordinate=grid_shifted[y + 1, x - 1, :]
                )
                top_right_quadrant = quadrant_from_coordinate(
                    coordinate=grid_shifted[y + 1, x + 1, :]
                )
                bottom_left_quadrant = quadrant_from_coordinate(
                    coordinate=grid_shifted[y - 1, x - 1, :]
                )
                bottom_right_quadrant = quadrant_from_coordinate(
                    coordinate=grid_shifted[y - 1, x + 1, :]
                )

                if (
                    top_left_quadrant
                    + top_right_quadrant
                    + bottom_left_quadrant
                    + bottom_right_quadrant
                ) == 6:

                    if (
                        top_left_quadrant
                        != top_right_quadrant
                        != bottom_left_quadrant
                        != bottom_right_quadrant
                    ):

                        pixels_at_coordinate.append((y, x))

    return pixels_at_coordinate
