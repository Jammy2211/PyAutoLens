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

        for i in range(5):

            grid = self.grid_within_circle_from(
                lensing_obj=lensing_obj,
                grid=grid,
                source_plane_coordinate=source_plane_coordinate,
                radius=pixel_scale,
            )
            grid = grids.GridCoordinates.from_grid_sparse_uniform_upscale(
                grid_sparse_uniform=grid, upscale_factor=3
            )
            pixel_scale /= pixel_scale / 3
            print(grid)

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

        grid_new_index = 0

        for grid_index in range(grid.shape[0]):
            if mask_within_circle[grid_index]:
                grid_new[grid_new_index] = grid[grid_index]
                grid_new_index += 1

        return grids.GridCoordinates(coordinates=grid_new)

    def mask_trough_from(self, lensing_obj, source_plane_coordinate, mask, buffer=1):

        grid = grids.Grid.from_mask(mask=mask)

        deflections = lensing_obj.deflections_from_grid(grid=grid)
        source_plane_grid = grid.grid_from_deflection_grid(deflection_grid=deflections)
        source_plane_squared_distances = source_plane_grid.squared_distances_from_coordinate(
            coordinate=source_plane_coordinate
        )

        trough_pixels = trough_pixels_from(
            array_2d=source_plane_squared_distances.in_2d, mask=mask
        )

        return msk.Mask.from_pixel_coordinates(
            shape_2d=grid.shape_2d,
            pixel_coordinates=trough_pixels,
            pixel_scales=grid.pixel_scales,
            sub_size=grid.sub_size,
            origin=grid.origin,
            buffer=buffer,
        )

    def image_plane_positions_from_old(self, lensing_obj, source_plane_coordinate):

        deflections = lensing_obj.deflections_from_grid(grid=self.initial_grid)
        source_plane_grid = self.initial_grid.grid_from_deflection_grid(
            deflection_grid=deflections
        )

        source_plane_squared_distances = source_plane_grid.squared_distances_from_coordinate(
            coordinate=source_plane_coordinate
        )

        trough_pixels = trough_pixels_from(
            array_2d=source_plane_squared_distances.in_2d, mask=self.initial_grid.mask
        )

        trough_mask = msk.Mask.from_pixel_coordinates(
            shape_2d=self.initial_grid.shape_2d,
            pixel_coordinates=trough_pixels,
            pixel_scales=self.initial_grid.pixel_scales,
            sub_size=self.initial_grid.sub_size,
            origin=self.initial_grid.origin,
            buffer=2,
        )

        multiple_image_pixels = positions_at_coordinate_from(
            grid_2d=source_plane_grid.in_2d,
            coordinate=source_plane_coordinate,
            mask=trough_mask,
        )

        return list(
            map(
                trough_mask.geometry.scaled_coordinates_from_pixel_coordinates,
                multiple_image_pixels,
            )
        )

    def image_plane_positions_from(self, lensing_obj, source_plane_coordinate):

        deflections = lensing_obj.deflections_from_grid(grid=self.initial_grid)
        source_plane_grid = self.initial_grid.grid_from_deflection_grid(
            deflection_grid=deflections
        )

        source_plane_squared_distances = source_plane_grid.squared_distances_from_coordinate(
            coordinate=source_plane_coordinate
        )

        trough_pixels = trough_pixels_from(
            array_2d=source_plane_squared_distances.in_2d, mask=self.initial_grid.mask
        )

        trough_mask = msk.Mask.from_pixel_coordinates(
            shape_2d=self.initial_grid.shape_2d,
            pixel_coordinates=trough_pixels,
            pixel_scales=self.initial_grid.pixel_scales,
            sub_size=self.initial_grid.sub_size,
            origin=self.initial_grid.origin,
            buffer=1,
        )

        trough_grid = grids.Grid.from_mask(mask=trough_mask)

        deflections = lensing_obj.deflections_from_grid(grid=trough_grid)
        source_plane_grid = trough_grid.grid_from_deflection_grid(
            deflection_grid=deflections
        )

        source_plane_squared_distances = source_plane_grid.squared_distances_from_coordinate(
            coordinate=source_plane_coordinate
        )


@decorator_util.jit()
def grid_neighbors_1d_from(grid_1d, pixel_scale):

    y_shape = int((np.max(grid_1d[:, 0]) - np.min(grid_1d[:, 0]) / pixel_scale)) + 3
    x_shape = int((np.max(grid_1d[:, 1]) - np.min(grid_1d[:, 1]) / pixel_scale)) + 3

    mask_2d = np.full(shape=(y_shape, x_shape), fill_value=True)

    grid_pixel_centres_1d = grid_util.grid_pixel_centres_1d_from(
        grid_scaled_1d=grid_1d,
        shape_2d=(y_shape, x_shape),
        pixel_scales=(pixel_scale, pixel_scale),
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


@decorator_util.jit()
def peak_pixels_from(array_2d, mask=None):

    if mask is None:
        mask = np.full(fill_value=False, shape=array_2d.shape)

    peak_pixels = []

    for y in range(1, array_2d.shape[0] - 1):
        for x in range(1, array_2d.shape[1] - 1):
            if not mask[y, x]:
                if (
                    array_2d[y, x] > array_2d[y + 1, x]
                    and array_2d[y, x] > array_2d[y + 1, x + 1]
                    and array_2d[y, x] > array_2d[y, x + 1]
                    and array_2d[y, x] > array_2d[y - 1, x + 1]
                    and array_2d[y, x] > array_2d[y - 1, x]
                    and array_2d[y, x] > array_2d[y - 1, x - 1]
                    and array_2d[y, x] > array_2d[y, x - 1]
                    and array_2d[y, x] > array_2d[y + 1, x - 1]
                ):

                    peak_pixels.append([y, x])

    return peak_pixels


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
