from autoarray import decorator_util
import numpy as np
from autoarray import mask as msk
from autoarray.util import mask_util
from autoarray.structures import grids
from autoarray.mask import mask as msk


class PositionsSolver:
    def __init__(self, initial_grid):

        self.initial_grid = initial_grid.in_1d_binned

    def mask_within_circle_from(
        self, lensing_obj, source_plane_coordinate, radius, grid=None
    ):

        if grid is None:
            grid = self.initial_grid

        deflections = lensing_obj.deflections_from_grid(grid=grid)
        source_plane_grid = grid.grid_from_deflection_grid(deflection_grid=deflections)
        source_plane_squared_distances = source_plane_grid.squared_distances_from_coordinate(
            coordinate=source_plane_coordinate
        )
        mask_within_circle = source_plane_squared_distances.in_2d < radius ** 2.0

        return msk.Mask.manual(
            mask=mask_within_circle, pixel_scales=grid.pixel_scales, invert=True
        )

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
