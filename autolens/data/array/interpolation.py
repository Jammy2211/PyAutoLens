import numpy as np

from autolens.data.array import grids, mask as msk


# TODO : Think carefully about demagnified centra pixels.


class InterpolationGeometry(object):

    def __init__(self, y_min, y_max, x_min, x_max, y_pixel_scale, x_pixel_scale):
        """The geometry of a rectangular grid, defining where the grids top-left, top-right, bottom-left and \
        bottom-right corners are in arc seconds. The arc-second size of each rectangular pixel is also computed.

        Parameters
        -----------

        """
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max
        self.x_size = self.x_max - self.x_min
        self.y_size = self.y_max - self.y_min
        self.y_pixel_scale = y_pixel_scale
        self.x_pixel_scale = x_pixel_scale
        self.x_start = self.x_min - self.x_pixel_scale / 2.0
        self.y_start = self.y_min - self.y_pixel_scale / 2.0

    def new_from_x_and_y_scale(self, x_scale, y_scale):
        return InterpolationGeometry(y_min=self.y_min * y_scale, y_max=self.y_max * y_scale,
                                     x_min=self.x_min * x_scale, x_max=self.x_max * x_scale,
                                     y_pixel_scale=self.y_pixel_scale * y_scale,
                                     x_pixel_scale=self.x_pixel_scale * x_scale)


class InterpolationScheme(object):

    def __init__(self, shape, image_coords, image_pixel_scale):
        """An interpolation scheme which enables efficient deflection angle computation. This works by computing a \
        sub-set of deflection angles on a uniform regular grid and bilinearly interpolating these values to determine \
        the masked_image-grid and sub-grid deflection angle values.

        This class describes the interpolation scheme that is used to do this, for howtolens the shape of the \
        interpolation-grid and the neighbors of every pixel on the grid in the 4 directions required for bilinear \
        interpolation.

        The interpolation scheme can be performed on masked_image and sub grids that are non-uniform (e.g. after tracing them \
        from the masked_image-plane to a source-plane). The x_pixel and y_pixel bounds of the new grids are used to scale the \
        interpolation scheme to the new plane.

        Parameters
        -----------
        shape : (int, int)
            The shape of the interpolation schemes interpolation-grid.
        image_coords : ndarray
            The masked_image-plane coordinates of each pixel on the interpolation-grid.
        """

        def bottom_right_neighbors():
            """For each pixel on the deflection-interpolation grid, compute pixels directly neighboring each pixel \
            to their right, downwards and down-right.

            These are the pixels bilinear interpolation will be performed using if a deflection angle lands in the \
            bottom-right quadrant of a pixel."""

            down_right_neighbors = -1 * np.ones((self.pixels, 3), dtype='int')

            for y_pixel in range(self.shape[0]):
                for x_pixel in range(self.shape[1]):

                    pixel_index = y_pixel * self.shape[1] + x_pixel

                    if x_pixel < self.shape[1] - 1:
                        down_right_neighbors[pixel_index, 0] = pixel_index + 1

                    if y_pixel < self.shape[0] - 1:
                        down_right_neighbors[pixel_index, 1] = pixel_index + self.shape[1]

                    if x_pixel < self.shape[1] - 1 and y_pixel < self.shape[0] - 1:
                        down_right_neighbors[pixel_index, 2] = pixel_index + self.shape[1] + 1

            return down_right_neighbors

        def bottom_left_neighbors():
            """For each pixel on the deflection-interpolation grid, compute pixels directly neighboring each pixel \
            to their left, downwards and down-left.

            These are the pixels bilinear interpolation will be performed using if a deflection angle lands in the \
            bottom-left quadrant of a pixel."""

            down_left_neighbors = -1 * np.ones((self.pixels, 3), dtype='int')

            for y_pixel in range(self.shape[0]):
                for x_pixel in range(self.shape[1]):

                    pixel_index = y_pixel * self.shape[1] + x_pixel

                    if x_pixel > 0:
                        down_left_neighbors[pixel_index, 0] = pixel_index - 1

                    if x_pixel > 0 and y_pixel < self.shape[0] - 1:
                        down_left_neighbors[pixel_index, 1] = pixel_index + self.shape[1] - 1

                    if y_pixel < self.shape[0] - 1:
                        down_left_neighbors[pixel_index, 2] = pixel_index + self.shape[1]

            return down_left_neighbors

        def top_right_neighbors():
            """For each pixel on the deflection-interpolation grid, compute pixels directly neighboring each pixel \
            to their right, upwards and up-right.

            These are the pixels bilinear interpolation will be performed using if a deflection angle lands in the \
            top-right quadrant of a pixel."""
            up_right_neighbors = -1 * np.ones((self.pixels, 3), dtype='int')

            for y_pixel in range(self.shape[0]):
                for x_pixel in range(self.shape[1]):

                    pixel_index = y_pixel * self.shape[1] + x_pixel

                    if y_pixel > 0:
                        up_right_neighbors[pixel_index, 0] = pixel_index - self.shape[1]

                    if x_pixel < self.shape[1] - 1 and y_pixel > 0:
                        up_right_neighbors[pixel_index, 1] = pixel_index - self.shape[1] + 1

                    if x_pixel < self.shape[1] - 1:
                        up_right_neighbors[pixel_index, 2] = pixel_index + 1

            return up_right_neighbors

        def top_left_neighbors():
            """For each pixel on the deflection-interpolation grid, compute pixels directly neighboring each pixel \
            to their left, upwards and up-left.

            These are the pixels bilinear interpolation will be performed using if a deflection angle lands in the \
            top-left quadrant of a pixel."""
            up_left_neighbors = -1 * np.ones((self.pixels, 3), dtype='int')

            for y_pixel in range(self.shape[0]):
                for x_pixel in range(self.shape[1]):

                    pixel_index = y_pixel * self.shape[1] + x_pixel

                    if x_pixel > 0 and y_pixel > 0:
                        up_left_neighbors[pixel_index, 0] = pixel_index - self.shape[1] - 1

                    if y_pixel > 0:
                        up_left_neighbors[pixel_index, 1] = pixel_index - self.shape[1]

                    if x_pixel > 0:
                        up_left_neighbors[pixel_index, 2] = pixel_index - 1

            return up_left_neighbors

        self.image_coords = image_coords
        self.image_pixel_scale = image_pixel_scale

        self.geometry = InterpolationGeometry(y_min=np.min(image_coords[:, 1]), y_max=np.max(image_coords[:, 1]),
                                              x_min=np.min(image_coords[:, 0]), x_max=np.max(image_coords[:, 0]),
                                              y_pixel_scale=image_pixel_scale, x_pixel_scale=image_pixel_scale)

        self.shape = shape
        self.pixels = self.shape[0] * self.shape[1]

        self.bottom_right_neighbors = bottom_right_neighbors()
        self.bottom_left_neighbors = bottom_left_neighbors()
        self.top_right_neighbors = top_right_neighbors()
        self.top_left_neighbors = top_left_neighbors()

    @classmethod
    def from_mask(cls, mask, shape):
        """Determine the interpolation scheme from an masked_image-masks. This uses the x / y_pixel bounds of the masks to setup the \
        grid 'over' the masks, padded by the pixel-scale to ensure edge pixels have their deflection angles interpolated \
        correctly.

        Parameters
        -----------
        mask: msk.Mask
            The masks the interpolation scheme is generated based on.
        shape : (int, int)
            The shape of the interpolation schemes interpolation-grid.
        """
        image_grid = grids.RegularGrid.from_mask(mask)

        x_max = np.max(image_grid[:, 0]) + mask.pixel_scale
        x_min = np.min(image_grid[:, 0]) - mask.pixel_scale
        y_max = np.max(image_grid[:, 1]) + mask.pixel_scale
        y_min = np.min(image_grid[:, 1]) - mask.pixel_scale

        image_coords = np.zeros((shape[0] * shape[1], 2))

        for y_pixel in range(shape[0]):
            for x_pixel in range(shape[1]):
                pixel_index = y_pixel * shape[1] + x_pixel

                image_coords[pixel_index, 1] = x_min + 2.0 * (x_pixel / (shape[1] - 1)) * x_max
                image_coords[pixel_index, 0] = y_min + 2.0 * (y_pixel / (shape[0] - 1)) * y_max

        return InterpolationScheme(shape, image_coords, image_pixel_scale=mask.pixel_scale)

    def interpolation_coordinates_from_sizes(self, new_x_size, new_y_size):
        """Setup a set of interpolation coordinates, which represent a uniform-grid of coordinates which will be used \
        for deflection angle interpolation. These coordinates use the interpolation-scheme."""

        x_scale = new_x_size / self.geometry.x_size
        y_scale = new_y_size / self.geometry.y_size

        new_geometry = self.geometry.new_from_x_and_y_scale(x_scale, y_scale)

        interp_coords = np.zeros((self.pixels, 2), dtype='float64')

        interp_coords[:, 0] = self.image_coords[:, 0] * x_scale
        interp_coords[:, 1] = self.image_coords[:, 1] * y_scale

        return InterpolationCoordinates(array=interp_coords, geometry=new_geometry, scheme=self)


class InterpolationCoordinates(np.ndarray):

    def __new__(cls, array, geometry, scheme, *args, **kwargs):
        coords = np.array(array).view(cls)
        coords.geometry = geometry
        coords.scheme = scheme
        return coords

    def apply_function(self, func):
        return InterpolationDeflections(func(self), self, self.geometry, self.scheme)

    def interpolation_deflections_from_coordinates_and_galaxies(self, galaxies):
        def calculate_deflections(grid):
            return sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))

        return self.apply_function(calculate_deflections)


class InterpolationDeflections(np.ndarray):

    def __new__(cls, array, coords, geometry, scheme, *args, **kwargs):
        defls = np.array(array).view(cls)
        defls.interp_coords = coords
        defls.geometry = geometry
        defls.scheme = scheme
        return defls

    def grid_to_interp_from_grid(self, grid):

        y_pixels = np.floor((grid[:, 1] - self.geometry.y_start) / self.geometry.y_pixel_scale)
        x_pixels = np.floor((grid[:, 0] - self.geometry.x_start) / self.geometry.x_pixel_scale)

        return np.floor((x_pixels * self.scheme.shape[1]) + y_pixels)

    def interpolate_values_from_grid(self, grid):

        grid_to_interp = self.grid_to_interp_from_grid(grid)

        interpolated = np.zeros(grid.shape)

        for i in range(grid.shape[0]):

            interp_index = grid_to_interp[i]

            print(interp_index)
            print(self.interp_coords[interp_index, 0])

            if grid[i, 0] < self.interp_coords[interp_index, 0]:
                if grid[i, 1] < self.interp_coords[interp_index, 1]:
                    interpolated[i, 0] = self.interpolate_in_top_left_of_pixel(grid[i, 0], grid[i, 1], interp_index,
                                                                               self[:, 0])
                    interpolated[i, 1] = self.interpolate_in_top_left_of_pixel(grid[i, 0], grid[i, 1], interp_index,
                                                                               self[:, 1])

        return interpolated

    def interpolate_in_top_left_of_pixel(self, x, y, interp_index, deflections):

        bottom_right_index = interp_index
        top_left_index = self.scheme.top_left_neighbors[interp_index, 0]
        top_right_index = self.scheme.top_left_neighbors[interp_index, 1]
        bottom_left_index = self.scheme.top_left_neighbors[interp_index, 2]
        x0 = self.interp_coords[top_left_index, 0]
        x1 = self.interp_coords[top_right_index, 0]
        y0 = self.interp_coords[top_left_index, 1]
        y1 = self.interp_coords[bottom_left_index, 1]
        weight0 = ((x1 - x) / (x1 - x0)) * deflections[bottom_left_index] + ((x - x0) / (x1 - x0)) * deflections[
            bottom_right_index]
        weight1 = ((x1 - x) / (x1 - x0)) * deflections[bottom_left_index] + ((x - x0) / (x1 - x0)) * deflections[
            top_right_index]
        return ((y1 - y) / (y1 - y0)) * weight0 + ((y - y0) / (y1 - y0)) * weight1
