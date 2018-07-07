from src.imaging import scaled_array
from src.imaging import grids
from src import exc
import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask(scaled_array.ScaledArray):

    @classmethod
    def empty_for_shape_arc_seconds_and_pixel_scale(cls, shape_arc_seconds, pixel_scale):
        return cls(np.full(tuple(map(lambda d: int(d / pixel_scale), shape_arc_seconds)), True), pixel_scale)

    @classmethod
    def circular(cls, shape_arc_seconds, pixel_scale, radius_mask, centre=(0., 0.)):
        """
        Setup the mask as a circle, using a specified arc second radius.

        Parameters
        ----------
        shape_arc_seconds: (float, float)
            The (x,y) dimensions_2d
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        radius_mask : float
            The radius of the circular mask in arc seconds.
        centre: (float, float)
            The centre of the mask.
        """

        grid = Mask.empty_for_shape_arc_seconds_and_pixel_scale(shape_arc_seconds, pixel_scale)

        for x in range(int(grid.shape[0])):
            for y in range(int(grid.shape[1])):
                x_arcsec, y_arcsec = grid.pixel_coordinates_to_arc_second_coordinates((x, y))

                x_arcsec -= centre[0]
                y_arcsec -= centre[1]

                radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

                grid[x, y] = radius_arcsec > radius_mask

        return cls(grid, pixel_scale)

    @classmethod
    def annular(cls, shape_arc_seconds, pixel_scale, inner_radius_mask, outer_radius_mask, centre=(0., 0.)):
        """
        Setup the mask as a circle, using a specified inner and outer radius in arc seconds.

        Parameters
        ----------
        shape_arc_seconds : (float, float)
            The (x,y) dimensions_2d of the mask
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        inner_radius_mask : float
            The inner radius of the annulus mask in arc seconds.
        outer_radius_mask : float
            The outer radius of the annulus mask in arc seconds.
        centre: (float, float)
            The centre of the mask.
        """

        grid = Mask.empty_for_shape_arc_seconds_and_pixel_scale(shape_arc_seconds, pixel_scale)

        for x in range(int(grid.shape[0])):
            for y in range(int(grid.shape[1])):
                x_arcsec, y_arcsec = grid.pixel_coordinates_to_arc_second_coordinates((x, y))

                x_arcsec -= centre[0]
                y_arcsec -= centre[1]

                radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

                grid[x, y] = radius_arcsec > outer_radius_mask or radius_arcsec < inner_radius_mask

        return cls(grid, pixel_scale)

    @classmethod
    def unmasked(cls, shape_arc_seconds, pixel_scale):
        """
        Setup the mask such that all values are unmasked, thus corresponding to the entire image.

        Parameters
        ----------
        shape_arc_seconds : (float, float)
            The (x,y) dimensions_2d of the mask
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        grid = Mask.empty_for_shape_arc_seconds_and_pixel_scale(shape_arc_seconds, pixel_scale)
        return cls(np.ma.make_mask_none(grid.shape), pixel_scale)

    @classmethod
    def for_simulate(cls, shape_arc_seconds, pixel_scale, psf_size):

        if psf_size[0] % 2 == 0 or psf_size[1] % 2 == 0 or psf_size[0] != psf_size[1]:
            raise exc.KernelException("PSF Kernel must be odd and square")

        ma = cls.unmasked(shape_arc_seconds, pixel_scale)
        pad_size = (int(psf_size[0] / 2) + 1, int(psf_size[1] / 2) + 1)
        return ma.pad(new_dimensions=(ma.shape[0] + pad_size[0], ma.shape[1] + pad_size[1]), pad_value=1)

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    @property
    def coordinate_grid(self):
        """
        Compute the image grid_coords grids from a mask, using the center of every unmasked pixel.
        """
        coordinates = self.grid_coordinates

        pixels = self.pixels_in_mask

        grid = np.zeros(shape=(pixels, 2))
        pixel_count = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    grid[pixel_count, :] = coordinates[x, y]
                    pixel_count += 1

        return grids.CoordinateGrid(grid)

    def coordinates_collection_for_subgrid_size_and_blurring_shape(self, sub_grid_size, blurring_shape):
        image = self.coordinate_grid
        sub = self.sub_coordinate_grid_with_size(sub_grid_size)
        blurring = self.blurring_coordinate_grid(blurring_shape)

        return grids.CoordsCollection(image, sub, blurring)

    def sub_coordinate_grid_with_size(self, size):
        """ Compute the image sub-grid_coords grids from a mask, using the center of every unmasked pixel.

        Parameters
        ----------
        size : int
            The (grid_size_sub x grid_size_sub) of the sub-grid_coords of each image pixel.
        """

        image_pixels = self.pixels_in_mask
        image_pixel_count = 0

        grid = np.zeros(shape=(image_pixels * size ** 2, 2))

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    x_arcsec, y_arcsec = self.pixel_coordinates_to_arc_second_coordinates((x, y))

                    sub_pixel_count = 0

                    for x1 in range(size):
                        for y1 in range(size):
                            grid[image_pixel_count * size ** 2 + sub_pixel_count, 0] = \
                                self.sub_pixel_to_coordinate(x1, x_arcsec, size)

                            grid[image_pixel_count * size ** 2 + sub_pixel_count, 1] = \
                                self.sub_pixel_to_coordinate(y1, y_arcsec, size)

                            sub_pixel_count += 1

                    image_pixel_count += 1

        return grids.SubCoordinateGrid(grid, size)

    def blurring_coordinate_grid(self, psf_size):
        """ Compute the blurring grid_coords grids from a mask, using the center of every unmasked pixel.

        The blurring grid_coords contains all data_to_pixels which are not in the mask, but close enough to it that a
        fraction of their will be blurred into the mask region (and therefore they are needed for the analysis). They
        are located by scanning for all data_to_pixels which are outside the mask but within the psf size.

        Parameters
        ----------
        psf_size : (int, int)
           The size of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        if psf_size[0] % 2 == 0 or psf_size[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = self.blurring_mask_for_kernel_shape(psf_size)

        return blurring_mask.coordinate_grid

    def compute_grid_data(self, grid_data):
        """Compute a data grid, which represents the data values of a data-set (e.g. an image, noise, in the mask.

        Parameters
        ----------
        grid_data

        """
        pixels = self.pixels_in_mask

        grid = np.zeros(shape=pixels)
        pixel_count = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    grid[pixel_count] = grid_data[x, y]
                    pixel_count += 1

        return grid

    def compute_grid_mapper_data_to_pixel(self):
        """
        Compute the mapping of every pixel in the mask to its 2D pixel coordinates.
        """
        pixels = self.pixels_in_mask

        grid = np.zeros(shape=(pixels, 2), dtype='int')
        pixel_count = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    grid[pixel_count, :] = x, y
                    pixel_count += 1

        return grid

    def compute_grid_mapper_sparse(self, sparse_grid_size):
        """Given an image.Mask, compute the sparse cluster image data_to_pixels, defined as the sub-set of
        image-data_to_pixels used to perform KMeans clustering (this is used purely for speeding up the KMeans
        clustering algorithm).

        This sparse grid_coords is a uniform subsample of the masked image and is computed by only including image
        data_to_pixels which, when divided by the sparse_grid_size, do not give a remainder.

        Parameters
        ----------
        sparse_grid_size : int
            The spacing of the sparse image pixel grid_coords (e.g. a value of 2 will compute a sparse grid_coords of
            data_to_pixels which are two data_to_pixels apart)

        Returns
        -------
        cluster_to_image : ndarray
            The mapping between every sparse clustering image pixel and image pixel, where each entry gives the 1D index
            of the image pixel in the mask.
        image_to_cluster : ndarray
            The mapping between every image pixel and its closest sparse clustering pixel, where each entry give the 1D
            index of the sparse pixel in sparse_pixel arrays.
        """

        sparse_mask = self.compute_sparse_uniform_mask(sparse_grid_size)
        logger.debug("sparse_mask = {}".format(sparse_mask))
        sparse_index_image = self.compute_sparse_index_image(sparse_mask)
        logger.debug("sparse_index_image = {}".format(sparse_index_image))
        sparse_to_image = self.compute_sparse_to_image(sparse_mask)
        logger.debug("sparse_to_image = {}".format(sparse_to_image))
        image_to_sparse = self.compute_image_to_sparse(sparse_mask, sparse_index_image)
        logger.debug("image_to_sparse = {}".format(image_to_sparse))

        return sparse_to_image, image_to_sparse

    def compute_grid_border(self):
        """Compute the border image data_to_pixels from a mask, where a border pixel is a pixel inside the mask but on its \
        edge, therefore neighboring a pixel with a *True* value.
        """

        # TODO : Border data_to_pixels for a circular mask and annulus mask are different (the inner annulus
        # TODO : data_to_pixels should be ignored. Should we turn this to classes for Masks?

        border_pixels = np.empty(0)
        image_pixel_index = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    if self[x + 1, y] == 1 or self[x - 1, y] == 1 or self[x, y + 1] == 1 or \
                            self[x, y - 1] == 1 or self[x + 1, y + 1] == 1 or self[x + 1, y - 1] == 1 \
                            or self[x - 1, y + 1] == 1 or self[x - 1, y - 1] == 1:
                        border_pixels = np.append(border_pixels, image_pixel_index)

                    image_pixel_index += 1

        return border_pixels

    def blurring_mask_for_kernel_shape(self, kernel_shape):
        """Compute the blurring mask, which represents all data_to_pixels not in the mask but close enough to it that a
        fraction of their light will be blurring in the image.

        Parameters
        ----------
        kernel_shape : (int, int)
           The size of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        blurring_mask = np.full(self.shape, True)

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    for y1 in range((-kernel_shape[1] + 1) // 2, (kernel_shape[1] + 1) // 2):
                        for x1 in range((-kernel_shape[0] + 1) // 2, (kernel_shape[0] + 1) // 2):
                            if 0 <= x + x1 <= self.shape[0] - 1 \
                                    and 0 <= y + y1 <= self.shape[1] - 1:
                                if self[x + x1, y + y1]:
                                    blurring_mask[x + x1, y + y1] = False
                            else:
                                raise exc.MaskException(
                                    "setup_blurring_mask extends beyond the size of the mask - pad the image"
                                    "before masking")

        return Mask(blurring_mask, self.pixel_scale)

    def compute_sparse_uniform_mask(self, sparse_grid_size):
        """Setup a two-dimensional sparse mask of image data_to_pixels, by keeping all image data_to_pixels which do not
        give a remainder when divided by the sub-grid_coords size. """
        sparse_mask = np.full(self.shape, True)

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    if x % sparse_grid_size == 0 and y % sparse_grid_size == 0:
                        sparse_mask[x, y] = False

        return Mask(sparse_mask, self.pixel_scale)

    def compute_sparse_index_image(self, sparse_mask):
        """Setup an image which, for each *False* entry in the sparse mask, puts the sparse pixel index in that pixel.

         This is used for computing the image_to_cluster vector, whereby each image pixel is paired to the sparse
         pixel in this image via a neighbor search."""

        sparse_index_2d = np.zeros(self.shape, dtype=int)
        sparse_pixel_index = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not sparse_mask[x, y]:
                    sparse_pixel_index += 1
                    sparse_index_2d[x, y] = sparse_pixel_index

        return sparse_index_2d

    def compute_sparse_to_image(self, sparse_mask):
        """Compute the mapping of each sparse image pixel to its closest image pixel, defined using a mask of image \
        data_to_pixels.

        Parameters
        ----------
        sparse_mask : ndarray
            The two-dimensional boolean image of the sparse grid_coords.

        Returns
        -------
        cluster_to_image : ndarray
            The mapping between every sparse clustering image pixel and image pixel, where each entry gives the 1D index
            of the image pixel in the self.
        """
        sparse_to_image = np.empty(0, dtype=int)
        image_pixel_index = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):

                if not sparse_mask[x, y]:
                    sparse_to_image = np.append(sparse_to_image, image_pixel_index)

                if not self[x, y]:
                    image_pixel_index += 1

        return sparse_to_image

    def compute_image_to_sparse(self, sparse_mask, sparse_index_image):
        """Compute the mapping between every image pixel in the mask and its closest sparse clustering pixel.

        This is performed by going to each image pixel in the *mask*, and pairing it with its nearest neighboring pixel
        in the *sparse_mask*. The index of the *sparse_mask* pixel is drawn from the *sparse_index_image*. This
        neighbor search continue grows larger and larger around a pixel, until a pixel contained in the *sparse_mask* is
        successfully found.

        Parameters
        ----------
        sparse_index_image
        sparse_mask : ndarray
            The two-dimensional boolean image of the sparse grid_coords.

        Returns
        -------
        image_to_cluster : ndarray
            The mapping between every image pixel and its closest sparse clustering pixel, where each entry give the 1D
            index of the sparse pixel in sparse_pixel arrays.

        """
        image_to_sparse = np.empty(0, dtype=int)

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    iboarder = 0
                    pixel_match = False
                    while not pixel_match:
                        for x1 in range(x - iboarder, x + iboarder + 1):
                            for y1 in range(y - iboarder, y + iboarder + 1):
                                if 0 <= x1 < self.shape[0] and 0 <= y1 < self.shape[1]:
                                    if not sparse_mask[x1, y1] and not pixel_match:
                                        image_to_sparse = np.append(image_to_sparse, sparse_index_image[x1, y1] - 1)
                                        pixel_match = True

                        iboarder += 1
                        if iboarder == 100:
                            raise exc.MaskException('compute_image_to_sparse - Stuck in infinite loop')

        return image_to_sparse
