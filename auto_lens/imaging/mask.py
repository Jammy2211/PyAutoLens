from auto_lens.imaging import data
import numpy as np


class Mask(data.DataGrid):

    # TODO : The mask class methods are a bit messy with how we use DataGrid to make them. Can this be done cleaner?

    @classmethod
    def empty_for_shape_arc_seconds_and_pixel_scale(cls, shape_arc_seconds, pixel_scale):
        return cls(np.zeros(map(lambda d: d / pixel_scale, shape_arc_seconds)), pixel_scale)

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

        for y in range(int(grid.shape[0])):
            for x in range(int(grid.shape[1])):
                x_arcsec, y_arcsec = grid.pixel_coordinates_to_arc_second_coordinates((x, y))

                x_arcsec -= centre[0]
                y_arcsec -= centre[1]

                radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

                grid[y, x] = radius_arcsec > radius_mask

        return grid

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

        for y in range(int(grid.shape[0])):
            for x in range(int(grid.shape[1])):
                x_arcsec, y_arcsec = grid.pixel_coordinates_to_arc_second_coordinates((x, y))

                x_arcsec -= centre[0]
                y_arcsec -= centre[1]

                radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

                grid[y, x] = radius_arcsec > outer_radius_mask or radius_arcsec < inner_radius_mask

        return grid

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
        return np.ma.make_mask_none(grid.shape)

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    def compute_grid_coords_image(self):
        """
        Compute the image grid_coords grids from a mask, using the center of every unmasked pixel.
        """
        coordinates = self.grid_coordinates

        pixels = self.pixels_in_mask

        grid = np.zeros(shape=(pixels, 2))
        pixel_count = 0

        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                if not self[y, x]:
                    grid[pixel_count, :] = coordinates[y, x]
                    pixel_count += 1

        return grid

    def compute_grid_coords_image_sub(self, grid_size_sub):
        """ Compute the image sub-grid_coords grids from a mask, using the center of every unmasked pixel.

        Parameters
        ----------
        grid_size_sub : int
            The (grid_size_sub x grid_size_sub) of the sub-grid_coords of each image pixel.
        """

        image_pixels = self.pixels_in_mask
        image_pixel_count = 0

        grid = np.zeros(shape=(image_pixels, grid_size_sub ** 2, 2))

        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                if not self[y, x]:
                    x_arcsec, y_arcsec = self.pixel_coordinates_to_arc_second_coordinates((x, y))

                    sub_pixel_count = 0

                    for y1 in range(grid_size_sub):
                        for x1 in range(grid_size_sub):
                            grid[image_pixel_count, sub_pixel_count, 0] = \
                                self.sub_pixel_to_coordinate(x1, x_arcsec, grid_size_sub)

                            grid[image_pixel_count, sub_pixel_count, 1] = \
                                self.sub_pixel_to_coordinate(y1, y_arcsec, grid_size_sub)

                            sub_pixel_count += 1

                    image_pixel_count += 1

        return grid

    def compute_grid_coords_blurring(self, psf_size):
        """ Compute the blurring grid_coords grids from a mask, using the center of every unmasked pixel.

        The blurring grid_coords contains all data_to_pixels which are not in the mask, but close enough to it that a
        fraction of their will be blurred into the mask region (and therefore they are needed for the analysis). They
        are located by scanning for all data_to_pixels which are outside the mask but within the psf size.

        Parameters
        ----------
        psf_size : (int, int)
           The size of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        blurring_mask = self.compute_blurring_mask(psf_size)

        return blurring_mask.compute_grid_coords_image()

    def compute_grid_data(self, grid_data):
        """Compute a data grid, which represents the data values of a data-set (e.g. an image, noise, in the mask.

        Parameters
        ----------
        grid_data

        """
        pixels = self.pixels_in_mask

        grid = np.zeros(shape=pixels)
        pixel_count = 0

        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                if not self[y, x]:
                    grid[pixel_count] = grid_data[y, x]
                    pixel_count += 1

        return grid

    def compute_grid_mapper_data_to_pixel(self):
        """
        Compute the mapping of every pixel in the mask to its 2D pixel coordinates.
        """
        pixels = self.pixels_in_mask

        grid = np.zeros(shape=(pixels, 2), dtype='int')
        pixel_count = 0

        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                if not self[y, x]:
                    grid[pixel_count, :] = y, x
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
        clustering_to_image : ndarray
            The mapping between every sparse clustering image pixel and image pixel, where each entry gives the 1D index
            of the image pixel in the mask.
        image_to_clustering : ndarray
            The mapping between every image pixel and its closest sparse clustering pixel, where each entry give the 1D
            index of the sparse pixel in sparse_pixel arrays.
        """

        sparse_mask = self.compute_sparse_uniform_mask(sparse_grid_size)
        print("sparse_mask = {}".format(sparse_mask))
        sparse_index_image = self.compute_sparse_index_image(sparse_mask)
        print("sparse_index_image = {}".format(sparse_index_image))
        sparse_to_image = self.compute_sparse_to_image(sparse_mask)
        print("sparse_to_image = {}".format(sparse_to_image))
        image_to_sparse = self.compute_image_to_sparse(sparse_mask, sparse_index_image)
        print("image_to_sparse = {}".format(image_to_sparse))

        return sparse_to_image, image_to_sparse

    def compute_grid_border(self):
        """Compute the border image data_to_pixels from a mask, where a border pixel is a pixel inside the mask but on its \
        edge, therefore neighboring a pixel with a *True* value.
        """

        # TODO : Border data_to_pixels for a circular mask and annulus mask are different (the inner annulus
        # TODO : data_to_pixels should be ignored. Should we turn this to classes for Masks?

        border_pixels = np.empty(0)
        image_pixel_index = 0

        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                if not self[y, x]:
                    if self[y + 1, x] == 1 or self[y - 1, x] == 1 or self[y, x + 1] == 1 or \
                            self[y, x - 1] == 1 or self[y + 1, x + 1] == 1 or self[y + 1, x - 1] == 1 \
                            or self[y - 1, x + 1] == 1 or self[y - 1, x - 1] == 1:
                        border_pixels = np.append(border_pixels, image_pixel_index)

                    image_pixel_index += 1

        return border_pixels

    def compute_blurring_mask(self, psf_size):
        """Compute the blurring mask, which represents all data_to_pixels not in the mask but close enough to it that a \
        fraction of their light will be blurring in the image.

        Parameters
        ----------
        psf_size : (int, int)
           The size of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        blurring_mask = np.ones(self.shape)

        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                if not self[y, x]:
                    for y1 in range((-psf_size[1] + 1) // 2, (psf_size[1] + 1) // 2):
                        for x1 in range((-psf_size[0] + 1) // 2, (psf_size[0] + 1) // 2):
                            if 0 <= y + y1 <= self.shape[0] - 1 \
                                    and 0 <= x + x1 <= self.shape[1] - 1:
                                if self[y + y1, x + x1]:
                                    blurring_mask[y + y1, x + x1] = False
                            else:
                                raise MaskException(
                                    "setup_blurring_mask extends beynod the size of the mask - pad the image"
                                    "before masking")

        return Mask(blurring_mask, self.pixel_scale)

    def compute_sparse_uniform_mask(self, sparse_grid_size):
        """Setup a two-dimensional sparse mask of image data_to_pixels, by keeping all image data_to_pixels which do not
        give a remainder when divided by the sub-grid_coords size. """
        sparse_mask = np.ones(self.shape)

        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                if not self[y, x]:
                    if x % sparse_grid_size == 0 and y % sparse_grid_size == 0:
                        sparse_mask[y, x] = False

        return Mask(sparse_mask, self.pixel_scale)

    def compute_sparse_index_image(self, sparse_mask):
        """Setup an image which, for each *False* entry in the sparse mask, puts the sparse pixel index in that pixel.

         This is used for computing the image_to_clustering vector, whereby each image pixel is paired to the sparse
         pixel in this image via a neighbor search."""

        sparse_index_2d = np.zeros(self.shape)
        sparse_pixel_index = 0

        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                if not sparse_mask[y, x]:
                    sparse_pixel_index += 1
                    sparse_index_2d[y, x] = sparse_pixel_index

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
        clustering_to_image : ndarray
            The mapping between every sparse clustering image pixel and image pixel, where each entry gives the 1D index
            of the image pixel in the self.
        """
        sparse_to_image = np.empty(0)
        image_pixel_index = 0

        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                print("for {}:{}".format(y, x))

                if not sparse_mask[y, x]:
                    sparse_to_image = np.append(sparse_to_image, image_pixel_index)

                if not self[y, x]:
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
        image_to_clustering : ndarray
            The mapping between every image pixel and its closest sparse clustering pixel, where each entry give the 1D
            index of the sparse pixel in sparse_pixel arrays.

        """
        image_to_sparse = np.empty(0)

        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                if not self[y, x]:
                    iboarder = 0
                    pixel_match = False
                    while not pixel_match:
                        for y1 in range(y - iboarder, y + iboarder + 1):
                            for x1 in range(x - iboarder, x + iboarder + 1):
                                if 0 <= y1 < self.shape[0] and 0 <= x1 < self.shape[1]:
                                    if not sparse_mask[y1, x1] and not pixel_match:
                                        image_to_sparse = np.append(image_to_sparse, sparse_index_image[y1, x1] - 1)
                                        pixel_match = True

                        iboarder += 1
                        if iboarder == 100:
                            raise MaskException('compute_image_to_sparse - Stuck in infinite loop')

        return image_to_sparse


class MaskException(Exception):
    pass
