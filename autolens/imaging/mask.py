from autolens.imaging import scaled_array
from autolens import exc
import numpy as np
from functools import wraps
import inspect

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)


class Memoizer(object):
    def __init__(self):
        """
        Class to store the results of a function given a set of inputs.
        """
        self.results = {}
        self.calls = 0
        self.arg_names = None

    def __call__(self, func):
        """
        Memoize decorator. Any time a function is called that a memoizer has been attached to its results are stored in
        the results dictionary or retrieved from the dictionary if the function has already been called with those
        arguments.

        Note that the same memoizer persists over all instances of a class. Any state for a given instance that is not
        given in the representation of that instance will be ignored. That is, it is possible that the memoizer will
        give incorrect results if instance state does not affect __str__ but does affect the value returned by the
        memoized method.

        Parameters
        ----------
        func: function
            A function for which results should be memoized

        Returns
        -------
        decorated: function
            A function that memoizes results
        """
        if self.arg_names is not None:
            raise AssertionError("Instantiate a new Memoizer for each function")
        self.arg_names = inspect.getfullargspec(func).args

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = ", ".join(
                ["('{}', {})".format(arg_name, arg) for arg_name, arg in
                 list(zip(self.arg_names, args)) + [(k, v) for k, v in kwargs.items()]])
            if key not in self.results:
                self.calls += 1
            self.results[key] = func(*args, **kwargs)
            return self.results[key]

        return wrapper


class Mask(scaled_array.ScaledArray):
    """
    A mask represented by an ndarray where True is masked.
    """

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
            The (x,y) image_shape
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        radius_mask : float
            The radius of the circular mask in arc seconds.
        centre: (float, float)
            The centre of the mask.
        """

        grid = Mask.empty_for_shape_arc_seconds_and_pixel_scale(shape_arc_seconds, pixel_scale)

        def fill_grid(x, y):
            x_arcsec, y_arcsec = grid.pixel_coordinates_to_arc_second_coordinates((x, y))

            x_arcsec -= centre[0]
            y_arcsec -= centre[1]

            radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            grid[x, y] = radius_arcsec > radius_mask

        grid.map(fill_grid)

        return cls(grid, pixel_scale)

    @classmethod
    def annular(cls, shape_arc_seconds, pixel_scale, inner_radius, outer_radius, centre=(0., 0.)):
        """
        Setup the mask as a circle, using a specified inner and outer radius in arc seconds.

        Parameters
        ----------
        shape_arc_seconds : (float, float)
            The (x,y) image_shape of the mask
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        inner_radius : float
            The inner radius of the annulus mask in arc seconds.
        outer_radius : float
            The outer radius of the annulus mask in arc seconds.
        centre: (float, float)
            The centre of the mask.
        """

        grid = Mask.empty_for_shape_arc_seconds_and_pixel_scale(shape_arc_seconds, pixel_scale)

        def fill_grid(x, y):
            x_arcsec, y_arcsec = grid.pixel_coordinates_to_arc_second_coordinates((x, y))

            x_arcsec -= centre[0]
            y_arcsec -= centre[1]

            radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

            grid[x, y] = radius_arcsec > outer_radius or radius_arcsec < inner_radius

        grid.map(fill_grid)

        return cls(grid, pixel_scale)

    @classmethod
    def unmasked(cls, shape_arc_seconds, pixel_scale):
        """
        Setup the mask such that all values are unmasked, thus corresponding to the entire image.

        Parameters
        ----------
        shape_arc_seconds : (float, float)
            The (x,y) image_shape of the mask
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
        return ma.pad(new_dimensions=(ma.shape[0] + psf_size[0] - 1, ma.shape[1] + psf_size[1] - 1), pad_value=1)

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

        return ImageGrid(grid)

    def map_to_1d(self, grid_data):
        """Compute a data grid, which represents the data values of a data-set (e.g. an image, noise, in the mask.

        Parameters
        ----------
        grid_data: ndarray | float | None

        """
        if grid_data is None or isinstance(grid_data, float):
            return grid_data

        pixels = self.pixels_in_mask

        grid = np.zeros(shape=pixels)
        pixel_count = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    grid[pixel_count] = grid_data[x, y]
                    pixel_count += 1

        return grid

    def grid_to_pixel(self):
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

    def __getitem__(self, coords):
        try:
            return super(Mask, self).__getitem__(coords)
        except IndexError:
            return True

    @property
    def border_pixel_indices(self):
        """Compute the border image data_to_pixels from a mask, where a border pixel is a pixel inside the mask but on
        its edge, therefore neighboring a pixel with a *True* value.
        """

        border_pixels = np.empty(0, dtype='int')
        image_pixel_index = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    if self[x + 1, y] or self[x - 1, y] or self[x, y + 1] or \
                            self[x, y - 1] or self[x + 1, y + 1] or self[x + 1, y - 1] \
                            or self[x - 1, y + 1] or self[x - 1, y - 1]:
                        border_pixels = np.append(border_pixels, image_pixel_index)

                    image_pixel_index += 1

        return border_pixels

    def border_sub_pixel_indices(self, sub_grid_size):
        """Compute the border image data_to_pixels from a mask, where a border pixel is a pixel inside the mask but on
        its edge, therefore neighboring a pixel with a *True* value.
        """

        border_sub_pixels = np.empty(0, dtype='int')
        image_pixel_index = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    if self[x + 1, y] or self[x - 1, y] or self[x, y + 1] or \
                            self[x, y - 1] or self[x + 1, y + 1] or self[x + 1, y - 1] \
                            or self[x - 1, y + 1] or self[x - 1, y - 1]:

                        x_arcsec, y_arcsec = self.pixel_coordinates_to_arc_second_coordinates((x, y))
                        sub_grid = np.zeros((sub_grid_size ** 2, 2))
                        sub_pixel_count = 0

                        for x1 in range(sub_grid_size):
                            for y1 in range(sub_grid_size):
                                sub_grid[sub_pixel_count, 0] = self.sub_pixel_to_coordinate(x1, x_arcsec, sub_grid_size)
                                sub_grid[sub_pixel_count, 1] = self.sub_pixel_to_coordinate(y1, y_arcsec, sub_grid_size)
                                sub_pixel_count += 1

                        sub_grid_radii = np.add(np.square(sub_grid[:, 0]), np.square(sub_grid[:, 1]))
                        border_sub_pixel_index = image_pixel_index * (sub_grid_size ** 2) + np.argmax(sub_grid_radii)
                        border_sub_pixels = np.append(border_sub_pixels, border_sub_pixel_index)

                    image_pixel_index += 1

        return border_sub_pixels

    @Memoizer()
    def blurring_mask_for_kernel_shape(self, kernel_shape):
        """Compute the blurring mask, which represents all data_to_pixels not in the mask but close enough to it that a
        fraction of their light will be blurring in the image.

        Parameters
        ----------
        kernel_shape : (int, int)
           The sub_grid_size of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = np.full(self.shape, True)

        def fill_grid(x, y):
            if not self[x, y]:
                for y1 in range((-kernel_shape[1] + 1) // 2, (kernel_shape[1] + 1) // 2):
                    for x1 in range((-kernel_shape[0] + 1) // 2, (kernel_shape[0] + 1) // 2):
                        if 0 <= x + x1 <= self.shape[0] - 1 and 0 <= y + y1 <= self.shape[1] - 1:
                            if self[x + x1, y + y1]:
                                blurring_mask[x + x1, y + y1] = False
                        else:
                            raise exc.MaskException(
                                "setup_blurring_mask extends beyond the sub_grid_size of the mask - pad the image"
                                "before masking")

        self.map(fill_grid)

        return Mask(blurring_mask, self.pixel_scale)

    def map_to_2d(self, data):
        """Use mapper to map an input data-set from a *GridData* to its original 2D image.
        Parameters
        -----------
        data : ndarray
            The grid-data which is mapped to its 2D image.
        """
        data_2d = np.zeros(self.shape)

        for (i, pixel) in enumerate(self.grid_to_pixel()):
            data_2d[pixel[0], pixel[1]] = data[i]

        return data_2d


class SparseMask(Mask):
    def __new__(cls, mask, sparse_grid_size, *args, **kwargs):
        sparse_mask = np.full(mask.shape, True)

        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if not mask[x, y]:
                    if x % sparse_grid_size == 0 and y % sparse_grid_size == 0:
                        sparse_mask[x, y] = False

        return np.array(sparse_mask).view(cls)

    def __init__(self, mask, sparse_grid_size):
        super().__init__(mask)
        self.mask = mask
        self.sparse_grid_size = sparse_grid_size

    @property
    @Memoizer()
    def index_image(self):
        """
        Setup an image which, for each *False* entry in the sparse mask, puts the sparse pixel index in that pixel.

         This is used for computing the image_to_cluster vector, whereby each image pixel is paired to the sparse
         pixel in this image via a neighbor search."""

        sparse_index_2d = np.zeros(self.shape, dtype=int)
        sparse_pixel_index = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    sparse_pixel_index += 1
                    sparse_index_2d[x, y] = sparse_pixel_index

        return sparse_index_2d

    @property
    @Memoizer()
    def sparse_to_image(self):
        """
        Compute the mapping of each sparse image pixel to its closest image pixel, defined using a mask of image \
        data_to_pixels.

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

                if not self[x, y]:
                    sparse_to_image = np.append(sparse_to_image, image_pixel_index)

                if not self.mask[x, y]:
                    image_pixel_index += 1

        return sparse_to_image

    @property
    @Memoizer()
    def image_to_sparse(self):
        """Compute the mapping between every image pixel in the mask and its closest sparse clustering pixel.

        This is performed by going to each image pixel in the *mask*, and pairing it with its nearest neighboring pixel
        in the *sparse_mask*. The index of the *sparse_mask* pixel is drawn from the *sparse_index_image*. This
        neighbor search continue grows larger and larger around a pixel, until a pixel contained in the *sparse_mask* is
        successfully found.

        Returns
        -------
        image_to_cluster : ndarray
            The mapping between every image pixel and its closest sparse clustering pixel, where each entry give the 1D
            index of the sparse pixel in sparse_pixel arrays.

        """
        image_to_sparse = np.empty(0, dtype=int)

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self.mask[x, y]:
                    iboarder = 0
                    pixel_match = False
                    while not pixel_match:
                        for x1 in range(x - iboarder, x + iboarder + 1):
                            for y1 in range(y - iboarder, y + iboarder + 1):
                                if 0 <= x1 < self.shape[0] and 0 <= y1 < self.shape[1]:
                                    if not self[x1, y1] and not pixel_match:
                                        image_to_sparse = np.append(image_to_sparse, self.index_image[x1, y1] - 1)
                                        pixel_match = True

                        iboarder += 1
                        if iboarder == 100:
                            raise exc.MaskException('compute_image_to_sparse - Stuck in infinite loop')

        return image_to_sparse


class ImageGrid(np.ndarray):
    """Abstract class for a regular grid of coordinates. On a regular grid, each pixel's arc-second coordinates \
    are represented by the value at the centre of the pixel.

    Coordinates are defined from the top-left corner, such that data_to_image in the top-left corner of an \
    image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds. The image pixel indexes are \
    also counted from the top-left.

    A regular *grid_coords* is a NumPy array of image_shape [image_pixels, 2]. Therefore, the first element maps \
    to the image pixel index, and second element to its (x,y) arc second coordinates. For example, the value \
    [3,1] gives the 4th image pixel's y coordinate.

    Below is a visual illustration of a regular grid, where a total of 10 data_to_image are unmasked and therefore \
    included in the grid.

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an example image.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|o|o|x|x|x|x|     x = True (Pixel is masked and excluded from analysis)
    |x|x|x|o|o|o|o|x|x|x|     o = False (Pixel is not masked and included in analysis)
    |x|x|x|o|o|o|o|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    This image pixel index's will come out like this (and the direction of arc-second coordinates is highlighted
    around the image.

    pixel_scale = 1.0"

    <--- -ve  x  +ve -->

    |x|x|x|x|x|x|x|x|x|x|  ^   grid_coords[0] = [-0.5,  1.5]
    |x|x|x|x|x|x|x|x|x|x|  |   grid_coords[1] = [ 0.5,  1.5]
    |x|x|x|x|x|x|x|x|x|x|  |   grid_coords[2] = [-1.5,  0.5]
    |x|x|x|x|0|1|x|x|x|x| +ve  grid_coords[3] = [-0.5,  0.5]
    |x|x|x|2|3|4|5|x|x|x|  y   grid_coords[4] = [ 0.5,  0.5]
    |x|x|x|6|7|8|9|x|x|x| -ve  grid_coords[5] = [ 1.5,  0.5]
    |x|x|x|x|x|x|x|x|x|x|  |   grid_coords[6] = [-1.5, -0.5]
    |x|x|x|x|x|x|x|x|x|x|  |   grid_coords[7] = [-0.5, -0.5]
    |x|x|x|x|x|x|x|x|x|x| \/   grid_coords[8] = [ 0.5, -0.5]
    |x|x|x|x|x|x|x|x|x|x|      grid_coords[9] = [ 1.5, -0.5]
    """

    @property
    def no_pixels(self):
        return self.shape[0]

    def __new__(cls, arr, *args, **kwargs):
        return arr.view(cls)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(ImageGrid, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super(ImageGrid, self).__setstate__(state[0:-1])


class SubGrid(ImageGrid):
    """Abstract class for a sub of coordinates. On a sub-grid, each pixel is sub-gridded into a uniform grid of
     sub-coordinates, which are used to perform over-sampling in the lens analysis.

    Coordinates are defined from the top-left corner, such that data_to_image in the top-left corner of an
    image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds. The image pixel indexes are
    also counted from the top-left.

    A sub *grid_coords* is a NumPy array of image_shape [image_pixels, sub_grid_pixels, 2]. Therefore, the first
    element maps to the image pixel index, the second element to the sub-pixel index and third element to that
    sub pixel's (x,y) arc second coordinates. For example, the value [3, 6, 1] gives the 4th image pixel's
    7th sub-pixel's y coordinate.

    Below is a visual illustration of a sub grid. Like the regular grid, the indexing of each sub-pixel goes from
    the top-left corner. In contrast to the regular grid above, our illustration below restricts the mask to just
    2 data_to_image, to keep the illustration brief.

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an example image.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from analysis)
    |x|x|x|x|o|o|x|x|x|x|     o = False (Pixel is not masked and included in analysis)
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    Our regular-grid looks like it did before:

    pixel_scale = 1.0"

    <--- -ve  x  +ve -->

    |x|x|x|x|x|x|x|x|x|x|  ^
    |x|x|x|x|x|x|x|x|x|x|  |
    |x|x|x|x|x|x|x|x|x|x|  |
    |x|x|x|x|x|x|x|x|x|x| +ve  grid_coords[0] = [-1.5,  0.5]
    |x|x|x|0|1|x|x|x|x|x|  y   grid_coords[1] = [-0.5,  0.5]
    |x|x|x|x|x|x|x|x|x|x| -ve
    |x|x|x|x|x|x|x|x|x|x|  |
    |x|x|x|x|x|x|x|x|x|x|  |
    |x|x|x|x|x|x|x|x|x|x| \/
    |x|x|x|x|x|x|x|x|x|x|

    However, we now go to each image-pixel and derive a sub-pixel grid for it. For example, for pixel 0,
    if *sub_grid_size=2*, we use a 2x2 sub-grid:

    Pixel 0 - (2x2):

           grid_coords[0,0] = [-1.66, 0.66]
    |0|1|  grid_coords[0,1] = [-1.33, 0.66]
    |2|3|  grid_coords[0,2] = [-1.66, 0.33]
           grid_coords[0,3] = [-1.33, 0.33]

    Now, we'd normally sub-grid all data_to_image using the same *sub_grid_size*, but for this illustration lets
    pretend we used a sub_grid_size of 3x3 for pixel 1:

             grid_coords[0,0] = [-0.75, 0.75]
             grid_coords[0,1] = [-0.5,  0.75]
             grid_coords[0,2] = [-0.25, 0.75]
    |0|1|2|  grid_coords[0,3] = [-0.75,  0.5]
    |3|4|5|  grid_coords[0,4] = [-0.5,   0.5]
    |6|7|8|  grid_coords[0,5] = [-0.25,  0.5]
             grid_coords[0,6] = [-0.75, 0.25]
             grid_coords[0,7] = [-0.5,  0.25]
             grid_coords[0,8] = [-0.25, 0.25]

    """

    def __init__(self, array, mask, sub_grid_size=1):
        # noinspection PyArgumentList
        super(SubGrid, self).__init__()
        self.sub_grid_size = sub_grid_size
        self.sub_grid_length = int(sub_grid_size ** 2.0)
        self.sub_grid_fraction = 1.0 / self.sub_grid_length
        self.mask = mask

    def __array_finalize__(self, obj):
        if isinstance(obj, SubGrid):
            self.sub_grid_size = obj.sub_grid_size
            self.sub_grid_length = obj.sub_grid_length
            self.sub_grid_fraction = obj.sub_grid_fraction
            self.mask = obj.mask

    @classmethod
    def from_mask(cls, mask, sub_grid_size=1):
        sub_pixel_count = 0

        grid = np.zeros(shape=(mask.pixels_in_mask * sub_grid_size ** 2, 2))

        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if not mask[x, y]:
                    x_arcsec, y_arcsec = mask.pixel_coordinates_to_arc_second_coordinates((x, y))

                    for x1 in range(sub_grid_size):
                        for y1 in range(sub_grid_size):
                            grid[sub_pixel_count, 0] = mask.sub_pixel_to_coordinate(x1, x_arcsec, sub_grid_size)

                            grid[sub_pixel_count, 1] = mask.sub_pixel_to_coordinate(y1, y_arcsec, sub_grid_size)

                            sub_pixel_count += 1
        return SubGrid(grid, mask, sub_grid_size=sub_grid_size)

    def sub_data_to_image(self, data):
        return np.multiply(self.sub_grid_fraction, data.reshape(-1, self.sub_grid_length).sum(axis=1))

    @property
    @Memoizer()
    def sub_to_image(self):
        """ Compute the pairing of every sub-pixel to its original image pixel from a mask. """
        sub_to_image = np.zeros(shape=(self.mask.pixels_in_mask * self.sub_grid_size ** 2,), dtype='int')
        image_pixel_count = 0
        sub_pixel_count = 0

        for x in range(self.mask.shape[0]):
            for y in range(self.mask.shape[1]):
                if not self.mask[x, y]:
                    for x1 in range(self.sub_grid_size):
                        for y1 in range(self.sub_grid_size):
                            sub_to_image[sub_pixel_count] = image_pixel_count
                            sub_pixel_count += 1

                    image_pixel_count += 1

        return sub_to_image


class GridCollection(object):

    def __init__(self, image, sub, blurring):
        """
        A collection of grids which contain the coordinates of an image. This includes the image's regular grid,
        sub-grid, blurring region, etc. Coordinate grids are passed through the ray-tracing module to set up the image,
        lens and source planes.

        Parameters
        -----------
        image : GridCoordsImage
            A grid of coordinates for the regular image grid.
        sub : GridCoordsImageSub
            A grid of coordinates for the sub-gridded image grid.
        blurring : GridCoordsBlurring
            A grid of coordinates for the blurring regions.
        """
        self.image = image
        self.sub = sub
        self.blurring = blurring

    @classmethod
    def from_mask_sub_grid_size_and_blurring_shape(cls, mask, sub_grid_size, blurring_shape):
        image_coords = mask.coordinate_grid
        sub_grid_coords = SubGrid.from_mask(mask, sub_grid_size)
        blurring_coords = mask.blurring_mask_for_kernel_shape(blurring_shape).coordinate_grid
        return GridCollection(image_coords, sub_grid_coords, blurring_coords)

    def apply_function(self, func):
        return GridCollection(func(self.image), func(self.sub), func(self.blurring))

    def map_function(self, func, *arg_lists):
        return GridCollection(*[func(*args) for args in zip(self, *arg_lists)])

    @property
    def sub_pixels(self):
        return self.sub.shape[0]

    def __getitem__(self, item):
        return [self.image, self.sub, self.blurring][item]


class ImageGridBorder(np.ndarray):

    @property
    def no_pixels(self):
        return self.shape[0]

    def __new__(cls, arr, polynomial_degree=3, centre=(0.0, 0.0), *args, **kwargs):
        border = arr.view(cls)
        border.polynomial_degree = polynomial_degree
        border.centre = centre
        return border

    @classmethod
    def from_mask(cls, mask, polynomial_degree=3, centre=(0.0, 0.0)):
        return cls(mask.border_pixel_indices, polynomial_degree, centre)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(ImageGridBorder, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super(ImageGridBorder, self).__setstate__(state[0:-1])

    def grid_to_radii(self, grid):
        """
        Convert coordinates to a circular radius.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid

        Returns
        -------
        The radius at those coordinates
        """

        return np.sqrt(np.add(np.square(np.subtract(grid[:, 0], self.centre[0])),
                              np.square(np.subtract(grid[:, 1], self.centre[1]))))

    def grid_to_thetas(self, grid):
        """
        Compute the angle in degrees between the image_grid and plane positive x-axis, defined counter-clockwise.

        Parameters
        ----------
        grid : Union((float, float), ndarray)
            The x and y image_grid of the plane.

        Returns
        ----------
        The angle between the image_grid and the x-axis.
        """
        shifted_grid = np.subtract(grid, self.centre)
        theta_from_x = np.degrees(np.arctan2(shifted_grid[:, 1], shifted_grid[:, 0]))
        theta_from_x[theta_from_x < 0.0] += 360.
        return theta_from_x

    def polynomial_fit_to_border(self, grid):

        border_grid = grid[self]

        return np.polyfit(self.grid_to_thetas(border_grid), self.grid_to_radii(border_grid), self.polynomial_degree)

    def move_factors_from_grid(self, grid):
        """Get the move factor of a coordinate.
         A move-factor defines how far a coordinate outside the source-plane setup_border_pixels must be moved in order
         to lie on it. PlaneCoordinates already within the setup_border_pixels return a move-factor of 1.0, signifying
         they are already within the setup_border_pixels.

        Parameters
        ----------
        grid : ndarray
            The x and y image_grid of the pixel to have its move-factor computed.
        """
        grid_thetas = self.grid_to_thetas(grid)
        grid_radii = self.grid_to_radii(grid)
        poly = self.polynomial_fit_to_border(grid)

        with np.errstate(divide='ignore'):
            move_factors = np.divide(np.polyval(poly, grid_thetas), grid_radii)
        move_factors[move_factors > 1.0] = 1.0

        return move_factors

    def relocated_grid_from_grid(self, grid):
        move_factors = self.move_factors_from_grid(grid)
        return np.multiply(grid, move_factors[:, None])


class SubGridBorder(ImageGridBorder):

    @classmethod
    def from_mask(cls, mask, sub_grid_size, polynomial_degree=3, centre=(0.0, 0.0)):
        return cls(mask.border_sub_pixel_indices(sub_grid_size), polynomial_degree, centre)


class BorderCollection(object):

    def __init__(self, image, sub):
        self.image = image
        self.sub = sub

    @classmethod
    def from_mask_and_sub_grid_size(cls, mask, sub_grid_size, polynomial_degree=3, centre=(0.0, 0.0)):
        image_border = ImageGridBorder.from_mask(mask, polynomial_degree, centre)
        sub_border = SubGridBorder.from_mask(mask, sub_grid_size, polynomial_degree, centre)
        return BorderCollection(image_border, sub_border)

    def relocated_grids_from_grids(self, grids):
        return GridCollection(image=self.image.relocated_grid_from_grid(grids.image),
                              sub=self.sub.relocated_grid_from_grid(grids.sub),
                              blurring=None)
