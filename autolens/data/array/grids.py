import numpy as np
import numba

from autolens.data.array import mask as msk, scaled_array
from autolens.data.array.util import grid_util
from autolens.data.array.util import mapping_util, array_util, mask_util


class DataGrids(object):

    def __init__(self, regular, sub, blurring, pix=None):
        """The grids containing the (y,x) arc-second coordinates of padded pixels in a masks. There are 3 grids:

        Regular - the (y,x) coordinate at the center of every padded pixel.
        Sub - the (y,x) coordinates of every sub-pixel in every padded pixel, each using a grid of size \
        sub_grid_size x sub_grid_size.
        Blurring - the (y,x) coordinates of all blurring pixels, which are masked pixels whose light is \
        blurred into masked pixels during PSF convolution.

        The grids are stored as 1D arrays, where each entry corresponds to an padded (sub-)pixel. The 1D array is \
        ordered such pixels begin from the top-row of the masks 2D array and then downwards.

        Parameters
        -----------
        regular : RegularGrid
            The grid of (y,x) arc-second coordinates at the centre of every data value (e.g. image-pixels).
        sub : SubGrid
            The grid of (y,x) arc-second coordinates at the centre of every regular pixel's sub-pixels.
        blurring : RegularGrid | ndarray | None
            The grid of (y,x) arc-second coordinates at the centre of every blurring-mask pixel.
        pix : RegularGrid | ndarray | None
            The grid of (y,x) arc-second coordinates of every image-plane pixelization grid used for adaptive source \
            -plane pixelizations.
        """
        self.regular = regular
        self.sub = sub
        self.blurring = blurring
        if pix is None:
            self.pix = np.array([[0.0, 0.0]])
        else:
            self.pix = pix

    @classmethod
    def grids_from_mask_sub_grid_size_and_psf_shape(cls, mask, sub_grid_size, psf_shape):
        """Setup the *DataGrids* from a masks, sub-grid size and psf-shape.

        Parameters
        -----------
        mask : Mask
            The masks whose padded pixels the imaging-grids are setup using.
        sub_grid_size : int
            The size of a sub-pixels sub-grid (sub_grid_size x sub_grid_size).
        psf_shape : (int, int)
            the shape of the PSF used in the analysis, which defines the masks's blurring-region.
        """
        image_grid = RegularGrid.from_mask(mask)
        sub_grid = SubGrid.from_mask_and_sub_grid_size(mask, sub_grid_size)
        blurring_grid = RegularGrid.blurring_grid_from_mask_and_psf_shape(mask, psf_shape)
        return DataGrids(image_grid, sub_grid, blurring_grid)

    @classmethod
    def from_shape_and_pixel_scale(cls, shape, pixel_scale, sub_grid_size=2, psf_shape=(1, 1)):
        image_grid = RegularGrid.from_shape_and_pixel_scale(shape=shape, pixel_scale=pixel_scale)
        sub_grid = SubGrid.from_shape_pixel_scale_and_sub_grid_size(shape=shape, pixel_scale=pixel_scale,
                                                                    sub_grid_size=sub_grid_size)
        blurring_grid = np.array([[0.0, 0.0]])
        return DataGrids(image_grid, sub_grid, blurring_grid)

    @classmethod
    def padded_grids_from_mask_sub_grid_size_and_psf_shape(cls, mask, sub_grid_size, psf_shape):
        """Setup the collection of padded imaging-grids from a masks, using also an input sub-grid size to setup the \
        sub-grid and psf-shape to setup the blurring grid.

        Parameters
        -----------
        mask : Mask
            The masks whose padded pixels the imaging-grids are setup using.
        sub_grid_size : int
            The size of a sub-pixels sub-grid (sub_grid_size x sub_grid_size).
        psf_shape : (int, int)
            the shape of the PSF used in the analysis, which therefore defines the masks's blurring-region.
        """
        image_padded_grid = PaddedRegularGrid.padded_grid_from_shapes_and_pixel_scale(shape=mask.shape,
                                                                                      psf_shape=psf_shape,
                                                                                      pixel_scale=mask.pixel_scale)
        sub_padded_grid = PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                          sub_grid_size=sub_grid_size,
                                                                                          psf_shape=psf_shape)
        # TODO : The blurring grid is not used when the grid mapper is called, the 0.0 0.0 stops errors inr ayT_racing
        # TODO : implement a more explicit solution
        return DataGrids(regular=image_padded_grid, sub=sub_padded_grid, blurring=np.array([[0.0, 0.0]]))

    @classmethod
    def grids_for_simulation(cls, shape, pixel_scale, psf_shape, sub_grid_size=1):
        """Setup a collection of imaging-grids for simulating an datas_ of a strong lens.

        This routine uses padded-grids which ensure that the PSF blurring in the simulation routine \
        (*imaging.PrepatoryImage.simulate*) is not degraded due to edge effects.

        Parameters
        -----------
        pixel_scale
        shape
        sub_grid_size : int
            The size of a sub-pixels sub-grid (sub_grid_size x sub_grid_size).
        psf_shape : (int, int)
            the shape of the PSF used in the analysis, which therefore defines the masks's blurring-region.
        """
        return cls.padded_grids_from_mask_sub_grid_size_and_psf_shape(mask=msk.Mask(array=np.full(shape, False),
                                                                                pixel_scale=pixel_scale),
                                                                      sub_grid_size=sub_grid_size,
                                                                      psf_shape=psf_shape)

    def data_grids_with_pix_grid(self, pix_grid, regular_to_nearest_regular_pix):
        pix = PixGrid(arr=pix_grid, regular_to_nearest_regular_pix=regular_to_nearest_regular_pix)
        return DataGrids(regular=self.regular, sub=self.sub, blurring=self.blurring, pix=pix)

    def apply_function(self, func):
        if self.blurring is not None and self.pix is not None:
            return DataGrids(func(self.regular), func(self.sub), func(self.blurring), func(self.pix))
        elif self.blurring is None and self.pix is not None:
            return DataGrids(func(self.regular), func(self.sub), self.blurring, func(self.pix))
        elif self.blurring is not None and self.pix is None:
            return DataGrids(func(self.regular), func(self.sub), func(self.blurring), self.pix)
        else:
            return DataGrids(func(self.regular), func(self.sub), self.blurring, self.pix)

    def map_function(self, func, *arg_lists):
        return DataGrids(*[func(*args) for args in zip(self, *arg_lists)])

    @property
    def sub_pixels(self):
        return self.sub.shape[0]

    def __getitem__(self, item):
        return [self.regular, self.sub, self.blurring, self.pix][item]


class RegularGrid(np.ndarray):
    """Abstract class for a regular grid of coordinates, where each padded pixel's (y,x) arc-second coordinates \
    are represented by the value at the origin of the pixel.

    Coordinates are defined from the top-left corner, such that pixels in the top-left corner of a masks (e.g. [0,0]) \
    have a negative x and y-value in arc seconds. The masks pixel indexes are also counted from the top-left.

    An *RegularGrid* is a ndarray of shape [regular_pixels, 2], where regular_pixels is the total number of padded \
    datas_-pixels. The first element maps of the ndarray corresponds to the padded pixel index and second element the \
    x or y arc second coordinates. For howtolens:

    - regular_grid[3,1] = the 4th padded pixel's y-coordinate.
    - regular_grid[6,0] = the 5th padded pixel's x-coordinate.

    Below is a visual illustration of a regular, where a total of 10 pixels are padded and are included in the \
    grid.

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an howtolens masks.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|o|o|x|x|x|x|     x = True (Pixel is masked and excluded from lensing)
    |x|x|x|o|o|o|o|x|x|x|     o = False (Pixel is not masked and included in lensing)
    |x|x|x|o|o|o|o|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    The masks pixel index's will come out like this (and the direction of arc-second coordinates is highlighted
    around the masks.

    pixel_scale = 1.0"

    <--- -ve  x  +ve -->

    |x|x|x|x|x|x|x|x|x|x|  ^   regular_grid[0] = [-0.5, -1.5]
    |x|x|x|x|x|x|x|x|x|x|  |   regular_grid[1] = [ 0.5, -1.5]
    |x|x|x|x|x|x|x|x|x|x|  |   regular_grid[2] = [-1.5, -0.5]
    |x|x|x|x|0|1|x|x|x|x| -ve  regular_grid[3] = [-0.5, -0.5]
    |x|x|x|2|3|4|5|x|x|x|  y   regular_grid[4] = [ 0.5, -0.5]
    |x|x|x|6|7|8|9|x|x|x| +ve  regular_grid[5] = [ 1.5, -0.5]
    |x|x|x|x|x|x|x|x|x|x|  |   regular_grid[6] = [-1.5,  0.5]
    |x|x|x|x|x|x|x|x|x|x|  |   regular_grid[7] = [-0.5,  0.5]
    |x|x|x|x|x|x|x|x|x|x| \/   regular_grid[8] = [ 0.5,  0.5]
    |x|x|x|x|x|x|x|x|x|x|      regular_grid[9] = [ 1.5,  0.5]
    """

    def __new__(cls, arr, mask, *args, **kwargs):
        obj = arr.view(cls)
        obj.mask = mask
        return obj

    def __array_finalize__(self, obj):
        if hasattr(obj, "mask"):
            self.mask = obj.mask

    @property
    def masked_shape_arcsec(self):
        return (np.amax(self[: ,0]) - np.amin(self[: ,0]), np.amax(self[: ,1]) - np.amin(self[: ,1]))

    @property
    def unlensed_grid(self):
        return RegularGrid(arr=grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=self.mask,
                                                                             pixel_scales=self.mask.pixel_scales),
                           mask=self.mask)

    @property
    def unlensed_unmasked_grid(self):
        return RegularGrid(arr=grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(shape=self.mask.shape,
                                                                                            pixel_scales=self.mask.pixel_scales),
                           mask=self.mask)

    @classmethod
    def from_mask(cls, mask):
        """Setup an *RegularGrid* of the regular regular from a masks. The center of every padded pixel gives \
        the grid's (y,x) arc-second coordinates.

        Parameters
        -----------
        mask : Mask
            The masks whose padded pixels are used to setup the sub-pixel grids."""
        array = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=mask.pixel_scales)
        return cls(array, mask)

    @classmethod
    def from_shape_and_pixel_scale(cls, shape, pixel_scale):
        mask = msk.Mask.padded_for_shape_and_pixel_scale(shape=shape, pixel_scale=pixel_scale)
        array = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask, pixel_scales=mask.pixel_scales)
        return cls(array, mask)

    @classmethod
    def blurring_grid_from_mask_and_psf_shape(cls, mask, psf_shape):
        """Setup an *RegularGrid* of the blurring-grid from a masks. The center of every padded blurring-pixel gives \
        the grid's (y,x) arc-second coordinates."""
        blurring_mask = mask.blurring_mask_for_psf_shape(psf_shape)
        return RegularGrid.from_mask(blurring_mask)

    def map_to_2d(self, array_1d):
        """ Map a 1D array the same dimension as the grid to its original 2D array.

        Parameters
        -----------
        array_1d : ndarray
            The 1D array of masked datas which is mapped to 2D.
        """
        return mapping_util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(array_1d,
                                                                                               self.mask.shape,
                                                                                               self.mask.masked_grid_index_to_pixel)

    def scaled_array_from_array_1d(self, array_1d):
        return scaled_array.ScaledSquarePixelArray(array=self.map_to_2d(array_1d), pixel_scale=self.mask.pixel_scale,
                                                   origin=self.mask.origin)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(RegularGrid, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super(RegularGrid, self).__setstate__(state[0:-1])

    @property
    def total_pixels(self):
        return self.shape[0]

    @property
    def pixel_scale(self):

        return self.mask.pixel_scale

    @property
    def yticks(self):
        """Compute the yticks labels of this grid, used for plotting the y-axis ticks when visualizing a regular"""
        return np.linspace(np.min(self[:, 0]), np.max(self[:, 0]), 4)

    @property
    def xticks(self):
        """Compute the xticks labels of this grid, used for plotting the x-axis ticks when visualizing a regular"""
        return np.linspace(np.min(self[:, 1]), np.max(self[:, 1]), 4)


class SubGrid(RegularGrid):
    """Abstract class for a sub-grid of coordinates. On a sub-grid, each padded pixel is sub-gridded into a uniform \
     grid of (y,x) sub-coordinates..

    Coordinates are defined from the top-left corner, such that pixels in the top-left corner of an \
    masks (e.g. [0,0]) have negative x and y-values in arc seconds. The masks pixel indexes are also counted from the \
    top-left.

    Sub-pixels follow the same convention as above, where the top-left sub-pixel has the lowest x and y-values in each \
    datas_-pixel. Sub-pixel indexes include all previous sub-pixels in all previous regular-pixels.

    A *SubGrid* is a NumPy array of shape [image_pixels*sub_grid_pixels**2, 2]. The first element of the ndarray \
    corresponds to the padded sub-pixel index, and second element the sub-pixel's (y,x) arc second coordinates. \
    For howtolens:

    - sub_grid[9, 1] - using a 2x2 sub-grid, gives the 3rd datas_-pixel's 2nd sub-pixel y-coordinate.
    - sub_grid[9, 1] - using a 3x3 sub-grid, gives the 2nd datas_-pixel's 1st sub-pixel y-coordinate.
    - sub_grid[27, 0] - using a 3x3 sub-grid, gives the 4th datas_-pixel's 1st sub-pixel x-coordinate.

    Below is a visual illustration of a sub grid. Like the regular grid, the indexing of each sub-pixel goes from \
    the top-left corner. In contrast to the regular grid above, our illustration below restricts the masks to just
    2 pixels, to keep the illustration brief.

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an howtolens masks.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from lensing)
    |x|x|x|x|o|o|x|x|x|x|     o = False (Pixel is not masked and included in lensing)
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
    |x|x|x|x|x|x|x|x|x|x| +ve  regular_grid[0] = [-1.5,  0.5]
    |x|x|x|0|1|x|x|x|x|x|  y   regular_grid[1] = [-0.5,  0.5]
    |x|x|x|x|x|x|x|x|x|x| -ve
    |x|x|x|x|x|x|x|x|x|x|  |
    |x|x|x|x|x|x|x|x|x|x|  |
    |x|x|x|x|x|x|x|x|x|x| \/
    |x|x|x|x|x|x|x|x|x|x|

    However, we now go to each masks-pixel and derive a sub-pixel grid for it. For howtolens, for pixel 0,
    if *sub_grid_size=2*, we use a 2x2 sub-grid:

    Pixel 0 - (2x2):

           regular_grid[0] = [-1.66, 0.66]
    |0|1|  regular_grid[1] = [-1.33, 0.66]
    |2|3|  regular_grid[2] = [-1.66, 0.33]
           regular_grid[3] = [-1.33, 0.33]

    Now, we'd normally sub-grid all pixels using the same *sub_grid_size*, but for this illustration lets
    pretend we used a sub_grid_size of 3x3 for pixel 1:

             regular_grid[0] = [-0.75, 0.75]
             regular_grid[1] = [-0.5,  0.75]
             regular_grid[2] = [-0.25, 0.75]
    |0|1|2|  regular_grid[3] = [-0.75,  0.5]
    |3|4|5|  regular_grid[4] = [-0.5,   0.5]
    |6|7|8|  regular_grid[5] = [-0.25,  0.5]
             regular_grid[6] = [-0.75, 0.25]
             regular_grid[7] = [-0.5,  0.25]
             regular_grid[8] = [-0.25, 0.25]
    """

    # noinspection PyUnusedLocal
    def __init__(self, array, mask, sub_grid_size=1):
        # noinspection PyArgumentList
        super(SubGrid, self).__init__()
        self.mask = mask
        self.sub_grid_size = sub_grid_size
        self.sub_grid_length = int(sub_grid_size ** 2.0)
        self.sub_grid_fraction = 1.0 / self.sub_grid_length

    @property
    def unlensed_grid(self):
        return SubGrid(grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=self.mask,
            pixel_scales=self.mask.pixel_scales,
            sub_grid_size=self.sub_grid_size),
            self.mask, self.sub_grid_size)

    @property
    def unlensed_unmasked_grid(self):
        return SubGrid(grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full(self.mask.shape, False), pixel_scales=self.mask.pixel_scales,
            sub_grid_size=self.sub_grid_size),
            mask=self.mask, sub_grid_size=self.sub_grid_size)

    @classmethod
    def from_mask_and_sub_grid_size(cls, mask, sub_grid_size=1):
        """Setup a *SubGrid* of the regular-pixels, using a masks and a specified sub-grid size. The center of \
        every padded pixel's sub-pixels give the grid's (y,x) arc-second coordinates.

        Parameters
        -----------
        mask : Mask
            The masks whose padded pixels are used to setup the sub-pixel grids.
        sub_grid_size : int
            The size (sub_grid_size x sub_grid_size) of each datas_-pixels sub-grid.
        """
        sub_grid_masked = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask,
                                                                                                pixel_scales=mask.pixel_scales,
                                                                                                sub_grid_size=sub_grid_size)
        return SubGrid(sub_grid_masked, mask, sub_grid_size)

    @classmethod
    def from_shape_pixel_scale_and_sub_grid_size(cls, shape, pixel_scale, sub_grid_size):
        mask = msk.Mask.padded_for_shape_and_pixel_scale(shape=shape, pixel_scale=pixel_scale)
        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask,
                                                                                         pixel_scales=mask.pixel_scales,
                                                                                         sub_grid_size=sub_grid_size)
        return SubGrid(sub_grid, mask, sub_grid_size)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if isinstance(obj, SubGrid):
            self.sub_grid_size = obj.sub_grid_size
            self.sub_grid_length = obj.sub_grid_length
            self.sub_grid_fraction = obj.sub_grid_fraction
            self.mask = obj.mask

    def sub_data_to_regular_data(self, sub_array):
        """For an input sub-gridded array, mappers it from the sub-grid to a 1D regular by summing each set of
        each set of sub-pixels values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_array : ndarray
            A 1D sub-gridded array of values (e.g. the intensities, surface-densities, potential) which is mapped to
            a 1d regular array.
        """
        return np.multiply(self.sub_grid_fraction, sub_array.reshape(-1, self.sub_grid_length).sum(axis=1))

    @property
    @array_util.Memoizer()
    def sub_to_regular(self):
        """ Compute the mapping between every sub-pixel and its host regular-pixel.

        For howtolens:

        - sub_to_pixel[8] = 2 -  The seventh sub-pixel is within the 3rd regular pixel.
        - sub_to_pixel[20] = 4 -  The nineteenth sub-pixel is within the 5th regular pixel.
        """
        return mapping_util.sub_to_regular_from_mask(self.mask, self.sub_grid_size).astype('int')


class PixGrid(np.ndarray):

    def __new__(cls, arr, regular_to_nearest_regular_pix, *args, **kwargs):
        obj = arr.view(cls)
        obj.regular_to_nearest_regular_pix = regular_to_nearest_regular_pix
        return obj

    def __array_finalize__(self, obj):
        if hasattr(obj, "regular_to_nearest_regular_pix"):
            self.regular_to_nearest_regular_pix = obj.regular_to_nearest_regular_pix


class SparseToRegularGrid(scaled_array.RectangularArrayGeometry):

    def __init__(self, unmasked_sparse_grid_shape, pixel_scales, regular_grid):
        """Abstract class which represents a sparse grid overlaid over a regular-grid. The regular-grids mask is used to \
         determine which sparse-grid pixels are in the mask, and thse pixels form the centers of the sparse grid.

        This sparse grid is used to determine the pixel centers of an adaptive grid pixelization.

        Parameters
        ----------
        unmasked_sparse_grid_shape : (int, int)
            The shape of the unmasked sparse-grid whose centres form the centres of pixelization pixels.
        pixel_scales : (float, float)
            The pixel-to-arcsecond scale of a pixel in the y and x directions.
        regular_grid : RegularGrid
            The regular-grid the sparse grid is compared to.
        """

        self.shape = unmasked_sparse_grid_shape
        self.origin = (0.0, 0.0)
        self.total_unmasked_sparse_pixels = int(self.shape[0] * self.shape[1])
        self.pixel_scales = pixel_scales

        self.regular_grid = regular_grid
        self.unmasked_sparse_grid = self.grid_1d
        self.unmasked_sparse_grid_pixel_centres = \
            regular_grid.mask.grid_arc_seconds_to_grid_pixel_centres(self.unmasked_sparse_grid)

        self.total_sparse_pixels = mask_util.total_sparse_pixels_from_mask(mask=self.regular_grid.mask,
               unmasked_sparse_grid_pixel_centres=self.unmasked_sparse_grid_pixel_centres)

    @property
    def unmasked_sparse_to_sparse(self):
        """The 1D index mappings between the unmasked sparse-grid and masked sparse grid."""
        return mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(mask=self.regular_grid.mask,
                     unmasked_sparse_grid_pixel_centres=self.unmasked_sparse_grid_pixel_centres,
                      total_sparse_pixels=self.total_sparse_pixels).astype('int')

    @property
    def sparse_to_unmasked_sparse(self):
        """The 1D index mappings between the masked sparse-grid and unmasked sparse grid."""
        return mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=self.total_sparse_pixels, mask=self.regular_grid.mask,
            unmasked_sparse_grid_pixel_centres=self.unmasked_sparse_grid_pixel_centres).astype('int')

    @property
    def regular_to_unmasked_sparse(self):
        """The 1D index mapping between the regular-grid and unmasked sparse-grid."""
        return self.grid_arc_seconds_to_grid_pixel_indexes(grid_arc_seconds=self.regular_grid)

    @property
    def regular_to_sparse(self):
        """The 1D index mappings between the regular-grid and masked sparse-grid."""
        return mapping_util.regular_to_sparse_from_sparse_mappings(
            regular_to_unmasked_sparse=self.regular_to_unmasked_sparse,
            unmasked_sparse_to_sparse=self.unmasked_sparse_to_sparse).astype('int')

    @property
    def sparse_grid(self):
        """The (y,x) arc-second coordinates of the masked sparse-grid."""
        return mapping_util.sparse_grid_from_unmasked_sparse_grid(unmasked_sparse_grid=self.unmasked_sparse_grid,
                                                                  sparse_to_unmasked_sparse=self.sparse_to_unmasked_sparse)


class PaddedRegularGrid(RegularGrid):

    def __new__(cls, arr, mask, image_shape, *args, **kwargs):
        """An *PaddedRegularGrid* stores the (y,x) arc-second coordinates of a masks's pixels in 1D, in an analogous
        fashion to an *RegularGrid*. An *PaddedRegularGrid* deviate from a normal grid in that:

        - All pixels are used (as opposed to just padded pixels)
        - The masks is padded when computing the grid, such that additional pixels beyond its edge are included.

        Padded-grids allow quantities like intensities to be computed on large 2D arrays spanning the entire datas_, as
        opposed to just within the masked region.
        """
        arr = arr.view(cls)
        arr.mask = mask
        arr.image_shape = image_shape
        return arr

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if hasattr(obj, "image_shape"):
            self.image_shape = obj.image_shape

    @classmethod
    def padded_grid_from_shapes_and_pixel_scale(self, shape, psf_shape, pixel_scale):
        """Setup an *PaddedRegularGrid* of the regular regular for an input datas_-shape, psf-shape and pixel-scale.

        The center of every pixel is used to setup the grid's (y,x) arc-second coordinates, including padded-pixels
        which are beyond the input shape but will have light blurred into them given the psf-shape.

        Parameters
        ----------
        pixel_scale : float
            The scale of each pixel in arc seconds
        shape : (int, int)
            The (y,x) shape of the padded-grid's 2D datas_ in units of pixels.
        psf_shape : (int, int)
           The shape of the psf which defines the blurring region and therefore size of padding.
        """
        padded_shape = (shape[0] + psf_shape[0] - 1, shape[1] + psf_shape[1] - 1)
        padded_image_grid = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
            mask=np.full(padded_shape, False), pixel_scales=(pixel_scale, pixel_scale))
        padded_mask = msk.Mask.padded_for_shape_and_pixel_scale(shape=padded_shape, pixel_scale=pixel_scale)
        return PaddedRegularGrid(arr=padded_image_grid, mask=padded_mask, image_shape=shape)

    def convolve_array_1d_with_psf(self, padded_array_1d, psf):
        """Convolve a 2d padded array of values (e.g. intensities beforoe PSF blurring) with a PSF, and then trim \
        the convolved array to its original 2D shape.

        Parameters
        -----------
        psf
        padded_array_1d: ndarray
            A 1D array of values which were computed using the *PaddedRegularGrid*.
        """
        padded_array_2d = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(padded_array_1d,
                                                                                                 self.mask.shape)
        blurred_padded_array_2d = psf.convolve(padded_array_2d)
        return mapping_util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(array_2d=blurred_padded_array_2d,
                                                                                   mask=np.full(self.mask.shape,
                                                                                                False))

    def map_to_2d(self, padded_array_1d):
        """ Map a padded 1D array of values to its origianl 2D array.

        Parameters
        -----------
        padded_array_1d : ndarray
            A 1D array of values which were computed using the *PaddedRegularGrid*.
        """
        padded_array_2d = self.map_to_2d_keep_padded(padded_array_1d)
        pad_size_0 = self.mask.shape[0] - self.image_shape[0]
        pad_size_1 = self.mask.shape[1] - self.image_shape[1]
        return (padded_array_2d[pad_size_0 // 2:self.mask.shape[0] - pad_size_0 // 2,
                pad_size_1 // 2:self.mask.shape[1] - pad_size_1 // 2])

    def map_to_2d_keep_padded(self, padded_array_1d):
        """ Map a padded 1D array of values to its padded 2D array.

        Parameters
        -----------
        padded_array_1d : ndarray
            A 1D array of values which were computed using the *PaddedRegularGrid*.
        """
        return mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(padded_array_1d, self.mask.shape)

    @property
    def padded_shape(self):
        return self.mask.shape


class PaddedSubGrid(SubGrid, PaddedRegularGrid):

    def __init__(self, arr, mask, image_shape, sub_grid_size=1):
        """A *PaddedSubGrid* stores the (y,x) arc-second coordinates of a masks's sub-pixels in 1D, in an analogous
        fashion to a *SubGrid*. A *PaddedSubGrid* deviate from a normal grid in that:

        - All pixels are used (as opposed to just padded pixels)
        - The masks is padded when computing the grid, such that additional pixels beyond its edge are included.

        Padded-grids allow quantities like intensities to be computed on large 2D arrays spanning the entire datas_, as
        opposed to just within the masked region.
        """
        super(PaddedSubGrid, self).__init__(arr, mask, sub_grid_size)
        self.image_shape = image_shape

    @classmethod
    def padded_grid_from_mask_sub_grid_size_and_psf_shape(cls, mask, sub_grid_size, psf_shape):
        """Setup an *PaddedSubGrid* for an input masks, sub-grid size and psf-shape.

        The center of every sub-pixel is used to setup the grid's (y,x) arc-second coordinates, including \
        padded-pixels which are beyond the input shape but will have light blurred into them given the psf-shape.

        Parameters
        ----------
        mask : Mask
            The masks whose padded pixels are used to setup the sub-pixel grids.
        sub_grid_size : int
            The size (sub_grid_size x sub_grid_size) of each datas_-pixels sub-grid.
        psf_shape : (int, int)
           The shape of the psf which defines the blurring region and therefore size of padding.
        """

        padded_shape = (mask.shape[0] + psf_shape[0] - 1, mask.shape[1] + psf_shape[1] - 1)

        padded_sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full(padded_shape, False), pixel_scales=mask.pixel_scales, sub_grid_size=sub_grid_size)

        padded_mask = msk.Mask.padded_for_shape_and_pixel_scale(shape=padded_shape, pixel_scale=mask.pixel_scale)

        return PaddedSubGrid(arr=padded_sub_grid, mask=padded_mask, image_shape=mask.shape,
                             sub_grid_size=sub_grid_size)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if isinstance(obj, PaddedSubGrid):
            self.image_shape = obj.image_shape


class RegularGridBorder(np.ndarray):

    def __new__(cls, arr, *args, **kwargs):
        """The border of a regular, containing the pixel-index's of all padded pixels that are on the \
        masks's border (e.g. they are next to a *True* value in at least one of the surrounding 8 pixels).

        A polynomial is fitted to the (y,x) coordinates of the border's pixels. This allows us to relocate \
        demagnified pixel's in a grid to its border, so that they do not disrupt an adaptive inversion.

        Parameters
        -----------
        arr : ndarray
            A 1D array of the integer indexes of an *RegularGrid*'s border pixels.
        polynomial_degree : int
            The degree of the polynomial that is used to fit_normal the border when relocating pixels outside the border to \
            its edge.
        centre : (float, float)
            The origin of the border, which can be shifted relative to its coordinates.
        """
        border = arr.view(cls)
        return border

    @classmethod
    def from_mask(cls, mask):
        """Setup the *RegularGridBorder* from a masks.

        Parameters
        -----------
        mask : Mask
            The masks the padded border pixel index's are computed from.
        polynomial_degree : int
            The degree of the polynomial that is used to fit_normal the border when relocating pixels outside the border to \
            its edge.
        centre : (float, float)
            The origin of the border, which can be shifted relative to its coordinates.
        """
        return cls(mask.border_pixels)

    def relocated_grids_from_grids(self, grids):
        """Determine a set of relocated grids from an input set of grids, by relocating their pixels based on the \
        border.

        The blurring-grid does not have its coordinates relocated, as it is only used for computing analytic \
        light-profiles and not inversion-grids.

        Parameters
        -----------
        grids : DataGrids
            The imaging-grids (datas_, sub) which have their coordinates relocated.
        """
        border_grid = grids.regular[self]
        return DataGrids(regular=self.relocated_grid_from_grid_jit(grid=grids.regular, border_grid=border_grid),
                         sub=self.relocated_grid_from_grid_jit(grid=grids.sub, border_grid=border_grid),
                         blurring=None,
                         pix=self.relocated_grid_from_grid_jit(grid=grids.pix, border_grid=border_grid))

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def relocated_grid_from_grid_jit(grid, border_grid):

        border_origin = np.zeros(2)
        border_origin[0] = np.mean(border_grid[: ,0])
        border_origin[1] = np.mean(border_grid[: ,1])
        border_grid_radii = np.sqrt(np.add(np.square(np.subtract(border_grid[:, 0], border_origin[0])),
                                           np.square(np.subtract(border_grid[:, 1], border_origin[1]))))
        border_min_radii = np.min(border_grid_radii)

        grid_radii = np.sqrt(np.add(np.square(np.subtract(grid[:, 0], border_origin[0])),
                                    np.square(np.subtract(grid[:, 1], border_origin[1]))))

        for pixel_index in range(grid.shape[0]):

            if grid_radii[pixel_index] > border_min_radii:

                closest_pixel_index = np.argmin(np.square(grid[pixel_index, 0] - border_grid[:, 0]) +
                                                np.square(grid[pixel_index, 1] - border_grid[:, 1]))

                move_factor = border_grid_radii[closest_pixel_index] / grid_radii[pixel_index]
                if move_factor < 1.0:
                    grid[pixel_index, :] = move_factor *(grid[pixel_index, :] - border_origin[:]) + border_origin[:]

        return grid

    @property
    def total_pixels(self):
        return self.shape[0]

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(RegularGridBorder, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super(RegularGridBorder, self).__setstate__(state[0:-1])