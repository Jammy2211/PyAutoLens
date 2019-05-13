import numpy as np
import scipy.spatial.qhull as qhull
from functools import wraps

from autolens import decorator_util
from autolens.data.array import mask as msk, scaled_array
from autolens.data.array.util import grid_util
from autolens.data.array.util import mapping_util, array_util, mask_util


def sub_to_image_grid(func):
    """
    Wrap the function in a function that, if the grid is a sub-grid (grids.SubGrid), rebins the computed \
    values to the sub-grids corresponding regular-grid by taking the mean of each set of sub-gridded values.

    Parameters
    ----------
    func : (profiles, *args, **kwargs) -> Object
        A function that requires the sub-grid and galaxies.
    """

    @wraps(func)
    def wrapper(grid, galaxies, *args, **kwargs):
        """

        Parameters
        ----------
        grid : RegularGrid
            The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
        galaxies : [Galaxy]
            The list of galaxies a profile quantity (e.g. intensities) is computed for which is rebinned if it \
            is a sub-grid.

        Returns
        -------
        ndarray
            If a RegularGrid is input, the profile quantity of the galaxy (e.g. intensities) evaluated using this grid.

            If a SubGrid is input, the profile quantity of the galaxy (e.g. intensities) evaluated using the sub-grid \
            and rebinned to the regular-grids array dimensions.

        Examples
        ---------
        @grids.sub_to_image_grid
        def intensities_of_galaxies_from_grid(grid, galaxies):
            return sum(map(lambda g: g.intensities_from_grid(grid), galaxies))

        galaxy_util.intensities_of_galaxies_from_grid(grid=grid_stack.sub, galaxies=galaxies)
        """

        result = func(grid, galaxies, *args, *kwargs)

        if isinstance(grid, SubGrid):
            return grid.regular_data_1d_from_sub_data_1d(result)
        else:
            return result

    return wrapper


class GridStack(object):

    def __init__(self, regular, sub, blurring, pix=None):
        """A 'stack' of grid_stack which contain the (y,x) arc-second coordinates of pixels in a mask. The stack \
        contains at least 3 grid_stack:

        regular - the (y,x) coordinate at the center of every unmasked pixel.

        sub - the (y,x) coordinates of every sub-pixel in every unmasked pixel, each using a grid of size \
        (sub_grid_size x sub_grid_size).

        blurring - the (y,x) coordinates of all blurring pixels, which are masked pixels whose light is \
        blurred into unmasked pixels during PSF convolution.

        There are also optional grid_stack, used if specific PyAutoLens functionality is required:

        pix - the (y,x) coordinates of the grid which is used to form the pixels of a \
        *pixelizations.AdaptivePixelization* pixelization.

        The grid_stack are stored as 2D arrays, where each entry corresponds to the (y,x) coordinates of a pixel. The
        positive y-axis is upwards and poitive x-axis to the right. The array is ordered such pixels begin from the \
        top-row of the mask and go rightwards and then downwards.

        Parameters
        -----------
        regular : RegularGrid
            The grid of (y,x) arc-second coordinates at the centre of every unmasked pixel.
        sub : SubGrid | np.ndarray
            The grid of (y,x) arc-second coordinates at the centre of every unmasked pixel's sub-pixels.
        blurring : RegularGrid | ndarray | None
            The grid of (y,x) arc-second coordinates at the centre of every blurring-mask pixel.
        pix : PixGrid | ndarray | None
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

    def unmasked_blurred_image_from_psf_and_unmasked_image(self, psf, unmasked_image_1d):
        """For a padded grid-stack and psf, compute an unmasked blurred image from an unmasked unblurred image.

        This relies on using the lens data's padded-grid, which is a grid of (y,x) coordinates which extends over the \
        entire image as opposed to just the masked region.

        Parameters
        ----------
        psf : ccd.PSF
            The PSF of the image used for convolution.
        unmasked_image_1d : ndarray
            The 1D unmasked image which is blurred.
        """
        blurred_image_1d = self.regular.convolve_array_1d_with_psf(padded_array_1d=unmasked_image_1d,
                                                                   psf=psf)

        return self.regular.scaled_array_2d_from_array_1d(array_1d=blurred_image_1d)

    @classmethod
    def grid_stack_from_mask_sub_grid_size_and_psf_shape(cls, mask, sub_grid_size, psf_shape, optimal_sub_grid=True):
        """Setup a grid-stack of grid_stack from a mask, sub-grid size and psf-shape.

        Parameters
        -----------
        mask : Mask
            The mask whose unmasked pixels (*False*) are used to generate the grid-stack's grid_stack.
        sub_grid_size : int
            The size of a sub-pixel's sub-grid (sub_grid_size x sub_grid_size).
        psf_shape : (int, int)
            the shape of the PSF used in the analysis, which defines the mask's blurring-region.
        """
        regular_grid = RegularGrid.from_mask(mask=mask)
        sub_grid = SubGrid.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=sub_grid_size,
                                                       optimal_sub_grid=optimal_sub_grid)
        blurring_grid = RegularGrid.blurring_grid_from_mask_and_psf_shape(mask=mask, psf_shape=psf_shape)
        return GridStack(regular_grid, sub_grid, blurring_grid)

    @classmethod
    def from_shape_pixel_scale_and_sub_grid_size(cls, shape, pixel_scale, sub_grid_size=2, optimal_sub_grid=True):
        """Setup a grid-stack of grid_stack from a 2D array shape, a pixel scale and a sub-grid size.
        
        This grid corresponds to a fully unmasked 2D array.

        Parameters
        -----------
        shape : (int, int)
            The 2D shape of the array, where all pixels are used to generate the grid-stack's grid_stack.
        pixel_scale : float
            The size of each pixel in arc seconds.            
        sub_grid_size : int
            The size of a sub-pixel's sub-grid (sub_grid_size x sub_grid_size).
        """
        regular_grid = RegularGrid.from_shape_and_pixel_scale(shape=shape, pixel_scale=pixel_scale)
        sub_grid = SubGrid.from_shape_pixel_scale_and_sub_grid_size(
            shape=shape, pixel_scale=pixel_scale, sub_grid_size=sub_grid_size, optimal_sub_grid=optimal_sub_grid)
        blurring_grid = np.array([[0.0, 0.0]])
        return GridStack(regular_grid, sub_grid, blurring_grid)

    @classmethod
    def padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(cls, mask, sub_grid_size, psf_shape,
                                                                optimal_sub_grid=True):
        """Setup a grid-stack of masked grid_stack from a mask,  sub-grid size and psf-shape.

        Parameters
        -----------
        mask : Mask
            The mask whose masked pixels the grid-stack are setup using.
        sub_grid_size : int
            The size of a sub-pixels sub-grid (sub_grid_size x sub_grid_size).
        psf_shape : (int, int)
            The shape of the PSF used in the analysis, which defines the mask's blurring-region.
        """
        regular_padded_grid = PaddedRegularGrid.padded_grid_from_shape_psf_shape_and_pixel_scale(
            shape=mask.shape, psf_shape=psf_shape, pixel_scale=mask.pixel_scale)
        sub_padded_grid = PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(
            mask=mask, sub_grid_size=sub_grid_size, psf_shape=psf_shape, optimal_sub_grid=optimal_sub_grid)
        # TODO : The blurring grid is not used when the grid mapper is called, the 0.0 0.0 stops errors inr ayT_racing
        # TODO : implement a more explicit solution
        return GridStack(regular=regular_padded_grid, sub=sub_padded_grid, blurring=np.array([[0.0, 0.0]]))

    @classmethod
    def grid_stack_for_simulation(cls, shape, pixel_scale, psf_shape, sub_grid_size=2, optimal_sub_grid=True):
        """Setup a grid-stack of grid_stack for simulating an image of a strong lens, whereby the grid's use \
        padded-grid_stack to ensure that the PSF blurring in the simulation routine (*ccd.PrepatoryImage.simulate*) \
        is not degraded due to edge effects.

        Parameters
        -----------
        shape : (int, int)
            The 2D shape of the array, where all pixels are used to generate the grid-stack's grid_stack.
        pixel_scale : float
            The size of each pixel in arc seconds.            
        psf_shape : (int, int)
            The shape of the PSF used in the analysis, which defines how much the grid's must be masked to mitigate \
            edge effects.
        sub_grid_size : int
            The size of a sub-pixel's sub-grid (sub_grid_size x sub_grid_size).
        """
        return cls.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=msk.Mask(array=np.full(shape, False), pixel_scale=pixel_scale), sub_grid_size=sub_grid_size,
            psf_shape=psf_shape, optimal_sub_grid=optimal_sub_grid)

    def new_grid_stack_with_interpolator_added_to_each_grid(self, interp_pixel_scale):
        regular = self.regular.new_grid_with_interpolator(interp_pixel_scale=interp_pixel_scale)
        sub = self.sub.new_grid_with_interpolator(interp_pixel_scale=interp_pixel_scale)

        # TODO : Like the TODO above, we need to elegently handle a blurring grid of None.

        if self.blurring.shape != (1, 2):
            blurring = self.blurring.new_grid_with_interpolator(interp_pixel_scale=interp_pixel_scale)
        else:
            blurring = np.array([[0.0, 0.0]])
        return GridStack(regular=regular, sub=sub, blurring=blurring, pix=self.pix)

    def new_grid_stack_with_pix_grid_added(self, pix_grid, regular_to_nearest_pix):
        """Setup a grid-stack of grid_stack using an existing grid-stack.
        
        The new grid-stack has the same grid_stack (regular, sub, blurring, etc.) as before, but adds a pix-grid as a \
        new attribute.

        Parameters
        -----------
        pix_grid : ndarray
            The grid of (y,x) arc-second coordinates of every image-plane pixelization grid used for adaptive \
             pixelizations.
        regular_to_nearest_pix : ndarray
            A 1D array that maps every regular-grid pixel to its nearest pix-grid pixel.
        """
        pix = PixGrid(arr=pix_grid, regular_to_nearest_pix=regular_to_nearest_pix)
        return GridStack(regular=self.regular, sub=self.sub, blurring=self.blurring, pix=pix)

    def apply_function(self, func):
        """Apply a function to all grid_stack in the grid-stack.
        
        This is used by the *ray-tracing* module to easily apply tracing operations to all grid_stack."""
        if self.blurring is not None and self.pix is not None:
            return GridStack(func(self.regular), func(self.sub), func(self.blurring), func(self.pix))
        elif self.blurring is None and self.pix is not None:
            return GridStack(func(self.regular), func(self.sub), self.blurring, func(self.pix))
        elif self.blurring is not None and self.pix is None:
            return GridStack(func(self.regular), func(self.sub), func(self.blurring), self.pix)
        else:
            return GridStack(func(self.regular), func(self.sub), self.blurring, self.pix)

    def map_function(self, func, *arg_lists):
        """Map a function to all grid_stack in a grid-stack"""
        return GridStack(*[func(*args) for args in zip(self, *arg_lists)])

    def scaled_array_2d_from_array_1d(self, array_1d):
        return self.regular.scaled_array_2d_from_array_1d(array_1d=array_1d)

    @property
    def sub_pixels(self):
        return self.sub.shape[0]

    def __getitem__(self, item):
        return [self.regular, self.sub, self.blurring, self.pix][item]


class RegularGrid(np.ndarray):

    def __new__(cls, arr, mask, *args, **kwargs):
        """A regular grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of an \
        unmasked pixel. The positive y-axis is upwards and poitive x-axis to the right. 
        
        A *RegularGrid* is ordered such pixels begin from the top-row of the mask and go rightwards and then \ 
        downwards. Therefore, it is a ndarray of shape [total_unmasked_pixels, 2]. The first element of the ndarray \
        thus corresponds to the regular pixel index and second element the y or x arc -econd coordinates. For example:

        - regular_grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - regular_grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Below is a visual illustration of a regular-grid, where a total of 10 pixels are unmasked and are included in \
        the grid.

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|o|o|x|x|x|x|     x = True (Pixel is masked and excluded from the regular grid)
        |x|x|x|o|o|o|o|x|x|x|     o = False (Pixel is not masked and included in the regular grid)
        |x|x|x|o|o|o|o|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        The mask pixel index's will come out like this (and the direction of arc-second coordinates is highlighted
        around the mask.

        pixel_scale = 1.0"

        <--- -ve  x  +ve -->
                                                        y      x
        |x|x|x|x|x|x|x|x|x|x|  ^   regular_grid[0] = [ 1.5, -0.5]
        |x|x|x|x|x|x|x|x|x|x|  |   regular_grid[1] = [ 1.5,  0.5]
        |x|x|x|x|x|x|x|x|x|x|  |   regular_grid[2] = [ 0.5, -1.5]
        |x|x|x|x|0|1|x|x|x|x| +ve  regular_grid[3] = [ 0.5, -0.5]
        |x|x|x|2|3|4|5|x|x|x|  y   regular_grid[4] = [ 0.5,  0.5]
        |x|x|x|6|7|8|9|x|x|x| -ve  regular_grid[5] = [ 0.5,  1.5]
        |x|x|x|x|x|x|x|x|x|x|  |   regular_grid[6] = [-0.5, -1.5]
        |x|x|x|x|x|x|x|x|x|x|  |   regular_grid[7] = [-0.5, -0.5]
        |x|x|x|x|x|x|x|x|x|x| \/   regular_grid[8] = [-0.5,  0.5]
        |x|x|x|x|x|x|x|x|x|x|      regular_grid[9] = [-0.5,  1.5]
        """
        obj = arr.view(cls)
        obj.mask = mask
        obj.interpolator = None
        return obj

    def __array_finalize__(self, obj):
        if hasattr(obj, "mask"):
            self.mask = obj.mask
        if hasattr(obj, "interpolator"):
            self.interpolator = obj.interpolator

    @property
    def masked_shape_arcsec(self):
        return np.amax(self[:, 0]) - np.amin(self[:, 0]), np.amax(self[:, 1]) - np.amin(self[:, 1])

    @property
    def unlensed_grid(self):
        return RegularGrid(arr=grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
            mask=self.mask,
            pixel_scales=self.mask.pixel_scales),
            mask=self.mask)

    @property
    def unlensed_unmasked_grid(self):
        return RegularGrid(arr=grid_util.regular_grid_1d_from_shape_pixel_scales_and_origin(
            shape=self.mask.shape,
            pixel_scales=self.mask.pixel_scales),
            mask=self.mask)

    def convolve_array_1d_with_psf(self, padded_array_1d, psf):
        """Convolve a 1d padded array of values (e.g. intensities before PSF blurring) with a PSF, and then trim \
        the convolved array to its original 2D shape.

        Parameters
        -----------
        padded_array_1d: ndarray
            A 1D array of values which were computed using the *PaddedRegularGrid*.
        psf : ndarray
            An array describing the PSF kernel of the image.
        """
        padded_array_2d = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(
            array_1d=padded_array_1d, shape=self.mask.shape)
        # noinspection PyUnresolvedReferences
        blurred_padded_array_2d = psf.convolve(array=padded_array_2d)
        return mapping_util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(array_2d=blurred_padded_array_2d,
                                                                                   mask=np.full(self.mask.shape, False))

    @classmethod
    def from_mask(cls, mask):
        """Setup a regular-grid from a mask, wehere the center of every unmasked pixel gives the grid's (y,x) \
        arc-second coordinates.

        Parameters
        -----------
        mask : Mask
            The mask whose unmasked pixels are used to setup the regular-pixel grid.
        """
        array = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask,
                                                                                   pixel_scales=mask.pixel_scales)
        return cls(array, mask)

    @classmethod
    def from_shape_and_pixel_scale(cls, shape, pixel_scale):
        """Setup a regular-grid from a 2D array shape and pixel scale. Here, the center of every pixel on the 2D \
        array gives the grid's (y,x) arc-second coordinates. 
         
        This is equivalent to using a 2D mask consisting entirely of unmasked pixels.

        Parameters
        -----------
        shape : (int, int)
            The 2D shape of the array, where all pixels are used to generate the grid-stack's grid_stack.
        pixel_scale : float
            The size of each pixel in arc seconds.                 
        """
        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=shape, pixel_scale=pixel_scale)
        array = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=mask,
                                                                                   pixel_scales=mask.pixel_scales)
        return cls(array, mask)

    @classmethod
    def blurring_grid_from_mask_and_psf_shape(cls, mask, psf_shape):
        """Setup a blurring-grid from a mask, where a blurring grid consists of all pixels that are masked, but they \
        are close enough to the unmasked pixels that a fraction of their light will be blurred into those pixels \
        via PSF convolution. For example, if our mask is as follows:
        
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an ccd.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from lens)
        |x|x|x|o|o|o|x|x|x|x|     o = False (Pixel is not masked and included in lens)
        |x|x|x|o|o|o|x|x|x|x|
        |x|x|x|o|o|o|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        
        For a PSF of shape (3,3), the following blurring mask is computed (noting that only pixels that are direct \
        neighbors of the unmasked pixels above will blur light into an unmasked pixel):

        |x|x|x|x|x|x|x|x|x|     This is an example regular.Mask, where:
        |x|x|x|x|x|x|x|x|x|
        |x|x|o|o|o|o|o|x|x|     x = True (Pixel is masked and excluded from lens)
        |x|x|o|x|x|x|o|x|x|     o = False (Pixel is not masked and included in lens)
        |x|x|o|x|x|x|o|x|x|
        |x|x|o|x|x|x|o|x|x|
        |x|x|o|o|o|o|o|x|x|
        |x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|
        
        Thus, the blurring grid coordinates and indexes will be as follows:
        
        pixel_scale = 1.0"

        <--- -ve  x  +ve -->
                                                            y     x
        |x|x|x |x |x |x |x |x|x|  |   blurring_grid[0] = [2.0, -2.0]  blurring_grid[9] =  [-1.0, -2.0]
        |x|x|x |x |x |x |x |x|x|  |   blurring_grid[1] = [2.0, -1.0]  blurring_grid[10] = [-1.0,  2.0]
        |x|x|0 |1 |2 |3 |4 |x|x| +ve  blurring_grid[2] = [2.0,  0.0]  blurring_grid[11] = [-2.0, -2.0]
        |x|x|5 |x |x |x |6 |x|x|  y   blurring_grid[3] = [2.0,  1.0]  blurring_grid[12] = [-2.0, -1.0]
        |x|x|7 |x |x |x |8 |x|x| -ve  blurring_grid[4] = [2.0,  2.0]  blurring_grid[13] = [-2.0,  0.0]
        |x|x|9 |x |x |x |10|x|x|  |   blurring_grid[5] = [1.0, -2.0]  blurring_grid[14] = [-2.0,  1.0]
        |x|x|11|12|13|14|15|x|x|  |   blurring_grid[6] = [1.0,  2.0]  blurring_grid[15] = [-2.0,  2.0]
        |x|x|x |x |x |x |x |x|x| \/   blurring_grid[7] = [0.0, -2.0]
        |x|x|x |x |x |x |x |x|x|      blurring_grid[8] = [0.0,  2.0]
        
        For a PSF of shape (5,5), the following blurring mask is computed (noting that pixels that are 2 pixels from an
        direct unmasked pixels now blur light into an unmasked pixel):

        |x|x|x|x|x|x|x|x|x|     This is an example regular.Mask, where:
        |x|o|o|o|o|o|o|o|x|
        |x|o|o|o|o|o|o|o|x|     x = True (Pixel is masked and excluded from lens)
        |x|o|o|x|x|x|o|o|x|     o = False (Pixel is not masked and included in lens)
        |x|o|o|x|x|x|o|o|x|
        |x|o|o|x|x|x|o|o|x|
        |x|o|o|o|o|o|o|o|x|
        |x|o|o|o|o|o|o|o|x|
        |x|x|x|x|x|x|x|x|x|
        """
        blurring_mask = mask.blurring_mask_for_psf_shape(psf_shape)
        return RegularGrid.from_mask(blurring_mask)

    def new_grid_with_interpolator(self, interp_pixel_scale):
        # noinspection PyAttributeOutsideInit
        # TODO: This function doesn't do what it says on the tin. The returned grid would be the same as the grid
        # TODO: on which the function was called but with a new interpolator set.
        self.interpolator = Interpolator.from_mask_grid_and_interp_pixel_scales(mask=self.mask, grid=self[:, :],
                                                                                interp_pixel_scale=interp_pixel_scale)
        return self

    def array_2d_from_array_1d(self, array_1d):
        """ Map a 1D array the same dimension as the grid to its original masked 2D array.

        Parameters
        -----------
        array_1d : ndarray
            The 1D array which is mapped to its masked 2D array.
        """
        return mapping_util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(
            array_1d=array_1d, shape=self.mask.shape, one_to_two=self.mask.masked_grid_index_to_pixel)

    def scaled_array_2d_from_array_1d(self, array_1d):
        """ Map a 1D array the same dimension as the grid to its original masked 2D array and return it as a scaled \
        array.

        Parameters
        -----------
        array_1d : ndarray
            The 1D array of which is mapped to a 2D scaled array.
        """
        return scaled_array.ScaledSquarePixelArray(array=self.array_2d_from_array_1d(array_1d),
                                                   pixel_scale=self.mask.pixel_scale,
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

    # noinspection PyUnusedLocal
    def __init__(self, array, mask, sub_grid_size=1):
        """ A sub-grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of each \
        sub-pixel of an unmasked pixel (e.g. the pixels of a regular-grid). The positive y-axis is upwards and poitive \
        x-axis to the right, and this convention is followed for the sub-pixels in each unmasked pixel.

        A *SubGrid* is ordered such that pixels begin from the first (top-left) sub-pixel in the first unmasked pixel. \
        Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel. Therefore, \
        the sub-grid is an ndarray of shape [total_unmasked_pixels*(sub_grid_shape)**2, 2]. For example:

        - sub_grid[9, 1] - using a 2x2 sub-grid, gives the 3rd unmasked pixel's 2nd sub-pixel x-coordinate.
        - sub_grid[9, 1] - using a 3x3 sub-grid, gives the 2nd unmasked pixel's 1st sub-pixel x-coordinate.
        - sub_grid[27, 0] - using a 3x3 sub-grid, gives the 4th unmasked pixel's 1st sub-pixel y-coordinate.

        Below is a visual illustration of a sub grid. Like the regular grid, the indexing of each sub-pixel goes from \
        the top-left corner. In contrast to the regular grid above, our illustration below restricts the mask to just \
        2 pixels, to keep the illustration brief.

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from lens)
        |x|x|x|x|o|o|x|x|x|x|     o = False (Pixel is not masked and included in lens)
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
        |x|x|x|x|x|x|x|x|x|x|  |                        y     x
        |x|x|x|x|x|x|x|x|x|x| +ve  regular_grid[0] = [0.5,  -1.5]
        |x|x|x|0|1|x|x|x|x|x|  y   regular_grid[1] = [0.5,  -0.5]
        |x|x|x|x|x|x|x|x|x|x| -ve
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x| \/
        |x|x|x|x|x|x|x|x|x|x|

        However, we now go to each unmasked pixel and derive a sub-pixel grid for it. For example, for pixel 0,
        if *sub_grid_size=2*, we use a 2x2 sub-grid:

        Pixel 0 - (2x2):
                                y      x
               sub_grid[0] = [0.66, -1.66]
        |0|1|  sub_grid[1] = [0.66, -1.33]
        |2|3|  sub_grid[2] = [0.33, -1.66]
               sub_grid[3] = [0.33, -1.33]

        Now, we'd normally sub-grid all pixels using the same *sub_grid_size*, but for this illustration lets
        pretend we used a sub_grid_size of 3x3 for pixel 1:

                                  y      x
                 sub_grid[0] = [0.75, -0.75]
                 sub_grid[1] = [0.75, -0.5]
                 sub_grid[2] = [0.75, -0.25]
        |0|1|2|  sub_grid[3] = [0.5,  -0.75]
        |3|4|5|  sub_grid[4] = [0.5,  -0.5]
        |6|7|8|  sub_grid[5] = [0.5,  -0.25]
                 sub_grid[6] = [0.25, -0.75]
                 sub_grid[7] = [0.25, -0.5]
                 sub_grid[8] = [0.25, -0.25]
        """
        # noinspection PyArgumentList
        super(SubGrid, self).__init__()
        self.mask = mask
        self.sub_grid_size = sub_grid_size
        self.sub_grid_length = int(sub_grid_size ** 2.0)
        self.sub_grid_fraction = 1.0 / self.sub_grid_length

    @property
    def unlensed_grid(self):
        return SubGrid(grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size_optimal_spacing(
            mask=self.mask, pixel_scales=self.mask.pixel_scales, sub_grid_size=self.sub_grid_size),
            self.mask, self.sub_grid_size)

    @property
    def unlensed_unmasked_grid(self):
        return SubGrid(grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size_optimal_spacing(
            mask=np.full(self.mask.shape, False), pixel_scales=self.mask.pixel_scales,
            sub_grid_size=self.sub_grid_size),
            mask=self.mask, sub_grid_size=self.sub_grid_size)

    @classmethod
    def from_mask_and_sub_grid_size(cls, mask, sub_grid_size=1, optimal_sub_grid=True):
        """Setup a sub-grid of the unmasked pixels, using a mask and a specified sub-grid size. The center of \
        every unmasked pixel's sub-pixels give the grid's (y,x) arc-second coordinates.

        Parameters
        -----------
        mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid_stack.
        sub_grid_size : int
            The size (sub_grid_size x sub_grid_size) of each unmasked pixels sub-grid.
        """
        sub_grid_masked = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=mask, pixel_scales=mask.pixel_scales, sub_grid_size=sub_grid_size, optimal_sub_grid=optimal_sub_grid)
        return SubGrid(sub_grid_masked, mask, sub_grid_size)

    @classmethod
    def from_shape_pixel_scale_and_sub_grid_size(cls, shape, pixel_scale, sub_grid_size, optimal_sub_grid=True):
        """Setup a sub-grid from a 2D array shape and pixel scale. Here, the center of every pixel on the 2D \
        array gives the grid's (y,x) arc-second coordinates, where each pixel has sub-pixels specified by the \
        sub-grid size.

        This is equivalent to using a 2D mask consisting entirely of unmasked pixels.

        Parameters
        -----------
        shape : (int, int)
            The 2D shape of the array, where all pixels are used to generate the grid-stack's grid_stack.
        pixel_scale : float
            The size of each pixel in arc seconds.
        sub_grid_size : int
            The size (sub_grid_size x sub_grid_size) of each unmasked pixels sub-grid.
        """
        mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=shape, pixel_scale=pixel_scale)
        sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=mask, pixel_scales=mask.pixel_scales, sub_grid_size=sub_grid_size, optimal_sub_grid=optimal_sub_grid)
        return SubGrid(sub_grid, mask, sub_grid_size)

    def sub_array_2d_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        sub_shape = (self.mask.shape[0] * self.sub_grid_size, self.mask.shape[1] * self.sub_grid_size)
        sub_one_to_two = self.mask.masked_sub_grid_index_to_sub_pixel(sub_grid_size=self.sub_grid_size)
        return mapping_util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(
            array_1d=sub_array_1d, shape=sub_shape, one_to_two=sub_one_to_two)

    def scaled_array_2d_with_sub_dimensions_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid to its original masked 2D sub-array and return it as
        a scaled array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub-array of which is mapped to a 2D scaled sub-array the dimensions.
        """
        return scaled_array.ScaledSquarePixelArray(array=self.sub_array_2d_from_sub_array_1d(sub_array_1d=sub_array_1d),
                                                   pixel_scale=self.mask.pixel_scale / self.sub_grid_size,
                                                   origin=self.mask.origin)

    def scaled_array_2d_with_regular_dimensions_from_binned_up_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid to its original masked 2D sub-array and return it as
        a scaled array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub-array of which is mapped to a 2D scaled sub-array the dimensions.
        """
        array_1d = self.regular_data_1d_from_sub_data_1d(sub_array_1d=sub_array_1d)
        return scaled_array.ScaledSquarePixelArray(array=self.array_2d_from_array_1d(array_1d=array_1d),
                                                   pixel_scale=self.mask.pixel_scale,
                                                   origin=self.mask.origin)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if isinstance(obj, SubGrid):
            self.sub_grid_size = obj.sub_grid_size
            self.sub_grid_length = obj.sub_grid_length
            self.sub_grid_fraction = obj.sub_grid_fraction
            self.mask = obj.mask

    def regular_data_1d_from_sub_data_1d(self, sub_array_1d):
        """For an input sub-gridded array, map its hyper-values from the sub-gridded values to a 1D regular grid of \
        values by summing each set of each set of sub-pixels values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_array_1d : ndarray
            A 1D sub-gridded array of values (e.g. the intensities, surface-densities, potential) which is mapped to
            a 1d regular array.
        """
        return np.multiply(self.sub_grid_fraction, sub_array_1d.reshape(-1, self.sub_grid_length).sum(axis=1))

    @property
    @array_util.Memoizer()
    def sub_to_regular(self):
        """The mapping between every sub-pixel and its host regular-pixel.

        For example:

        - sub_to_pixel[8] = 2 -  The ninth sub-pixel is within the 3rd regular pixel.
        - sub_to_pixel[20] = 4 -  The twenty first sub-pixel is within the 5th regular pixel.
        """
        return mapping_util.sub_to_regular_from_mask(self.mask, self.sub_grid_size).astype('int')


class PixGrid(np.ndarray):

    def __new__(cls, arr, regular_to_nearest_pix, *args, **kwargs):
        """A pix-grid of (y,x) coordinates which are used to form the pixel centres of adaptive pixelizations in the \
        *pixelizations* module.

        A *PixGrid* is ordered such pixels begin from the top-row of the mask and go rightwards and then \
        downwards. Therefore, it is a ndarray of shape [total_pix_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pix pixel index and second element the y or x arc -econd coordinates. For example:

        - pix_grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - pix_grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Parameters
        -----------
        pix_grid : ndarray
            The grid of (y,x) arc-second coordinates of every image-plane pixelization grid used for adaptive source \
            -plane pixelizations.
        regular_to_nearest_pix : ndarray
            A 1D array that maps every regular-grid pixel to its nearest pix-grid pixel.
        """
        obj = arr.view(cls)
        obj.regular_to_nearest_pix = regular_to_nearest_pix
        obj.interpolator = None
        return obj

    def __array_finalize__(self, obj):
        if hasattr(obj, "regular_to_nearest_pix"):
            self.regular_to_nearest_pix = obj.regular_to_nearest_pix
        if hasattr(obj, "interpolator"):
            self.interpolator = obj.interpolator


class SparseToRegularGrid(scaled_array.RectangularArrayGeometry):

    def __init__(self, unmasked_sparse_grid_shape, pixel_scales, regular_grid, origin=(0.0, 0.0)):
        """A sparse grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of a \
        pixel on the sparse grid. To setup the sparse-grid, it is laid over a regular-grid of unmasked pixels, such \
        that all sparse-grid pixels which map inside of an unmasked regular-grid pixel are included on the sparse grid.

        To setup this sparse grid, we thus have two sparse grid_stack:

        - The unmasked sparse-grid, which corresponds to a uniform 2D array of pixels. The edges of this grid \
          correspond to the 4 edges of the mask (e.g. the higher and lowest (y,x) arc-second unmasked pixels) and the \
          grid's shape is speciifed by the unmasked_sparse_grid_shape parameter.

        - The (masked) sparse-grid, which is all pixels on the unmasked sparse-grid above which fall within unmasked \
          regular-grid pixels. These are the pixels which are actually used for other modules in PyAutoLens.

        The origin of the unmasked sparse grid can be changed to allow off-center pairings with sparse-grid pixels, \
        which is necessary when a mask has a centre offset from (0.0", 0.0"). However, the sparse grid itself \
        retains an origin of (0.0", 0.0"), ensuring its arc-second grid uses the same coordinate system as the \
        other grid_stack.

        The sparse grid is used to determine the pixel centers of an adaptive grid pixelization.

        Parameters
        ----------
        unmasked_sparse_grid_shape : (int, int)
            The shape of the unmasked sparse-grid whose centres form the sparse-grid.
        pixel_scales : (float, float)
            The pixel-to-arcsecond scale of a pixel in the y and x directions.
        regular_grid : RegularGrid
            The regular-grid used to determine which pixels are in the sparse grid.
        origin : (float, float)
            The centre of the unmasked sparse grid, which matches the centre of the mask.
        """

        self.shape = unmasked_sparse_grid_shape
        self.origin = origin
        self.total_unmasked_sparse_pixels = int(self.shape[0] * self.shape[1])
        self.pixel_scales = pixel_scales

        self.regular_grid = regular_grid
        self.unmasked_sparse_grid = self.grid_1d
        self.unmasked_sparse_grid_pixel_centres = \
            regular_grid.mask.grid_arcsec_to_grid_pixel_centres(self.unmasked_sparse_grid)

        self.total_sparse_pixels = mask_util.total_sparse_pixels_from_mask(
            mask=self.regular_grid.mask,
            unmasked_sparse_grid_pixel_centres=self.unmasked_sparse_grid_pixel_centres)

    @property
    def unmasked_sparse_to_sparse(self):
        """The 1D index mappings between the unmasked sparse-grid and masked sparse grid."""

        return mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(
            mask=self.regular_grid.mask,
            unmasked_sparse_grid_pixel_centres=self.unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=self.total_sparse_pixels).astype(
            'int')

    @property
    def sparse_to_unmasked_sparse(self):
        """The 1D index mappings between the masked sparse-grid and unmasked sparse grid."""
        return mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=self.total_sparse_pixels, mask=self.regular_grid.mask,
            unmasked_sparse_grid_pixel_centres=self.unmasked_sparse_grid_pixel_centres).astype('int')

    @property
    def regular_to_unmasked_sparse(self):
        """The 1D index mapping between the regular-grid and unmasked sparse-grid."""
        return self.grid_arcsec_to_grid_pixel_indexes(grid_arcsec=self.regular_grid)

    @property
    def regular_to_sparse(self):
        """The 1D index mappings between the regular-grid and masked sparse-grid."""

        return mapping_util.regular_to_sparse_from_sparse_mappings(
            regular_to_unmasked_sparse=self.regular_to_unmasked_sparse,
            unmasked_sparse_to_sparse=self.unmasked_sparse_to_sparse).astype('int')

    @property
    def sparse_grid(self):
        """The (y,x) arc-second coordinates of the masked sparse-grid."""
        return mapping_util.sparse_grid_from_unmasked_sparse_grid(
            unmasked_sparse_grid=self.unmasked_sparse_grid,
            sparse_to_unmasked_sparse=self.sparse_to_unmasked_sparse)


class PaddedRegularGrid(RegularGrid):

    def __new__(cls, arr, mask, image_shape, *args, **kwargs):
        """A padded regular-grid stores the (y,x) arc-second coordinates of a 2D array, analogous to a *RegularGrid*. \
         Howeve, the padded grid deviates from a regular grid in that:

        - All pixels are used to form the grid (as opposed to just masked pixels)
        - The mask is padded when computing the grid, such that additional pixels beyond its edge are included. This \
          padding extends based on the size of the PSF kernel, such that PSF blurring effects at the edge of the \
          array do not degrade the image.

        Padded-grid_stack are therefore used to compute quantities like light-profile intensities large 2D arrays \
        that span the entire image (as opposed to just the masked region).
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
    def padded_grid_from_shape_psf_shape_and_pixel_scale(cls, shape, psf_shape, pixel_scale):
        """Setup a regular padded grid from a 2D array shape, psf-shape and pixel-scale.

        The center of every pixel is used to setup the grid's (y,x) arc-second coordinates, including padded pixels \
        which are beyond the input shape but will blurred light into the 2D array's shape due to the psf.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the masked-grid's 2D image in units of pixels.
        psf_shape : (int, int)
           The shape of the psf which defines the blurring region and therefore size of padding.
        pixel_scale : float
            The scale of each pixel in arc seconds
        """
        padded_shape = (shape[0] + psf_shape[0] - 1, shape[1] + psf_shape[1] - 1)
        padded_regular_grid = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
            mask=np.full(padded_shape, False), pixel_scales=(pixel_scale, pixel_scale))
        padded_mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=padded_shape, pixel_scale=pixel_scale)
        return PaddedRegularGrid(arr=padded_regular_grid, mask=padded_mask, image_shape=shape)

    def padded_blurred_image_2d_from_padded_image_1d_and_psf(self, padded_image_1d, psf):
        """Compute a 2D padded blurred image from a 1D padded image.

        Parameters
        ----------
        padded_image_1d : ndarray
            A 1D unmasked image which is blurred with the PSF.
        psf : ndarray
            An array describing the PSF kernel of the image.
        """
        padded_model_image_1d = self.convolve_array_1d_with_psf(padded_array_1d=padded_image_1d, psf=psf)
        return self.scaled_array_2d_from_array_1d(array_1d=padded_model_image_1d)

    def array_2d_from_array_1d(self, padded_array_1d):
        """ Map a padded 1D array of values to its original 2D array, trimming all edge values.

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
        return mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=padded_array_1d,
                                                                                      shape=self.mask.shape)

    @property
    def padded_shape(self):
        return self.mask.shape


class PaddedSubGrid(SubGrid, PaddedRegularGrid):

    def __init__(self, arr, mask, image_shape, sub_grid_size=1):
        """A *PaddedSubGrid* stores the (y,x) arc-second coordinates of a mask's sub-pixels in 1D, in an analogous
        fashion to a *SubGrid*. A *PaddedSubGrid* deviate from a normal grid in that:

        - All pixels are used (as opposed to just masked pixels)
        - The mask is masked when computing the grid, such that additional pixels beyond its edge are included.

        Padded-grid_stack allow quantities like intensities to be computed on large 2D arrays spanning the entire
        image, as opposed to just within the masked region.
        """
        super(PaddedSubGrid, self).__init__(arr, mask, sub_grid_size)
        self.image_shape = image_shape

    @classmethod
    def padded_grid_from_mask_sub_grid_size_and_psf_shape(cls, mask, sub_grid_size, psf_shape, optimal_sub_grid=True):
        """Setup an *PaddedSubGrid* for an input mask, sub-grid size and psf-shape.

        The center of every sub-pixel is used to setup the grid's (y,x) arc-second coordinates, including \
        masked-pixels which are beyond the input shape but will have light blurred into them given the psf-shape.

        Parameters
        ----------
        mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid_stack.
        sub_grid_size : int
            The size (sub_grid_size x sub_grid_size) of each image-pixels sub-grid.
        psf_shape : (int, int)
           The shape of the psf which defines the blurring region and therefore size of padding.
        """

        padded_shape = (mask.shape[0] + psf_shape[0] - 1, mask.shape[1] + psf_shape[1] - 1)

        padded_sub_grid = grid_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full(padded_shape, False), pixel_scales=mask.pixel_scales, sub_grid_size=sub_grid_size,
            optimal_sub_grid=optimal_sub_grid)

        padded_mask = msk.Mask.unmasked_for_shape_and_pixel_scale(shape=padded_shape, pixel_scale=mask.pixel_scale)

        return PaddedSubGrid(arr=padded_sub_grid, mask=padded_mask, image_shape=mask.shape,
                             sub_grid_size=sub_grid_size)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if isinstance(obj, PaddedSubGrid):
            self.image_shape = obj.image_shape


class RegularGridBorder(np.ndarray):

    def __new__(cls, arr, *args, **kwargs):
        """The borders of a regular grid, containing the pixel-index's of all masked pixels that are on the \
        mask's border (e.g. they are next to a *True* value in at least one of the surrounding 8 pixels and at one of \
        the exterior edge's of the mask).

        This is used to relocate demagnified pixel's in a grid to its border, so that they do not disrupt an \
        adaptive pixelization's inversion.

        Parameters
        -----------
        arr : ndarray
            A 1D array of the integer indexes of an *RegularGrid*'s borders pixels.
        """
        border = arr.view(cls)
        return border

    @classmethod
    def from_mask(cls, mask):
        """Setup the regular-grid border using a mask.

        Parameters
        -----------
        mask : Mask
            The mask the masked borders pixel index's are computed from.
        """
        return cls(mask.border_pixels)

    def relocated_grid_stack_from_grid_stack(self, grid_stack):
        """Determine a set of relocated grid_stack from an input set of grid_stack, by relocating their pixels based on the \
        borders.

        The blurring-grid does not have its coordinates relocated, as it is only used for computing analytic \
        light-profiles and not inversion-grid_stack.

        Parameters
        -----------
        grid_stack : GridStack
            The grid-stack, whose grid_stack coordinates are relocated.
        """
        border_grid = grid_stack.regular[self]
        return GridStack(regular=self.relocated_grid_from_grid_jit(grid=grid_stack.regular, border_grid=border_grid),
                         sub=self.relocated_grid_from_grid_jit(grid=grid_stack.sub, border_grid=border_grid),
                         blurring=None,
                         pix=self.relocated_grid_from_grid_jit(grid=grid_stack.pix, border_grid=border_grid))

    @staticmethod
    @decorator_util.jit()
    def relocated_grid_from_grid_jit(grid, border_grid):
        """ Relocate the coordinates of a grid to its border if they are outside the border. This is performed as \
        follows:

        1) Use the mean value of the grid's y and x coordinates to determine the origin of the grid.
        2) Compute the radial distance of every grid coordinate from the origin.
        3) For every coordinate, find its nearest pixel in the border.
        4) Determine if it is outside the border, by comparing its radial distance from the origin to its paid \
           border pixel's radial distance.
        5) If its radial distance is larger, use the ratio of radial distances to move the coordinate to the border \
           (if its inside the border, do nothing).
        """
        border_origin = np.zeros(2)
        border_origin[0] = np.mean(border_grid[:, 0])
        border_origin[1] = np.mean(border_grid[:, 1])
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
                    grid[pixel_index, :] = move_factor * (grid[pixel_index, :] - border_origin[:]) + border_origin[:]

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


class Interpolator(object):

    def __init__(self, grid, interp_grid):
        self.grid = grid
        self.interp_grid = interp_grid

        self.vtx, self.wts = self.interp_weights

    @property
    def interp_weights(self):
        tri = qhull.Delaunay(self.interp_grid)
        simplex = tri.find_simplex(self.grid)
        # noinspection PyUnresolvedReferences
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = self.grid - temp[:, 2]
        bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    @classmethod
    def from_mask_grid_and_interp_pixel_scales(cls, mask, grid, interp_pixel_scale):
        rescale_factor = mask.pixel_scale / interp_pixel_scale
        rescaled_mask = mask_util.rescaled_mask_from_mask_and_rescale_factor(mask=mask, rescale_factor=rescale_factor)
        interp_mask = mask_util.edge_buffed_mask_from_mask(mask=rescaled_mask)
        interp_grid = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(
            mask=interp_mask,
            pixel_scales=(
                interp_pixel_scale,
                interp_pixel_scale),
            origin=mask.origin)
        return Interpolator(grid=grid, interp_grid=interp_grid)

    def interpolated_values_from_values(self, values):
        return np.einsum('nj,nj->n', np.take(values, self.vtx), self.wts)


def grid_interpolate(func):
    """
    Decorate a profile method that accepts a coordinate grid and returns a data grid.

    If an interpolator attribute is associated with the input grid then that interpolator is used to down sample the
    coordinate grid prior to calling the function and up sample the result of the function.

    If no interpolator attribute is associated with the input grid then the function is called as normal.

    Parameters
    ----------
    func
        Some method that accepts a grid

    Returns
    -------
    decorated_function
        The function with optional interpolation
    """

    @wraps(func)
    def wrapper(profile, grid, grid_radial_minimum=None, *args, **kwargs):
        if hasattr(grid, "interpolator"):
            interpolator = grid.interpolator
            if grid.interpolator is not None:
                values = func(profile, interpolator.interp_grid, grid_radial_minimum, *args, **kwargs)
                if values.ndim == 1:
                    return interpolator.interpolated_values_from_values(values=values)
                elif values.ndim == 2:
                    y_values = interpolator.interpolated_values_from_values(values=values[:, 0])
                    x_values = interpolator.interpolated_values_from_values(values=values[:, 1])
                    return np.asarray([y_values, x_values]).T
        return func(profile, grid, grid_radial_minimum, *args, **kwargs)

    return wrapper
