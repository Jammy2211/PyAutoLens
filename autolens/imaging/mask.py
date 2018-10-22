import logging

import numpy as np

from autolens import exc
from autolens.imaging import imaging_util
from autolens.imaging import scaled_array

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask(scaled_array.ScaledSquarePixelArray):
    """
    A mask represented by an ndarray where True is masked.
    """

    def __getitem__(self, coords):
        try:
            return super(Mask, self).__getitem__(coords)
        except IndexError:
            return True

    @classmethod
    def padded_for_shape_and_pixel_scale(cls, shape, pixel_scale):
        """
        Setup the mask such that all pixels are padded.

        Parameters
        ----------
        shape : (int, int)
            The (x,y) shape of the mask in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls(np.full(tuple(map(lambda d: int(d), shape)), False, dtype='bool'), pixel_scale)

    @classmethod
    def masked_for_shape_and_pixel_scale(cls, shape, pixel_scale):
        """
        Setup the mask such that all pixels are masked.

        Parameters
        ----------
        shape : (int, int)
            The (x,y) shape of the mask in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls(np.full(tuple(map(lambda d: int(d), shape)), True, dtype='bool'), pixel_scale)

    @classmethod
    def circular(cls, shape, pixel_scale, radius_mask_arcsec, centre=(0., 0.)):
        """
        Setup the mask as a circle, using a specified arc second radius.

        Parameters
        ----------
        shape: (int, int)
            The (x,y) shape of the mask in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        radius_mask_arcsec : float
            The radius of the circular mask in arc seconds.
        centre: (float, float)
            The centre of the mask.
        """
        mask = imaging_util.mask_circular_from_shape_pixel_scale_and_radius(shape, pixel_scale, radius_mask_arcsec,
                                                                            centre)
        return cls(mask.astype('bool'), pixel_scale)

    @classmethod
    def annular(cls, shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec, centre=(0., 0.)):
        """
        Setup the mask as an annulus, using a specified inner and outer radius in arc seconds.

        Parameters
        ----------
        shape : (int, int)
            The (x,y) shape of the mask in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        outer_radius_arcsec : float
            The inner radius of the annulus mask in arc seconds.
        outer_radius_arcsec : float
            The outer radius of the annulus mask in arc seconds.
        centre: (float, float)
            The centre of the mask.
        """
        mask = imaging_util.mask_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec,
                                                                          outer_radius_arcsec, centre)
        return cls(mask.astype('bool'), pixel_scale)

    @classmethod
    def anti_annular(cls, shape, pixel_scale, inner_radius_arcsec, outer_radius_arcsec, outer_radius_2_arcsec,
                     centre=(0., 0.)):
        """
        Setup the mask as an annulus, using a specified inner and outer radius in arc seconds.

        Parameters
        ----------
        shape : (int, int)
            The (x,y) shape of the mask in units of pixels.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        inner_radius_arcsec : float
            The inner radius of the annulus mask in arc seconds.
        outer_radius_arcsec : float
            The outer radius of the annulus mask in arc seconds.
        outer_radius_2_arcsec
        centre: (float, float)
            The centre of the mask.
        """
        mask = imaging_util.mask_anti_annular_from_shape_pixel_scale_and_radii(shape, pixel_scale, inner_radius_arcsec,
                                                                               outer_radius_arcsec,
                                                                               outer_radius_2_arcsec, centre)
        return cls(mask, pixel_scale)

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    @property
    def grid_to_pixel(self):
        """A 1D array of mappings between every padded pixel and its 2D pixel coordinates."""
        return imaging_util.grid_to_pixel_from_mask(self).astype('int')

    def map_2d_array_to_masked_1d_array(self, array_2d):
        """For a 2D data-array (e.g. the _data, noise-mappers, etc.) mappers it to a masked 1D array of values usin this mask.

        Parameters
        ----------
        array_2d : ndarray | None | float
            The data to be mapped to a masked 1D array.
        """
        if array_2d is None or isinstance(array_2d, float):
            return array_2d
        return imaging_util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(self, array_2d)

    @imaging_util.Memoizer()
    def blurring_mask_for_psf_shape(self, psf_shape):
        """Compute the blurring mask, which represents all masked pixels whose light will be blurred into padded \
        pixels via PSF convolution.

        Parameters
        ----------
        psf_shape : (int, int)
           The shape of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        if psf_shape[0] % 2 == 0 or psf_shape[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = imaging_util.mask_blurring_from_mask_and_psf_shape(self, psf_shape)

        return Mask(blurring_mask, self.pixel_scale)

    @property
    def border_pixels(self):
        """The indicies of the mask's border pixels, where a border pixel is a pixel inside the mask but on its edge \
        (next to at least one pixel with a *True* value).
        """
        return imaging_util.border_pixels_from_mask(self).astype('int')

    def border_sub_pixels(self, sub_grid_size):
        """The indicies of the mask's border sub-pixels, where a border sub-pixel is the sub-pixel in a mask border \
        _data-pixel which is closest to the edge."""
        return imaging_util.border_sub_pixels_from_mask_pixel_scales_and_sub_grid_size(mask=self,
                                                                                       pixel_scales=self.pixel_scales,
                                                                                       sub_grid_size=sub_grid_size).astype(
            'int')


class ImagingGrids(object):

    def __init__(self, image, sub, blurring):
        """The grids containing the (x,y) arc-second coordinates of padded pixels in a mask. There are 3 grids:

        The _data-grid - the (x,y) coordinate at the center of every padded pixel.
        The sub-grid - the (x,y) coordinates of every sub-pixel in every padded pixel, each using a grid of size \
        sub_grid_size x sub_grid_size.
        The blurring-grid - the (x,y) coordinates of all blurring pixels, which are masked pixels whose light is \
        blurred into masked pixels during PSF convolution.

        The grids are stored as 1D arrays, where each entry corresponds to an padded (sub-)pixel. The 1D array is \
        ordered such pixels begin from the top-row of the masks 2D array and then downwards.

        Parameters
        -----------
        image : ImageGrid
            The grid of (x,y) arc-second coordinates at the centre of every padded pixel.
        sub : SubGrid
            The grid of (x,y) arc-second coordinates at the centre of every padded pixel's sub-pixels.
        blurring : ImageGrid | ndarray | None
            The grid of (x,y) arc-second coordinates at the centre of every blurring-mask pixel.
        """
        self.image = image
        self.sub = sub
        self.blurring = blurring

    @classmethod
    def grids_from_mask_sub_grid_size_and_psf_shape(cls, mask, sub_grid_size, psf_shape):
        """Setup the *ImagingGrids* from a mask, sub-grid size and psf-shape.

        Parameters
        -----------
        mask : Mask
            The mask whose padded pixels the imaging-grids are setup using.
        sub_grid_size : int
            The size of a sub-pixels sub-grid (sub_grid_size x sub_grid_size).
        psf_shape : (int, int)
            the shape of the PSF used in the analysis, which defines the mask's blurring-region.
        """
        image_grid = ImageGrid.from_mask(mask)
        sub_grid = SubGrid.from_mask_and_sub_grid_size(mask, sub_grid_size)
        blurring_grid = ImageGrid.blurring_grid_from_mask_and_psf_shape(mask, psf_shape)
        return ImagingGrids(image_grid, sub_grid, blurring_grid)

    @classmethod
    def from_shape_and_pixel_scale(cls, shape, pixel_scale, sub_grid_size=2, psf_shape=(1, 1)):
        image_grid = ImageGrid.from_shape_and_pixel_scale(shape=shape, pixel_scale=pixel_scale)
        sub_grid = SubGrid.from_shape_pixel_scale_and_sub_grid_size(shape=shape, pixel_scale=pixel_scale,
                                                                    sub_grid_size=sub_grid_size)
        blurring_grid = np.array([[0.0, 0.0]])
        return ImagingGrids(image_grid, sub_grid, blurring_grid)

    @classmethod
    def padded_grids_from_mask_sub_grid_size_and_psf_shape(cls, mask, sub_grid_size, psf_shape):
        """Setup the collection of padded imaging-grids from a mask, using also an input sub-grid size to setup the \
        sub-grid and psf-shape to setup the blurring grid.

        Parameters
        -----------
        mask : Mask
            The mask whose padded pixels the imaging-grids are setup using.
        sub_grid_size : int
            The size of a sub-pixels sub-grid (sub_grid_size x sub_grid_size).
        psf_shape : (int, int)
            the shape of the PSF used in the analysis, which therefore defines the mask's blurring-region.
        """
        image_padded_grid = PaddedImageGrid.padded_grid_from_shapes_and_pixel_scale(shape=mask.shape,
                                                                                    psf_shape=psf_shape,
                                                                                    pixel_scale=mask.pixel_scale)
        sub_padded_grid = PaddedSubGrid.padded_grid_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                          sub_grid_size=sub_grid_size,
                                                                                          psf_shape=psf_shape)
        # TODO : The blurring grid is not used when the grid mapper is called, the 0.0 0.0 stops errors inr ayT_racing
        # TODO : implement a more explicit solution
        return ImagingGrids(image=image_padded_grid, sub=sub_padded_grid, blurring=np.array([[0.0, 0.0]]))

    @classmethod
    def grids_for_simulation(cls, shape, pixel_scale, psf_shape, sub_grid_size=1):
        """Setup a collection of imaging-grids for simulating an _data of a strong lens. 
        
        This routine uses padded-grids which ensure that the PSF blurring in the simulation routine \ 
        (*imaging.PrepatoryImage.simulate*) is not degraded due to edge effects.

        Parameters
        -----------
        pixel_scale
        shape
        sub_grid_size : int
            The size of a sub-pixels sub-grid (sub_grid_size x sub_grid_size).
        psf_shape : (int, int)
            the shape of the PSF used in the analysis, which therefore defines the mask's blurring-region.
        """
        return cls.padded_grids_from_mask_sub_grid_size_and_psf_shape(mask=Mask(array=np.full(shape, False),
                                                                                pixel_scale=pixel_scale),
                                                                      sub_grid_size=sub_grid_size,
                                                                      psf_shape=psf_shape)

    def apply_function(self, func):
        if self.blurring is not None:
            return ImagingGrids(func(self.image), func(self.sub), func(self.blurring))
        else:
            return ImagingGrids(func(self.image), func(self.sub), self.blurring)

    def map_function(self, func, *arg_lists):
        return ImagingGrids(*[func(*args) for args in zip(self, *arg_lists)])

    @property
    def sub_pixels(self):
        return self.sub.shape[0]

    def __getitem__(self, item):
        return [self.image, self.sub, self.blurring][item]


class ImageGrid(np.ndarray):
    """Abstract class for a regular grid of coordinates, where each padded pixel's (x,y) arc-second coordinates \
    are represented by the value at the centre of the pixel.

    Coordinates are defined from the top-left corner, such that pixels in the top-left corner of a mask (e.g. [0,0]) \
    have a negative x and y-value in arc seconds. The mask pixel indexes are also counted from the top-left.

    An *ImageGrid* is a ndarray of shape [image_pixels, 2], where image_pixels is the total number of padded \
    _data-pixels. The first element maps of the ndarray corresponds to the padded pixel index and second element the \
    x or y arc second coordinates. For howtolens:

    - image_grid[3,1] = the 4th padded pixel's y-coordinate.
    - image_grid[6,0] = the 5th padded pixel's x-coordinate.

    Below is a visual illustration of an _data-grid, where a total of 10 pixels are padded and are included in the \
    grid.

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an howtolens mask.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|o|o|x|x|x|x|     x = True (Pixel is masked and excluded from lensing)
    |x|x|x|o|o|o|o|x|x|x|     o = False (Pixel is not masked and included in lensing)
    |x|x|x|o|o|o|o|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    The mask pixel index's will come out like this (and the direction of arc-second coordinates is highlighted
    around the mask.

    pixel_scale = 1.0"

    <--- -ve  x  +ve -->

    |x|x|x|x|x|x|x|x|x|x|  ^   image_grid[0] = [-0.5, -1.5]
    |x|x|x|x|x|x|x|x|x|x|  |   image_grid[1] = [ 0.5, -1.5]
    |x|x|x|x|x|x|x|x|x|x|  |   image_grid[2] = [-1.5, -0.5]
    |x|x|x|x|0|1|x|x|x|x| -ve  image_grid[3] = [-0.5, -0.5]
    |x|x|x|2|3|4|5|x|x|x|  y   image_grid[4] = [ 0.5, -0.5]
    |x|x|x|6|7|8|9|x|x|x| +ve  image_grid[5] = [ 1.5, -0.5]
    |x|x|x|x|x|x|x|x|x|x|  |   image_grid[6] = [-1.5,  0.5]
    |x|x|x|x|x|x|x|x|x|x|  |   image_grid[7] = [-0.5,  0.5]
    |x|x|x|x|x|x|x|x|x|x| \/   image_grid[8] = [ 0.5,  0.5]
    |x|x|x|x|x|x|x|x|x|x|      image_grid[9] = [ 1.5,  0.5]
    """

    def __new__(cls, arr, mask, *args, **kwargs):
        obj = arr.view(cls)
        obj.mask = mask
        return obj

    def __array_finalize__(self, obj):
        if hasattr(obj, "mask"):
            self.mask = obj.mask

    @property
    def unlensed_grid(self):
        return ImageGrid(arr=imaging_util.image_grid_1d_masked_from_mask_and_pixel_scales(mask=self.mask,
                                                                                          pixel_scales=self.mask.pixel_scales),
                         mask=self.mask)

    @property
    def unlensed_unmasked_grid(self):
        return ImageGrid(arr=imaging_util.image_grid_1d_from_shape_and_pixel_scales(shape=self.mask.shape,
                                                                                    pixel_scales=self.mask.pixel_scales),
                         mask=self.mask)

    @classmethod
    def from_mask(cls, mask):
        """Setup an *ImageGrid* of the regular _data-grid from a mask. The center of every padded pixel gives \
        the grid's (x,y) arc-second coordinates.

        Parameters
        -----------
        mask : Mask
            The mask whose padded pixels are used to setup the sub-pixel grids."""
        array = imaging_util.image_grid_1d_masked_from_mask_and_pixel_scales(mask=mask, pixel_scales=mask.pixel_scales)
        return cls(array, mask)

    @classmethod
    def from_shape_and_pixel_scale(cls, shape, pixel_scale):
        mask = Mask.padded_for_shape_and_pixel_scale(shape=shape, pixel_scale=pixel_scale)
        array = imaging_util.image_grid_1d_masked_from_mask_and_pixel_scales(mask=mask, pixel_scales=mask.pixel_scales)
        return cls(array, mask)

    @classmethod
    def blurring_grid_from_mask_and_psf_shape(cls, mask, psf_shape):
        """Setup an *ImageGrid* of the blurring-grid from a mask. The center of every padded blurring-pixel gives \
        the grid's (x,y) arc-second coordinates."""
        blurring_mask = mask.blurring_mask_for_psf_shape(psf_shape)
        return ImageGrid.from_mask(blurring_mask)

    def map_to_2d(self, array_1d):
        """ Map a 1D array the same dimension as the grid to its original 2D array.

        Parameters
        -----------
        array_1d : ndarray
            The 1D array of masked data which is mapped to 2D.
        """
        return imaging_util.map_masked_1d_array_to_2d_array_from_array_1d_shape_and_one_to_two(array_1d,
                                                                                               self.mask.shape,
                                                                                               self.mask.grid_to_pixel)

    def scaled_array_from_array_1d(self, array_1d):
        return scaled_array.ScaledSquarePixelArray(array=self.map_to_2d(array_1d), pixel_scale=self.mask.pixel_scale)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(ImageGrid, self).__reduce__()
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
        super(ImageGrid, self).__setstate__(state[0:-1])

    @property
    def total_pixels(self):
        return self.shape[0]

    @property
    def pixel_scale(self):

        return self.mask.pixel_scale

    @property
    def yticks(self):
        """Compute the yticks labels of this grid, used for plotting the y-axis ticks when visualizing an _data-grid"""
        return np.linspace(np.min(self[:, 0]), np.max(self[:, 0]), 4)

    @property
    def xticks(self):
        """Compute the xticks labels of this grid, used for plotting the x-axis ticks when visualizing an _data-grid"""
        return np.linspace(np.min(self[:, 1]), np.max(self[:, 1]), 4)


class SubGrid(ImageGrid):
    """Abstract class for a sub-grid of coordinates. On a sub-grid, each padded pixel is sub-gridded into a uniform \
     grid of (x,y) sub-coordinates..

    Coordinates are defined from the top-left corner, such that pixels in the top-left corner of an \
    mask (e.g. [0,0]) have negative x and y-values in arc seconds. The mask pixel indexes are also counted from the \
    top-left.

    Sub-pixels follow the same convention as above, where the top-left sub-pixel has the lowest x and y-values in each \
    _data-pixel. Sub-pixel indexes include all previous sub-pixels in all previous padded _data-pixels.

    A *SubGrid* is a NumPy array of shape [image_pixels*sub_grid_pixels**2, 2]. The first element of the ndarray \
    corresponds to the padded sub-pixel index, and second element the sub-pixel's (x,y) arc second coordinates. \
    For howtolens:

    - sub_grid[9, 1] - using a 2x2 sub-grid, gives the 3rd _data-pixel's 2nd sub-pixel y-coordinate.
    - sub_grid[9, 1] - using a 3x3 sub-grid, gives the 2nd _data-pixel's 1st sub-pixel y-coordinate.
    - sub_grid[27, 0] - using a 3x3 sub-grid, gives the 4th _data-pixel's 1st sub-pixel x-coordinate.

    Below is a visual illustration of a sub grid. Like the regular grid, the indexing of each sub-pixel goes from \
    the top-left corner. In contrast to the regular grid above, our illustration below restricts the mask to just
    2 pixels, to keep the illustration brief.

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an howtolens mask.Mask, where:
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
    |x|x|x|x|x|x|x|x|x|x| +ve  image_grid[0] = [-1.5,  0.5]
    |x|x|x|0|1|x|x|x|x|x|  y   image_grid[1] = [-0.5,  0.5]
    |x|x|x|x|x|x|x|x|x|x| -ve
    |x|x|x|x|x|x|x|x|x|x|  |
    |x|x|x|x|x|x|x|x|x|x|  |
    |x|x|x|x|x|x|x|x|x|x| \/
    |x|x|x|x|x|x|x|x|x|x|

    However, we now go to each mask-pixel and derive a sub-pixel grid for it. For howtolens, for pixel 0,
    if *sub_grid_size=2*, we use a 2x2 sub-grid:

    Pixel 0 - (2x2):

           image_grid[0] = [-1.66, 0.66]
    |0|1|  image_grid[1] = [-1.33, 0.66]
    |2|3|  image_grid[2] = [-1.66, 0.33]
           image_grid[3] = [-1.33, 0.33]

    Now, we'd normally sub-grid all pixels using the same *sub_grid_size*, but for this illustration lets
    pretend we used a sub_grid_size of 3x3 for pixel 1:

             image_grid[0] = [-0.75, 0.75]
             image_grid[1] = [-0.5,  0.75]
             image_grid[2] = [-0.25, 0.75]
    |0|1|2|  image_grid[3] = [-0.75,  0.5]
    |3|4|5|  image_grid[4] = [-0.5,   0.5]
    |6|7|8|  image_grid[5] = [-0.25,  0.5]
             image_grid[6] = [-0.75, 0.25]
             image_grid[7] = [-0.5,  0.25]
             image_grid[8] = [-0.25, 0.25]
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
        return SubGrid(imaging_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=self.mask,
            pixel_scales=self.mask.pixel_scales,
            sub_grid_size=self.sub_grid_size),
            self.mask, self.sub_grid_size)

    @property
    def unlensed_unmasked_grid(self):
        return SubGrid(imaging_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full(self.mask.shape, False), pixel_scales=self.mask.pixel_scales,
            sub_grid_size=self.sub_grid_size),
            mask=self.mask, sub_grid_size=self.sub_grid_size)

    @classmethod
    def from_mask_and_sub_grid_size(cls, mask, sub_grid_size=1):
        """Setup a *SubGrid* of the padded _data-pixels, using a mask and a specified sub-grid size. The center of \
        every padded pixel's sub-pixels give the grid's (x,y) arc-second coordinates.

        Parameters
        -----------
        mask : Mask
            The mask whose padded pixels are used to setup the sub-pixel grids.
        sub_grid_size : int
            The size (sub_grid_size x sub_grid_size) of each _data-pixels sub-grid.
        """
        sub_grid_masked = imaging_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask,
                                                                                                   pixel_scales=mask.pixel_scales,
                                                                                                   sub_grid_size=sub_grid_size)
        return SubGrid(sub_grid_masked, mask, sub_grid_size)

    @classmethod
    def from_shape_pixel_scale_and_sub_grid_size(cls, shape, pixel_scale, sub_grid_size):
        mask = Mask.padded_for_shape_and_pixel_scale(shape=shape, pixel_scale=pixel_scale)
        sub_grid = imaging_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(mask=mask,
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

    def sub_data_to_image(self, sub_array):
        """For an input sub-gridded array, mappers it from the sub-grid to a 1D _data-grid by summing each set of
        each set of sub-pixels values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_array : ndarray
            A 1D sub-gridded array of values (e.g. the intensities, surface-densities, potential) which is mapped to
            a 1d _data-grid array.
        """
        return np.multiply(self.sub_grid_fraction, sub_array.reshape(-1, self.sub_grid_length).sum(axis=1))

    @property
    @imaging_util.Memoizer()
    def sub_to_image(self):
        """ Compute the mapping between every sub-pixel and its host padded _data-pixel.

        For howtolens:

        - sub_to_pixel[8] = 2 -  The seventh sub-pixel is within the 3rd padded _data pixel.
        - sub_to_pixel[20] = 4 -  The nineteenth sub-pixel is within the 5th padded _data pixel.
        """
        return imaging_util.sub_to_image_from_mask(self.mask, self.sub_grid_size).astype('int')


class PaddedImageGrid(ImageGrid):

    def __new__(cls, arr, mask, image_shape, *args, **kwargs):
        """An *PaddedImageGrid* stores the (x,y) arc-second coordinates of a mask's pixels in 1D, in an analogous
        fashion to an *ImageGrid*. An *PaddedImageGrid* deviate from a normal grid in that:

        - All pixels are used (as opposed to just padded pixels)
        - The mask is padded when computing the grid, such that additional pixels beyond its edge are included.

        Padded-grids allow quantities like intensities to be computed on large 2D arrays spanning the entire _data, as
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
        """Setup an *PaddedImageGrid* of the regular _data-grid for an input _data-shape, psf-shape and pixel-scale.

        The center of every pixel is used to setup the grid's (x,y) arc-second coordinates, including padded-pixels
        which are beyond the input shape but will have light blurred into them given the psf-shape.

        Parameters
        ----------
        pixel_scale : float
            The scale of each pixel in arc seconds
        shape : (int, int)
            The (x,y) shape of the padded-grid's 2D _data in units of pixels.
        psf_shape : (int, int)
           The shape of the psf which defines the blurring region and therefore size of padding.
        """
        padded_shape = (shape[0] + psf_shape[0] - 1, shape[1] + psf_shape[1] - 1)
        padded_image_grid = imaging_util.image_grid_1d_masked_from_mask_and_pixel_scales(
            mask=np.full(padded_shape, False), pixel_scales=(pixel_scale, pixel_scale))
        padded_mask = Mask.padded_for_shape_and_pixel_scale(shape=padded_shape, pixel_scale=pixel_scale)
        return PaddedImageGrid(arr=padded_image_grid, mask=padded_mask, image_shape=shape)

    def convolve_array_1d_with_psf(self, padded_array_1d, psf):
        """Convolve a 2d padded array of values (e.g. intensities beforoe PSF blurring) with a PSF, and then trim \
        the convolved array to its original 2D shape.

        Parameters
        -----------
        psf
        padded_array_1d: ndarray
            A 1D array of values which were computed using the *PaddedImageGrid*.
        """
        padded_array_2d = imaging_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(padded_array_1d,
                                                                                                 self.mask.shape)
        blurred_padded_array_2d = psf.convolve(padded_array_2d)
        return imaging_util.map_2d_array_to_masked_1d_array_from_array_2d_and_mask(array_2d=blurred_padded_array_2d,
                                                                                   mask=np.full(self.mask.shape,
                                                                                                False))

    def map_to_2d(self, padded_array_1d):
        """ Map a padded 1D array of values to its origianl 2D array.

        Parameters
        -----------
        padded_array_1d : ndarray
            A 1D array of values which were computed using the *PaddedImageGrid*.
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
            A 1D array of values which were computed using the *PaddedImageGrid*.
        """
        return imaging_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(padded_array_1d, self.mask.shape)

    @property
    def padded_shape(self):
        return self.mask.shape


class PaddedSubGrid(SubGrid, PaddedImageGrid):

    def __init__(self, arr, mask, image_shape, sub_grid_size=1):
        """A *PaddedSubGrid* stores the (x,y) arc-second coordinates of a mask's sub-pixels in 1D, in an analogous
        fashion to a *SubGrid*. A *PaddedSubGrid* deviate from a normal grid in that:

        - All pixels are used (as opposed to just padded pixels)
        - The mask is padded when computing the grid, such that additional pixels beyond its edge are included.

        Padded-grids allow quantities like intensities to be computed on large 2D arrays spanning the entire _data, as
        opposed to just within the masked region.
        """
        super(PaddedSubGrid, self).__init__(arr, mask, sub_grid_size)
        self.image_shape = image_shape

    @classmethod
    def padded_grid_from_mask_sub_grid_size_and_psf_shape(cls, mask, sub_grid_size, psf_shape):
        """Setup an *PaddedSubGrid* for an input mask, sub-grid size and psf-shape.

        The center of every sub-pixel is used to setup the grid's (x,y) arc-second coordinates, including \
        padded-pixels which are beyond the input shape but will have light blurred into them given the psf-shape.

        Parameters
        ----------
        mask : Mask
            The mask whose padded pixels are used to setup the sub-pixel grids.
        sub_grid_size : int
            The size (sub_grid_size x sub_grid_size) of each _data-pixels sub-grid.
        psf_shape : (int, int)
           The shape of the psf which defines the blurring region and therefore size of padding.
        """

        padded_shape = (mask.shape[0] + psf_shape[0] - 1, mask.shape[1] + psf_shape[1] - 1)

        padded_sub_grid = imaging_util.sub_grid_1d_masked_from_mask_pixel_scales_and_sub_grid_size(
            mask=np.full(padded_shape, False), pixel_scales=mask.pixel_scales, sub_grid_size=sub_grid_size)

        padded_mask = Mask.padded_for_shape_and_pixel_scale(shape=padded_shape, pixel_scale=mask.pixel_scale)

        return PaddedSubGrid(arr=padded_sub_grid, mask=padded_mask, image_shape=mask.shape,
                             sub_grid_size=sub_grid_size)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if isinstance(obj, PaddedSubGrid):
            self.image_shape = obj.image_shape


class ImagingGridBorders(object):

    def __init__(self, image, sub):
        """The borders containing the pixel-index's of all padded pixels that are on the mask's border (e.g. \
        they are next to a *True* value in at least one of the surrounding 8 pixels). There are 2 borders:

        The _data-border - The pixel-indexes of all padded pixels in an *ImageGrid* that are border-pixels.
        The sub-grid - The sub-pixel-indexes of all padded pixels in a *SubGrid* that are border-pixels.

        A polynomial is fitted to the (x,y) coordinates of each border's pixels. This allows us to relocate \
        demagnified pixel's in a grid to its border, so that they do not disrupt an adaptive inversion.

        Parameters
        -----------
        image : ImageGridBorder
            The pixel-index's of all _data-pixels which are on the mask border.
        sub : SubGridBorder
            The sub-pixel-index's of all sub-pixels which are on the mask border.
        """
        self.image = image
        self.sub = sub

    @classmethod
    def from_mask_and_sub_grid_size(cls, mask, sub_grid_size, polynomial_degree=3, centre=(0.0, 0.0)):
        """Setup the *ImagingGridBorders* from a mask, sub-grid size and polynomial degree.

        Parameters
        -----------
        mask : Mask
            The mask the padded border (sub-)pixel index's are computed from.
        sub_grid_size : int
            The size of a sub-pixels sub-grid (sub_grid_size x sub_grid_size).
        polynomial_degree : int
            The degree of the polynomial that is used to fit the border when relocating pixels outside the border to \
            its edge.
        centre : (float, float)
            The centre of the border, which can be shifted relative to its coordinates.
        """
        image_border = ImageGridBorder.from_mask(mask, polynomial_degree, centre)
        sub_border = SubGridBorder.from_mask(mask, sub_grid_size, polynomial_degree, centre)
        return ImagingGridBorders(image_border, sub_border)

    def relocated_grids_from_grids(self, grids):
        """Determine a set of relocated grids from an input set of grids, by relocating their pixels based on the \
        border.

        The blurring-grid does not have its coordinates relocated, as it is only used for computing analytic \
        light-profiles and not inversion-grids.

        Parameters
        -----------
        grids : ImagingGrids
            The imaging-grids (_data, sub) which have their coordinates relocated.
        """
        return ImagingGrids(image=self.image.relocated_grid_from_grid(grids.image),
                            sub=self.sub.relocated_grid_from_grid(grids.sub),
                            blurring=None)


class ImageGridBorder(np.ndarray):

    def __new__(cls, arr, polynomial_degree=3, centre=(0.0, 0.0), *args, **kwargs):
        """The borders of an _data-grid, containing the pixel-index's of all padded pixels that are on the \
        mask's border (e.g. they are next to a *True* value in at least one of the surrounding 8 pixels).

        A polynomial is fitted to the (x,y) coordinates of the border's pixels. This allows us to relocate \
        demagnified pixel's in a grid to its border, so that they do not disrupt an adaptive inversion.

        Parameters
        -----------
        arr : ndarray
            A 1D array of the integer indexes of an *ImageGrid*'s border pixels.
        polynomial_degree : int
            The degree of the polynomial that is used to fit the border when relocating pixels outside the border to \
            its edge.
        centre : (float, float)
            The centre of the border, which can be shifted relative to its coordinates.
        """
        border = arr.view(cls)
        border.polynomial_degree = polynomial_degree
        border.centre = centre
        return border

    @classmethod
    def from_mask(cls, mask, polynomial_degree=3, centre=(0.0, 0.0)):
        """Setup the *ImageGridBorder* from a mask.

        Parameters
        -----------
        mask : Mask
            The mask the padded border pixel index's are computed from.
        polynomial_degree : int
            The degree of the polynomial that is used to fit the border when relocating pixels outside the border to \
            its edge.
        centre : (float, float)
            The centre of the border, which can be shifted relative to its coordinates.
        """
        return cls(mask.border_pixels, polynomial_degree, centre)

    @property
    def total_pixels(self):
        return self.shape[0]

    def grid_to_radii(self, grid):
        """ Convert a grid of (x,y) arc-second coordinates to their circular radii values, based on the grid's centre.

        Parameters
        ----------
        grid : ImageGrid
            The (x, y) coordinates of each padded _data-pixel.
        """

        return np.sqrt(np.add(np.square(np.subtract(grid[:, 0], self.centre[0])),
                              np.square(np.subtract(grid[:, 1], self.centre[1]))))

    def grid_to_thetas(self, grid):
        """
        Converted a grid of (x,y) arc-second coordinates to the angle in degrees between their location and the \
        positive x-axis counter-clockwise.

        Parameters
        ----------
        grid : ImageGrid
            The (x, y) coordinates of each padded _data-pixel.
        """
        shifted_grid = np.subtract(grid, self.centre)
        theta_from_x = np.degrees(np.arctan2(shifted_grid[:, 1], shifted_grid[:, 0]))
        theta_from_x[theta_from_x < 0.0] += 360.
        return theta_from_x

    def polynomial_fit_to_border(self, grid):
        """Fit the border's radial coordinates with a polynomial.

        Parameters
        ----------
        grid : ImageGrid
            The (x, y) coordinates of each padded _data-pixel.
        """

        border_grid = grid[self]

        return np.polyfit(self.grid_to_thetas(border_grid), self.grid_to_radii(border_grid), self.polynomial_degree)

    def move_factors_from_grid(self, grid):
        """ Compute the move-factor of every (x,y) arc-second coordinate on the *ImageGrid*.

        A move-factor defines how far a coordinate must be moved towards the border's centre in order to lie on it. \
        If a pixel is already within the border, the move-factor 1.0 (i.e. no movement).

        Parameters
        ----------
        grid : ImageGrid
            The (x, y) coordinates of each padded _data-pixel.
        """
        grid_thetas = self.grid_to_thetas(grid)
        grid_radii = self.grid_to_radii(grid)
        poly = self.polynomial_fit_to_border(grid)

        with np.errstate(divide='ignore'):
            move_factors = np.divide(np.polyval(poly, grid_thetas), grid_radii)
        move_factors[move_factors > 1.0] = 1.0

        return move_factors

    def relocated_grid_from_grid(self, grid):
        """Compute the relocated (x,y) grid coordinates of every (x,y) arc-second coordinate on an *ImageGrid*.

        Parameters
        -----------
        grid : ImageGrid
            The (x, y) coordinates of each padded _data-pixel.
        """
        move_factors = self.move_factors_from_grid(grid)
        return np.multiply(grid, move_factors[:, None])

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

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super(ImageGridBorder, self).__setstate__(state[0:-1])


class SubGridBorder(ImageGridBorder):

    @classmethod
    def from_mask(cls, mask, sub_grid_size, polynomial_degree=3, centre=(0.0, 0.0)):
        """Setup the *SubGridBorder* from a mask.

        Parameters
        -----------
        mask : Mask
            The mask the padded border pixel index's are computed from.
        sub_grid_size : int
            The size of a sub-pixels sub-grid (sub_grid_size x sub_grid_size).
        polynomial_degree : int
            The degree of the polynomial that is used to fit the border when relocating pixels outside the border to \
            its edge.
        centre : (float, float)
            The centre of the border, which can be shifted relative to its coordinates.
        """
        return cls(mask.border_sub_pixels(sub_grid_size), polynomial_degree, centre)
