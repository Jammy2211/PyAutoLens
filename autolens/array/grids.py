import numpy as np
import scipy.spatial.qhull as qhull
from functools import wraps
from sklearn.cluster import KMeans

from autolens import decorator_util
from autolens import exc
from autolens.array import mask as msk, scaled_array
from autolens.array.util import grid_util, array_util, mask_util, binning_util
from autolens.array.mapping_util import (
    array_mapping_util,
    grid_mapping_util,
    sparse_mapping_util,
)


class Grid(np.ndarray):
    def __new__(cls, arr, mask, binned=None, *args, **kwargs):
        """A grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of an \
        unmasked pixel. The positive y-axis is upwards and poitive x-axis to the right. 
        
        A *Grid* is ordered such pixels begin from the top-row of the mask and go rightwards and then \ 
        downwards. Therefore, it is a ndarray of shape [total_unmasked_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pixel index and second element the y or x arc -econd coordinates. For example:

        - grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Below is a visual illustration of a grid, where a total of 10 pixels are unmasked and are included in \
        the grid.

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|o|o|x|x|x|x|     x = True (Pixel is masked and excluded from the grid)
        |x|x|x|o|o|o|o|x|x|x|     o = False (Pixel is not masked and included in the grid)
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
        |x|x|x|x|x|x|x|x|x|x|  ^   grid[0] = [ 1.5, -0.5]
        |x|x|x|x|x|x|x|x|x|x|  |   grid[1] = [ 1.5,  0.5]
        |x|x|x|x|x|x|x|x|x|x|  |   grid[2] = [ 0.5, -1.5]
        |x|x|x|x|0|1|x|x|x|x| +ve  grid[3] = [ 0.5, -0.5]
        |x|x|x|2|3|4|5|x|x|x|  y   grid[4] = [ 0.5,  0.5]
        |x|x|x|6|7|8|9|x|x|x| -ve  grid[5] = [ 0.5,  1.5]
        |x|x|x|x|x|x|x|x|x|x|  |   grid[6] = [-0.5, -1.5]
        |x|x|x|x|x|x|x|x|x|x|  |   grid[7] = [-0.5, -0.5]
        |x|x|x|x|x|x|x|x|x|x| \/   grid[8] = [-0.5,  0.5]
        |x|x|x|x|x|x|x|x|x|x|      grid[9] = [-0.5,  1.5]

        A sub-grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of each \
        sub-pixel of an unmasked pixel (e.g. the pixels of a grid). The positive y-axis is upwards and poitive \
        x-axis to the right, and this convention is followed for the sub-pixels in each unmasked pixel.

        A *SubGrid* is ordered such that pixels begin from the first (top-left) sub-pixel in the first unmasked pixel. \
        Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel. Therefore, \
        the sub-grid is an ndarray of shape [total_unmasked_pixels*(sub_grid_shape)**2, 2]. For example:

        - grid[9, 1] - using a 2x2 sub-grid, gives the 3rd unmasked pixel's 2nd sub-pixel x-coordinate.
        - grid[9, 1] - using a 3x3 sub-grid, gives the 2nd unmasked pixel's 1st sub-pixel x-coordinate.
        - grid[27, 0] - using a 3x3 sub-grid, gives the 4th unmasked pixel's 1st sub-pixel y-coordinate.

        Below is a visual illustration of a sub grid. Like the grid, the indexing of each sub-pixel goes from \
        the top-left corner. In contrast to the grid above, our illustration below restricts the mask to just \
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

        Our grid looks like it did before:

        pixel_scale = 1.0"

        <--- -ve  x  +ve -->

        |x|x|x|x|x|x|x|x|x|x|  ^
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x|  |                        y     x
        |x|x|x|x|x|x|x|x|x|x| +ve  grid[0] = [0.5,  -1.5]
        |x|x|x|0|1|x|x|x|x|x|  y   grid[1] = [0.5,  -0.5]
        |x|x|x|x|x|x|x|x|x|x| -ve
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x| \/
        |x|x|x|x|x|x|x|x|x|x|

        However, we now go to each unmasked pixel and derive a sub-pixel grid for it. For example, for pixel 0,
        if *sub_size=2*, we use a 2x2 sub-grid:

        Pixel 0 - (2x2):
                                y      x
               grid[0] = [0.66, -1.66]
        |0|1|  grid[1] = [0.66, -1.33]
        |2|3|  grid[2] = [0.33, -1.66]
               grid[3] = [0.33, -1.33]

        Now, we'd normally sub-grid all pixels using the same *sub_size*, but for this illustration lets
        pretend we used a sub_size of 3x3 for pixel 1:

                                  y      x
                 grid[0] = [0.75, -0.75]
                 grid[1] = [0.75, -0.5]
                 grid[2] = [0.75, -0.25]
        |0|1|2|  grid[3] = [0.5,  -0.75]
        |3|4|5|  grid[4] = [0.5,  -0.5]
        |6|7|8|  grid[5] = [0.5,  -0.25]
                 grid[6] = [0.25, -0.75]
                 grid[7] = [0.25, -0.5]
                 grid[8] = [0.25, -0.25]

        """
        obj = arr.view(cls)
        obj.mask = mask
        obj.sub_border_1d_indexes = mask.sub_border_1d_indexes
        obj.interpolator = None
        obj.binned = None
        return obj

    def __array_finalize__(self, obj):

        if isinstance(obj, Grid):
            self.mask = obj.mask
            self.sub_border_1d_indexes = obj.sub_border_1d_indexes
            self.interpolator = obj.interpolator
            self.binned = obj.binned

    @classmethod
    def from_mask(cls, mask):
        """Setup a sub-grid of the unmasked pixels, using a mask and a specified sub-grid size. The center of \
        every unmasked pixel's sub-pixels give the grid's (y,x) arc-second coordinates.

        Parameters
        -----------
        mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        """

        sub_grid_masked = grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, pixel_scales=mask.pixel_scales, sub_size=mask.sub_size
        )

        return Grid(arr=sub_grid_masked, mask=mask)

    @classmethod
    def from_shape_pixel_scale_and_sub_size(cls, shape, pixel_scale, sub_size):
        """Setup a sub-grid from a 2D array shape and pixel scale. Here, the center of every pixel on the 2D \
        array gives the grid's (y,x) arc-second coordinates, where each pixel has sub-pixels specified by the \
        sub-grid size.

        This is equivalent to using a 2D mask consisting entirely of unmasked pixels.

        Parameters
        -----------
        shape : (int, int)
            The 2D shape of the array, where all pixels are used to generate the grid's grid.
        pixel_scale : float
            The size of each pixel in arc seconds.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        """

        mask = msk.Mask.unmasked_from_shape_pixel_scale_and_sub_size(
            shape=shape, pixel_scale=pixel_scale, sub_size=sub_size
        )

        grid = grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, pixel_scales=mask.pixel_scales, sub_size=sub_size
        )

        return Grid(arr=grid, mask=mask)

    @classmethod
    def blurring_grid_from_mask_and_psf_shape(cls, mask, psf_shape):
        """Setup a blurring-grid from a mask, where a blurring grid consists of all pixels that are masked, but they \
        are close enough to the unmasked pixels that a fraction of their light will be blurred into those pixels \
        via PSF convolution. For example, if our mask is as follows:
        
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an imaging.Mask, where:
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

        |x|x|x|x|x|x|x|x|x|     This is an example grid.Mask, where:
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

        |x|x|x|x|x|x|x|x|x|     This is an example grid.Mask, where:
        |x|o|o|o|o|o|o|o|x|
        |x|o|o|o|o|o|o|o|x|     x = True (Pixel is masked and excluded from lens)
        |x|o|o|x|x|x|o|o|x|     o = False (Pixel is not masked and included in lens)
        |x|o|o|x|x|x|o|o|x|
        |x|o|o|x|x|x|o|o|x|
        |x|o|o|o|o|o|o|o|x|
        |x|o|o|o|o|o|o|o|x|
        |x|x|x|x|x|x|x|x|x|
        """

        blurring_mask = mask.blurring_mask_from_psf_shape(psf_shape=psf_shape)

        return Grid.from_mask(mask=blurring_mask)

    @property
    def sub_size(self):
        return self.mask.sub_size

    @property
    def mapping(self):
        return self.mask.mapping

    @property
    def in_2d(self):
        return self.mapping.sub_grid_2d_with_sub_dimensions_from_sub_grid_1d(
            sub_grid_1d=self
        )

    @property
    @array_util.Memoizer()
    def unlensed_unsubbed_1d(self):
        return Grid(
            arr=grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
                mask=self.mask, pixel_scales=self.mask.pixel_scales, sub_size=1
            ),
            mask=self.mask,
            sub_size=1,
        )

    @property
    def unlensed_1d(self):
        return Grid(
            arr=grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
                mask=self.mask,
                pixel_scales=self.mask.pixel_scales,
                sub_size=self.sub_size,
            ),
            mask=self.mask,
            sub_size=self.sub_size,
        )

    @property
    def unlensed_unmasked_1d(self):
        return Grid(
            grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
                mask=np.full(self.mask.shape, False),
                pixel_scales=self.mask.pixel_scales,
                sub_size=self.sub_size,
            ),
            mask=self.mask,
            sub_size=self.sub_size,
        )

    @property
    def unlensed_2d(self):
        return Grid(
            arr=grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
                sub_grid_1d=self.unlensed_1d, mask=self.mask, sub_size=self.sub_size
            ),
            mask=self.mask,
            sub_size=self.sub_size,
        )

    @property
    def unlensed_unmasked_2d(self):
        return Grid(
            arr=grid_mapping_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
                mask=np.full(self.mask.shape, False),
                sub_grid_1d=self.unlensed_unmasked_1d,
                sub_size=self.sub_size,
            ),
            mask=self.mask,
            sub_size=self.sub_size,
        )

    @property
    def total_pixels(self):
        return self.shape[0]

    @property
    def pixel_scale(self):
        return self.mask.pixel_scale

    def blurring_grid_from_psf_shape(self, psf_shape):

        blurring_mask = self.mask.blurring_mask_from_psf_shape(psf_shape=psf_shape)

        return Grid.from_mask(mask=blurring_mask)

    def marching_squares_grid_pixels_to_grid_arcsec(self, grid_pixels, shape):

        grid_arcsec = grid_util.grid_pixels_1d_to_grid_arcsec_1d(
            grid_pixels_1d=grid_pixels,
            shape=shape,
            pixel_scales=(
                self.pixel_scale / self.sub_size,
                self.pixel_scale / self.sub_size,
            ),
            origin=self.mask.origin,
        )

        grid_arcsec[:, 0] -= self.pixel_scale / (2.0 * self.sub_size)
        grid_arcsec[:, 1] += self.pixel_scale / (2.0 * self.sub_size)

        return grid_arcsec

    def new_grid_with_binned_grid(self, binned_grid):
        # noinspection PyAttributeOutsideInit
        # TODO: This function doesn't do what it says on the tin. The returned grid would be the same as the grid
        # TODO: on which the function was called but with a new interpolator set.
        self.binned = binned_grid
        return self

    def new_grid_with_interpolator(self, pixel_scale_interpolation_grid):
        # noinspection PyAttributeOutsideInit
        # TODO: This function doesn't do what it says on the tin. The returned grid would be the same as the grid
        # TODO: on which the function was called but with a new interpolator set.
        self.interpolator = Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=self.mask,
            grid=self[:, :],
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        )
        return self

    def padded_grid_from_psf_shape(self, psf_shape):

        shape = self.mask.shape

        padded_shape = (shape[0] + psf_shape[0] - 1, shape[1] + psf_shape[1] - 1)

        padded_mask = msk.Mask.unmasked_from_shape_pixel_scale_and_sub_size(
            shape=padded_shape,
            pixel_scale=self.mask.pixel_scale,
            sub_size=self.sub_size,
        )

        padded_sub_grid = Grid.from_mask(mask=padded_mask)

        if self.interpolator is None:
            return padded_sub_grid
        else:
            return padded_sub_grid.new_grid_with_interpolator(
                pixel_scale_interpolation_grid=self.interpolator.pixel_scale_interpolation_grid
            )

    def trimmed_array_2d_from_padded_array_1d_and_image_shape(
        self, padded_array_1d, image_shape
    ):
        """ Map a padded 1D array of values to its original 2D array, trimming all edge values.

        Parameters
        -----------
        padded_array_1d : ndarray
            A 1D array of values which were computed using a padded grid
        """

        # TODO : Raise error if padded grid not used here.

        padded_array_2d = self.mapping.array_2d_from_array_1d(array_1d=padded_array_1d)
        pad_size_0 = self.mask.shape[0] - image_shape[0]
        pad_size_1 = self.mask.shape[1] - image_shape[1]
        return padded_array_2d[
            pad_size_0 // 2 : self.mask.shape[0] - pad_size_0 // 2,
            pad_size_1 // 2 : self.mask.shape[1] - pad_size_1 // 2,
        ]

    def convolve_array_1d_with_psf(self, padded_array_1d, psf):
        """Convolve a 1d padded array of values (e.g. image before PSF blurring) with a PSF, and then trim \
        the convolved array to its original 2D shape.

        Parameters
        -----------
        padded_array_1d: ndarray
            A 1D array of values which were computed using the a padded grid.
        psf : ndarray
            An array describing the PSF kernel of the image.
        """

        padded_array_2d = array_mapping_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=padded_array_1d,
            mask=np.full(fill_value=False, shape=self.mask.shape),
            sub_size=1,
        )

        # noinspection PyUnresolvedReferences
        blurred_padded_array_2d = psf.convolve(array_2d=padded_array_2d)

        return array_mapping_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=blurred_padded_array_2d,
            mask=np.full(self.mask.shape, False),
            sub_size=1,
        )

    def unmasked_blurred_array_2d_from_padded_array_1d_psf_and_image_shape(
        self, padded_array_1d, psf, image_shape
    ):
        """For a padded grid and psf, compute an unmasked blurred image from an unmasked unblurred image.

        This relies on using the lens data's padded-grid, which is a grid of (y,x) coordinates which extends over the \
        entire image as opposed to just the masked region.

        Parameters
        ----------
        psf : abstract_data.PSF
            The PSF of the image used for convolution.
        unmasked_image_1d : ndarray
            The 1D unmasked image which is blurred.
        """

        padded_array_2d = self.mapping.scaled_array_2d_from_array_1d(
            array_1d=padded_array_1d
        )

        blurred_image_2d = psf.convolve(array_2d=padded_array_2d)

        blurred_image_1d = self.mapping.array_1d_from_array_2d(
            array_2d=blurred_image_2d
        )

        return self.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=blurred_image_1d, image_shape=image_shape
        )

    @property
    def sub_border_grid(self):
        return self[self.mask.sub_border_1d_indexes]

    def relocated_grid_from_grid(self, grid):
        """Determine a set of relocated grid from an input set of grid, by relocating their pixels based on the \
        borders.

        The blurring-grid does not have its coordinates relocated, as it is only used for computing analytic \
        light-profiles and not inversion-grid.

        Parameters
        -----------
        grid : GridStack
            The grid, whose grid coordinates are relocated.
        """

        return Grid(
            arr=self.relocated_grid_from_grid_jit(
                grid=grid, border_grid=self.sub_border_grid
            ),
            mask=grid.mask,
            sub_size=grid.sub_size,
        )

    def relocated_pixelization_grid_from_pixelization_grid(self, pixelization_grid):
        """Determine a set of relocated grid from an input set of grid, by relocating their pixels based on the \
        borders.

        The blurring-grid does not have its coordinates relocated, as it is only used for computing analytic \
        light-profiles and not inversion-grid.

        Parameters
        -----------
        grid : GridStack
            The grid, whose grid coordinates are relocated.
        """

        return PixelizationGrid(
            arr=self.relocated_grid_from_grid_jit(
                grid=pixelization_grid, border_grid=self.sub_border_grid
            ),
            mask_1d_index_to_nearest_pixelization_1d_index=pixelization_grid.mask_1d_index_to_nearest_pixelization_1d_index,
        )

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
        border_grid_radii = np.sqrt(
            np.add(
                np.square(np.subtract(border_grid[:, 0], border_origin[0])),
                np.square(np.subtract(border_grid[:, 1], border_origin[1])),
            )
        )
        border_min_radii = np.min(border_grid_radii)

        grid_radii = np.sqrt(
            np.add(
                np.square(np.subtract(grid[:, 0], border_origin[0])),
                np.square(np.subtract(grid[:, 1], border_origin[1])),
            )
        )

        for pixel_index in range(grid.shape[0]):

            if grid_radii[pixel_index] > border_min_radii:

                closest_pixel_index = np.argmin(
                    np.square(grid[pixel_index, 0] - border_grid[:, 0])
                    + np.square(grid[pixel_index, 1] - border_grid[:, 1])
                )

                move_factor = (
                    border_grid_radii[closest_pixel_index] / grid_radii[pixel_index]
                )

                if move_factor < 1.0:
                    grid[pixel_index, :] = (
                        move_factor * (grid[pixel_index, :] - border_origin[:])
                        + border_origin[:]
                    )

        return grid

    @property
    @array_util.Memoizer()
    def in_radians(self):
        return (self * np.pi) / 648000.0

    @property
    def masked_shape_arcsec(self):
        return (
            np.amax(self[:, 0]) - np.amin(self[:, 0]),
            np.amax(self[:, 1]) - np.amin(self[:, 1]),
        )

    @property
    def yticks(self):
        """Compute the yticks labels of this grid, used for plotting the y-axis ticks when visualizing a grid"""
        return np.linspace(np.min(self[:, 0]), np.max(self[:, 0]), 4)

    @property
    def xticks(self):
        """Compute the xticks labels of this grid, used for plotting the x-axis ticks when visualizing a grid"""
        return np.linspace(np.min(self[:, 1]), np.max(self[:, 1]), 4)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Grid, self).__reduce__()
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
        super(Grid, self).__setstate__(state[0:-1])


class BinnedGrid(Grid):
    def __init__(
        self,
        arr,
        mask,
        bin_up_factor,
        binned_mask_1d_index_to_mask_1d_indexes,
        binned_mask_1d_index_to_mask_1d_sizes,
        total_unbinned_pixels,
    ):
        # noinspection PyArgumentList
        super(BinnedGrid, self).__init__()
        self.mask = mask
        self.bin_up_factor = bin_up_factor
        self.binned_mask_1d_index_to_mask_1d_indexes = (
            binned_mask_1d_index_to_mask_1d_indexes
        )
        self.binned_mask_1d_index_to_mask_1d_sizes = (
            binned_mask_1d_index_to_mask_1d_sizes
        )
        self.total_unbinned_pixels = total_unbinned_pixels

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if isinstance(obj, BinnedGrid):
            self.mask = obj.mask
            self.bin_up_factor = obj.bin_up_factor
            self.binned_mask_1d_index_to_mask_1d_indexes = (
                obj.binned_mask_1d_index_to_mask_1d_indexes
            )
            self.binned_mask_1d_index_to_mask_1d_sizes = (
                obj.binned_mask_1d_index_to_mask_1d_sizes
            )
            self.total_unbinned_pixels = obj.total_unbinned_pixels

    @classmethod
    def from_mask_and_pixel_scale_binned_grid(cls, mask, pixel_scale_binned_grid):

        if pixel_scale_binned_grid > mask.pixel_scale:

            bin_up_factor = int(pixel_scale_binned_grid / mask.pixel_scale)

        else:

            bin_up_factor = 1

        binned_mask = mask.binned_up_mask_sub_size_1_from_mask(
            bin_up_factor=bin_up_factor
        )

        binned_grid = Grid.from_mask(mask=binned_mask)
        binned_mask_1d_index_to_mask_1d_indexes, binned_mask_1d_index_to_mask_1d_sizes = binning_util.binned_masked_array_1d_to_masked_array_1d_all_frommask_and_bin_up_factor(
            mask_2d=mask, bin_up_factor=bin_up_factor
        )

        return BinnedGrid(
            arr=binned_grid,
            mask=binned_mask,
            bin_up_factor=bin_up_factor,
            binned_mask_1d_index_to_mask_1d_indexes=binned_mask_1d_index_to_mask_1d_indexes.astype(
                "int"
            ),
            binned_mask_1d_index_to_mask_1d_sizes=binned_mask_1d_index_to_mask_1d_sizes.astype(
                "int"
            ),
            total_unbinned_pixels=mask.pixels_in_mask,
        )


class PixelizationGrid(np.ndarray):
    def __new__(
        cls, arr, mask_1d_index_to_nearest_pixelization_1d_index, *args, **kwargs
    ):
        """A pixelization-grid of (y,x) coordinates which are used to form the pixel centres of adaptive pixelizations in the \
        *pixelizations* module.

        A *PixGrid* is ordered such pixels begin from the top-row of the mask and go rightwards and then \
        downwards. Therefore, it is a ndarray of shape [total_pix_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pixelization pixel index and second element the y or x arc -econd coordinates. For example:

        - pix_grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - pix_grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Parameters
        -----------
        pix_grid : ndarray
            The grid of (y,x) arc-second coordinates of every image-plane pixelization grid used for adaptive source \
            -plane pixelizations.
        mask_1d_index_to_nearest_pixelization_1d_index : ndarray
            A 1D array that maps every grid pixel to its nearest pixelization-grid pixel.
        """
        obj = arr.view(cls)
        obj.mask_1d_index_to_nearest_pixelization_1d_index = (
            mask_1d_index_to_nearest_pixelization_1d_index
        )
        obj.interpolator = None
        return obj

    @classmethod
    def from_grid_and_unmasked_2d_grid_shape(cls, unmasked_sparse_shape, grid):

        sparse_grid = SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=unmasked_sparse_shape, grid=grid
        )

        return PixelizationGrid(
            arr=sparse_grid.sparse,
            mask_1d_index_to_nearest_pixelization_1d_index=sparse_grid.mask_1d_index_to_sparse_1d_index,
        )

    def __array_finalize__(self, obj):
        if hasattr(obj, "mask_1d_index_to_nearest_pixelization_1d_index"):
            self.mask_1d_index_to_nearest_pixelization_1d_index = (
                obj.mask_1d_index_to_nearest_pixelization_1d_index
            )
        if hasattr(obj, "interpolator"):
            self.interpolator = obj.interpolator


class SparseToGrid(scaled_array.RectangularArrayGeometry):
    def __init__(self, sparse_grid, mask_1d_index_to_sparse_1d_index):
        """A sparse grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of a \
        pixel on the sparse grid. To setup the sparse-grid, it is laid over a grid of unmasked pixels, such \
        that all sparse-grid pixels which map inside of an unmasked grid pixel are included on the sparse grid.

        To setup this sparse grid, we thus have two sparse grid:

        - The unmasked sparse-grid, which corresponds to a uniform 2D array of pixels. The edges of this grid \
          correspond to the 4 edges of the mask (e.g. the higher and lowest (y,x) arc-second unmasked pixels) and the \
          grid's shape is speciifed by the unmasked_sparse_grid_shape parameter.

        - The (masked) sparse-grid, which is all pixels on the unmasked sparse-grid above which fall within unmasked \
          grid pixels. These are the pixels which are actually used for other modules in PyAutoLens.

        The origin of the unmasked sparse grid can be changed to allow off-center pairings with sparse-grid pixels, \
        which is necessary when a mask has a centre offset from (0.0", 0.0"). However, the sparse grid itself \
        retains an origin of (0.0", 0.0"), ensuring its arc-second grid uses the same coordinate system as the \
        other grid.

        The sparse grid is used to determine the pixel centers of an adaptive grid pixelization.

        Parameters
        ----------
        unmasked_sparse_shape : (int, int)
            The shape of the unmasked sparse-grid whose centres form the sparse-grid.
        pixel_scales : (float, float)
            The pixel-to-arcsecond scale of a pixel in the y and x directions.
        grid : Grid
            The grid used to determine which pixels are in the sparse grid.
        origin : (float, float)
            The centre of the unmasked sparse grid, which matches the centre of the mask.
        """
        self.sparse = sparse_grid
        self.mask_1d_index_to_sparse_1d_index = mask_1d_index_to_sparse_1d_index

    @classmethod
    def from_grid_and_unmasked_2d_grid_shape(cls, grid, unmasked_sparse_shape):
        """Calculate the image-plane pixelization from a grid of coordinates (and its mask).

        See *grid_stacks.SparseToGrid* for details on how this grid is calculated.

        Parameters
        -----------
        grid : grids.Grid
            The grid of (y,x) arc-second coordinates at the centre of every image value (e.g. image-pixels).
        """

        pixel_scale = grid.mask.pixel_scale

        pixel_scales = (
            (grid.masked_shape_arcsec[0] + pixel_scale) / (unmasked_sparse_shape[0]),
            (grid.masked_shape_arcsec[1] + pixel_scale) / (unmasked_sparse_shape[1]),
        )

        origin = grid.mask.centre

        unmasked_sparse_grid = grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
            shape=unmasked_sparse_shape,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
        )

        unmasked_sparse_grid_pixel_centres = grid.mask.grid_arcsec_to_grid_pixel_centres(
            grid_arcsec=unmasked_sparse_grid
        )

        total_sparse_pixels = mask_util.total_sparse_pixels_from_mask(
            mask=grid.mask,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        unmasked_sparse_to_sparse = sparse_mapping_util.unmasked_sparse_to_sparse_from_mask_and_pixel_centres(
            mask=grid.mask,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=total_sparse_pixels,
        ).astype(
            "int"
        )

        sparse_to_unmasked_sparse = sparse_mapping_util.sparse_to_unmasked_sparse_from_mask_and_pixel_centres(
            total_sparse_pixels=total_sparse_pixels,
            mask=grid.mask,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        ).astype(
            "int"
        )

        regular_to_unmasked_sparse = grid_util.grid_arcsec_1d_to_grid_pixel_indexes_1d(
            grid_arcsec_1d=grid.unlensed_unsubbed_1d,
            shape=unmasked_sparse_shape,
            pixel_scales=pixel_scales,
            origin=origin,
        ).astype("int")

        mask_1d_index_to_sparse_1d_index = sparse_mapping_util.mask_1d_index_to_sparse_1d_index_from_sparse_mappings(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            unmasked_sparse_to_sparse=unmasked_sparse_to_sparse,
        ).astype(
            "int"
        )

        sparse_grid = sparse_mapping_util.sparse_grid_from_unmasked_sparse_grid(
            unmasked_sparse_grid=unmasked_sparse_grid,
            sparse_to_unmasked_sparse=sparse_to_unmasked_sparse,
        )

        return SparseToGrid(
            sparse_grid=sparse_grid,
            mask_1d_index_to_sparse_1d_index=mask_1d_index_to_sparse_1d_index,
        )

    @classmethod
    def from_total_pixels_binned_grid_and_weight_map(
        cls,
        total_pixels,
        binned_grid,
        binned_weight_map,
        n_iter=1,
        max_iter=5,
        seed=None,
    ):
        """Calculate the image-plane pixelization from a grid of coordinates (and its mask).

        See *grid_stacks.SparseToGrid* for details on how this grid is calculated.

        Parameters
        -----------
        binned_grid : grids.Grid
            The grid of (y,x) arc-second coordinates at the centre of every image value (e.g. image-pixels).
        """

        if total_pixels > binned_grid.shape[0]:
            raise exc.PixelizationException

        kmeans = KMeans(
            n_clusters=total_pixels, random_state=seed, n_init=n_iter, max_iter=max_iter
        )

        kmeans = kmeans.fit(X=binned_grid, sample_weight=binned_weight_map)

        mask_1d_index_to_sparse_1d_index = sparse_mapping_util.mask_1d_index_to_sparse_1d_index_from_binned_grid(
            sparse_labels=kmeans.labels_,
            binned_mask_1d_index_to_mask_1d_indexes=binned_grid.binned_mask_1d_index_to_mask_1d_indexes,
            binned_mask_1d_index_to_mask_1d_sizes=binned_grid.binned_mask_1d_index_to_mask_1d_sizes,
            total_unbinned_pixels=binned_grid.total_unbinned_pixels,
        )

        return SparseToGrid(
            sparse_grid=kmeans.cluster_centers_,
            mask_1d_index_to_sparse_1d_index=mask_1d_index_to_sparse_1d_index.astype(
                "int"
            ),
        )

    @property
    def total_sparse_pixels(self):
        return len(self.sparse)


class Interpolator(object):
    def __init__(self, grid, interp_grid, pixel_scale_interpolation_grid):
        self.grid = grid
        self.interp_grid = interp_grid
        self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid
        self.vtx, self.wts = self.interp_weights

    @property
    def interp_weights(self):
        tri = qhull.Delaunay(self.interp_grid)
        simplex = tri.find_simplex(self.grid)
        # noinspection PyUnresolvedReferences
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = self.grid - temp[:, 2]
        bary = np.einsum("njk,nk->nj", temp[:, :2, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    @classmethod
    def from_mask_grid_and_pixel_scale_interpolation_grids(
        cls, mask, grid, pixel_scale_interpolation_grid
    ):

        rescale_factor = mask.pixel_scale / pixel_scale_interpolation_grid

        rescaled_mask = mask_util.rescaledmask_frommask_and_rescale_factor(
            mask_2d=mask, rescale_factor=rescale_factor
        )

        interp_mask = mask_util.edge_buffed_mask_from_mask(mask=rescaled_mask).astype(
            "bool"
        )

        interp_grid = grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=interp_mask,
            pixel_scales=(
                pixel_scale_interpolation_grid,
                pixel_scale_interpolation_grid,
            ),
            sub_size=1,
            origin=mask.origin,
        )

        return Interpolator(
            grid=grid,
            interp_grid=interp_grid,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        )

    def interpolated_values_from_values(self, values):
        return np.einsum("nj,nj->n", np.take(values, self.vtx), self.wts)


def grid_interpolate(func):
    """
    Decorate a profile method that accepts a coordinate grid and returns a data_type grid.

    If an interpolator attribute is associated with the input grid then that interpolator is used to down sample the
    coordinate grid prior to calling the function and up sample the result of the function.

    If no interpolator attribute is associated with the input grid then the function is called as hyper.

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
                values = func(
                    profile,
                    interpolator.interp_grid,
                    grid_radial_minimum,
                    *args,
                    **kwargs
                )
                if values.ndim == 1:
                    return interpolator.interpolated_values_from_values(values=values)
                elif values.ndim == 2:
                    y_values = interpolator.interpolated_values_from_values(
                        values=values[:, 0]
                    )
                    x_values = interpolator.interpolated_values_from_values(
                        values=values[:, 1]
                    )
                    return np.asarray([y_values, x_values]).T
        return func(profile, grid, grid_radial_minimum, *args, **kwargs)

    return wrapper
