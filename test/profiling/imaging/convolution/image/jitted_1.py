import numba
import numpy as np
import pytest
from profiling import profiling_data
from profiling import tools

from autolens import exc
from autolens.model.profiles import light_profiles


class Convolver(object):
    """Class to setup the 1D convolution of an masked_image / mapping_matrix matrix.

    IMAGE FRAMES:
    ------------

    For a masked masked_image, in 2D, one can compute for every unmasked pixel all other unmasked pixels it will blur light \
    into for a given PSF psf size, e.g.:

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an howtolens masked_image.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from lensing)
    |x|x|x|o|o|o|x|x|x|x|     o = False (Pixel is not masked and included in lensing)
    |x|x|x|o|o|o|x|x|x|x|
    |x|x|x|o|o|o|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    Here, there are 9 unmasked pixels. Indexing of each unmasked pixel goes from the top-left corner right,
    and downwards, therefore:

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|0|1|2|x|x|x|x|
    |x|x|x|3|4|5|x|x|x|x|
    |x|x|x|6|7|8|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    For every unmasked pixel, the Convolver over-lays the PSF psf and computes three quantities;

    image_frame_indexes - The indexes of all masked_image pixels it will blur light into.
    image_frame_psfs - The psf values that overlap each masked_image pixel it will blur light into.
    image_frame_length - The number of masked_image-pixels it will blur light into (because unmasked pixels are excluded)

    For howtolens, if we had the following 3x3 psf:

    |0.1|0.2|0.3|
    |0.4|0.5|0.6|
    |0.7|0.8|0.9|

    For pixel 0 above, when we overlap the psf 4 unmasked pixels overlap this psf, such that:

    image_frame_indexes = [0, 1, 3, 4]
    image_frame_psfs = [0.5, 0.6, 0.8, 0.9]
    image_frame_length = 4

    Noting that the other 5 psf values (0.1, 0.2, 0.3, 0.4, 0.7) overlap masked pixels and are thus discard.

    For pixel 1, we get the following results:

    image_frame_indexes = [0, 1, 2, 3, 4, 5]
    image_frame_psfs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    image_frame_lengths = 6

    In the majority of cases, there will be no unmasked pixels when the psf overlaps. This is the case above for \
    central pixel 4, where:

    image_frame_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    image_frame_psfs = [0,1, 0.2, 0,3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    image_frame_lengths = 6

    Once we have set up all these quantities, the convolution routine simply uses them to convolve_image a 1D array of masked_image
    datas / a mapping_matrix matrix masked_image.

    BLURRING FRAMES:
    --------------

    Whilst the scheme above accounts for all blurred light within the masks, it does not account for the fact that
    pixels outside of the masks will also blur light into it. For model_galaxy light profiles, this effect is accounted for \
    using blurring frames, however it is omitted for mapping_matrix matrix regular.

    First, a blurring masks is computed from a masks, which describes all pixels which are close enough to the masks \
    to blur light into it for a given psf size. Following the howtolens above, the following blurring masks is \
    computed:

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an howtolens masked_image.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|o|o|o|o|o|x|x|x|     x = True (Pixel is masked and excluded from lensing)
    |x|x|o|x|x|x|o|x|x|x|     o = False (Pixel is not masked and included in lensing)
    |x|x|o|x|x|x|o|x|x|x|
    |x|x|o|x|x|x|o|x|x|x|
    |x|x|o|o|o|o|o|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    Indexing again goes from the top-left corner right and downwards:

    |x|x| x| x| x| x| x|x|x|x|
    |x|x| x| x| x| x| x|x|x|x|
    |x|x| x| x| x| x| x|x|x|x|
    |x|x| 0| 1| 2| 3| 4|x|x|x|
    |x|x| 5| x| x| x| 6|x|x|x|
    |x|x| 7| x| x| x| 8|x|x|x|
    |x|x| 9| x| x| x|10|x|x|x|
    |x|x|11|12|13|14|15|x|x|x|
    |x|x| x| x| x| x| x|x|x|x|
    |x|x| x| x| x| x| x|x|x|x|

    For every unmasked blurring-pixel, the Convolver over-lays the PSF psf and computes three quantities;

    blurring_frame_indexes - The indexes of all unmasked masked_image pixels it will blur light into.
    bluring_frame_kernels - The psf values that overlap each unmasked masked_image pixel it will blur light into.
    blurring_frame_length - The number of masked_image-pixels it will blur light into (because unmasked pixels are excluded)

    Note, therefore, that the blurring frame does not perform any blurring which blurs light into other blurring pixels.
    It only performs computations which add light inside of the masks, which is the most computationally efficient method.

    For pixel 0 above, when we overlap the 3x3 psf above only 1 unmasked masked_image pixels overlap the psf, such that:

    blurring_frame_indexes = [0]
    blurring_frame_psfs = [0.9]
    blurring_frame_length = 1

    For pixel 1 above, when we overlap the 3x3 psf above 2 unmasked masked_image pixels overlap the psf, such that:

    blurring_frame_indexes = [0, 1]
    blurring_frame_psfs = [0.8, 0.9]
    blurring_frame_length = 2

    For pixel 3 above, when we overlap the 3x3 psf above 3 unmasked masked_image pixels overlap the psf, such that:

    blurring_frame_indexes = [0, 1, 2]
    blurring_frame_psfs = [0.7, 0.8, 0.9]
    blurring_frame_length = 3
    """

    mask = None
    pixels_in_mask = None
    mask_index_array = None
    kernel = None
    kernel_shape = None
    kernel_max_size = None

    def setup_mask_index_array(self):
        count = 0
        for x in range(self.mask.shape[0]):
            for y in range(self.mask.shape[1]):
                if not self.mask[x, y]:
                    self.mask_index_array[x, y] = count
                    count += 1

    def setup_image_frames(self):
        image_index = 0
        self.image_frame_indexes = np.zeros((self.pixels_in_mask, self.kernel_max_size), dtype='int')
        self.image_frame_kernels = np.zeros((self.pixels_in_mask, self.kernel_max_size))
        self.image_frame_lengths = np.zeros((self.pixels_in_mask), dtype='int')
        for x in range(self.mask_index_array.shape[0]):
            for y in range(self.mask_index_array.shape[1]):
                if not self.mask[x][y]:
                    image_frame_indexes, image_frame_kernels = self.frame_at_coords_jit((x, y), self.mask,
                                                                                        self.mask_index_array,
                                                                                        self.kernel)
                    self.image_frame_indexes[image_index, :] = image_frame_indexes
                    self.image_frame_kernels[image_index, :] = image_frame_kernels
                    self.image_frame_lengths[image_index] = image_frame_indexes[image_frame_indexes >= 0].shape[0]
                    image_index += 1

    def frame_at_coords(self, coords):
        """
        Parameters
        ----------
        coords: (int, int)
            The regular_grid of mask_index_array on which the frame should be centred
        psf_shape: (int, int)
            The shape of the psf for which this frame will be used
        Returns
        -------
        frame: ndarray
            A subset of mask_index_array of shape psf_shape where elements with regular_grid outside of image_frame_indexes have
            value -1
        """
        half_x = int(self.kernel_shape[0] / 2)
        half_y = int(self.kernel_shape[1] / 2)

        frame = -1 * np.ones((self.kernel_max_size))
        kernel_frame = -1.0 * np.ones((self.kernel_max_size))

        count = 0
        for i in range(self.kernel_shape[0]):
            for j in range(self.kernel_shape[1]):
                x = coords[0] - half_x + i
                y = coords[1] - half_y + j
                if 0 <= x < self.mask_index_array.shape[0] and 0 <= y < self.mask_index_array.shape[1]:
                    value = self.mask_index_array[x, y]
                    if value >= 0 and not self.mask[x, y]:
                        frame[count] = value
                        kernel_frame[count] = self.kernel[i, j]
                        count += 1

        return frame, kernel_frame

    @staticmethod
    @numba.jit(nopython=True)
    def frame_at_coords_jit(coords, mask, mask_index_array, kernel):
        """
        Parameters
        ----------
        coords: (int, int)
            The regular_grid of mask_index_array on which the frame should be centred
        psf_shape: (int, int)
            The shape of the psf for which this frame will be used
        Returns
        -------
        frame: ndarray
            A subset of mask_index_array of shape psf_shape where elements with regular_grid outside of image_frame_indexes have
            value -1
        """

        kernel_shape = kernel.shape
        kernel_max_size = kernel_shape[0] * kernel_shape[1]

        half_x = int(kernel_shape[0] / 2)
        half_y = int(kernel_shape[1] / 2)

        frame = -1 * np.ones((kernel_max_size))
        kernel_frame = -1.0 * np.ones((kernel_max_size))

        count = 0
        for i in range(kernel_shape[0]):
            for j in range(kernel_shape[1]):
                x = coords[0] - half_x + i
                y = coords[1] - half_y + j
                if 0 <= x < mask_index_array.shape[0] and 0 <= y < mask_index_array.shape[1]:
                    value = mask_index_array[x, y]
                    if value >= 0 and not mask[x, y]:
                        frame[count] = value
                        kernel_frame[count] = kernel[i, j]
                        count += 1

        return frame, kernel_frame


class ConvolverImage(Convolver):

    def __init__(self, mask, blurring_mask, kernel):
        """
        Class to create masked_image frames and blurring frames used to convolve_image a psf with a 1D masked_image of non-masked \
        values.

        Parameters
        ----------
        mask : Mask
            A masks where True eliminates datas.
        burring_mask : Mask
            A masks of pixels outside the masks but whose light blurs into it after convolution.
        kernel : masked_image.PSF or ndarray
            An array representing a PSF psf.
        """

        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise exc.KernelException("Kernel must be odd")
        if mask.shape != blurring_mask.shape:
            raise exc.KernelException("Mask and Blurring masks must be same shape to generate Convolver")

        self.mask = mask
        self.blurring_mask = blurring_mask
        self.mask_index_array = np.full(self.mask.shape, -1)
        self.pixels_in_mask = int(np.size(self.mask) - np.sum(self.mask))
        self.pixels_in_blurring_mask = int(np.size(blurring_mask) - np.sum(blurring_mask))

        self.setup_mask_index_array()

        self.kernel = kernel
        self.kernel_shape = kernel.shape
        self.kernel_max_size = self.kernel_shape[0] * self.kernel_shape[1]

        self.setup_image_frames()
        self.setup_blurring_frames()

    def setup_blurring_frames(self):

        image_index = 0
        self.blurring_frame_indexes = np.zeros((self.pixels_in_blurring_mask, self.kernel_max_size), dtype='int')
        self.blurring_frame_kernels = np.zeros((self.pixels_in_blurring_mask, self.kernel_max_size))
        self.blurring_frame_lengths = np.zeros((self.pixels_in_blurring_mask), dtype='int')
        for x in range(self.mask.shape[0]):
            for y in range(self.mask.shape[1]):
                if self.mask[x][y] and not self.blurring_mask[x, y]:
                    image_frame_indexes, image_frame_kernels = self.frame_at_coords_jit((x, y), self.mask,
                                                                                        self.mask_index_array,
                                                                                        self.kernel)
                    self.blurring_frame_indexes[image_index, :] = image_frame_indexes
                    self.blurring_frame_kernels[image_index, :] = image_frame_kernels
                    self.blurring_frame_lengths[image_index] = image_frame_indexes[image_frame_indexes >= 0].shape[0]
                    image_index += 1

    def convolve_image(self, image_array, blurring_array):
        """
        Parameters
        ----------
        blurring_array: [Float]
            An array representing the mapping_matrix of a source pixel to a set of masked_image pixels within the blurring region.
        sub_shape: (int, int)
            Defines a sub_grid-region of the psf for which the result should be calculated
        image_array: [float]
            A 1D array
        Returns
        -------
        convolved_vector: [float]
            A vector convolved with the psf
        """

        new_array = np.zeros(image_array.shape)

        for image_index in range(len(image_array)):

            frame_indexes = self.image_frame_indexes[image_index]
            frame_kernels = self.image_frame_kernels[image_index]
            frame_length = self.image_frame_lengths[image_index]
            value = image_array[image_index]

            for kernel_index in range(frame_length):
                vector_index = frame_indexes[kernel_index]
                kernel = frame_kernels[kernel_index]
                new_array[vector_index] += value * kernel

        for image_index in range(len(blurring_array)):

            frame_indexes = self.blurring_frame_indexes[image_index]
            frame_kernels = self.blurring_frame_kernels[image_index]
            frame_length = self.blurring_frame_lengths[image_index]
            value = blurring_array[image_index]

            for kernel_index in range(frame_length):
                vector_index = frame_indexes[kernel_index]
                kernel = frame_kernels[kernel_index]
                new_array[vector_index] += value * kernel

        return new_array

    def convolve_image_jitted(self, image_array, blurring_array):
        return self.convolve_image_jit(image_array, self.image_frame_indexes,
                                       self.image_frame_kernels, self.image_frame_lengths,
                                       blurring_array, self.blurring_frame_indexes,
                                       self.blurring_frame_kernels, self.blurring_frame_lengths)

    @staticmethod
    @numba.jit(nopython=True)
    def convolve_image_jit(image_array, image_frame_indexes, image_frame_kernels, image_frame_lengths,
                           blurring_array, blurring_frame_indexes, blurring_frame_kernels, blurring_frame_lengths):

        new_array = np.zeros(image_array.shape)

        for image_index in range(len(image_array)):

            frame_indexes = image_frame_indexes[image_index]
            frame_kernels = image_frame_kernels[image_index]
            frame_length = image_frame_lengths[image_index]
            value = image_array[image_index]

            for kernel_index in range(frame_length):
                vector_index = frame_indexes[kernel_index]
                kernel = frame_kernels[kernel_index]
                new_array[vector_index] += value * kernel

        for image_index in range(len(blurring_array)):

            frame_indexes = blurring_frame_indexes[image_index]
            frame_kernels = blurring_frame_kernels[image_index]
            frame_length = blurring_frame_lengths[image_index]
            value = blurring_array[image_index]

            for kernel_index in range(frame_length):
                vector_index = frame_indexes[kernel_index]
                kernel = frame_kernels[kernel_index]
                new_array[vector_index] += value * kernel

        return new_array


sub_grid_size = 4
psf_shape = (21, 21)
# psf_shape = (41, 41)
sersic = light_profiles.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=0.1,
                                         effective_radius=0.8, sersic_index=4.0)

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size, psf_shape=psf_shape)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size, psf_shape=psf_shape)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size, psf_shape=psf_shape)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size, psf_shape=psf_shape)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=sub_grid_size, psf_shape=psf_shape)

lsst_image = sersic.intensities_from_grid(grid=lsst.grids.image_plane_images_)
lsst_blurring_image = sersic.intensities_from_grid(grid=lsst.grids.blurring)
assert lsst.masked_image.convolver_image.convolve_image(lsst_image, lsst_blurring_image) == \
       pytest.approx(lsst.masked_image.convolver_image.convolve_image(lsst_image, lsst_blurring_image), 1e-4)

euclid_image = sersic.intensities_from_grid(grid=euclid.grids.image_plane_images_)
euclid_blurring_image = sersic.intensities_from_grid(grid=euclid.grids.blurring)
hst_image = sersic.intensities_from_grid(grid=hst.grids.image_plane_images_)
hst_blurring_image = sersic.intensities_from_grid(grid=hst.grids.blurring)
hst_up_image = sersic.intensities_from_grid(grid=hst_up.grids.image_plane_images_)
hst_up_blurring_image = sersic.intensities_from_grid(grid=hst_up.grids.blurring)
ao_image = sersic.intensities_from_grid(grid=ao.grids.image_plane_images_)
ao_blurring_image = sersic.intensities_from_grid(grid=ao.grids.blurring)

euclid.masked_image.convolver_image.convolve_image(image_array=euclid_image,
                                                   blurring_array=euclid_blurring_image)
hst.masked_image.convolver_image.convolve_image(image_array=hst_image, blurring_array=hst_blurring_image)
hst_up.masked_image.convolver_image.convolve_image(image_array=hst_up_image,
                                                   blurring_array=hst_up_blurring_image)
ao.masked_image.convolver_image.convolve_image(image_array=ao_image, blurring_array=ao_blurring_image)


@tools.tick_toc_x1
def lsst_solution():
    lsst.masked_image.convolver_image.convolve_image(image_array=lsst_image, blurring_array=lsst_blurring_image)


@tools.tick_toc_x1
def euclid_solution():
    euclid.masked_image.convolver_image.convolve_image(image_array=euclid_image, blurring_array=euclid_blurring_image)


@tools.tick_toc_x1
def hst_solution():
    hst.masked_image.convolver_image.convolve_image(image_array=hst_image, blurring_array=hst_blurring_image)


@tools.tick_toc_x1
def hst_up_solution():
    hst_up.masked_image.convolver_image.convolve_image(image_array=hst_up_image, blurring_array=hst_up_blurring_image)


@tools.tick_toc_x1
def ao_solution():
    ao.masked_image.convolver_image.convolve_image(image_array=ao_image, blurring_array=ao_blurring_image)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
