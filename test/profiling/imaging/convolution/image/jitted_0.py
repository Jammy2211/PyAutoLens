import numba
import numpy as np
import pytest
from profiling import profiling_data
from profiling import tools

from autolens.model.profiles import light_profiles


class FrameMakerOriginal(object):
    def __init__(self, mask):
        """
        Class to create number array and frames used in 1D convolution
        Parameters
        ----------
        mask: Mask
                A masks where True eliminates data_vector
        """
        self.mask = mask

        self.number_array = np.full(self.mask.shape, -1)

        count = 0

        for x in range(self.mask.shape[0]):
            for y in range(self.mask.shape[1]):
                if not self.mask[x, y]:
                    self.number_array[x, y] = count
                    count += 1

    def make_frame_array(self, kernel_shape):
        """
        Parameters
        ----------
            An array in which non-masked elements have been numbered 0, 1, 2,...N
        kernel_shape: (int, int)
            The shape of the psf for which frames will be used
        Returns
        -------
        image_frame_indexes: [ndarray]
            A list of frames where the position of a frame corresponds to the number at the origin of that frame
        """
        if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0:
            raise exc.KernelException("Kernel must be odd")
        frame_array = []

        # TODO: How would I avoid using for-loops here?
        for x in range(self.number_array.shape[0]):
            for y in range(self.number_array.shape[1]):
                if not self.mask[x][y]:
                    frame_array.append(self.frame_at_coords((x, y), kernel_shape))

        return frame_array

    def make_blurring_frame_array(self, kernel_shape, blurring_region_mask=None):
        """
        Parameters
        ----------
        kernel_shape: (int, int)
            The shape of the psf
        blurring_region_mask: Mask
            A masks describing the boundary of the region from which values may be convoluted

        Returns
        -------
        blurring_frame_indexes [ndarray]
            A list of frames where the position corresponds to a position in the blurring region data_vector grid and the
            entries correspond to positions in the primary data_vector grid
        """
        if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0:
            raise exc.KernelException("Kernel must be odd")
        if blurring_region_mask is not None and self.mask.shape != blurring_region_mask.shape:
            raise AssertionError("masks and blurring_region_mask must have the same shape")

        frame_array = []
        for x in range(self.mask.shape[0]):
            for y in range(self.mask.shape[1]):
                if self.mask[x][y] and (blurring_region_mask is None or not blurring_region_mask[x, y]):
                    frame = self.frame_at_coords((x, y), kernel_shape)
                    frame_array.append(frame)
        return frame_array

    def frame_at_coords(self, coords, kernel_shape):
        """
        Parameters
        ----------
        coords: (int, int)
            The image_grid of mask_index_array on which the frame should be centred
        kernel_shape: (int, int)
            The shape of the psf for which this frame will be used
        Returns
        -------
        frame: ndarray
            A subset of mask_index_array of shape psf_shape where elements with image_grid outside of image_frame_indexes have
            value -1
        """
        half_x = int(kernel_shape[0] / 2)
        half_y = int(kernel_shape[1] / 2)

        frame = np.full((kernel_shape[0] * kernel_shape[1],), -1)

        for i in range(kernel_shape[0]):
            for j in range(kernel_shape[1]):
                x = coords[0] - half_x + i
                y = coords[1] - half_y + j
                if 0 <= x < self.number_array.shape[0] and 0 <= y < self.number_array.shape[1]:
                    value = self.number_array[x, y]
                    if value >= 0 and not self.mask[x, y]:
                        frame[j + kernel_shape[1] * i] = value

        return frame

    def convolver_for_kernel_shape(self, kernel_shape, blurring_region_mask=None):
        """
        Create a convolver_image that can be used to apply a psf of any shape to a 1D vector of non-masked values
        Parameters
        ----------
        blurring_region_mask: Mask
            A masks describing the blurring region. If False then that pixel is included int he blurring region.
        kernel_shape: (int, int)
            The shape of the psf
        Returns
        -------
            convolver_image: Convolver
        """
        return Convolver(self.make_frame_array(kernel_shape),
                         self.make_blurring_frame_array(
                             kernel_shape,
                             blurring_region_mask) if blurring_region_mask is not None else None)

    def convolver_for_kernel(self, psf):
        return self.convolver_for_kernel_shape(kernel_shape=psf.shape,
                                               blurring_region_mask=self.mask.blurring_mask_for_psf_shape(
                                                   psf_shape=psf.shape)).convolver_for_kernel(kernel=psf)


class ConvolverOriginal(object):
    def __init__(self, frame_array, blurring_frame_array):
        """
        Class to convolve_image a psf with a 1D vector of non-masked values
        Parameters
        ----------
        blurring_frame_array: [ndarray]
            An array of frames created by the frame maker. Maps positions in the psf to values in the 1D vector for
            masked pixels.
        frame_array: [ndarray]
            An array of frames created by the frame maker. A frame maps positions in the psf to values in the 1D
            vector.
        """
        self.frame_array = frame_array
        self.blurring_frame_array = blurring_frame_array

    def convolver_for_kernel(self, kernel):
        """
        Parameters
        ----------
        kernel: ndarray
            An array representing a psf

        Returns
        -------
        convolver_image: KernelConvolver
            An object used to convolve_image image
        """
        return KernelConvolver(kernel, self.frame_array, self.blurring_frame_array)


class KernelConvolverOriginal(object):

    def __init__(self, kernel, frame_array, blurring_frame_array=None):
        self.shape = kernel.shape

        self.length = self.shape[0] * self.shape[1]
        self.kernel = kernel.flatten()
        self.frame_array = frame_array
        self.blurring_frame_array = blurring_frame_array

    def convolve_array(self, pixel_array, blurring_array=None, sub_shape=None):
        """
        Parameters
        ----------
        blurring_array: [Float]
            An array representing the mapping_matrix of a source pixel to a set of masked_image pixels within the blurring region.
        sub_shape: (int, int)
            Defines a sub_grid-region of the psf for which the result should be calculated
        pixel_array: [float]
            A 1D array
        Returns
        -------
        convolved_vector: [float]
            A vector convolved with the psf
        """

        new_array = np.zeros(pixel_array.shape)

        for pixel_index in range(len(pixel_array)):
            frame = self.frame_array[pixel_index]
            value = pixel_array[pixel_index]

            if value > 0:
                new_array = self.convolution_for_value_frame_and_new_array(value, frame, new_array, sub_shape)

        if blurring_array is not None:
            for pixel_index in range(len(blurring_array)):
                frame = self.blurring_frame_array[pixel_index]
                value = blurring_array[pixel_index]

                if value > 0:
                    new_array = self.convolution_for_value_frame_and_new_array(value, frame, new_array, sub_shape)

        return new_array

    def convolve_array_jitted(self, pixel_array, blurring_array=None):
        return self.convolve_array_jit(pixel_array, self.frame_array, blurring_array, self.blurring_frame_array,
                                       self.length, self.kernel)

    @staticmethod
    @numba.jit(nopython=True)
    def convolve_array_jit(pixel_array, frame_array, blurring_array, blurring_frame_array, length, kernel):

        new_array = np.zeros(pixel_array.shape)

        for pixel_index in range(len(pixel_array)):

            frame = frame_array[pixel_index]
            value = pixel_array[pixel_index]

            for kernel_index in range(length):

                vector_index = frame[kernel_index]

                if vector_index == -1:
                    continue
                result = value * kernel[kernel_index]
                if result > 0:
                    new_array[vector_index] += result

        for pixel_index in range(len(blurring_array)):

            frame = blurring_frame_array[pixel_index]
            value = blurring_array[pixel_index]

            for kernel_index in range(length):

                vector_index = frame[kernel_index]

                if vector_index == -1:
                    continue
                result = value * kernel[kernel_index]
                if result > 0:
                    new_array[vector_index] += result

        return new_array

    def convolve_mapping_matrix(self, mapping):
        """
        Simple version of function that applies this convolver_image to a whole mapping_matrix matrix.

        Parameters
        ----------
        blurring_array: [Float]
            An array representing the mapping_matrix of a source pixel to a set of masked_image pixels within the blurring region.
        array: [float]
            An array representing the mapping_matrix of a source pixel to a set of masked_image pixels.

        Returns
        -------
        convolved_array: [float]
            A matrix representing the mapping_matrix of source data_to_image to image_grid data_to_image accounting for
            convolution
        """
        blurred_mapping = np.zeros(mapping.shape)
        for i in range(mapping.shape[1]):
            blurred_mapping[:, i] = self.convolve_array(mapping[:, i])
        return blurred_mapping

    def convolve_mapping_matrix_jit(self, mapping):
        """
        Simple version of function that applies this convolver_image to a whole mapping_matrix matrix.

        Parameters
        ----------
        blurring_array: [Float]
            An array representing the mapping_matrix of a source pixel to a set of masked_image pixels within the blurring region.
        array: [float]
            An array representing the mapping_matrix of a source pixel to a set of masked_image pixels.

        Returns
        -------
        convolved_array: [float]
            A matrix representing the mapping_matrix of source data_to_image to image_grid data_to_image accounting for
            convolution
        """

        @numba.jit(nopython=True)
        def convolve_matrix_jitted(mapping_matrix, frame_array, length, kernel):

            blurred_mapping = np.zeros(mapping_matrix.shape)

            for pixel_index in range(mapping_matrix.shape[1]):
                for image_index in range(mapping_matrix.shape[0]):

                    frame = frame_array[image_index]
                    value = mapping_matrix[image_index, pixel_index]

                    if value > 0:

                        for kernel_index in range(length):

                            vector_index = frame[kernel_index]

                            if vector_index == -1:
                                continue
                            result = value * kernel[kernel_index]
                            if result > 0:
                                blurred_mapping[vector_index, pixel_index] += result

            return blurred_mapping

        return convolve_matrix_jitted(mapping, self.frame_array, self.length, self.kernel)

    def convolution_for_value_frame_and_new_array(self, value, frame, new_array, sub_shape=None):
        """
        Convolves a value with the psf and populates a new array according to the entries in the frame

        Parameters
        ----------
        value: float
            Some value
        frame: ndarray
            An array describing which entries in the new array convolved values should be inserted into
        new_array: ndarray
            An array into which convolved values are inserted
        sub_shape: (int, int)
            The shape of a reduced sub_grid_size psf

        Returns
        -------
        new_array: ndarray
            The array with values convolved into it
        """

        limits = None
        if sub_shape is not None:
            limits = calculate_limits(self.shape, sub_shape)

        for kernel_index in range(self.length):
            if sub_shape is not None and not is_in_sub_shape(kernel_index, limits, self.shape):
                continue

            vector_index = frame[kernel_index]

            if vector_index == -1:
                continue
            result = value * self.kernel[kernel_index]
            if result > 0:
                new_array[vector_index] += result

        return new_array


sub_grid_size = 2
psf_shape = (41, 41)
sersic = light_profiles.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=0.1,
                                         effective_radius=0.8, sersic_index=4.0)

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size, psf_shape=psf_shape)
lsst_kernel_convolver = KernelConvolverOriginal(kernel=lsst.image_plane_images_.psf.resized_scaled_array_from_array(psf_shape),
                                                frame_array=lsst.masked_image.convolver.frame_array,
                                                blurring_frame_array=lsst.masked_image.convolver.blurring_frame_array)
lsst_image = sersic.intensities_from_grid(grid=lsst.grids.image_plane_images_)
lsst_blurring_image = sersic.intensities_from_grid(grid=lsst.grids.blurring)

assert (lsst_kernel_convolver.convolve_array(lsst_image, lsst_blurring_image) ==
        pytest.approx(lsst_kernel_convolver.convolve_array_jitted(lsst_image, lsst_blurring_image)))

euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size, psf_shape=psf_shape)
euclid_kernel_convolver = KernelConvolverOriginal(kernel=euclid.image_plane_images_.psf.resized_scaled_array_from_array(psf_shape),
                                                  frame_array=euclid.masked_image.convolver.frame_array,
                                                  blurring_frame_array=euclid.masked_image.convolver.blurring_frame_array)
euclid_image = sersic.intensities_from_grid(grid=euclid.grids.image_plane_images_)
euclid_blurring_image = sersic.intensities_from_grid(grid=euclid.grids.blurring)
euclid_kernel_convolver.convolve_array_jitted(pixel_array=euclid_image, blurring_array=euclid_blurring_image)

hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size, psf_shape=psf_shape)
hst_kernel_convolver = KernelConvolverOriginal(kernel=hst.image_plane_images_.psf.resized_scaled_array_from_array(psf_shape),
                                               frame_array=hst.masked_image.convolver.frame_array,
                                               blurring_frame_array=hst.masked_image.convolver.blurring_frame_array)
hst_image = sersic.intensities_from_grid(grid=hst.grids.image_plane_images_)
hst_blurring_image = sersic.intensities_from_grid(grid=hst.grids.blurring)
hst_kernel_convolver.convolve_array_jitted(pixel_array=hst_image, blurring_array=hst_blurring_image)

hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size, psf_shape=psf_shape)
hst_up_kernel_convolver = KernelConvolverOriginal(kernel=hst_up.image_plane_images_.psf.resized_scaled_array_from_array(psf_shape),
                                                  frame_array=hst_up.masked_image.convolver.frame_array,
                                                  blurring_frame_array=hst_up.masked_image.convolver.blurring_frame_array)
hst_up_image = sersic.intensities_from_grid(grid=hst_up.grids.image_plane_images_)
hst_up_blurring_image = sersic.intensities_from_grid(grid=hst_up.grids.blurring)
hst_up_kernel_convolver.convolve_array_jitted(pixel_array=hst_up_image, blurring_array=hst_up_blurring_image)

ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=sub_grid_size, psf_shape=psf_shape)
ao_kernel_convolver = KernelConvolverOriginal(kernel=ao.image_plane_images_.psf.resized_scaled_array_from_array(psf_shape),
                                              frame_array=ao.masked_image.convolver.frame_array,
                                              blurring_frame_array=ao.masked_image.convolver.blurring_frame_array)
ao_image = sersic.intensities_from_grid(grid=ao.grids.image_plane_images_)
ao_blurring_image = sersic.intensities_from_grid(grid=ao.grids.blurring)
ao_kernel_convolver.convolve_array_jitted(pixel_array=ao_image, blurring_array=ao_blurring_image)


@tools.tick_toc_x1
def lsst_solution():
    lsst_kernel_convolver.convolve_array_jitted(pixel_array=lsst_image, blurring_array=lsst_blurring_image)


@tools.tick_toc_x1
def euclid_solution():
    euclid_kernel_convolver.convolve_array_jitted(pixel_array=euclid_image, blurring_array=euclid_blurring_image)


@tools.tick_toc_x1
def hst_solution():
    hst_kernel_convolver.convolve_array_jitted(pixel_array=hst_image, blurring_array=hst_blurring_image)


@tools.tick_toc_x1
def hst_up_solution():
    hst_up_kernel_convolver.convolve_array_jitted(pixel_array=hst_up_image, blurring_array=hst_up_blurring_image)


@tools.tick_toc_x1
def ao_solution():
    ao_kernel_convolver.convolve_array_jitted(pixel_array=ao_image, blurring_array=ao_blurring_image)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
