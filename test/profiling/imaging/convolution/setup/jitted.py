import numba
import numpy as np
import pytest
from profiling import profiling_data
from profiling import tools

from autolens import exc
from autolens.model.profiles import light_profiles


class Convolver(object):

    def __init__(self, mask, blurring_mask, kernel):
        """
        Class to create number array and frames used to convolve_image a psf with a 1D vector of non-masked values.

        Parameters
        ----------
        mask : Mask
            A masks where True eliminates datas.
        mask : Mask
            A masks of pixels outside the masks but whose light blurs into it after convolution.
        kernel : masked_image.PSF or ndarray
            An array representing a PSF psf.

        Attributes
        ----------
        blurring_frame_indexes: [ndarray]
            An array of frames created by the frame maker. Maps positions in the psf to values in the 1D vector for
            masked pixels.
        image_frame_indexes: [ndarray]
            An array of frames created by the frame maker. A frame maps positions in the psf to values in the 1D
            vector.
        """

        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise exc.KernelException("Kernel must be odd")
        if mask.shape != blurring_mask.shape:
            raise exc.KernelException("Mask and Blurring masks must be same shape to generate Convolver")

        self.mask = mask
        self.kernel = kernel
        self.kernel_shape = kernel.shape
        self.mask_index_array = np.full(self.mask.shape, -1)

        count = 0

        for x in range(self.mask.shape[0]):
            for y in range(self.mask.shape[1]):
                if not self.mask[x, y]:
                    self.mask_index_array[x, y] = count
                    count += 1

        self.frame_array = []
        self.frame_kernel_array = []
        for x in range(self.mask_index_array.shape[0]):
            for y in range(self.mask_index_array.shape[1]):
                if not self.mask[x][y]:
                    frame, kernel_frame = self.frame_at_coords((x, y), self.mask, self.mask_index_array, self.kernel)
                    self.frame_array.append(frame)
                    self.frame_kernel_array.append(kernel_frame)

        self.frame_lengths = np.asarray(list(map(lambda frame: frame.shape[0], self.frame_array)), dtype='int')

        self.blurring_frame_array = []
        self.blurring_frame_kernel_array = []
        for x in range(self.mask.shape[0]):
            for y in range(self.mask.shape[1]):
                if self.mask[x][y] and not blurring_mask[x, y]:
                    frame, kernel_frame = self.frame_at_coords((x, y), self.mask, self.mask_index_array, self.kernel)
                    self.blurring_frame_array.append(frame)
                    self.blurring_frame_kernel_array.append(kernel_frame)

        self.blurring_frame_lengths = np.asarray(list(map(lambda frame: frame.shape[0], self.blurring_frame_array)),
                                                 dtype='int')

    @staticmethod
    @numba.jit(nopython=True)
    def frame_at_coords(coords, mask, mask_index_array, kernel):
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

        frame = np.zeros((kernel_max_size))
        kernel_frame = np.zeros((kernel_max_size))

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

        return frame[0:count], kernel_frame[0:count]


sub_grid_size = 4
# psf_shape = (21, 21)
psf_shape = (41, 41)
sersic = light_profiles.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=0.1,
                                         effective_radius=0.8, sersic_index=4.0)

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size, psf_shape=psf_shape)

lsst_convolver = Convolver(mask=lsst.mask, blurring_mask=lsst.masked_image.blurring_mask,
                           kernel=lsst.image_plane_images_.psf.resized_scaled_array_from_array(psf_shape))

assert lsst.masked_image.convolver.frame_array[0] == pytest.approx(lsst_convolver.frame_array[0], 1e-2)
assert lsst.masked_image.convolver.frame_array[1] == pytest.approx(lsst_convolver.frame_array[1], 1e-2)
assert lsst.masked_image.convolver.frame_kernel_array[1] == pytest.approx(lsst_convolver.frame_kernel_array[1], 1e-2)
assert lsst.masked_image.convolver.blurring_frame_array[0] == pytest.approx(lsst_convolver.blurring_frame_array[0],
                                                                            1e-2)
assert lsst.masked_image.convolver.blurring_frame_array[1] == pytest.approx(lsst_convolver.blurring_frame_array[1],
                                                                            1e-2)
assert lsst.masked_image.convolver.blurring_frame_kernel_array[1] == pytest.approx(
    lsst_convolver.blurring_frame_kernel_array[1], 1e-2)

euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size, psf_shape=psf_shape)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size, psf_shape=psf_shape)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size, psf_shape=psf_shape)


# ao = profiling_data.setup_class(phase_name='AO', pixel_scales=0.01, sub_grid_size=sub_grid_size, psf_shape=psf_shape)

@tools.tick_toc_x1
def lsst_solution():
    Convolver(mask=lsst.mask, blurring_mask=lsst.masked_image.blurring_mask,
              kernel=lsst.image_plane_images_.psf.resized_scaled_array_from_array(psf_shape))


@tools.tick_toc_x1
def euclid_solution():
    Convolver(mask=euclid.mask, blurring_mask=euclid.masked_image.blurring_mask,
              kernel=euclid.image_plane_images_.psf.resized_scaled_array_from_array(psf_shape))


@tools.tick_toc_x1
def hst_solution():
    Convolver(mask=hst.mask, blurring_mask=hst.masked_image.blurring_mask,
              kernel=hst.image_plane_images_.psf.resized_scaled_array_from_array(psf_shape))


@tools.tick_toc_x1
def hst_up_solution():
    Convolver(mask=hst_up.mask, blurring_mask=hst_up.masked_image.blurring_mask,
              kernel=hst_up.image_plane_images_.psf.resized_scaled_array_from_array(psf_shape))


@tools.tick_toc_x1
def ao_solution():
    Convolver(mask=ao.mask, blurring_mask=ao.blurring_mask, kernel=ao.image_plane_images_.psf.resized_scaled_array_from_array(psf_shape))


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
