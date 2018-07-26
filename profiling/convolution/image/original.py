import numpy as np

from src.profiles import light_profiles
from profiling import profiling_data
from profiling import tools
from imaging import convolution


class KernelConvolverProfiling(convolution.KernelConvolver):

    def __init__(self, kernel, frame_array, blurring_frame_array=None):

        super(KernelConvolverProfiling, self).__init__(kernel, frame_array, blurring_frame_array)

    def convolve_array(self, pixel_array, blurring_array=None, sub_shape=None):
        """
        Parameters
        ----------
        blurring_array: [Float]
            An array representing the mapping of a source pixel to a set of image pixels within the blurring region.
        sub_shape: (int, int)
            Defines a sub_grid-region of the kernel for which the result should be calculated
        pixel_array: [float]
            A 1D array
        Returns
        -------
        convolved_vector: [float]
            A vector convolved with the kernel
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

subgrid_size=2
psf_shape = (41, 41)
sersic = light_profiles.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=0.1,
                                         effective_radius=0.8, sersic_index=4.0)

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, subgrid_size=subgrid_size, psf_shape=psf_shape)
lsst_kernel_convolver = KernelConvolverProfiling(kernel=lsst.image.psf.trim(psf_shape),
                                                 frame_array=lsst.masked_image.convolver.frame_array,
                                                 blurring_frame_array=lsst.masked_image.convolver.blurring_frame_array)

euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, subgrid_size=subgrid_size, psf_shape=psf_shape)
euclid_kernel_convolver = KernelConvolverProfiling(kernel=lsst.image.psf.trim(psf_shape),
                                                 frame_array=lsst.masked_image.convolver.frame_array,
                                                 blurring_frame_array=lsst.masked_image.convolver.blurring_frame_array)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, subgrid_size=subgrid_size, psf_shape=psf_shape)
hst_kernel_convolver = KernelConvolverProfiling(kernel=lsst.image.psf.trim(psf_shape),
                                                 frame_array=lsst.masked_image.convolver.frame_array,
                                                 blurring_frame_array=lsst.masked_image.convolver.blurring_frame_array)

hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, subgrid_size=subgrid_size, psf_shape=psf_shape)
hst_up_kernel_convolver = KernelConvolverProfiling(kernel=lsst.image.psf.trim(psf_shape),
                                                 frame_array=lsst.masked_image.convolver.frame_array,
                                                 blurring_frame_array=lsst.masked_image.convolver.blurring_frame_array)

ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, subgrid_size=subgrid_size, psf_shape=psf_shape)
ao_kernel_convolver = KernelConvolverProfiling(kernel=lsst.image.psf.trim(psf_shape),
                                                 frame_array=lsst.masked_image.convolver.frame_array,
                                                 blurring_frame_array=lsst.masked_image.convolver.blurring_frame_array)

lsst_image = sersic.intensity_from_grid(grid=lsst.coords.image_coords)
lsst_blurring_image = sersic.intensity_from_grid(grid=lsst.coords.blurring_coords)
euclid_image = sersic.intensity_from_grid(grid=euclid.coords.image_coords)
euclid_blurring_image = sersic.intensity_from_grid(grid=euclid.coords.blurring_coords)
hst_image = sersic.intensity_from_grid(grid=hst.coords.image_coords)
hst_blurring_image = sersic.intensity_from_grid(grid=hst.coords.blurring_coords)
hst_up_image = sersic.intensity_from_grid(grid=hst_up.coords.image_coords)
hst_up_blurring_image = sersic.intensity_from_grid(grid=hst_up.coords.blurring_coords)
ao_image = sersic.intensity_from_grid(grid=ao.coords.image_coords)
ao_blurring_image = sersic.intensity_from_grid(grid=ao.coords.blurring_coords)

@tools.tick_toc_x1
def lsst_solution():
    lsst_kernel_convolver.convolve_array(pixel_array=lsst_image, blurring_array=lsst_blurring_image)

@tools.tick_toc_x1
def euclid_solution():
    euclid_kernel_convolver.convolve_array(pixel_array=euclid_image, blurring_array=euclid_blurring_image)

@tools.tick_toc_x1
def hst_solution():
    hst_kernel_convolver.convolve_array(pixel_array=hst_image, blurring_array=hst_blurring_image)

@tools.tick_toc_x1
def hst_up_solution():
    hst_up_kernel_convolver.convolve_array(pixel_array=hst_up_image, blurring_array=hst_up_blurring_image)

@tools.tick_toc_x1
def ao_solution():
    ao_kernel_convolver.convolve_array(pixel_array=ao_image, blurring_array=ao_blurring_image)

# @tools.tick_toc_x1
# def jitted_solution():
#     kernel_convolver.convolve_array_jit(data)

if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()