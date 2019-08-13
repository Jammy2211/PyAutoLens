from autolens import decorator_util

import numpy as np

class Transformer(object):

    def __init__(self, uv_wavelengths, grid_radians):

        self.uv_wavelengths = uv_wavelengths
        self.grid_radians = grid_radians

        self.total_visibilities = uv_wavelengths.shape[0]
        self.total_image_pixels = grid_radians.shape[0]

        self.preloaded_reals = self.preload_real_visibilities(
            grid_radians=grid_radians,
            uv_wavelengths=uv_wavelengths,
            total_image_pixels=self.total_image_pixels)

    def real_visibilities_from_intensities(
            self, intensities_1d,
    ):

        return self.real_visibilities_jit(
            intensities_1d=intensities_1d, grid_radians=self.grid_radians,
            uv_wavelengths=self.uv_wavelengths, total_visibilities=self.total_visibilities,
            total_image_pixels=self.total_image_pixels)

    @staticmethod
    @decorator_util.jit()
    def real_visibilities_jit(intensities_1d, grid_radians, uv_wavelengths, total_visibilities, total_image_pixels):

        real_visibilities = np.zeros(shape=(total_visibilities))

        for i in range(total_image_pixels):
            for j in range(total_visibilities):
                real_visibilities[j] += intensities_1d[i] * np.cos(-2.0 * np.pi * (
                        grid_radians[i, 1] * uv_wavelengths[j, 0] - grid_radians[
                    i, 0] * uv_wavelengths[j, 1]))

        return real_visibilities

    @staticmethod
    @decorator_util.jit()
    def preload_real_visibilities(grid_radians, uv_wavelengths, total_image_pixels):

        preloaded_real_visibilities = np.zeros(shape=(total_image_pixels, uv_wavelengths.shape[0]))

        for i in range(total_image_pixels):
            for j in range(uv_wavelengths.shape[0]):
                preloaded_real_visibilities[i,j] += np.cos(-2.0 * np.pi * (
                grid_radians[i, 1] * uv_wavelengths[j, 0] - grid_radians[i, 0] * uv_wavelengths[j,1]))

        return preloaded_real_visibilities

    def real_visibilities_via_preload_from_intensities(
            self, intensities_1d,
    ):
        """Extract the 1D image-plane image and 1D blurring image-plane image of every plane and blur each with the \
        PSF using a convolver (see ccd.convolution).

        These are summed to give the tracer's overall blurred image-plane image in 1D.

        Parameters
        ----------
        convolver_image : hyper_galaxy.ccd.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        real_visibilities = np.zeros(shape=(self.uv_wavelengths.shape[0]))

        return self.real_visibilities_via_preload_jit(intensities_1d=intensities_1d, preloaded_reals=self.preloaded_reals,
                                                      total_visibilities=self.total_visibilities, total_image_pixels=self.total_image_pixels)

    @staticmethod
    @decorator_util.jit()
    def real_visibilities_via_preload_jit(intensities_1d, preloaded_reals, total_visibilities, total_image_pixels):

        real_visibilities = np.zeros(shape=(total_visibilities))

        for i in range(total_image_pixels):
            for j in range(total_visibilities):
                real_visibilities[j] += intensities_1d[i] * preloaded_reals[i,j]

        return real_visibilities