import numpy as np

class Transformer(object):

    def __init__(self, uv_wavelengths, grid_radians):

        self.uv_wavelengths = uv_wavelengths
        self.total_visibilities = len(uv_wavelengths)
        self.grid_radians = grid_radians

    def real_visibilities_from_intensities(
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

        for i in range(self.total_visibilities):
            real_visibilities += intensities_1d[i] * np.cos(2.0 * np.pi * (
                        self.grid_radians[i, 1] * self.uv_wavelengths[:, 0] - self.grid_radians[i, 0] * self.uv_wavelengths[:, 1]))

        return real_visibilities
