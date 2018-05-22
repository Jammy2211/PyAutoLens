import numpy as np
from matplotlib import pyplot

from auto_lens.imaging import imaging


def setup_random_seed(seed):
    """Setup the random seed. If the input seed is -1, the code will use a random seed for every run. If it is positive,
    that seed is used for all runs, thereby giving reproducible results

    Parameters
    ----------
    seed : int
        The seed of the random number generator, used for the random signal_to_noise_ratio maps.
    """
    if seed == -1:
        seed = np.random.randint(0, int(1e9))  # Use one seed, so all regions have identical column non-uniformity.
    np.random.seed(seed)


class SimulateImage(imaging.Image):

    def __init__(self, data, pixel_scale, exposure_time, sim_optics=None, sim_poisson_noise=None,
                 sim_background_noise=None, noise_seed=-1):
        """
        Creates a new simulated image.

        Parameters
        ----------
        data : ndarray
            The image of the lensed to be simulated.
        pixel_scale: float
            The scale of an image pixel.
        exposure_time : imaging.ExposureTime
            The exposure time in each image pixel, used to convert the image from electrons per second to counts.
        sim_optics : simulaate.SimulateOptics
            Blurs image with PSF.
        sim_poisson_noise : simulaate.SimulatePoissonNoise
            Adds Poisson noise to image.
        sim_background_noise : simulaate.SimulateBackgroundNoise
            Adds background noise to image.
        noise_seed : int
            The seed of the random number generator, used for the random signal_to_noise_ratio maps.
        """

        self.data_original = np.asarray(data)

        super(SimulateImage, self).__init__(data, pixel_scale)

        self.exposure_time = exposure_time

        self.sim_optics = sim_optics
        self.sim_poisson_noise = sim_poisson_noise
        self.sim_background_noise = sim_background_noise

        if self.sim_optics is not None:
            self.data = self.sim_optics.simulate_for_image(self.data)

        if self.sim_poisson_noise is not None:
            self.data += self.sim_poisson_noise.simulate_for_image(self.data, self.exposure_time.data)

        if self.sim_background_noise is not None:
            self.data += self.sim_poisson_noise.simulate_for_image(self.data)

        self.background_noise = sim_background_noise
        self.noise_seed = noise_seed

    #    self.estimate_noise_in_simulated_image()
    #    self.estimate_signal_to_noise_ratio_in_simulated_image()

    # TODO: This should match the super
    @classmethod
    def from_fits(cls, path, filename, hdu, pixel_scale, exposure_time, sim_optics=None, sim_poisson_noise=None,
                  sim_background_noise=None, noise_seed=-1):
        """
        Loads the image data from a .fits file.

        Parameters
        ----------
        path : str
            The directory path to the fits file.
        filename : str
            The file name of the fits file.
        hdu : int
            The HDU number in the fits file containing the image data.
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        data = imaging.numpy_array_from_fits(path + filename, hdu)
        return SimulateImage(data, pixel_scale, exposure_time, sim_optics, sim_poisson_noise, sim_background_noise,
                             noise_seed)

    # TODO : These will ultimately removed to the Data classed, which this routine inherites from

    def estimate_noise_in_simulated_image(self):
        """Estimate the signal_to_noise_ratio in the simulated image, using the exposure time and background
        signal_to_noise_ratio"""
        # TODO: These should be defined in the constructor or converted into properties
        self.noise = imaging.estimate_noise_from_image(self.data, self.exposure_time, self.background_noise)

    def estimate_signal_to_noise_ratio_in_simulated_image(self):
        """Estimate the signal_to_noise_ratio in the simulated image, using the exposure time and background
        signal_to_noise_ratio"""
        self.signal_to_noise_ratio = np.divide(self.data, self.noise)

    def plot(self):
        pyplot.imshow(self.data)
        pyplot.show()


class SimulateOptics(object):

    def __init__(self, psf):
        """Class to blur simulated image with a psf.

        Parameters
        -----------
        psf : imaging.PSF
            The PSF.
        """
        self.psf = psf

    def simulate_for_image(self, image):
        return self.psf.convolve_with_image(image)


class SimulatePoissonNoise(object):

    def __init__(self, noise_seed=-1):
        """Class to add Poisson noise (e.g. count statistics in an image) to a simulated image.

        Parameters
        ----------
        noise_seed : int
            The seed of the random number generator, used for the random noise maps.
        """
        self.noise_seed = noise_seed

    def simulate_for_image(self, image, exposure_time):
        """Generate a two-dimensional background noise-map for an image, generating values from a Gaussian \
        distribution with mean 0.0.

        Parameters
        ----------
        image : ndarray
            The 2D image background noise is added to.
        exposure_time : ndarray
            The 2D array of pixel exposure times.
        """
        setup_random_seed(self.noise_seed)
        image_counts = imaging.electrons_per_second_to_counts(image, exposure_time)
        # TODO: Should be __init__ or property
        self.poisson_noise_map = image - np.divide(np.random.poisson(image_counts, image.shape), exposure_time)
        return image + self.poisson_noise_map


def poisson_noise(image, exposure_time, seed=-1):
    """
    Generate a two-dimensional background noise-map for an image, generating values from a Gaussian
    distribution with mean 0.0.

    Parameters
    ----------
    image : ndarray
        The 2D image background noise is added to.
    exposure_time : ndarray
        The 2D array of pixel exposure times.
    seed : int
        The seed of the random number generator, used for the random noise maps.

    Returns
    -------
    poisson_noise: ndarray
        An array describing simulated poisson noise 
    """
    setup_random_seed(seed)
    image_counts = imaging.electrons_per_second_to_counts(image, exposure_time)
    return image - np.divide(np.random.poisson(image_counts, image.shape), exposure_time)


def background_noise(image, sigma, seed=-1):
    setup_random_seed(seed)
    background_noise_map = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    return background_noise_map
