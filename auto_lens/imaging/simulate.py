import numpy as np
from matplotlib import pyplot

from auto_lens.imaging import imaging

def generate_poisson_noise_map(image, exposure_time, seed=-1):
    """Generate a Poisson two-dimensional signal_to_noise_ratio map from an input image. This includes a conversion of the image \
    from electrons per second to counts (and back).

    NOTE : np.random.poisson returns a new image subjected to Poisson signal_to_noise_ratio. This is subtracted from the image to \
    generate the Poisson signal_to_noise_ratio map.

    Parameters
    ----------
    image : ndarray
        The image in electrons per second, used to generate the Poisson signal_to_noise_ratio map.
    exposure_time : float or ndarray
        The exposure time in each image pixel, used to convert the image from electrons per second to counts.
    seed : int
        The seed of the random number generator, used for the random signal_to_noise_ratio maps.
    """
    setup_random_seed(seed)
    image_counts = imaging.convert_array_to_counts(image, exposure_time)
    return image - np.divide(np.random.poisson(image_counts, image.shape), exposure_time)

def generate_background_noise_map(dimensions, background_noise, seed=-1):
    """Generate a Gaussian background two-dimensional signal_to_noise_ratio map.

    Parameters
    ----------
    dimensions : (int, int)
        The (x,y) pixel_dimensions of the generated Gaussian signal_to_noise_ratio map.
    background_noise : float or ndarray
        Standard deviation of the 1D Gaussian that each signal_to_noise_ratio value is drawn from
    seed : int
        The seed of the random number generator, used for the random signal_to_noise_ratio maps.
    """
    setup_random_seed(seed)
    return np.random.normal(0.0, background_noise, dimensions)

def setup_random_seed(seed):
    """Setup the random seed. If the input seed is -1, the code will use a random seed for every run. If it is positive,
    that seed is used for all runs, thereby giving reproducible results

    Parameters
    ----------
    seed : int
        The seed of the random number generator, used for the random signal_to_noise_ratio maps.
    """
    if seed == -1:
        seed = np.random.randint(0, 1e9)  # Use one seed, so all regions have identical column non-uniformity.
    np.random.seed(seed)


class SimulateImage(imaging.Data):

    def __init__(self, data, pixel_scale, psf=None, exposure_time=None, background_noise=None, noise_seed=-1):
        """
        Creates a new simulated image.

        Parameters
        ----------
        data : ndarray
            The image of the lensed to be simulated.
        pixel_scale: float
            The scale of an image pixel.
        psf : imaging.PSF
            The image of the simulated image.
        exposure_time : ndarray
            The exposure time in each image pixel, used to convert the image from electrons per second to counts.
        noise_seed : int
            The seed of the random number generator, used for the random signal_to_noise_ratio maps.
        """

        self.data_original = data

        super(SimulateImage,self).__init__(data, pixel_scale)

        self.psf = psf
        self.exposure_time = exposure_time
        self.background_noise = background_noise
        self.noise_seed = noise_seed

        if self.psf is not None:
            self.simulate_optics()

        if self.exposure_time is not None:
            self.simulate_poisson_noise()

        if self.background_noise is not None:
            self.simulate_background_noise()

        self.estimate_noise_in_simulated_image()
        self.estimate_signal_to_noise_ratio_in_simulated_image()

    @classmethod
    def from_fits(cls, path, filename, hdu, pixel_scale, psf=None, exposure_time=None, background_noise=None,
                  noise_seed=-1):
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
        sky_background_level : float
            An estimate of the level of background sky in the image (electrons per second).
        sky_background_noise : float
            An estimate of the signal_to_noise_ratio level in the background sky (electrons per second).
        """
        data = imaging.numpy_array_from_fits(path + filename, hdu)
        return SimulateImage(data, pixel_scale, psf, exposure_time, background_noise, noise_seed)

    def simulate_optics(self):
        """
        Blur simulated image with a psf.
        """
        self.data = self.psf.convolve_with_image(self.data)

    def simulate_poisson_noise(self):
        """Simulate Poisson signal_to_noise_ratio in image"""
        self.poisson_noise_map = generate_poisson_noise_map(self.data, self.exposure_time.data, self.noise_seed)

        self.data += self.poisson_noise_map

    def simulate_background_noise(self):
        """Simulate the background signal_to_noise_ratio"""
        self.background_noise_map = generate_background_noise_map(self.pixel_dimensions, self.background_noise.data,
                                                                  self.noise_seed)

        self.data += self.background_noise_map

    def estimate_noise_in_simulated_image(self):
        """Estimate the signal_to_noise_ratio in the simulated image, using the exposure time and background signal_to_noise_ratio"""
        self.noise = imaging.estimate_noise_from_image(self.data, self.exposure_time.data, self.background_noise.data)

    def estimate_signal_to_noise_ratio_in_simulated_image(self):
        """Estimate the signal_to_noise_ratio in the simulated image, using the exposure time and background signal_to_noise_ratio"""
        self.signal_to_noise_ratio = np.divide(self.data, self.noise)

    def plot(self):
        pyplot.imshow(self.data)
        pyplot.show()