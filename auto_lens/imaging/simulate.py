import numpy as np
from matplotlib import pyplot

from auto_lens.imaging import imaging

def generate_poisson_noise_map(image, exposure_time, seed=-1):
    """Generate a Poisson two-dimensional noise map from an input image. This includes a conversion of the image \
    from electrons per second to counts (and back).

    NOTE : np.random.poisson returns a new image subjected to Poisson noise. This is subtracted from the image to \
    generate the Poisson noise map.

    Parameters
    ----------
    image : ndarray
        The image in electrons per second, used to generate the Poisson noise map.
    exposure_time : float or ndarray
        The exposure time in each image pixel, used to convert the image from electrons per second to counts.
    seed : int
        The seed of the random number generator, used for the random noise maps.
    """
    setup_random_seed(seed)
    image_counts = imaging.convert_array_to_counts(image, exposure_time)
    return image - np.divide(np.random.poisson(image_counts, image.shape), exposure_time)

def generate_background_noise_map(dimensions, background_noise, seed=-1):
    """Generate a Gaussian background two-dimensional noise map.

    Parameters
    ----------
    dimensions : (int, int)
        The (x,y) pixel_dimensions of the generated Gaussian noise map.
    background_noise : float or ndarray
        Standard deviation of the 1D Gaussian that each noise value is drawn from
    seed : int
        The seed of the random number generator, used for the random noise maps.
    """
    setup_random_seed(seed)
    return np.random.normal(0.0, background_noise, dimensions)


class SimulateImage(imaging.Data):

    def __init__(self, data, pixel_scale, sky_level=0.0, psf=None, exposure_time_map=None, noise_seed=-1):
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
        exposure_time_map : ndarray
            The exposure time in each image pixel, used to convert the image from electrons per second to counts.
        noise_seed : int
            The seed of the random number generator, used for the random noise maps.
        """

        self.data_original = data

        super(SimulateImage,self).__init__(data, pixel_scale)

        self.psf = psf
        self.exposure_time_map = exposure_time_map
        self.sky_level = sky_level
        self.sky_noise_counts = np.sqrt(sky_level*np.max(self.exposure_time_map.data))
        self.noise_seed = noise_seed

        if self.psf is not None:
            self.simulate_optics()

        if self.exposure_time_map is not None:
            self.simulate_poisson_noise()

        self.noise = imaging.estimate_noise_from_image_and_background(self.data, self.exposure_time_map.data,
                                                                      self.sky_noise_counts)

    @classmethod
    def from_fits(cls, path, filename, hdu, pixel_scale, sky_level=0.0, psf=None, exposure_time_map=None,
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
            An estimate of the noise level in the background sky (electrons per second).
        """
        data = imaging.numpy_array_from_fits(path + filename, hdu)
        return SimulateImage(data, pixel_scale, sky_level, psf, exposure_time_map, noise_seed)

    def simulate_optics(self):
        """
        Blur simulated image with a psf.
        """
        self.data = self.psf.convolve_with_image(self.data)

    def simulate_poisson_noise(self):
        """Simulate Poisson noise in image"""
        self.poisson_noise_map = generate_poisson_noise_map(self.data, self.exposure_time_map.data, self.sky_level,
                                                            self.noise_seed)
        self.data = np.add(self.data, self.poisson_noise_map)

    def plot(self):
        pyplot.imshow(self.data)
        pyplot.show()


def setup_random_seed(seed):
    """Setup the random seed. If the input seed is -1, the code will use a random seed for every run. If it is positive,
    that seed is used for all runs, thereby giving reproducible results

    Parameters
    ----------
    seed : int
        The seed of the random number generator, used for the random noise maps.
    """
    if seed == -1:
        seed = np.random.randint(0, 1e9)  # Use one seed, so all regions have identical column non-uniformity.
    np.random.seed(seed)