import numpy as np

from auto_lens.imaging import imaging

def generate_poisson_noise_map(image, exposure_time_map, seed=-1):
    """Generate a Poisson two-dimensional noise map from an input image. This includes a conversion of the image \
    from electrons per second to counts (and back).

    NOTE : np.random.poisson returns a new image subjected to Poisson noise. This is subtracted from the image to \
    generate the Poisson noise map.

    Parameters
    ----------
    image : ndarray
        The image in electrons per second, used to generate the Poisson noise map.
    exposure_time_map : ndarray
        The exposure time in each image pixel, used to convert the image from electrons per second to counts.
    seed : int
        The seed of the random number generator, used for the random noise maps.
    """
    setup_random_seed(seed)
    image_counts = imaging.convert_array_to_counts(image, exposure_time_map)
    return image - np.divide(np.random.poisson(image_counts, image.shape), exposure_time_map)

def generate_gaussian_noise_map(dimensions, mean, sigma, seed=-1):
    """Generate a Gaussian two-dimensional noise map.

    Parameters
    ----------
    dimensions : (int, int)
        The (x,y) pixel_dimensions of the generated Gaussian noise map.
    sigma : mean
        Mean value of the 1D Gaussian that each noise value is drawn from
    sigma : float
        Standard deviation of the 1D Gaussian that each noise value is drawn from
    seed : int
        The seed of the random number generator, used for the random noise maps.
    """
    setup_random_seed(seed)
    return np.random.normal(mean, sigma, dimensions)


class SimulateImage(imaging.Data):

    def __init__(self, data, pixel_scale, background_sky_level=None, psf=None, exposure_time_map=None, noise_seed=-1):
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
        self.background_sky_level = background_sky_level
        self.exposure_time_map = exposure_time_map
        self.noise_seed = noise_seed

        if self.background_sky_level is not None:
            self.simulate_background_sky()

        if self.psf is not None:
            self.simulate_optics()

        if self.exposure_time_map is not None:
            self.simulate_poisson_noise()

    def simulate_background_sky(self):
        """Simulate a constant background sky by adding it to the image"""
        self.background_sky_map = np.ones((self.pixel_dimensions))*self.background_sky_level
        self.data = np.add(self.data, self.background_sky_map)

    def simulate_optics(self):
        """
        Blur simulated image with a psf.
        """
        self.data = self.psf.convolve_with_image(self.data)

    def simulate_poisson_noise(self):
        """Simulate Poisson noise in image"""
        self.poisson_noise_map = generate_poisson_noise_map(self.data, self.exposure_time_map, self.noise_seed)
        self.data = np.add(self.data, self.poisson_noise_map)



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