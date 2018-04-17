import numpy as np

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

def compute_residuals(observed_image, model_image):
    """ Calculate the residuals between an observed image and a model of that image, \
    by subtracting the model from the data.

    Residuals = (Data - Model).

    Parameters
    ----------
    observed_image : ndarray
        Two dimensional array containing observed image data
    model_image : nddarray
        Two dimensional array containing model_mapper of the observed image
    """
    return np.subtract(observed_image, model_image)

def compute_variances_from_noise(noise):
    """The variances are the noise (standard deviations) squared."""
    return np.square(noise)

def compute_chi_sq_image(observed_image, model_image, noise):
    """ Calculate the chi-squared image of an observed image, model image and noise map.

    The noise map gives the standard deviation of each image pixel, and is converted to variances (by squaring) in the
     chi squared calculation.

     Chi_Sqs = ((Data - Model)/Noise)**2.0 = (Residuals/Noise)**2.0 = (Residuals**2.0/Variances).

    Parameters
    ----------
    observed_image : ndarray
        Two dimensional array containing observed image data
    model_image : nddarray
        Two dimensional array containing model_mapper of the observed image
    noise : nddarray
        Two dimensional array containing the noise (standard deviation) in each observed image pixel.
    """
    residuals = compute_residuals(observed_image, model_image)
    variances = compute_variances_from_noise(noise)
    return np.divide(np.square(residuals), variances)

def compute_likelihood(observed_image, model_image, noise):
    """ Calculate the likelihood image of an observed image, model image and noise map. The likelihood is defined as
    the sum of chi squared values multiplied by 0.5.

    The noise map gives the standard deviation of each image pixel, and is converted to variances (by squaring) in the
    chi squared calculation.

    NOTE1 : For efficiency, this routine uses only numpy routines in one big calculation, thus it is 'unpythonic'.

    Parameters
    ----------
    observed_data : ndarray
        Two dimensional array containing the observed image data
    model_image : nddarray
        Two dimensional array containing the model of the observed image
    noise : nddarray
        Two dimensional array containing the noise (standard deviation) in each observed image pixel
    """
    return -0.5 * np.sum(np.square(np.divide(np.subtract(observed_image, model_image), noise)))

def convert_array_to_counts(array, exposure_time_array):
    """For an array (in electrons per second) and exposure time array, return an array in units counts.

    Parameters
    ----------
    array : ndarray
        The image from which the Poisson noise map is estimated.
    exposure_time_array : ndarray
        The exposure time in each image pixel."""
    return np.multiply(array, exposure_time_array)

def convert_array_to_electrons_per_second(array, exposure_time_array):
    """For an array (in counts) and exposure time array, convert the array to units electrons per second
    Parameters
    ----------
    array : ndarray
        The image from which the Poisson noise map is estimated.
    exposure_time_array : ndarray
        The exposure time in each image pixel.
    """
    return np.divide(array, exposure_time_array)

def estimate_noise_std_in_quadrature(sigma_std_counts, image_counts):
    return np.sqrt(np.square(sigma_std_counts) + image_counts)

def estimate_poisson_noise_std_from_image(image, exposure_time_map):
    """Estimate a Poisson two-dimensional noise map from an input image.

    Parameters
    ----------
    image : ndarray
        The image in electrons per second, used to estimate the Poisson noise map.
    exposure_time_map : ndarray
        The exposure time in each image pixel, used to convert the image from electrons per second to counts.
    """
    image_counts = convert_array_to_counts(image, exposure_time_map)
    noise_std_counts = estimate_noise_std_in_quadrature(sigma_std_counts=0.0, image_counts=image_counts)
    return convert_array_to_electrons_per_second(noise_std_counts, exposure_time_map)

def estimate_noise_from_image_and_background(image, exposure_time_map, sigma_background, exposure_time_mean):
    """Estimate a Poisson two-dimensional noise map from an input image.

    Parameters
    ----------
    image : ndarray
        The image in electrons per second, used to estimate the Poisson noise map.
    exposure_time_map : ndarray
        The exposure time in each image pixel, used to convert the image from electrons per second to counts.
    sigma_background : float
        The estimate standard deviation of the 1D Gaussian level of noise in the background.
    exposure_time_mean : float
        The mean exposure time of the image and therefore background.
    """
    sigma_counts = np.multiply(sigma_background, exposure_time_mean)
    image_counts = convert_array_to_counts(image, exposure_time_map)
    noise_counts = estimate_noise_std_in_quadrature(sigma_counts, image_counts)
    return convert_array_to_electrons_per_second(noise_counts, exposure_time_map)

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
    image_counts = convert_array_to_counts(image, exposure_time_map)
    return image - np.divide(np.random.poisson(image_counts, image.shape), exposure_time_map)