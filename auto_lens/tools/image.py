import numpy as np
import scipy.signal

class KernelException(Exception):
    pass

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
    """ Calculate the residuals between an observed image (data) and a model_mapper of that image (model_mapper),
    by subtracting the data from the model_mapper, residuals = data - model_mapper.

    Parameters
    ----------
    observed_image : ndarray
        Two dimensional array containing observed image data
    model_image : nddarray
        Two dimensional array containing model_mapper of the observed image
    """
    residuals = np.subtract(observed_image, model_image)
    return residuals

def compute_chi_sq_image(observed_data, model_image, noise_map):
    """ Calculate the chi-squared image that results from divding the residuals between an observed image (data) and \
    a model_mapper of that image (model_mapper) by the noise in the image (noise), chi_sq = (data - model_mapper)/noise = residuals/noise.

    Parameters
    ----------
    observed_data : ndarray
        Two dimensional array containing observed image data
    model_image : nddarray
        Two dimensional array containing model_mapper of the observed image
    noise_map : nddarray
        Two dimensional array containing noise in the observed image

    Examples
    --------
    chi_sq_image = CTI_Tools.compute_chi_sq_image(data, model_mapper, noise)

    """

    residuals = compute_residuals(observed_data, model_image)
    resi_noise = np.divide(residuals, noise_map)  # residualss / noise
    chi_sq_image = np.square(resi_noise)  # chi_sq = ( residuals / noise ) ^2
    return chi_sq_image

def compute_likelihood(observed_image, model_image, noise_map, mask):
    """Computes the likelihood of a charge injection line image, by taking the difference between an observed \
    image and model_mapper image. This gives the residuals, which are divided by the variance of each pixel and squared \
    to give their chi sq values. Finally, these are summed and multiplied by -0.5 to give the likelihood.

    This is performed for all pixels within the *ChargeInjectImage* instance's 'LIKELIHOOD' region(s). \
    Typically, this region will correspond to the whole image, but gives the user the option to remove \
    certains sections.

    Parameters
    ----------
    data : ndarray
        Two dimensional array containing observed image data
    model_mapper : nddarray
        Two dimensional array containing model_mapper of the observed image
    noise : nddarray
        Two dimensional array containing noise in the observed image

    Returns
    ----------
    likelihood : float
        The likelihood computed from comparing this *ChargeInjectImage* instance's observed and model_mapper images.

    Examples
    --------
    likelihood_total += self.image[no].compute_overall_likelihood()
    """
    masked_image = np.ma.masked_array(observed_image, mask)
    likelihood = -0.5 * np.sum(np.square(np.divide(np.subtract(masked_image, model_image), noise_map)))
    return likelihood

def convert_image_to_counts(image, exposure_time_map):
    """For an image (in electrons per second) and exposure time map, return the image in counts
    Parameters
    ----------
    image : ndarray
        The image from which the Poisson noise map is estimated.
    exposure_time_map : ndarray
        The exposure time in each image pixel."""
    return np.multiply(image, exposure_time_map)

def convolve_image_with_kernel(image, kernel):
    """
    Convolve a two-dimensional image with a two-dimensional kernel (e.g. a PSF)

    NOTE1 : The PSF kernel must be size odd x odd to avoid ambuiguities with convolution offsets.

    NOTE2 : SciPy has multiple 'mode' options for the size of the output array (e.g. does it include zero padding). We \
    require the output array to be the same size as the input image.

    Parameters
    ----------
    image : ndarray
        The image which is to be convolved with the kernel
    kernel : ndarray
        The kernel used for the convolution.
    """
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise KernelException("Kernel must be odd")

    return scipy.signal.convolve2d(image, kernel, mode='same')

def estimate_poisson_noise_from_image(image, exposure_time_map):
    """Estimate a Poisson two-dimensional noise map from an input image.

    Parameters
    ----------
    image : ndarray
        The image from which the Poisson noise map is estimated.
    exposure_time_map : ndarray
        The exposure time in each image pixel.
    """
    image_counts = convert_image_to_counts(image, exposure_time_map)
    return np.divide(np.sqrt(image_counts), exposure_time_map)

def estimate_noise_from_image_and_background(image, exposure_time_map, sigma_background, exposure_time_mean):
    """Estimate a Poisson two-dimensional noise map from an input image.

    Parameters
    ----------
    image : ndarray
        The image from which the Poisson noise map is estimated.
    exposure_time_map : ndarray
        The exposure time in each image pixel.
    sigma_background : float
        The estimate standard deviation of the 1D Gaussian level of noise in the background.
    exposure_time_mean : float
        The mean exposure time of the image and therefore background.
    """
    sigma_counts = np.multiply(sigma_background, exposure_time_mean)
    image_counts = convert_image_to_counts(image, exposure_time_map)
    return np.divide(np.sqrt(np.square(sigma_counts) + image_counts), exposure_time_map)

def generate_gaussian_noise_map(dimensions, mean, sigma, seed=-1):
    """Generate a Gaussian two-dimensional noise map.

    Parameters
    ----------
    dimensions : (int, int)
        The (x,y) dimensions of the generated Gaussian noise map.
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
    """Generate a Poisson two-dimensional noise map from an input image.

    NOTE : np.random.poisson returns a new image subjected to Poisson noise. This is subtracted from image_counts to \
    generate the Poisson noise map.

    Parameters
    ----------
    image : ndarray
        The image used to generate the Poisson noise map.
    exposure_time_map : ndarray
        The exposure time in each image pixel.
    seed : int
        The seed of the random number generator, used for the random noise maps.
    """
    setup_random_seed(seed)
    image_counts = convert_image_to_counts(image, exposure_time_map)
    return image - np.divide(np.random.poisson(image_counts, image.shape), exposure_time_map)