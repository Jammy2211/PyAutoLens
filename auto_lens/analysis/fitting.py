import numpy as np

from auto_lens.imaging import grids
from auto_lens.analysis import ray_tracing
from auto_lens.pixelization import pixelization
from auto_lens.pixelization import covariance_matrix


def fit_data_with_profiles(grid_data, kernel_convolver, tracer):
    """Fit the data using the ray_tracing model, where only light_profiles are used to represent the galaxy images.

    Parameters
    ----------
    grid_data : grids.GridDataCollection
        The collection of grid data-sets (image, noise, etc.)
    kernel_convolver : auto_lens.pixelization.frame_convolution.KernelConvolver
        The 2D Point Spread Function (PSF).
    tracer : ray_tracing.Tracer
        The ray-tracing configuration of the model galaxies and their profiles.
    """
    blurred_model_image = generate_blurred_light_profile_image(tracer, kernel_convolver)
    return compute_likelihood(grid_data.image, grid_data.noise, blurred_model_image)


def generate_blurred_light_profile_image(tracer, kernel_convolver):
    """For a given ray-tracing model, compute the light profile image(s) of its galaxies and blur them with the
    PSF.

    Parameters
    ----------
    tracer : ray_tracing.Tracer
        The ray-tracing configuration of the model galaxies and their profiles.
    kernel_convolver : auto_lens.pixelization.frame_convolution.KernelConvolver
        The 2D Point Spread Function (PSF).
    """
    image_light_profile = tracer.generate_image_of_galaxy_light_profiles()
    blurring_image_light_profile = tracer.generate_blurring_image_of_galaxy_light_profiles()
    return blur_image_including_blurring_region(image_light_profile, blurring_image_light_profile, kernel_convolver)


def blur_image_including_blurring_region(image, blurring_image, kernel_convolver):
    """For a given image and blurring region, convert them to 2D and blur with the PSF, then return as the 1D DataGrid.

    Parameters
    ----------
    image : ndarray
        The image data using the GridData 1D representation.
    blurring_image : ndarray
        The blurring region data, using the GridData 1D representation.
    kernel_convolver : auto_lens.pixelization.frame_convolution.KernelConvolver
        The 2D Point Spread Function (PSF).
    """
    return grids.GridData(kernel_convolver.convolve_array(image, blurring_image))


def fit_data_with_pixelization(grid_data, pix, kernel_convolver, tracer, mapper_cluster):
    """Fit the data using the ray_tracing model, where only pixelizations are used to represent the galaxy images.

    Parameters
    ----------
    grid_data : grids.GridDataCollection
        The collection of grid data-sets (image, noise, etc.)
    pix : pixelization.Pixelization
        The pixelization used to fit the data.
    kernel_convolver : auto_lens.pixelization.frame_convolution.KernelConvolver
        The 2D Point Spread Function (PSF).
    tracer : ray_tracing.Tracer
        The ray-tracing configuration of the model galaxies and their profiles.
    mapper_cluster : auto_lens.imaging.grids.GridMapperCluster
        The mapping between cluster-pixels and image / source pixels.
    """

    # TODO : If pixelization is in galaxy or tracer, we can compute the mapping matrix from it.

    mapping_matrix, regularization_matrix = pix.compute_mapping_and_regularization_matrix(
        source_coordinates=tracer.source_plane.grids.image, source_sub_coordinates=tracer.source_plane.grids.sub,
        mapper_cluster=mapper_cluster)

    # TODO : Build matrix convolution into frame_convolution?
    # Go over every column of mapping matrix, perform blurring
    blurred_mapping_matrix = np.zeros(mapping_matrix.shape)
    for i in range(mapping_matrix.shape[1]):
        blurred_mapping_matrix[:, i] = kernel_convolver.convolve_array(mapping_matrix[:, i])

    # TODO : Use fast routines once ready.

    cov_matrix = covariance_matrix.compute_covariance_matrix_exact(blurred_mapping_matrix, grid_data.noise)
    d_vector = covariance_matrix.compute_d_vector_exact(blurred_mapping_matrix, grid_data.image, grid_data.noise)

    cov_reg_matrix = cov_matrix + regularization_matrix

    s_vector = np.linalg.solve(cov_reg_matrix, d_vector)

    model_image = pixelization_model_image_from_s_vector(s_vector, blurred_mapping_matrix)

    return compute_bayesian_evidence(grid_data.image, grid_data.noise, model_image, s_vector, cov_reg_matrix,
                                     regularization_matrix)


# TODO : Put this here for now as it uses the blurred mapping matrix (and thus the PSF). Move to pixelization?
def pixelization_model_image_from_s_vector(s_vector, blurred_mapping_matrix):
    """ Map the reconstructioon source s_vecotr back to the image-plane to compute the pixelization's model-image.
    """
    pixelization_model_image = np.zeros(blurred_mapping_matrix.shape[0])
    for i in range(blurred_mapping_matrix.shape[0]):
        for j in range(len(s_vector)):
            pixelization_model_image[i] += s_vector[j] * blurred_mapping_matrix[i, j]

    return pixelization_model_image


def compute_likelihood(image, noise, model_image):
    """Compute the likelihood of a model image's fit to the data, by taking the difference between the observed \
    image and model ray-tracing image. The likelihood consists of two terms:

    Chi-squared term - The residuals (model - data) of every pixel divided by the noise in each pixel, all squared.
    [Chi_Squared_Term] = sum(([Residuals] / [Noise]) ** 2.0)

    The overall normalization of the noise is also included, by summing the log noise value in each pixel:
    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    These are summed and multiplied by -0.5 to give the likelihood:

    Likelihood = -0.5*[Chi_Squared_Term + Noise_Term]

    Parameters
    ----------
    image : grids.GridData
        The image data.
    noise : grids.GridData
        The noise in each pixel.
    model_image : grids.GridData
        The model image of the data.
    """
    return -0.5 * (compute_chi_sq_term(image, noise, model_image) + compute_noise_term(noise))


def compute_bayesian_evidence(image, noise, model_image, s_vector, cov_reg_matrix, regularization_matrix):
    return -0.5 * (compute_chi_sq_term(image, noise, model_image)
                   + compute_regularization_term(s_vector, regularization_matrix)
                   + compute_log_determinant_of_matrix_cholesky(cov_reg_matrix)
                   - compute_log_determinant_of_matrix_cholesky(regularization_matrix)
                   + compute_noise_term(noise))


def compute_chi_sq_term(image, noise, model_image):
    """Compute the chi-squared of a model image's fit to the data, by taking the difference between the observed \
    image and model ray-tracing image, dividing by the noise in each pixel and squaring:

    [Chi_Squared] = sum(([Data - Model] / [Noise]) ** 2.0)

    Parameters
    ----------
    image : grids.GridData
        The image data.
    noise : grids.GridData
        The noise in each pixel.
    model_image : grids.GridData
        The model image of the data.
    """
    return np.sum(((image - model_image) / noise) ** 2.0)


def compute_noise_term(noise):
    """Compute the noise normalization term of an image, which is computed by summing the noise in every pixel:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise : grids.GridData
        The noise in each pixel.
    """
    return np.sum(np.log(2 * np.pi * noise ** 2.0))


# TODO : Speed this up using source_pixel neighbors list to skip sparsity (see regularization matrix calculation)
def compute_regularization_term(s_vector, regularizaton_matrix):
    """ Compute the regularization term of a pixelization's Bayesian likelihood function. This represents the sum \
     of the difference in fluxes between every pair of neighboring source-pixels. This is computed as:

     s_T * H * s = s_vector.T * regularization_matrix * s_vector

     The term is referred to as 'G_l' in Warren & Dye 2003, Nightingale & Dye 2015.

     The above works include the regularization coefficient (lambda) in this calculation. In PyAutoLens, this is  \
     already in the regularization matrix and thus included in the matrix multiplication.

     Parameters
     -----------
     s_vector : ndarray
        1D vector of the reconstructed source fluxes.
     regularization_matrix : ndarray
        The matrix encoding which source-pixel pairs are regularized with one another.
     """
    return np.matmul(s_vector.T, np.matmul(regularizaton_matrix, s_vector))


def compute_log_determinant_of_matrix(matrix):
    """There are two terms in the pixelization's Bayesian likelihood funcition which require the log determinant of \
    a matrix. These are (Nightingale & Dye 2015, Nightingale, Dye and Massey 2018):

    ln[det(F + H)] = ln[det(cov_reg_matrix)]
    ln[det(H)]     = ln[det(regularization_matrix)]

    The regularization matrix is not necessarily positive-definite, thus its log determinant must be computed directly.

    Parameters
    -----------
    matrix : ndarray
        The positive-definite matrix the log determinant is computed for.
    """
    return np.sum(np.log(np.linalg.det(matrix)))


# TODO : Cholesky decomposition can also use source pixel neighbors list to skip sparsity.
def compute_log_determinant_of_matrix_cholesky(matrix):
    """There are two terms in the pixelization's Bayesian likelihood funcition which require the log determinant of \
    a matrix. These are (Nightingale & Dye 2015, Nightingale, Dye and Massey 2018):

    ln[det(F + H)] = ln[det(cov_reg_matrix)]
    ln[det(H)]     = ln[det(regularization_matrix)]

    The cov_reg_matrix is positive-definite, which means its log_determinant can be computed efficiently \
    (compared to using np.det) by using a Cholesky decomposition first and summing the log of each diagonal term.

    Parameters
    -----------
    matrix : ndarray
        The positive-definite matrix the log determinant is computed for.
    """
    return 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(matrix))))


def fit_data_with_pixelization_and_profiles(grid_data_collection, pixelization, kernel_convolver, tracer,
                                            mapper_cluster):
    return -1
    # TODO: implement me
