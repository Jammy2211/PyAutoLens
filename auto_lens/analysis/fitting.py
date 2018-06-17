import numpy as np

from auto_lens.imaging import grids
from auto_lens.analysis import ray_tracing
from auto_lens.pixelization import pixelization
from auto_lens.pixelization import covariance_matrix

def likelihood_for_image_tracer_pixelization_and_instrumentation(image, tracer, pixelization, instrumentation):
    # TODO: This function should take a tracer and return a likelihood. The ModelAnalysis class in the pipeline module
    # TODO: will construct the tracer using a non linear optimiser and priors.
    return 1.0

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
    return -0.5 * (np.sum(((image - model_image) / noise) ** 2.0 + np.log(2 * np.pi * noise ** 2.0)))


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
        blurred_mapping_matrix[:,i] = kernel_convolver.convolve_array(mapping_matrix[:,i])

    # TODO : Use fast routines once ready.

    cov_matrix = covariance_matrix.compute_covariance_matrix_exact(blurred_mapping_matrix, grid_data.noise)
    d_vector = covariance_matrix.compute_d_vector_exact(blurred_mapping_matrix, grid_data.image, grid_data.noise)

    cov_reg_matrix = cov_matrix + regularization_matrix

    s_vector = np.linalg.solve(cov_reg_matrix, d_vector)

    # TODO : The likelihood of a pixelization has additional terms (determinants of cov and reg matrices), so need to
    # TODO : Write routine which computes them.

    model_image = pixelization_model_image_from_s_vector(s_vector, blurred_mapping_matrix)

    # likelihood = compute_pixelization_likelihood(grid_data.image, grid_data.noise, model_image, s_vector,
    # cov_reg_matrix, regularization_matrix)

    return model_image

# TODO : Put this here for now as it uses the blurred mapping matrix (and thus the PSF). Move to pixelization?
def pixelization_model_image_from_s_vector(s_vector, blurred_mapping_matrix):
    """ Map the reconstructioon source s_vecotr back to the image-plane to compute the pixelization's model-image.
    """
    pixelization_model_image  = np.zeros(blurred_mapping_matrix.shape[0])
    for i in range(blurred_mapping_matrix.shape[0]):
        for j in range(len(s_vector)):
            pixelization_model_image[i] += s_vector[j] * blurred_mapping_matrix[i,j]

    return pixelization_model_image