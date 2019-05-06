import numpy as np


def blurred_image_1d_from_1d_unblurred_and_blurring_images(unblurred_image_1d, blurring_image_1d, convolver):

    return convolver.convolve_image(image_array=unblurred_image_1d, blurring_array=blurring_image_1d)


def likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(chi_squared,
                                                                                                regularization_term,
                                                                                                noise_normalization):
    """Compute the likelihood of an inversion's fit to the datas, including a regularization term which \
    comes from an inversion:

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared : float
        The chi-squared term of the inversion's fit to the observed datas.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    noise_normalization : float
        The normalization noise_map-term for the observed datas's noise-map.
    """
    return -0.5 * (chi_squared + regularization_term + noise_normalization)


def evidence_from_inversion_terms(chi_squared, regularization_term, log_curvature_regularization_term,
                                  log_regularization_term, noise_normalization):
    """Compute the evidence of an inversion's fit to the datas, where the evidence includes a number of \
    terms which quantify the complexity of an inversion's reconstruction (see the *inversion* module):

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Log(Covariance_Regularization_Term) -
                       Log(Regularization_Matrix_Term) + Noise_Term]

    Parameters
    ----------
    chi_squared : float
        The chi-squared term of the inversion's fit to the observed datas.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    log_curvature_regularization_term : float
        The log of the determinant of the sum of the curvature and regularization matrices.
    log_regularization_term : float
        The log of the determinant o the regularization matrix.
    noise_normalization : float
        The normalization noise_map-term for the observed datas's noise-map.
    """
    return -0.5 * (chi_squared + regularization_term + log_curvature_regularization_term - log_regularization_term
                   + noise_normalization)


def unmasked_blurred_image_of_planes_from_padded_grid_stack_and_psf(planes, padded_grid_stack, psf):
    """For lens data, compute the unmasked blurred image of every unmasked unblurred image of each plane. To do this, \
    this function iterates over all planes to extract their unmasked unblurred images.

    If a galaxy in a plane has a pixelization, the unmasked image is returned as None, as as the inversion's model \
    image cannot be mapped to an unmasked version.

    This relies on using the lens data's padded-grid, which is a grid of (y,x) coordinates which extends over the \
    entire image as opposed to just the masked region.

    This returns a list, where each list index corresponds to [plane_index].

    Parameters
    ----------
    planes : [plane.Plane]
        The list of planes the unmasked blurred images are computed using.
    padded_grid_stack : grids.GridStack
        A padded-grid_stack, whose padded grid is used for PSF convolution.
    psf : ccd.PSF
        The PSF of the image used for convolution.
    """
    unmasked_blurred_image_of_planes = []

    for plane in planes:

        if plane.has_pixelization:
            unmasked_blurred_image_of_plane = None
        else:
            unmasked_blurred_image_of_plane = \
                padded_grid_stack.unmasked_blurred_image_from_psf_and_unmasked_image(

                    psf=psf, unmasked_image_1d=plane.image_plane_image_1d)

        unmasked_blurred_image_of_planes.append(unmasked_blurred_image_of_plane)

    return unmasked_blurred_image_of_planes


def unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stack_and_psf(planes, padded_grid_stack, psf):
    """For lens data, compute the unmasked blurred image of every unmasked unblurred image of every galaxy in each \
    plane. To do this, this function iterates over all planes and then galaxies to extract their unmasked unblurred \
    images.

    If a galaxy in a plane has a pixelization, the unmasked image of that galaxy in the plane is returned as None \
    as as the inversion's model image cannot be mapped to an unmasked version.

    This relies on using the lens data's padded-grid, which is a grid of (y,x) coordinates which extends over the \
    entire image as opposed to just the masked region.

    This returns a list of lists, where each list index corresponds to [plane_index][galaxy_index].

    Parameters
    ----------
    planes : [plane.Plane]
        The list of planes the unmasked blurred images are computed using.
    padded_grid_stack : grids.GridStack
        A padded-grid_stack, whose padded grid is used for PSF convolution.
    psf : ccd.PSF
        The PSF of the image used for convolution.
    """
    return [plane.unmasked_blurred_image_of_galaxies_from_psf(padded_grid_stack, psf) for plane in planes]