import numpy as np


def blurred_image_1d_from_1d_unblurred_and_blurring_images(unblurred_image_1d, blurring_image_1d, convolver):
    """For a 1D masked image and 1D blurring image (the regions outside the mask whose light blurs \
    into the mask after PSF convolution), use both to compute the blurred image within the mask via PSF convolution.

    The convolution uses each image's convolver (*See ccd.convolution*).

    Parameters
    ----------
    unblurred_image_1d : ndarray
        The 1D masked datas which is blurred.
    blurring_image_1d : ndarray
        The 1D masked blurring image which is used for blurring.
    convolver : ccd.convolution.ConvolverImage
        The image-convolver which performs the convolution in 1D.
    """
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


def blurred_image_of_planes_from_1d_images_and_convolver(total_planes, image_plane_image_1d_of_planes,
                                                         image_plane_blurring_image_1d_of_planes, convolver,
                                                         map_to_scaled_array):
    """For a tracer, extract the image-plane image of every plane and blur it with the PSF.

    If none of the galaxies in a plane have a light profie or pixelization (and thus don't have an image) a *None* \
    is used.

    Parameters
    ----------
    total_planes : int
        The total number of planes that blurred images are computed for.
    image_plane_image_1d_of_planes : [ndarray]
        For every plane, the 1D image-plane image.
    image_plane_blurring_image_1d_of_planes : [ndarray]
        For every plane, the 1D image-plane blurring image.
    convolver : hyper.ccd.convolution.ConvolverImage
        Class which performs the PSF convolution of a masked image in 1D.
    map_to_scaled_array : func
        A function which maps a masked image from 1D to 2D.
    """

    blurred_image_of_planes = []

    for plane_index in range(total_planes):

        # If all entries are zero, there was no light profile / pixeization
        if np.count_nonzero(image_plane_image_1d_of_planes[plane_index]) > 0:

            blurred_image_1d_of_plane = blurred_image_1d_from_1d_unblurred_and_blurring_images(
                unblurred_image_1d=image_plane_image_1d_of_planes[plane_index],
                blurring_image_1d=image_plane_blurring_image_1d_of_planes[plane_index],
                convolver=convolver)

            blurred_image_of_plane = map_to_scaled_array(array_1d=blurred_image_1d_of_plane)

            blurred_image_of_planes.append(blurred_image_of_plane)

        else:

            blurred_image_of_planes.append(None)

    return blurred_image_of_planes


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


def contribution_maps_1d_from_hyper_images_and_galaxies(hyper_model_image_1d, hyper_galaxy_images_1d, hyper_galaxies,
                                                        hyper_minimum_values):
    """For a fitting hyper_galaxy_image, hyper_galaxy model image, list of hyper galaxies images and model hyper galaxies, compute
    their contribution maps, which are used to compute a scaled-noise_map map. All quantities are masked 1D arrays.

    The reason this is separate from the *contributions_from_fitting_hyper_images_and_hyper_galaxies* function is that
    each hyper_galaxy image has a list of hyper galaxies images and associated hyper galaxies (one for each galaxy). Thus,
    this function breaks down the calculation of each 1D masked contribution map and returns them in the same datas
    structure (2 lists with indexes [image_index][contribution_map_index].

    Parameters
    ----------
    hyper_model_image_1d : ndarray
        The best-fit model image to the datas (e.g. from a previous analysis phase).
    hyper_galaxy_images_1d : [ndarray]
        The best-fit model image of each hyper galaxy to the datas (e.g. from a previous analysis phase).
    hyper_galaxies : [galaxy.Galaxy]
        The hyper galaxies which represent the model components used to scale the noise_map, which correspond to
        individual galaxies in the image.
    hyper_minimum_values : [float]
        The minimum value of each hyper_galaxy-image contribution map, which ensure zero's don't impact the scaled noise-map.
    """
    # noinspection PyArgumentList
    return list(map(lambda hyper_galaxy, hyper_galaxy_image_1d, hyper_minimum_value:
                    hyper_galaxy.contributions_from_model_image_and_galaxy_image(model_image=hyper_model_image_1d,
                                                                                 galaxy_image=hyper_galaxy_image_1d,
                                                                                 minimum_value=hyper_minimum_value),
                    hyper_galaxies, hyper_galaxy_images_1d, hyper_minimum_values))


def scaled_noise_map_from_hyper_galaxies_and_contribution_maps(contribution_maps, hyper_galaxies, noise_map):
    """For a contribution map and noise-map, use the model hyper galaxies to compute a scaled noise-map.

    Parameters
    -----------
    contribution_maps : ndarray
        The image's list of 1D masked contribution maps (e.g. one for each hyper galaxy)
    hyper_galaxies : [galaxy.Galaxy]
        The hyper galaxies which represent the model components used to scale the noise_map, which correspond to
        individual galaxies in the image.
    noise_map : ccd.NoiseMap or ndarray
        An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
        second.
    """
    scaled_noise_maps = list(map(lambda hyper_galaxy, contribution_map:
                                 hyper_galaxy.hyper_noise_from_contributions(noise_map=noise_map,
                                                                             contributions=contribution_map),
                                 hyper_galaxies, contribution_maps))
    return noise_map + sum(scaled_noise_maps)
