import numpy as np
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing
from autolens.inversion import inversions
from autolens import exc

minimum_value_profile = 0.1


class AbstractFitter(object):

    def __init__(self, lensing_image, tracer):

        self._model_image = None
        self._residuals = None
        self._chi_squareds = None
        self._noise = None
        self._noise_term = None

        self.lensing_image = lensing_image
        self.tracer = tracer
        self._image = lensing_image[:]
        self.map_to_2d = lensing_image.grids.image.map_to_2d
        self.convolve_image = lensing_image.convolver_image.convolve_image

    @property
    def model_image(self):
        return self.map_to_2d(self._model_image)

    @property
    def residuals(self):
        return self.map_to_2d(self._residuals)

    @property
    def chi_squareds(self):
        return self.map_to_2d(self._chi_squareds)

    @property
    def chi_squared_term(self):
        return chi_squared_term_from_chi_squareds(self._chi_squareds)

    @property
    def noise_term(self):
        return noise_term_from_noise(self._noise)

    @property
    def likelihood(self):
        """
        Fit the data_vector using the ray_tracing model, where only light_profiles are used to represent the galaxy
        images.
        """
        return likelihood_from_chi_squared_and_noise_terms(self.chi_squared_term, self._noise_term)

    @property
    def noise(self):
        return self.map_to_2d(self._noise)

    @property
    def xticks(self):
        return self.lensing_image.image.xticks

    @property
    def yticks(self):
        return self.lensing_image.image.yticks

class AbstractHyperFitter(object):

    def __init__(self, lensing_image, hyper_model_image=None, hyper_galaxy_images=None,
                 hyper_minimum_values=None):
        self.map_to_2d = lensing_image.grids.image.map_to_2d
        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_images = hyper_galaxy_images
        self.hyper_minimum_values = hyper_minimum_values
        self._contributions = contributions_from_hyper_images_and_galaxies(hyper_model_image,
                              hyper_galaxy_images, self.tracer.hyper_galaxies, hyper_minimum_values)

    @property
    def contributions(self):
        return list(map(lambda contributions : self.map_to_2d(contributions), self._contributions))


class AbstractProfileFitter(object):

    def __init__(self, lensing_image):
        """
        Class to evaluate the fit between a model described by a tracer and an actual lensing_image.

        Parameters
        ----------
        lensing_image: li.LensingImage
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.TracerImageSourcePlanes
            An object describing the model
        """
        self._model_image = lensing_image.convolver_image.convolve_image(self.tracer._image_plane_image,
                                                                         self.tracer._image_plane_blurring_image)
        self._residuals = residuals_from_image_and_model(lensing_image[:], self._model_image)

    @property
    def model_images_of_planes(self):
        return list(map(lambda image_plane_image, image_plane_blurring_image :
                        self.map_to_2d(self.convolve_image(image_plane_image, image_plane_blurring_image)),
                        self.tracer._image_plane_images_of_planes, self.tracer._image_plane_blurring_images_of_planes))

    @property
    def model_images_of_galaxies(self):
        return list(map(lambda image_plane_image, image_plane_blurring_image :
                        self.map_to_2d(self.convolve_image(image_plane_image, image_plane_blurring_image)),
                        self.tracer._image_plane_images_of_galaxies, self.tracer._image_plane_blurring_images_of_galaxies))

    def images_of_planes(self, shape=(30, 30)):

        def map_to_2d(image, shape):

            image_2d = np.zeros(shape)

            for x in range(shape[0]):
                for y in range(shape[1]):

                    image_2d[y, x] = image[(x)*shape[0] + y]

            return image_2d

        return list(map(lambda plane_image : map_to_2d(plane_image, shape), self.tracer.plane_images_of_planes(shape)))


class AbstractPixelizationFitter(object):

    def __init__(self, lensing_image, noise):

        self.map_to_2d = lensing_image.grids.image.map_to_2d
        self._inversion = inversions.inversion_from_mapper_regularization_and_data(mapper=self.mapper,
                          regularization=self.regularization, image=lensing_image, noise_map=noise,
                          convolver=lensing_image.convolver_mapping_matrix)
        self._model_image = self._inversion.reconstructed_image
        self._residuals = residuals_from_image_and_model(lensing_image[:], self._model_image)

    @property
    def mapper(self):
        return self.tracer.mappers_of_planes[0]

    @property
    def regularization(self):
        return self.tracer.regularization_of_planes[0]

    @property
    def evidence(self):
        return evidence_from_reconstruction_terms(self.chi_squared_term,
                                                  self._inversion.regularization_term,
                                                  self._inversion.log_det_curvature_reg_matrix_term,
                                                  self._inversion.log_det_regularization_matrix_term,
                                                  self.noise_term)


class ProfileFitter(AbstractFitter, AbstractProfileFitter):

    def __init__(self, lensing_image, tracer):
        """
        Class to evaluate the fit between a model described by a tracer and an actual lensing_image.

        Parameters
        ----------
        lensing_image: li.LensingImage
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.TracerImageSourcePlanes
            An object describing the model
        """
        AbstractFitter.__init__(self, lensing_image, tracer)
        AbstractProfileFitter.__init__(self, lensing_image)
        self._noise = lensing_image.noise_map
        self._chi_squareds = chi_squareds_from_residuals_and_noise(self._residuals, self._noise)
        self._noise_term = noise_term_from_noise(self._noise)

    def pixelization_fitter_with_profile_subtracted_lensing_image(self, lensing_image):
        return PixelizationFitter(lensing_image[:] - self._model_image, self.tracer)


class HyperProfileFitter(AbstractFitter, AbstractHyperFitter, AbstractProfileFitter):

    def __init__(self, lensing_image, tracer, hyper_model_image, hyper_galaxy_images, hyper_minimum_values):
        """
        Class to evaluate the fit between a model described by a tracer and an actual lensing_image.

        Parameters
        ----------
        lensing_image: li.LensingImage
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.TracerImageSourcePlanes
            An object describing the model
        """
        AbstractFitter.__init__(self, lensing_image, tracer)
        AbstractHyperFitter.__init__(self, lensing_image, hyper_model_image, hyper_galaxy_images, hyper_minimum_values)
        AbstractProfileFitter.__init__(self, lensing_image)
        self._noise = scaled_noise_from_hyper_galaxies_and_contributions(self._contributions, self.tracer.hyper_galaxies,
                                                                         lensing_image.noise_map)
        self._noise_term = noise_term_from_noise(self._noise)
        self._chi_squareds = chi_squareds_from_residuals_and_noise(self._residuals, self._noise)

    def pixelization_fitter_with_profile_subtracted_lensing_image(self, lensing_image):
        return HyperPixelizationFitter(lensing_image - self._model_image, self.tracer,
                                       self.hyper_model_image, self.hyper_galaxy_images, self.hyper_minimum_values)


class PixelizationFitter(AbstractFitter, AbstractPixelizationFitter):

    def __init__(self, lensing_image, tracer):
        """
        Class to evaluate the fit between a model described by a tracer and an actual lensing_image.

        Parameters
        ----------
        lensing_image: li.LensingImage
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.TracerImageSourcePlanes
            An object describing the model
        """

        AbstractFitter.__init__(self, lensing_image, tracer)
        AbstractPixelizationFitter.__init__(self, lensing_image, lensing_image.noise_map)

        self._noise = lensing_image.noise_map
        self._chi_squareds = chi_squareds_from_residuals_and_noise(self._residuals, lensing_image.noise_map)
        self._noise_term = noise_term_from_noise(self._noise)


class HyperPixelizationFitter(AbstractFitter, AbstractHyperFitter, AbstractPixelizationFitter):

    def __init__(self, lensing_image, tracer, hyper_model_image, hyper_galaxy_images, hyper_minimum_values):
        """
        Class to evaluate the fit between a model described by a tracer and an actual lensing_image.

        Parameters
        ----------
        lensing_image: li.LensingImage
            An lensing_image that has been masked for efficiency
        tracer: ray_tracing.TracerImageSourcePlanes
            An object describing the model
        """

        AbstractFitter.__init__(self, lensing_image, tracer)
        AbstractHyperFitter.__init__(self, lensing_image, hyper_model_image, hyper_galaxy_images, hyper_minimum_values)
        self._noise = scaled_noise_from_hyper_galaxies_and_contributions(self._contributions, self.tracer.hyper_galaxies,
                                                                         lensing_image.noise_map)
        AbstractPixelizationFitter.__init__(self, lensing_image, self._noise)
        self._noise_term = noise_term_from_noise(self._noise)
        self._chi_squareds = chi_squareds_from_residuals_and_noise(self._residuals, self._noise)


class PositionFitter:

    def __init__(self, positions, noise):

        self.positions = positions
        self.noise = noise

    @property
    def chi_squareds(self):
        return np.square(np.divide(self.maximum_separations, self.noise))

    @property
    def likelihood(self):
        return -0.5*sum(self.chi_squareds)

    def maximum_separation_within_threshold(self, threshold):
        if max(self.maximum_separations) > threshold:
            return False
        else:
            return True

    @property
    def maximum_separations(self):
        return list(map(lambda positions : self.max_separation_of_grid(positions), self.positions))

    def max_separation_of_grid(self, grid):
        rdist_max = np.zeros((grid.shape[0]))
        for i in range(grid.shape[0]):
            xdists = np.square(np.subtract(grid[i,0], grid[:,0]))
            ydists = np.square(np.subtract(grid[i,1], grid[:,1]))
            rdist_max[i] = np.max(np.add(xdists, ydists))
        return np.max(np.sqrt(rdist_max))


def blur_image_including_blurring_region(image, blurring_image, convolver):
    """For a given lensing_image and blurring region, convert them to 2D and blur with the PSF, then return as
    the 1D DataGrid.

    Parameters
    ----------
    image : ndarray
        The lensing_image data_vector using the GridData 1D representation.
    blurring_image : ndarray
        The blurring region data_vector, using the GridData 1D representation.
    convolver : auto_lens.inversion.frame_convolution.KernelConvolver
        The 2D Point Spread Function (PSF).
    """
    return convolver.convolve_image(image, blurring_image)

def residuals_from_image_and_model(image, model):
    """Compute the residuals between an observed charge injection lensing_image and post-cti model lensing_image.

    Residuals = (Data - Model).

    Parameters
    -----------
    image : ChInj.CIImage
        The observed charge injection lensing_image data.
    model : np.ndarray
        The model lensing_image.
    """
    return np.subtract(image, model)

def chi_squareds_from_residuals_and_noise(residuals, noise):
    """Computes a chi-squared lensing_image, by calculating the squared residuals between an observed charge injection \
    images and post-cti hyper_model_image lensing_image and dividing by the variance (noises**2.0) in each pixel.

    Chi_Sq = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    This gives the residuals, which are divided by the variance of each pixel and squared to give their chi sq.

    Parameters
    -----------
    residuals
    noise : np.ndarray
        The noises in the lensing_image.
    """
    return np.square((np.divide(residuals, noise)))

def chi_squared_term_from_chi_squareds(chi_squareds):
    """Compute the chi-squared of a model lensing_image's fit to the data_vector, by taking the difference between the
    observed lensing_image and model ray-tracing lensing_image, dividing by the noise_map in each pixel and squaring:

    [Chi_Squared] = sum(([Data - Model] / [Noise]) ** 2.0)

    Parameters
    ----------
    chi_squareds
    """
    return np.sum(chi_squareds)

def noise_term_from_noise(noise):
    """Compute the noise_map normalization term of an lensing_image, which is computed by summing the noise_map in every pixel:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise : grids.GridData
        The noise_map in each pixel.
    """
    return np.sum(np.log(2 * np.pi * noise ** 2.0))

def likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term):
    """Compute the likelihood of a model lensing_image's fit to the data_vector, by taking the difference between the
    observed lensing_image and model ray-tracing lensing_image. The likelihood consists of two terms:

    Chi-squared term - The residuals (model - data_vector) of every pixel divided by the noise_map in each pixel, all
    squared.
    [Chi_Squared_Term] = sum(([Residuals] / [Noise]) ** 2.0)

    The overall normalization of the noise_map is also included, by summing the log noise_map value in each pixel:
    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    These are summed and multiplied by -0.5 to give the likelihood:

    Likelihood = -0.5*[Chi_Squared_Term + Noise_Term]

    Parameters
    ----------
    """
    return -0.5 * (chi_squared_term + noise_term)

def contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images, hyper_galaxies,
                                                 minimum_values):
    """Use the model lensing_image and galaxy lensing_image (computed in the previous phase of the pipeline) to determine the
    contributions of each hyper galaxy.

    Parameters
    -----------
    minimum_values
    hyper_model_image : ndarray
        The best-fit model lensing_image to the data_vector, from a previous phase of the pipeline
    hyper_galaxy_images : [ndarray]
        The best-fit model lensing_image of each hyper-galaxy, which can tell us how much flux each pixel contributes to.
    hyper_galaxies : [galaxy.HyperGalaxy]
        Each hyper-galaxy which is used to determine its contributions.
    """
    # noinspection PyArgumentList
    return list(map(lambda hyper, galaxy_image, minimum_value:
                    hyper.contributions_from_hyper_images(hyper_model_image, galaxy_image, minimum_value),
                    hyper_galaxies, hyper_galaxy_images, minimum_values))

def scaled_noise_from_hyper_galaxies_and_contributions(contributions, hyper_galaxies, noise):
    """Use the contributions of each hyper galaxy to compute the scaled noise_map.
    Parameters
    -----------
    noise
    hyper_galaxies
    contributions : [ndarray]
        The contribution of flux of each galaxy in each pixel (computed from galaxy.HyperGalaxy)
    """
    scaled_noises = list(map(lambda hyper, contribution: hyper.scaled_noise_from_contributions(noise, contribution),
                             hyper_galaxies, contributions))
    return noise + sum(scaled_noises)

def evidence_from_reconstruction_terms(chi_squared_term, regularization_term, log_covariance_regularization_term,
                                       log_regularization_term, noise_term):
    return -0.5 * (chi_squared_term + regularization_term + log_covariance_regularization_term -
                   log_regularization_term + noise_term)

def unmasked_model_image_from_tracer_and_lensing_image(tracer, lensing_image):

    if not tracer.has_unmasked_grids:
        raise exc.FittingException('An unmasked model image cannot be generated from fitting.py if the input '
                                   'tracer does not use unmasked grids')
    model_image_1d = lensing_image.unmasked_grids.image.convolve_array_1d_with_psf(tracer._image_plane_image,
                                                                                   lensing_image.psf)

    return lensing_image.unmasked_grids.image.map_to_2d(model_image_1d)

def unmasked_model_images_of_galaxies_from_tracer_and_lensing_image(tracer, lensing_image):

    if not tracer.has_unmasked_grids:
        raise exc.FittingException('An unmasked model image cannot be generated from fitting.py if the input '
                                   'tracer does not use unmasked grids')
    model_galaxy_images_1d = list(map(lambda image :
                             lensing_image.unmasked_grids.image.convolve_array_1d_with_psf(image, lensing_image.psf),
                             tracer._image_plane_images_of_galaxies))

    return list(map(lambda image : lensing_image.unmasked_grids.image.map_to_2d(image), model_galaxy_images_1d))