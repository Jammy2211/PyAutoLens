import numpy as np

from autolens import exc
from autolens.inversion import inversions
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing


class AbstractFit(object):

    def __init__(self, fitting_datas, model_datas_):
        """Abstract base class of a fit to a dataset.

        All fitting quantities (datas, model_datas, residuals, etc.) are stored a lists, where the different entries of
        the list correspond to different data-sets. To signify this, all quantities are pluralized. However, if a non-
        plural term is used (e.g. residual) the residuals of the first image are returned. This makes the interface
        for fitting single data-sets (the most common use-case) more intuitive.

        A fit is performed using masked data which has been mapped to 1D arrays, even if the data is originally a 2D
        image. The 1D representation of a data is followed by an underscore. For example, model_datas gives the 2D
        images of the data, whereas model_datas_ gives the 1D data representation.

        Parameters
        -----------
        fitting_datas : [fitting_data.FittingImage]
            The fitting images that are fitted.
        model_datas_ : [ndarray]
            The model datas the fitting images are fitted with.

        Attributes
        -----------
        datas : [scaled_array.ScaledSquarePixelArray]
            The data that is fitted.
        noise_maps : [scaled_array.ScaledSquarePixelArray]
            The noise-maps used to fit the data.
        masks : [image.Mask]
            The masks which define the regions of the data that are fitted.
        model_datas : [scaled_array.ScaledSquarePixelArray]
            The model dataset used to fit the data
        map_to_scaled_arrays : [func]
            Functions which map the masked 1D arrays of the data, model_data, residuals, etc. to their 2D data
            representation (if an image)
        residuals : [scaled_array.ScaledSquarePixelArray]
            The residuals of the fit (data - model_data).
        chi_squareds : [scaled_array.ScaledSquarePixelArray]
            The chi-sqaureds of the fit ((data - model_data) / noise_map ) **2.0
        chi_squared_terms : [float]
            The summed chi-squared of every data-point in a fit.
        chi_squareds_term : float
            The summed chi_squared_terms for all datasets.
        reduced_chi_squared_terms : [float]
            The reduced chi_squared of the fit to every data-set (chi_squared / number of data points).
        reduced_chi_squared_term : [float]
            The summed reduced chi_squared of the fit between data and model for all datasets.
        noise_terms : [float]
            The normalization term of a likelihood function assuming Gaussian noise in every data-point.
        noise_term : float
            The sum of all noise_terms for all datasets.
        likelihoods : [float]
            The likelihood of every fit between data and model -0.5 * (chi_squared_term + noise_term)
        likelihood : float
            The summed likelihood of the fit between data and model for all datasets.
        """

        self.datas_ = list(map(lambda fit_data : fit_data[:], fitting_datas))
        self.noise_maps_ = list(map(lambda fit_data : fit_data.noise_map_, fitting_datas))
        self.masks = list(map(lambda fit_data : fit_data.mask, fitting_datas))
        self.map_to_scaled_arrays = list(map(lambda fit_data: fit_data.grids.image.scaled_array_from_array_1d,
                                             fitting_datas))

        self.model_datas_ = model_datas_
        self.residuals_ = residuals_from_datas_and_model_datas(datas_=self.datas_, model_datas_=self.model_datas_)
        self.chi_squareds_ = chi_squareds_from_residuals_and_noise_maps(self.residuals_, self.noise_maps_)

    @property
    def chi_squared_terms(self):
        return chi_squared_terms_from_chi_squareds(self.chi_squareds_)

    @property
    def chi_squared_term(self):
        return sum(self.chi_squared_terms)

    @property
    def reduced_chi_squareds(self):
        return list(map(lambda chi_squared_term, mask : chi_squared_term / mask.pixels_in_mask,
                        self.chi_squared_terms, self.masks))

    @property
    def reduced_chi_squared(self):
        return sum(self.reduced_chi_squareds)

    @property
    def noise_terms(self):
        return noise_terms_from_noise_maps(self.noise_maps_)

    @property
    def noise_term(self):
        return sum(self.noise_terms)

    @property
    def likelihoods(self):
        return likelihoods_from_chi_squareds_and_noise_terms(self.chi_squared_terms, self.noise_terms)

    @property
    def likelihood(self):
        return sum(self.likelihoods)

    @property
    def noise_maps(self):
        return map_arrays_to_scaled_arrays(arrays_=self.noise_maps_, map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def model_datas(self):
        return map_arrays_to_scaled_arrays(arrays_=self.model_datas_, map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def residuals(self):
        return map_arrays_to_scaled_arrays(arrays_=self.residuals_, map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def chi_squareds(self):
        return map_arrays_to_scaled_arrays(arrays_=self.chi_squareds_, map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def noise_map(self):
        return self.noise_maps[0]

    @property
    def model_data(self):
        return self.model_datas[0]

    @property
    def residual(self):
        return self.residuals[0]

    @property
    def chi_squared(self):
        return self.chi_squareds[0]


class AbstractDataFit(AbstractFit):

    def __init__(self, fitting_datas, model_datas_):
        """Abstract Base class of a fit between data-set and model, see *AbstractFit*.

        Parameters
        -----------
        fitting_datas : [fitting_data.FittingImage]
            The fitting images that are fitted.
        model_datas_ : [ndarray]
            The model datas the fitting images are fitted with."""

        self.datas = list(map(lambda fit_data : fit_data.array, fitting_datas))
        super(AbstractDataFit, self).__init__(fitting_datas=fitting_datas, model_datas_=model_datas_)


class AbstractImageFit(AbstractFit):

    def __init__(self, fitting_images, model_images_):
        """Abstract Base class of a fit between data-set and model, see *AbstractFit*.

        This class is used to add the terms 'images' and 'model_images' to the attributes of a fit. However, they
        function identical to datas and model_datas, and this is done onoly for the user interface.

        Parameters
        -----------
        fitting_images : [fitting_data.FittingImage]
            The fitting images that are fitted.
        model_datas_ : [ndarray]
            The model images the fitting images are fitted with.
        """

        self.images = list(map(lambda fit_image : fit_image.image, fitting_images))
        super(AbstractImageFit, self).__init__(fitting_datas=fitting_images, model_datas_=model_images_)

    @property
    def model_images(self):
        return self.model_datas

    @property
    def model_image(self):
        return self.model_datas[0]


class AbstractHyperFit(object):

    def __init__(self, fitting_hyper_images, hyper_galaxies):
        """Abstract base class of a hyper-fit.

        A hyper-fit is a fit which performs a fit as described in the *AbstractFit*, but also includes a set of
        parameters which allow the noise-map of the data-set to be scaled. This is done to prevent over-fitting
        small regions of a data-set with high chi-squared values and therefore provide a global fit to the overall
        data-set.

        This is performed using an existing model of the data-set to compute a contributions image, which a set of
        hyper-parameters then use to increase the noise in localized regions of the data-set.

        Parameters
        -----------
        fitting_hyper_images : [fitting_data.FittingHyperImage]
            The fitting images that are fitted, which include the hyper-images used for scaling the noise-map.
        hyper_galaxies : [galaxy.Galaxy]
            The hyper-galaxies which represent the model components used to scale the noise, which correspnd to
            individual galaxies in the image.

        Attributes
        -----------
        contributions : [[scaled_array.ScaledSquarePixelArray]]
            The contribution map of every image, where there is an individual contribution map for each hyper-galaxy in
            the model.
        scaled_noise_maps : [scaled_array.ScaledSquarePixelArray]
            The scaled noise maps of the image, computed after using the hyper-galaxies.
        scaled_chi_squared_terms : [float]
            The summed scaled chi-squared of every data-point in a fit.
        scaled_chi_squareds_term : float
            The sum of all scaled_chi_squared_terms for all images.
        scaled_noise_terms : [float]
            The normalization term of a likelihood function assuming Gaussian noise in every data-point, using the
            scaled noise-map.
        scaled_noise_term : float
            The sum of all scaled_noise_terms for all images.
        scaled_likelihoods : [float]
            The likelihood of every fit between data and model using the scaled noise-map's fit \
            -0.5 * (scaled_chi_squared_term + scaled_noise_term)
        scaled_likelihood : float
            The summed scaled likelihood of the fit between data and model for all images.
        """

        self.is_hyper_fit = True
        self.contributions_ = \
            contributions_from_fitting_hyper_images_and_hyper_galaxies(fitting_hyper_images=fitting_hyper_images,
                                                                       hyper_galaxies=hyper_galaxies)


        self.scaled_noise_maps_ =\
            scaled_noises_from_fitting_hyper_images_contributions_and_hyper_galaxies(
                fitting_hyper_images=fitting_hyper_images, contributions_=self.contributions_,
                hyper_galaxies=hyper_galaxies)

    @property
    def scaled_chi_squared_terms(self):
        return chi_squared_terms_from_chi_squareds(self.scaled_chi_squareds_)

    @property
    def scaled_noise_terms(self):
        return noise_terms_from_noise_maps(self.scaled_noise_maps_)

    @property
    def scaled_noise_maps(self):
        return map_arrays_to_scaled_arrays(arrays_=self.scaled_noise_maps_,
                                           map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def scaled_chi_squareds(self):
        return map_arrays_to_scaled_arrays(arrays_=self.scaled_chi_squareds_,
                                           map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def contributions(self):
        contributions = [[] for _ in range(len(self.contributions_))]
        for image_index in range(len(contributions)):
            contributions[image_index] = list(map(lambda _contributions :
                                                  self.map_to_scaled_arrays[image_index](_contributions),
                                                  self.contributions_[image_index]))
        return contributions

    @property
    def scaled_noise_map(self):
        return self.scaled_noise_maps[0]

    @property
    def scaled_chi_squared(self):
        return self.scaled_chi_squareds[0]


class AbstractHyperImageFit(AbstractImageFit, AbstractHyperFit):

    def __init__(self, fitting_hyper_images, model_images_, hyper_galaxies):
       """Abstract base class for an image data-set which includes hyper noise-scaling. Seee *AbstractFit* and
       *AbstractHyperFit* for more details."""
       AbstractHyperFit.__init__(self=self, fitting_hyper_images=fitting_hyper_images, hyper_galaxies=hyper_galaxies)
       super(AbstractHyperImageFit, self).__init__(fitting_images=fitting_hyper_images, model_images_=model_images_)
       self.scaled_chi_squareds_ = chi_squareds_from_residuals_and_noise_maps(self.residuals_, self.scaled_noise_maps_)


class AbstractProfileFit(AbstractImageFit):

    def __init__(self, fitting_images, images_, blurring_images_):
        """Abstract base class for a fit to an image using a light-profile.

        This includes the blurring of the light-profile model image with the instrumental PSF.

        Parameters
        -----------
        fitting_images : [fitting_data.FittingImage]
            The fitting images that are fitted.
        images_ : [ndarray]
            The masked 1D representation of the light profile model images before PSF blurring.
        blurring_images_ : [ndarray]
            The 1D representation of the light profile model images blurring region, which corresponds to all pixels \
            which are not inside the mask but close enough that their light will be blurred into it via PSF convolution.
        """

        self.convolvers_image = list(map(lambda fit_image : fit_image.convolver_image, fitting_images))

        model_images_ = blur_images_including_blurring_regions(images_=images_, blurring_images_=blurring_images_,
                                                               convolvers=self.convolvers_image)

        super(AbstractProfileFit, self).__init__(fitting_images=fitting_images, model_images_=model_images_)


class AbstractInversionFit(AbstractImageFit):

    def __init__(self, fitting_images, mapper, regularization):
        """Abstract base class for a fit to an image which uses a linear inversion.

        This includes passing the image / noise_map / PSF and inversion objects to the inversons module to perform \
        the inversion.

        Parameters
        -----------
        fitting_images : [fitting_data.FittingImage]
            The fitting images that are fitted.
        mapper : inversion.mapper.Mapper
            Class storing the mappings between observed image-pixels and inversion's pixelization pixels.
        regularization : inversion.regularization.Regularization
            Class storing the regularization scheme of the inversion's pixelization.

        Attributes
        -----------
        likelihoods_with_regularization : [float]
            The likelihood of each fit to the image, including jsut 3 terms, the chi-squared term, regularization
            penalty factor and noise normalization term.
        likelihood_with_regularization : float
            The sum of all likelihoods_with_regularization for all images.
        evidences : [float]
            The Bayesian evidence of each fit to the image. The Bayesian evidence is described in Suyu et al. 2006 and
            the howtolens inversion tutorial chapter_4_inversions/tutorial_4_bayesian_regularization.
        evidence : float
            The sum of evidence values for all images.
        """

        self.inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(image=fitting_images[0][:],
        noise_map=fitting_images[0].noise_map_, convolver=fitting_images[0].convolver_mapping_matrix,
        mapper=mapper, regularization=regularization)

        super(AbstractInversionFit, self).__init__(fitting_images=fitting_images,
                                                   model_images_=[self.inversion.reconstructed_data_vector])

    @property
    def likelihoods_with_regularization(self):
        return likelihoods_with_regularization_from_chi_squared_regularization_and_noise_terms(self.chi_squared_terms,
               [self.inversion.regularization_term], self.noise_terms)

    @property
    def likelihood_with_regularization(self):
        return sum(self.likelihoods_with_regularization)

    @property
    def evidences(self):
        return evidences_from_reconstruction_terms(self.chi_squared_terms, [self.inversion.regularization_term],
                                                   [self.inversion.log_det_curvature_reg_matrix_term],
                                                   [self.inversion.log_det_regularization_matrix_term],
                                                   self.noise_terms)

    @property
    def evidence(self):
        return sum(self.evidences)


class AbstractProfileInversionFit(AbstractImageFit):
    
    def __init__(self, fitting_images, images_, blurring_images_, mapper, regularization):
        
        self.convolvers_image = list(map(lambda fit_image : fit_image.convolver_image, fitting_images))

        self.profile_model_images_ = blur_images_including_blurring_regions(images_=images_,
                                                                            blurring_images_=blurring_images_,
                                                                            convolvers=self.convolvers_image)

        self.profile_subtracted_images_ = list(map(lambda fitting_image, profile_model_image_ :
                                                   fitting_image[:] - profile_model_image_,
                                                   fitting_images, self.profile_model_images_))

        self.inversion = inversions.inversion_from_lensing_image_mapper_and_regularization(
            image=self.profile_subtracted_images_[0], noise_map=fitting_images[0].noise_map_,
            convolver=fitting_images[0].convolver_mapping_matrix, mapper=mapper, regularization=regularization)
        
        self.inversion_model_images_ = [self.inversion.reconstructed_data_vector]

        model_images_ = list(map(lambda profile_model_image_, inversion_model_image_ :
                                 profile_model_image_ + inversion_model_image_,
                                 self.profile_model_images_, self.inversion_model_images_))

        super(AbstractProfileInversionFit, self).__init__(fitting_images=fitting_images, model_images_=model_images_)

    @property
    def profile_subtracted_images(self):
        return map_arrays_to_scaled_arrays(arrays_=self.profile_subtracted_images_,
                                           map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def profile_model_images(self):
        return map_arrays_to_scaled_arrays(arrays_=self.profile_model_images_,
                                           map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def inversion_model_images(self):
        return map_arrays_to_scaled_arrays(arrays_=self.inversion_model_images_,
                                           map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def profile_subtracted_image(self):
        return self.profile_subtracted_images[0]

    @property
    def profile_model_image(self):
        return self.profile_model_images[0]

    @property
    def inversion_model_image(self):
        return self.inversion_model_images[0]

    @property
    def evidences(self):
        return evidences_from_reconstruction_terms(self.chi_squared_terms, [self.inversion.regularization_term],
                                                   [self.inversion.log_det_curvature_reg_matrix_term],
                                                   [self.inversion.log_det_regularization_matrix_term],
                                                   self.noise_terms)

    @property
    def evidence(self):
        return sum(self.evidences)


def map_arrays_to_scaled_arrays(arrays_, map_to_scaled_arrays):
    return list(map(lambda _array, map_to_scaled_array, : map_to_scaled_array(_array), arrays_, map_to_scaled_arrays))

def map_contributions_to_scaled_arrays(contributions_, map_to_scaled_array):
    return list(map(lambda _contribution : map_to_scaled_array(_contribution), contributions_))

def blur_images_including_blurring_regions(images_, blurring_images_, convolvers):
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
    return list(map(lambda image_, blurring_image_, convolver :
                    convolver.convolve_image(image_array=image_, blurring_array=blurring_image_),
                    images_, blurring_images_, convolvers))

def residuals_from_datas_and_model_datas(datas_, model_datas_):
    """Compute the residuals between an observed charge injection lensing_image and post-cti model_datas_ lensing_image.

    Residuals = (Data - Model).

    Parameters
    -----------
    datas_ : [np.ndarray]
        The observed charge injection lensing_image datas.
    model_datas_ : [np.ndarray]
        The model_datas_ lensing_image.
    """
    return list(map(lambda data_, model_data_ : np.subtract(data_, model_data_), datas_, model_datas_))

def chi_squareds_from_residuals_and_noise_maps(residuals_, noise_maps_):
    """Computes a chi-squared lensing_image, by calculating the squared residuals between an observed charge injection \
    image and post-cti hyper_model_image lensing_image and dividing by the variance (noises**2.0) in each pixel.

    Chi_Sq = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    This gives the residuals, which are divided by the variance of each pixel and squared to give their chi sq.

    Parameters
    -----------
    residuals_ : [np.ndarray]
        The residuals of the model fit_normal to the data_
    noise_maps_ : [np.ndarray]
        The noises in the lensing_image.
    """
    return list(map(lambda residual_, noise_map_ : np.square((np.divide(residual_, noise_map_))),
                    residuals_, noise_maps_))

def chi_squared_terms_from_chi_squareds(chi_squareds_):
    """Compute the chi-squared of a model lensing_image's fit_normal to the data_vector, by taking the difference between the
    observed lensing_image and model ray-tracing lensing_image, dividing by the noise_map_ in each pixel and squaring:

    [Chi_Squared] = sum(([Data - Model] / [Noise]) ** 2.0)

    Parameters
    ----------
    chi_squareds_
    """
    return list(map(lambda chi_squared_ : np.sum(chi_squared_), chi_squareds_))

def noise_terms_from_noise_maps(noise_maps_):
    """Compute the noise_map_ normalization term of an lensing_image, which is computed by summing the noise_map_ in every pixel:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise_maps_ : [np.ndarray]
        The noise_map_ in each pixel.
    """
    return list(map(lambda noise_map_ : np.sum(np.log(2 * np.pi * noise_map_ ** 2.0)), noise_maps_))

def likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms, noise_terms):
    """Compute the likelihood of a model lensing_image's fit_normal to the data_vector, by taking the difference between the
    observed lensing_image and model ray-tracing lensing_image. The likelihood consists of two terms:

    Chi-squared term - The residuals (model - data_vector) of every pixel divided by the noise_map_ in each pixel, all
    squared.
    [Chi_Squared_Term] = sum(([Residuals] / [Noise]) ** 2.0)

    The overall normalization of the noise_map_ is also included, by summing the log noise_map_ value in each pixel:
    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    These are summed and multiplied by -0.5 to give the likelihood:

    Likelihood = -0.5*[Chi_Squared_Term + Noise_Term]

    Parameters
    ----------
    """
    return list(map(lambda chi_squared_term, noise_term : -0.5 * (chi_squared_term + noise_term),
                    chi_squared_terms, noise_terms))

def likelihoods_with_regularization_from_chi_squared_regularization_and_noise_terms(chi_squared_terms,
                                                                                    regularization_terms, noise_terms):
    return list(map(lambda chi_squared_term, regularization_term, noise_term :
                    -0.5 * (chi_squared_term + regularization_term + noise_term),
                    chi_squared_terms, regularization_terms, noise_terms))

def evidences_from_reconstruction_terms(chi_squared_terms, regularization_terms, log_covariance_regularization_terms,
                                        log_regularization_terms, noise_terms):
    return list(map(lambda chi_squared_term, regularization_term, log_covariance_regularization_term,
                           log_regularization_term, noise_term :
                    -0.5 * (chi_squared_term + regularization_term + log_covariance_regularization_term -
                            log_regularization_term + noise_term),
                    chi_squared_terms, regularization_terms, log_covariance_regularization_terms,
                    log_regularization_terms, noise_terms))

def unmasked_model_images_from_fitting_images(fitting_images, unmasked_images_):

    return list(map(lambda fitting_image, _unmasked_image :
                    unmasked_model_image_from_fitting_image(fitting_image, _unmasked_image),
                    fitting_images, unmasked_images_))

def unmasked_model_image_from_fitting_image(fitting_image, unmasked_image_):

    _model_image = fitting_image.padded_grids.image.convolve_array_1d_with_psf(unmasked_image_,
                                                                               fitting_image.psf)

    return fitting_image.padded_grids.image.scaled_array_from_array_1d(_model_image)

def contributions_from_fitting_hyper_images_and_hyper_galaxies(fitting_hyper_images, hyper_galaxies):

    return list(map(lambda hyp :
                    contributions_from_hyper_images_and_galaxies(hyper_model_image=hyp.hyper_model_image,
                                                                 hyper_galaxy_images=hyp.hyper_galaxy_images,
                                                                 hyper_galaxies=hyper_galaxies,
                                                                 minimum_values=hyp.hyper_minimum_values),
                fitting_hyper_images))

def contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images, hyper_galaxies, minimum_values):
    """Use the model lensing_image and galaxy lensing_image (computed in the previous phase of the pipeline) to determine the
    contributions of each hyper galaxy.

    Parameters
    -----------
    minimum_values
    hyper_model_image : ndarray
        The best-fit_normal model lensing_image to the data_vector, from a previous phase of the pipeline
    hyper_galaxy_images : [ndarray]
        The best-fit_normal model lensing_image of each hyper-galaxy, which can tell us how much flux each pixel contributes to.
    hyper_galaxies : [galaxy.HyperGalaxy]
        Each hyper-galaxy which is used to determine its contributions.
    """
    # noinspection PyArgumentList
    return list(map(lambda hyper, galaxy_image, minimum_value:
                    hyper.contributions_from_hyper_images(hyper_model_image, galaxy_image, minimum_value),
                    hyper_galaxies, hyper_galaxy_images, minimum_values))

def scaled_noises_from_fitting_hyper_images_contributions_and_hyper_galaxies(fitting_hyper_images, contributions_,
                                                                             hyper_galaxies):
    return list(map(lambda hyp, contribution_ :
                    scaled_noise_from_hyper_galaxies_and_contributions(contributions_=contribution_,
                                                                       hyper_galaxies=hyper_galaxies,
                                                                       noise_map_=hyp.noise_map_),
                    fitting_hyper_images, contributions_))

def scaled_noise_from_hyper_galaxies_and_contributions(contributions_, hyper_galaxies, noise_map_):
    """Use the contributions of each hyper galaxy to compute the scaled noise_map_.
    Parameters
    -----------
    noise_map_
    hyper_galaxies
    contributions_ : [ndarray]
        The contribution_ of flux of each galaxy in each pixel (computed from galaxy.HyperGalaxy)
    """
    scaled_noises_ = list(map(lambda hyper, contribution_: hyper.scaled_noise_from_contributions(noise_map_, contribution_),
                             hyper_galaxies, contributions_))
    return noise_map_ + sum(scaled_noises_)