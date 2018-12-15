from autolens.data.fitting import fitter, fitting_util
from autolens.lensing import ray_tracing

class AbstractSensitivityFit(object):

    def __init__(self, tracer_normal, tracer_sensitive):

        self.tracer_normal = tracer_normal
        self.tracer_sensitive = tracer_sensitive


class SensitivityProfileFit(AbstractSensitivityFit):

    def __init__(self, sensitivity_images, tracer_normal, tracer_sensitive):
        """
        Class to evaluate the fit between a model described by a tracer_normal and an actual sensitivity_image.

        Parameters
        ----------
        sensitivity_images: [li.LensingImage]
            An sensitivity_image that has been masked for efficiency
        tracer_normal: ray_tracing.AbstractTracer
            An object describing the model
        """
        AbstractSensitivityFit.__init__(self=self, tracer_normal=tracer_normal, tracer_sensitive=tracer_sensitive)

        self.fit_normal = fitter.AbstractConvolutionFit(fitting_images=sensitivity_images,
                                                        images_=tracer_normal.image_plane_images_,
                                                        blurring_images_=tracer_normal.image_plane_blurring_images_)
        
        self.fit_sensitive = fitter.AbstractConvolutionFit(fitting_images=sensitivity_images,
                                                           images_=tracer_sensitive.image_plane_image_1d,
                                                           blurring_images_=tracer_sensitive.image_plane_blurring_image_1d)

    @classmethod
    def fast_likelihood(cls, sensitivity_images, tracer_normal, tracer_sensitive):
        """
        Fast calculation of likelihood

        Parameters
        ----------
        sensitivity_images: [li.LensingImage]
            An sensitivity_image that has been masked for efficiency
        tracer_normal: ray_tracing.AbstractTracer
            An object describing the model
        """

        convolvers = list(map(lambda lensing_image : lensing_image.convolver_image, sensitivity_images))
        noise_maps_ = list(map(lambda lensing_image : lensing_image.noise_map_, sensitivity_images))
        
        model_images_normal_ = fitting_util.blur_image_including_blurring_region(
            image_=tracer_normal.image_plane_images_,
            blurring_image_=tracer_normal.image_plane_blurring_images_, convolver=convolvers)
        
        residuals_normal_ = fitting_util.residual_map_from_data_mask_and_model_data(data=sensitivity_images,
                                                                                    model_data=model_images_normal_)
        
        chi_squareds_normal_ = fitting_util.chi_squareds_from_residual_map_and_noise_map(residual_map=residuals_normal_,
                                                                                         noise_map=noise_maps_)
        
        chi_squared_terms_normal = fitting_util.chi_squared_term_from_chi_squared_map(
            chi_squared_map=chi_squareds_normal_)
        
        noise_terms_normal = fitting_util.noise_term_from_mask_and_noise_map(noise_map=noise_maps_)
        likelihoods_normal = fitting_util.likelihood_from_chi_squared_term_and_noise_term(
            chi_squared_term=chi_squared_terms_normal,
            noise_term=noise_terms_normal)
        
        model_images_sensitive_ = fitting_util.blur_image_including_blurring_region(
            image_=tracer_sensitive.image_plane_image_1d, blurring_image_=tracer_sensitive.image_plane_blurring_image_1d,
            convolver=convolvers)
        
        residuals_sensitive_ = fitting_util.residual_map_from_data_mask_and_model_data(data=sensitivity_images,
                                                                                       model_data=model_images_sensitive_),
        chi_squareds_sensitive_ = fitting_util.chi_squareds_from_residual_map_and_noise_map(
            residual_map=residuals_sensitive_,
            noise_map=noise_maps_),
        chi_squared_terms_sensitive = fitting_util.chi_squared_term_from_chi_squared_map(
            chi_squared_map=chi_squareds_sensitive_)
        noise_terms_sensitive = fitting_util.noise_term_from_mask_and_noise_map(noise_map=noise_maps_)
        
        likelihoods_sensitive = fitting_util.likelihood_from_chi_squared_term_and_noise_term(
            chi_squared_term=chi_squared_terms_sensitive,
            noise_term=noise_terms_sensitive)

        return sum(likelihoods_sensitive) - sum(likelihoods_normal)

    @property
    def likelihood(self):
        return self.fit_sensitive.likelihood - self.fit_normal.likelihood