import numpy as np

from autolens import exc
from autolens.imaging import image as im
from autolens.inversion import inversions
from autolens.fitting import fitting
from autolens.fitting import fitting_data
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing

class AbstractSensitivityFit(object):

    def __init__(self, tracer_normal, tracer_sensitive):

        self.tracer_normal = tracer_normal
        self.tracer_sensitive = tracer_sensitive


class SensitivityProfileFit(AbstractSensitivityFit):

    def __init__(self, sensitivity_images, tracer_normal, tracer_sensitive):
        """
        Class to evaluate the fit between a model described by a tracer_normal and an actual lensing_image.

        Parameters
        ----------
        sensitivity_images: [li.LensingImage]
            An lensing_image that has been masked for efficiency
        tracer_normal: ray_tracing.AbstractTracer
            An object describing the model
        """
        AbstractSensitivityFit.__init__(self=self, tracer_normal=tracer_normal, tracer_sensitive=tracer_sensitive)

        self.fit_normal = fitting.AbstractProfileFit(fitting_images=sensitivity_images,
                                                     images_=tracer_normal.image_plane_images_,
                                                     blurring_images_=tracer_normal.image_plane_blurring_images_)
        
        self.fit_sensitive = fitting.AbstractProfileFit(fitting_images=sensitivity_images,
                                                        images_=tracer_sensitive.image_plane_images_,
                                                        blurring_images_=tracer_sensitive.image_plane_blurring_images_)

    @classmethod
    def fast_likelihood(cls, sensitivity_images, tracer_normal, tracer_sensitive):
        """
        Fast calculation of likelihood

        Parameters
        ----------
        sensitivity_images: [li.LensingImage]
            An lensing_image that has been masked for efficiency
        tracer_normal: ray_tracing.AbstractTracer
            An object describing the model
        """

        convolvers = list(map(lambda lensing_image : lensing_image.convolver_image, sensitivity_images))
        noise_maps_ = list(map(lambda lensing_image : lensing_image.noise_map_, sensitivity_images))
        
        model_images_normal_ = fitting.blur_images_including_blurring_regions(images_=tracer_normal.image_plane_images_,
                                     blurring_images_=tracer_normal.image_plane_blurring_images_, convolvers=convolvers)
        
        residuals_normal_ = fitting.residuals_from_datas_and_model_datas(datas_=sensitivity_images,
                                                                         model_datas_=model_images_normal_)
        
        chi_squareds_normal_ = fitting.chi_squareds_from_residuals_and_noise_maps(residuals_=residuals_normal_,
                                                                                  noise_maps_=noise_maps_)
        
        chi_squared_terms_normal = fitting.chi_squared_terms_from_chi_squareds(chi_squareds_=chi_squareds_normal_)
        
        noise_terms_normal = fitting.noise_terms_from_noise_maps(noise_maps_=noise_maps_)
        likelihoods_normal = fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms=chi_squared_terms_normal, 
                                                                                   noise_terms=noise_terms_normal)
        
        model_images_sensitive_ = fitting.blur_images_including_blurring_regions(
            images_=tracer_sensitive.image_plane_images_, blurring_images_=tracer_sensitive.image_plane_blurring_images_,
            convolvers=convolvers)
        
        residuals_sensitive_ = fitting.residuals_from_datas_and_model_datas(datas_=sensitivity_images,
                                                                            model_datas_=model_images_sensitive_),
        chi_squareds_sensitive_ = fitting.chi_squareds_from_residuals_and_noise_maps(residuals_=residuals_sensitive_,
                                                                                     noise_maps_=noise_maps_),
        chi_squared_terms_sensitive = fitting.chi_squared_terms_from_chi_squareds(chi_squareds_=chi_squareds_sensitive_)
        noise_terms_sensitive = fitting.noise_terms_from_noise_maps(noise_maps_=noise_maps_)
        
        likelihoods_sensitive = fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms=chi_squared_terms_sensitive, 
                                                                                   noise_terms=noise_terms_sensitive)

        return sum(likelihoods_normal) - sum(likelihoods_sensitive)

    @property
    def likelihood(self):
        return self.fit_normal.likelihood - self.fit_sensitive.likelihood