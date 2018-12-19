import numpy as np

from autolens import exc
from autolens.lensing.fitting import lensing_fitting_util
from autolens.data.fitting import fitter
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing

minimum_value_profile = 0.1

def fit_multiple_lensing_images_with_tracer(lensing_images, tracer, padded_tracer=None):
    """Fit a list of lensing unblurred_image_1d with a model tracer, automatically determining the type of fit based on the \
    properties of the galaxies in the tracer.

    Parameters
    -----------
    lensing_image : [li.LensingImage] or [li.LensingHyperImage]
        List of the lensing-unblurred_image_1d that are to be fitted.
    tracer : ray_tracing.AbstractTracer
        The tracer, which describes the ray-tracing of the strong lensing configuration.
    padded_tracer : ray_tracing.AbstractTracer
        A tracer with an identical strong lensing configuration to the tracer above, but using the lensing regular's \
        padded grid_stacks such that unmasked model-unblurred_image_1d can be computed.
    """

    if tracer.has_light_profile and not tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileFitter(lensing_image=lensing_images, tracer=tracer, padded_tracer=padded_tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileFit(lensing_hyper_images=lensing_images, tracer=tracer, padded_tracer=padded_tracer)

    elif not tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingInversionFit(lensing_images=lensing_images, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingInversionFit(lensing_hyper_images=lensing_images, tracer=tracer)

    elif tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileInversionFit(lensing_images=lensing_images, tracer=tracer, padded_tracer=padded_tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileInversionFit(lensing_hyper_images=lensing_images, tracer=tracer,
                                                   padded_tracer=padded_tracer)

    else:

        raise exc.FittingException('The fit_normal routine did not call a Fit class - check the '
                                   'properties of the tracer_normal')


def fast_likelihood_from_multiple_lensing_images_and_tracer(lensing_images, tracer):
    """Fit a list of lensing unblurred_image_1d with a model tracer, automatically determining the type of fit based on the \
    properties of the galaxies in the tracer.

    The likelihood is computed in the fastest, least memory-intensive, way possible, for efficient non-linear sampling.

    Parameters
    -----------
    lensing_image : [li.LensingImage] or [li.LensingHyperImage]
        List of the lensing-unblurred_image_1d that are to be fitted.
    tracer : ray_tracing.AbstractTracer
        The tracer, which describes the ray-tracing of the strong lensing configuration.
    """
    if tracer.has_light_profile and not tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileFitter.fast_fit(lensing_hyper_image=lensing_images, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileFit.fast_fit(lensing_hyper_images=lensing_images, tracer=tracer)

    elif not tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingInversionFit.fast_fit(lensing_images=lensing_images, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingInversionFit.fast_scaled_evidence(lensing_hyper_images=lensing_images, tracer=tracer)

    elif tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileInversionFit.fast_fit(lensing_images=lensing_images, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return HyperLensingProfileInversionFit.fast_scaled_evidence(lensing_hyper_images=lensing_images, tracer=tracer)

    else:

        raise exc.FittingException('The fast likelihood routine did not call a likelihood functions - check the '
                                   'properties of the tracer_normal')


class LensingConvolutionFitterMulti(fitter.DataFitterMulti):

    def __init__(self, lensing_images, unblurred_images_1d, blurring_images_1d):
        """Abstract base class for a fit to an regular using a light-profile.

        This includes the blurring of the light-profile model regular with the instrumental PSF.

        Parameters
        -----------
        lensing_image : lensing_image.LensingImage
            The lensing image that is being fitted.
        unblurred_image_1d : [ndarray]
            The masked 1D representation of the unblurred light profile image before PSF blurring.
        blurring_image_1d : [ndarray]
            The 1D representation of the light profile image's blurring region, which corresponds to all pixels \
            which are not inside the masks but close enough that their light will be blurred into it via PSF convolution.
        """

        model_images_1d = list(map(lambda lensing_image, unblurred_image_1d, blurring_image_1d :
        lensing_fitting_util.blurred_image_1d_from_1d_unblurred_and_blurring_images(unblurred_image_1d=unblurred_image_1d,
                                                                                    blurring_image_1d=blurring_image_1d, convolver=lensing_image.convolver_image),
                                   lensing_images, unblurred_images_1d, blurring_images_1d))

        model_images = list(map(lambda lensing_image, model_image_1d :
                                lensing_image.map_to_scaled_array(array_1d=model_image_1d),
                                lensing_images, model_images_1d))

        super(LensingConvolutionFitterMulti, self).__init__(fit_data=lensing_images, model_data=model_images)

    @property
    def model_images(self):
        return self.model_data

    @property
    def model_image_of_planes(self):

        model_images_of_planes = [[] for _ in range(self.tracer.total_grid_stacks)]

        for image_index in range(self.tracer.total_grid_stacks):
            convolver = self.convolvers_image[image_index]
            map_to_scaled_array = self.map_to_scaled_arrays[image_index]
            for plane_index in range(self.tracer.total_planes):

                if np.count_nonzero(self.tracer.image_plane_images_1d_of_planes[
                                        plane_index]):  # If all entries are zero, there was no light profile

                    model_image_of_plane = lensing_fitting_util.blurred_image_from_1d_unblurred_and_blurring_images(
                        unblurred_image_1d=self.tracer.image_plane_image_1d[0],
                        blurring_image_1d=self.tracer.image_plane_blurring_image_1d[0],
                        convolver=self.convolver_image, map_to_scaled_array=self.map_to_scaled_array)
                    model_images_of_planes[image_index].append(model_image_of_plane)

                else:

                    model_images_of_planes[image_index].append(None)

        return model_images_of_planes