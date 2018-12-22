import numpy as np

from autolens import exc
from autolens.data.fitting import fitter
from autolens.model.inversion import inversions
from autolens.lensing import lensing_image as li, lensing_util
from autolens.lensing import ray_tracing


def fit_lensing_image_stack_with_tracer(lensing_image, tracer, padded_tracer=None):
    """Fit a lensing image with a model tracer, automatically determining the type of fit based on the \
    properties of the galaxies in the tracer.

    Parameters
    -----------
    lensing_image : li.LensingImage or li.LensingHyperImage
        The lensing-images that is fitted.
    tracer : ray_tracing.Tracer
        The tracer, which describes the ray-tracing and strong lensing configuration.
    padded_tracer : ray_tracing.AbstractTracerNonStack or None
        A tracer with an identical strong lensing configuration to the tracer above, but using the lensing image's \
        padded grid_stack such that unmasked model-images can be computed.
    """

    if tracer.has_light_profile and not tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileFitter(lensing_image=lensing_image, tracer=tracer, padded_tracer=padded_tracer)
        elif tracer.has_hyper_galaxy:
            return LensingProfileHyperFitter(lensing_hyper_image=lensing_image, tracer=tracer,
                                             padded_tracer=padded_tracer)

    elif not tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingInversionFitter(lensing_image=lensing_image, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return LensingInversionHyperFitter(lensing_hyper_image=lensing_image, tracer=tracer)

    elif tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileInversionFitter(lensing_image=lensing_image, tracer=tracer,
                                                 padded_tracer=padded_tracer)
        elif tracer.has_hyper_galaxy:
            return LensingProfileInversionHyperFitter(lensing_hyper_image=lensing_image, tracer=tracer,
                                                      padded_tracer=padded_tracer)

    else:

        raise exc.FittingException('The fit routine did not call a Fit class - check the '
                                   'properties of the tracer')


def fast_fit_from_lensing_image_and_tracer(lensing_image, tracer):
    """Fit a lensing image with a model tracer, automatically determining the type of fit based on the properties of \
     the galaxies in the tracer.

    The likelihood is computed in the fastest, least memory-intensive, way possible, for efficient non-linear sampling.

    Parameters
    -----------
    lensing_image : li.LensingImage or li.LensingHyperImage
        The lensing-images that is fitted.
    tracer : ray_tracing.AbstractTracerNonStack
        The tracer, which describes the ray-tracing and strong lensing configuration.
    padded_tracer : ray_tracing.Tracer or None
        A tracer with an identical strong lensing configuration to the tracer above, but using the lensing image's \
        padded grid_stack such that unmasked model-images can be computed.
    """
    if tracer.has_light_profile and not tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileFitter.fast_fit(lensing_image=lensing_image, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return LensingProfileHyperFitter.fast_fit(lensing_hyper_image=lensing_image, tracer=tracer)

    elif not tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingInversionFitter.fast_fit(lensing_image=lensing_image, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return LensingInversionHyperFitter.fast_fit(lensing_hyper_image=lensing_image, tracer=tracer)

    elif tracer.has_light_profile and tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensingProfileInversionFitter.fast_fit(lensing_image=lensing_image, tracer=tracer)
        elif tracer.has_hyper_galaxy:
            return LensingProfileInversionHyperFitter.fast_fit(lensing_hyper_image=lensing_image, tracer=tracer)

    else:

        raise exc.FittingException('The fast likelihood routine did not call a likelihood functions - check the '
                                   'properties of the tracer')


class AbstractLensingProfileFitter(AbstractLensingFitter):

    def __init__(self, lensing_image, tracer, padded_tracer):
        """ An abstract lensing profile fitter, which generates the image-plane image of all galaxies (with light \
        profiles) in the tracer and blurs it with the lensing image's PSF.

        If a padded tracer is supplied, the blurred profile image's can be generated over the entire image and thus \
        without the mask.

        Parameters
        -----------
        lensing_image : li.LensingImage
            The lensing-image that is fitted.
        tracer : ray_tracing.Tracer
            The tracer, which describes the ray-tracing and strong lensing configuration.
        padded_tracer : ray_tracing.AbstractTracerNonStack or None
            A tracer with an identical strong lensing configuration to the tracer above, but using the lensing image's \
            padded grid_stack such that unmasked model-images can be computed.
        """
        super(AbstractLensingProfileFitter, self).__init__(tracer=tracer, padded_tracer=padded_tracer,
                                                           map_to_scaled_array=lensing_image.map_to_scaled_array)

        self.psf = lensing_image.psf
        self.convolver_image = lensing_image.convolver_image
        self.blurred_profile_image = lensing_util.blurred_image_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=tracer.image_plane_image_1d, blurring_image_1d=tracer.image_plane_blurring_image_1d,
            convolver=self.convolver_image, map_to_scaled_array=self.map_to_scaled_array)

    @property
    def model_image_of_planes(self):
        return lensing_util.blurred_image_of_planes_from_tracer_and_convolver(tracer=self.tracer,
                                                                              convolver_image=self.convolver_image, map_to_scaled_array=self.map_to_scaled_array)

    @property
    def unmasked_model_image(self):
        if self.padded_tracer is None:
            return None
        elif self.padded_tracer is not None:
            return lensing_util.unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(
                padded_grid_stack=self.padded_tracer.image_plane.grid_stack, psf=self.psf,
                unmasked_image_1d=self.padded_tracer.image_plane_image_1d)

    @property
    def unmasked_model_image_of_galaxies(self):
        if self.padded_tracer is None:
            return None
        elif self.padded_tracer is not None:
            return lensing_util.unmasked_blurred_image_of_galaxies_from_padded_grid_stack_psf_and_tracer(
                padded_grid_stack=self.padded_tracer.image_plane.grid_stack, psf=self.psf, tracer=self.padded_tracer)