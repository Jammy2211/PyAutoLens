from autofit.tools import fit
from autolens import exc
from autolens.lens.util import lens_fit_stack_util as stack_util
from autolens.lens import ray_tracing


def fit_lens_image_stack_with_tracer(lens_data_stack, tracer, padded_tracer=None):
    """Fit a lens image with a model tracer, automatically determining the type of fit based on the \
    properties of the galaxies in the tracer.

    Parameters
    -----------
    lens_data_stack : li.LensImageStack
        The lens-image stack that is fitted.
    tracer : ray_tracing.AbstractTracerStack
        The tracer, which describes the ray-tracing and strong lens configuration.
    padded_tracer : ray_tracing.AbstractTracerStack or None
        A tracer with an identical strong lens configuration to the tracer above, but using the lens image's \
        padded grid_stack such that unmasked model-images can be computed.
    """

    if tracer.has_light_profile and not tracer.has_pixelization:

        if not tracer.has_hyper_galaxy:
            return LensProfileFitStack(lens_data_stack=lens_data_stack, tracer=tracer,
                                       padded_tracer=padded_tracer)

    else:

        raise exc.FittingException('The fit routine did not call a Fit class - check the '
                                   'properties of the tracer')


class AbstractLensFitStack(object):

    def __init__(self, tracer, padded_tracer, map_to_scaled_arrays):
        """ An abstract lens fitter, which contains the tracer's used to perform the fit and functions to manipulate \
        the lens image's hyper.

        Parameters
        -----------
        tracer : ray_tracing_stack.AbstractTracerStack
            The tracer, which describes the ray-tracing and strong lens configuration.
        padded_tracer : ray_tracing_stack.AbstractTracerStack or None
            A tracer with an identical strong lens configuration to the tracer above, but using the lens image's \
            padded grid_stack such that unmasked model-images can be computed.
        map_to_scaled_arrays : func
            A list of functions which maps the 1D lens hyper to its unmasked 2D scaled-array.
        """
        self.tracer = tracer
        self.padded_tracer = padded_tracer
        self.map_to_scaled_arrays = map_to_scaled_arrays

    @property
    def total_inversions(self):
        return len(self.tracer.mappers_of_planes)


class AbstractLensProfileFitStack(AbstractLensFitStack):

    def __init__(self, lens_data_stack, tracer, padded_tracer):
        """ An abstract lens profile fitter, which generates the image-plane image of all galaxies (with light \
        profiles) in the tracer and blurs it with the lens image's PSF.

        If a padded tracer is supplied, the blurred profile image's can be generated over the entire image and thus \
        without the mask.

        Parameters
        -----------
        lens_data_stack : li.LensImageStack
            The lens-image stack that is fitted.
        tracer : ray_tracing_stack.AbstractTracerStack
            The tracer, which describes the ray-tracing and strong lens configuration.
        padded_tracer : ray_tracing_stack.AbstractTracerStack or None
            A tracer with an identical strong lens configuration to the tracer above, but using the lens image's \
            padded grid_stack such that unmasked model-images can be computed.
        """
        super(AbstractLensProfileFitStack, self).__init__(tracer=tracer, padded_tracer=padded_tracer,
                                                          map_to_scaled_arrays=lens_data_stack.map_to_scaled_arrays)

        self.psfs = lens_data_stack.psfs
        self.convolvers_image = lens_data_stack.convolvers_image

        blurred_profile_images_1d = stack_util.blurred_images_1d_of_images_from_1d_unblurred_and_bluring_images(
            unblurred_images_1d=tracer.image_plane_images_1d,
            blurring_images_1d=tracer.image_plane_blurring_images_1d,
            convolvers=self.convolvers_image)

        self.blurred_profile_images = stack_util.map_arrays_1d_to_scaled_arrays(arrays_1d=blurred_profile_images_1d,
                                                                     map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def model_images_of_planes(self):

        return stack_util.blurred_images_of_images_and_planes_from_1d_images_and_convolver(
            total_planes=self.tracer.total_planes,
            image_plane_images_1d_of_planes=self.tracer.image_plane_images_1d_of_planes,
            image_plane_blurring_images_1d_of_planes=self.tracer.image_plane_blurring_images_1d_of_planes,
            convolvers=self.convolvers_image, map_to_scaled_arrays=self.map_to_scaled_arrays)

    @property
    def unmasked_model_images(self):
        if self.padded_tracer is None:
            return None
        elif self.padded_tracer is not None:
            return stack_util.unmasked_blurred_image_of_datas_from_padded_grid_stacks_psfs_and_unmasked_images(
                padded_grid_stacks=self.padded_tracer.image_plane.grid_stacks, psfs=self.psfs,
                unmasked_images_1d=self.padded_tracer.image_plane_images_1d)

    @property
    def unmasked_model_images_of_planes_and_galaxies(self):
        if self.padded_tracer is None:
            return None
        elif self.padded_tracer is not None:
            return stack_util.unmasked_blurred_image_of_datas_planes_and_galaxies_from_padded_grid_stacks_and_psf(
                   planes=self.padded_tracer.planes, padded_grid_stacks=self.padded_tracer.image_plane.grid_stacks,
                   psfs=self.psfs)


class LensDataFitStack(fit.DataFitStack):

    def __init__(self, images, noise_maps, masks, model_images):
        """Class to fit a lens image with a model image.

        Parameters
        -----------
        images : ndarray
            List of the observed images that are fitted.
        noise_map : ndarray
            List of the noise-maps of the observed images.
        mask: msk.Mask
            List of the masks that are applied to the images.
        model_images : ndarray
            List of the model images the observed images are fitted with.
        """
        super(LensDataFitStack, self).__init__(datas=images, noise_maps=noise_maps,
                                               masks=masks, model_datas=model_images)

    @property
    def images(self):
        return self.datas

    @property
    def model_images(self):
        return self.model_datas

    @property
    def figure_of_merit(self):
        return self.likelihood


class LensProfileFitStack(LensDataFitStack, AbstractLensProfileFitStack):

    def __init__(self, lens_data_stack, tracer, padded_tracer=None):
        """ Fit a lens image with galaxy light-profiles, as follows:

        1) Generates the image-plane image of all galaxies with light profiles in the tracer.
        2) Blur this image-plane image with the lens image's PSF to generate the model-image.
        3) Fit the observed with this model-image.

        If a padded tracer is supplied, the blurred profile image's can be generated over the entire image and thus \
        without the mask.

        Parameters
        -----------
        lensing_image : li.Lensimage
            The lens-image that is fitted.
        tracer : ray_tracing.AbstractTracerStack
            The tracer, which describes the ray-tracing and strong lens configuration.
        padded_tracer : ray_tracing.Tracer or None
            A tracer with an identical strong lens configuration to the tracer above, but using the lens image's \
            padded grid_stack such that unmasked model-images can be computed.
        """
        AbstractLensProfileFitStack.__init__(self=self, lens_data_stack=lens_data_stack, tracer=tracer,
                                             padded_tracer=padded_tracer)

        super(LensProfileFitStack, self).__init__(images=lens_data_stack.images,
                                                  noise_maps=lens_data_stack.noise_maps,
                                                  masks=lens_data_stack.masks,
                                                  model_images=self.blurred_profile_images)