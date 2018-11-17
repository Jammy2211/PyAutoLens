import numpy as np

from autolens.imaging import convolution
from autolens.imaging import image as im
from autolens.imaging import mask as msk


class FittingImage(im.Image):

    def __new__(cls, image, mask, sub_grid_size=2, image_psf_shape=None):
        return np.array(mask.map_2d_array_to_masked_1d_array(image)).view(cls)

    def __init__(self, image, mask, sub_grid_size=2, image_psf_shape=None):
        """A fitting image is the collection of data components (e.g. the image, noise-maps, PSF, etc.) which are used \
        to fit a generate a model image of the fitting image and fit it to its observed image.

        The fitting image is masked, by converting all relevent data-components to masked 1D arrays (which are \
        syntacically followed by an underscore, e.g. noise_map_). 1D arrays are converted back to 2D masked arrays
        after the fit, using the mask's *scaled_array_from_1d_array* function.

        A fitting image also includes a number of attributes which are used to performt the fit, including (y,x) \
        grids of coordinates, convolvers and other utilities.

        Parameters
        ----------
        image : im.Image
            The 2D observed image and other observed quantities (noise-map, PSF, exposure-time map, etc.)
        mask: msk.Mask
            The 2D mask that is applied to image data.
        sub_grid_size : int
            The size of the sub-grid used for computing the SubGrid (see imaging.mask.SubGrid).
        image_psf_shape : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.

        Attributes
        ----------
        image : ScaledSquarePixelArray
            The 2D observed image data (not an instance of im.Image, so does not include the other data attributes,
            which are explicitly made as new attributes of the fitting image).
        image_ : ndarray
            The masked 1D array of the image.
        noise_map : NoiseMap
            An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        noise_map_ : ndarray
            The masked 1D array of the noise-map
        background_noise_map : NoiseMap or None
            An array describing the RMS standard deviation error in each pixel due to the background sky noise,
            preferably in units of electrons per second.
        background_noise_map_ : ndarray or None
            The masked 1D array of the background noise-map
        poisson_noise_map : NoiseMap or None
            An array describing the RMS standard deviation error in each pixel due to the Poisson counts of the source,
            preferably in units of electrons per second.
        poisson_noise_map_ : ndarray or None
            The masked 1D array of the poisson noise-map.
        exposure_time_map : ScaledSquarePixelArray
            An array describing the effective exposure time in each image pixel.
        exposure_time_map_ : ndarray or None
            The masked 1D array of the exposure time-map.
        background_sky_map : ScaledSquarePixelArray
            An array describing the background sky.
        background_sky_map_ : ndarray or None
            The masked 1D array of the background sky map.
        sub_grid_size : int
            The size of the sub-grid used for computing the SubGrid (see imaging.mask.SubGrid).
        image_psf_shape : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        convolver_image : imaging.convolution.ConvolverImage
            A convolver which convoles a 1D masked image (using the input mask) with the 2D PSF kernel.
        grids : imaging.mask.ImagingGrids
            Grids of (y,x) Cartesian coordinates which map over the masked 1D observed image's pixels (includes an \
            image-grid, sub-grid, etc.)
        padded_grids : imaging.mask.ImagingGrids
            Grids of padded (y,x) Cartesian coordinates which map over the every observed image's pixel in 1D and a \
            padded regioon to include edge's for accurate PSF convolution (includes an image-grid, sub-grid, etc.)
        border  imaging.mask.ImagingGridsBorders
            The border of the image-grid and sub-grid (see *ImagingGridsBorders* for their use).
        """
        super().__init__(array=image, pixel_scale=image.pixel_scale, noise_map=image.noise_map, psf=image.psf,
                         background_noise_map=image.background_noise_map, poisson_noise_map=image.poisson_noise_map,
                         exposure_time_map=image.exposure_time_map, background_sky_map=image.background_sky_map)

        self.image = image[:,:]
        self.mask = mask
        self.noise_map_ = mask.map_2d_array_to_masked_1d_array(image.noise_map)
        self.background_noise_map_ = mask.map_2d_array_to_masked_1d_array(image.background_noise_map)
        self.poisson_noise_map_ = mask.map_2d_array_to_masked_1d_array(image.poisson_noise_map)
        self.exposure_time_map_ = mask.map_2d_array_to_masked_1d_array(image.exposure_time_map)
        self.background_sky_map_ = mask.map_2d_array_to_masked_1d_array(image.background_sky_map)

        self.sub_grid_size = sub_grid_size

        if image_psf_shape is None:
            image_psf_shape = self.psf.shape

        self.convolver_image = convolution.ConvolverImage(self.mask, mask.blurring_mask_for_psf_shape(image_psf_shape),
                                                          self.psf.resized_scaled_array_from_array(image_psf_shape))

        self.grids = msk.ImagingGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                  sub_grid_size=sub_grid_size,
                                                                                  psf_shape=image_psf_shape)

        self.padded_grids = msk.ImagingGrids.padded_grids_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                        sub_grid_size=sub_grid_size, psf_shape=image_psf_shape)

        self.border = msk.ImageGridBorder.from_mask(mask=mask)

    def __array_finalize__(self, obj):
        super(FittingImage, self).__array_finalize__(obj)
        if isinstance(obj, FittingImage):
            self.noise_map_ = obj.noise_map_
            self.background_noise_map_ = obj.background_noise_map_
            self.poisson_noise_map_ = obj.poisson_noise_map_
            self.exposure_time_map_ = obj.exposure_time_map_
            self.background_sky_map_ = obj.background_sky_map_
            self.mask = obj.mask
            self.convolver_image = obj.convolver_image
            self.grids = obj.grids
            self.border = obj.border


class FittingHyperImage(FittingImage):

    def __new__(cls, image, mask, hyper_model_image, hyper_galaxy_images, hyper_minimum_values, sub_grid_size=2,
                image_psf_shape=None):
        return np.array(mask.map_2d_array_to_masked_1d_array(image)).view(cls)

    def __init__(self, image, mask, hyper_model_image, hyper_galaxy_images, hyper_minimum_values, sub_grid_size=2,
                 image_psf_shape=None):
        """A fitting hyper image is a fitting_image (see *FittingImage) which additionally includes a set of \
        'hyper_data'. This hyper-data is the best-fit model images of the observed data from a previous analysis, \
        and it is used to scale the noise in the image, so as to avoid over-fitting localized regions of the image \
        where the model does not provide a good fit.

        Look at the *FittingImage* docstring for all parameters / attributes not specific to a hyper image.

        Parameters
        ----------
        hyper_model_images : [ndarray]
            List of the masked 1D array best-fit model image's to each observed image in a previous analysis.
        hyper_galaxy_images : [[ndarray]]
            List of the masked 1D array best-fit model image's of every galaxy to each observed image in a previous \
            analysis.

        Attributes
        ----------

        """
        super(FittingHyperImage, self).__init__(image=image, mask=mask, sub_grid_size=sub_grid_size,
                                                image_psf_shape=image_psf_shape)

        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_images = hyper_galaxy_images
        self.hyper_minimum_values = hyper_minimum_values

    def __array_finalize__(self, obj):
        super(FittingImage, self).__array_finalize__(obj)
        if isinstance(obj, FittingHyperImage):
            self.noise_map_ = obj.noise_map_
            self.background_noise_map_ = obj.background_noise_map_
            self.poisson_noise_map_ = obj.poisson_noise_map_
            self.exposure_time_map_ = obj.exposure_time_map_
            self.background_sky_map_ = obj.background_sky_map_
            self.mask = obj.mask
            self.convolver_image = obj.convolver_image
            self.grids = obj.grids
            self.border = obj.border
            self.hyper_model_image = obj.hyper_model_image
            self.hyper_galaxy_images = obj.hyper_galaxy_images
            self.hyper_minimum_values = obj.hyper_minimum_values