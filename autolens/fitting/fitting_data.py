import numpy as np

from autolens.imaging import convolution
from autolens.imaging import image as im
from autolens.imaging import mask as msk


class FittingImage(im.Image):

    def __new__(cls, image, mask, sub_grid_size=2, image_psf_shape=None):
        return np.array(mask.map_2d_array_to_masked_1d_array(image)).view(cls)

    def __init__(self, image, mask, sub_grid_size=2, image_psf_shape=None):
        """
        The lensing _datas is the collection of datas (images, noise-maps, PSF), a masks, grids, convolvers and other \
        utilities that are used for modeling and fitting an _datas of a strong lens.

        Whilst the _datas datas is initially loaded in 2D, for the lensing _datas the masked-_datas (and noise-maps) \
        are reduced to 1D arrays for faster calculations.

        Parameters
        ----------
        image: im.Image
            The original _datas datas in 2D.
        mask: msk.Mask
            The 2D masks that is applied to the _datas.
        sub_grid_size : int
            The size of the sub-grid used for each lensing SubGrid. E.g. a value of 2 grids each _datas-pixel on a 2x2 \
            sub-grid.
        image_psf_shape : (int, int)
            The shape of the PSF used for convolving model images generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input _datas PSF, giving a faster analysis run-time.
        """
        super().__init__(array=image, pixel_scale=image.pixel_scale,
                         noise_map=mask.map_2d_array_to_masked_1d_array(image.noise_map), psf=image.psf,
                         background_noise_map=image.background_noise_map)

        self.image = image
        self.mask = mask
        self.sub_grid_size = sub_grid_size

        if image_psf_shape is None:
            image_psf_shape = self.image.psf.shape

        self.blurring_mask = mask.blurring_mask_for_psf_shape(image_psf_shape)
        self.convolver_image = convolution.ConvolverImage(self.mask,
                                                          self.blurring_mask, self.image.psf.trim_around_centre(image_psf_shape))

        self.grids = msk.ImagingGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                  sub_grid_size=sub_grid_size,
                                                                                  psf_shape=image_psf_shape)

        self.padded_grids = msk.ImagingGrids.padded_grids_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                        sub_grid_size=sub_grid_size, psf_shape=image_psf_shape)

        self.borders = msk.ImagingGridBorders.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=sub_grid_size)

    def __array_finalize__(self, obj):
        super(FittingImage, self).__array_finalize__(obj)
        if isinstance(obj, FittingImage):
            self.image = obj.image
            self.mask = obj.mask
            self.blurring_mask = obj.blurring_mask
            self.convolver_image = obj.convolver_image
            self.grids = obj.grids
            self.borders = obj.borders


class FittingHyperImage(FittingImage):

    def __new__(cls, image, mask, hyper_model_image, hyper_galaxy_images, hyper_minimum_values, sub_grid_size=2,
                image_psf_shape=None):
        return np.array(mask.map_2d_array_to_masked_1d_array(image)).view(cls)

    def __init__(self, image, mask, hyper_model_image, hyper_galaxy_images, hyper_minimum_values, sub_grid_size=2,
                 image_psf_shape=None):
        """
        The lensing _datas is the collection of datas (images, noise-maps, PSF), a masks, grids, convolvers and other \
        utilities that are used for modeling and fitting an _datas of a strong lens.

        Whilst the _datas datas is initially loaded in 2D, for the lensing _datas the masked-_datas (and noise-maps) \
        are reduced to 1D arrays for faster calculations.

        Parameters
        ----------
        image: im.Image
            The original _datas datas in 2D.
        mask: msk.Mask
            The 2D masks that is applied to the _datas.
        sub_grid_size : int
            The size of the sub-grid used for each lensing SubGrid. E.g. a value of 2 grids each _datas-pixel on a 2x2 \
            sub-grid.
        image_psf_shape : (int, int)
            The shape of the PSF used for convolving model images generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input _datas PSF, giving a faster analysis run-time.
        """

        super(FittingHyperImage, self).__init__(image=image, mask=mask, sub_grid_size=sub_grid_size,
                                                image_psf_shape=image_psf_shape)

        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_images = hyper_galaxy_images
        self.hyper_minimum_values = hyper_minimum_values

    def __array_finalize__(self, obj):
        super(FittingImage, self).__array_finalize__(obj)
        if isinstance(obj, FittingHyperImage):
            self.image = obj.image
            self.mask = obj.mask
            self.blurring_mask = obj.blurring_mask
            self.convolver_image = obj.convolver_image
            self.grids = obj.grids
            self.borders = obj.borders
            self.hyper_model_image = obj.hyper_model_image
            self.hyper_galaxy_images = obj.hyper_galaxy_images
            self.hyper_minimum_values = obj.hyper_minimum_values