from autolens.imaging import image as im
from autolens.imaging import mask as msk
from autolens.imaging import convolution
import numpy as np


class MaskedImage(im.Image):

    def __new__(cls, image, mask, sub_grid_size=2, image_psf_shape=None, pixelization_psf_shape=None, positions=None):
        return np.array(mask.map_2d_array_to_masked_1d_array(image)).view(cls)

    def __init__(self, image, mask, sub_grid_size=2, image_psf_shape=None, pixelization_psf_shape=None,
                 positions=None):
        """
        An masked_image that has been masked. Only data within the mask is kept. This data is kept in 1D with a corresponding
        array mapping_matrix data back to 2D.

        Parameters
        ----------
        image: im.Image
            A 2D masked_image
        mask: msk.Mask
            A mask to be applied to the masked_image
        sub_grid_size : int

        """
        super().__init__(array=image, pixel_scale=image.pixel_scale, noise=mask.map_2d_array_to_masked_1d_array(image.noise), psf=image.psf)

        self.image = image
        self.image_shape = image.shape
        self.mask = mask
        self.sub_grid_size = sub_grid_size

        if image_psf_shape is None:
            image_psf_shape = self.image.psf.shape
        if pixelization_psf_shape is None:
            pixelization_psf_shape = self.image.psf.shape

        self.blurring_mask = mask.blurring_mask_for_psf_shape(image_psf_shape)
        self.convolver_image = convolution.ConvolverImage(self.mask,
                                                          self.blurring_mask, self.image.psf.trim(image_psf_shape))
        self.convolver_mapping_matrix = convolution.ConvolverMappingMatrix(self.mask,
                                                                           self.image.psf.trim(pixelization_psf_shape))
        self.grids = msk.GridCollection.from_mask_sub_grid_size_and_blurring_shape(mask=mask, sub_grid_size=sub_grid_size,
                                                                                   blurring_shape=image_psf_shape)

        self.grid_mappers = msk.GridCollection.grid_mappers_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                            sub_grid_size=sub_grid_size, psf_shape=image_psf_shape)

        self.borders = msk.BorderCollection.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=sub_grid_size)
        self.positions = positions

    def __array_finalize__(self, obj):
        super(MaskedImage, self).__array_finalize__(obj)
        if isinstance(obj, MaskedImage):
            self.image = obj.image
            self.image_shape = obj.image_shape
            self.mask = obj.mask
            self.blurring_mask = obj.blurring_mask
            self.convolver_image = obj.convolver_image
            self.convolver_mapping_matrix = obj.convolver_mapping_matrix
            self.grids = obj.grids
            self.borders = obj.borders
            self.positions = obj.positions