from autolens.imaging import image as im
from autolens.imaging import mask as msk
from autolens.imaging import convolution
import numpy as np


class MaskedImage(im.Image):
    def __new__(cls, image, mask, sub_grid_size=2, profile_psf_shape=None, pixelization_psf_shape=None):
        return np.array(mask.map_to_1d(image)).view(cls)

    def __init__(self, image, mask, sub_grid_size=2, profile_psf_shape=None, pixelization_psf_shape=None):
        """
        An image that has been masked. Only data within the mask is kept. This data is kept in 1D with a corresponding
        array mapping data back to 2D.

        Parameters
        ----------
        image: im.Image
            A 2D image
        mask: msk.Mask
            A mask to be applied to the image
        sub_grid_size : int

        """
        super().__init__(array=image, pixel_scale=image.pixel_scale, noise=mask.map_to_1d(image.noise), psf=image.psf)

        self.image = image
        self.image_shape = image.shape
        self.mask = mask
        self.sub_grid_size = sub_grid_size

        if profile_psf_shape is None:
            profile_psf_shape = self.image.psf.shape
        if pixelization_psf_shape is None:
            pixelization_psf_shape = self.image.psf.shape

        self.blurring_mask = mask.blurring_mask_for_kernel_shape(profile_psf_shape)
        self.convolver_image = convolution.ConvolverImage(self.mask,
                                                          self.blurring_mask, self.image.psf.trim(profile_psf_shape))
        self.convolver_mapping_matrix = convolution.ConvolverMappingMatrix(self.mask,
                                                                           self.image.psf.trim(pixelization_psf_shape))
        self.grids = msk.GridCollection.from_mask_sub_grid_size_and_blurring_shape(mask=mask, sub_grid_size=sub_grid_size,
                                                                                   blurring_shape=profile_psf_shape)

        self.borders = msk.BorderCollection.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=sub_grid_size)

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



    def map_to_2d(self, data):
        """Use mapper to map an input data-set from a *GridData* to its original 2D image.
        Parameters
        -----------
        data : ndarray
            The grid-data which is mapped to its 2D image.
        """
        data_2d = np.zeros(self.image_shape)

        for (i, pixel) in enumerate(self.mask.grid_to_pixel()):
            data_2d[pixel[0], pixel[1]] = data[i]

        return data_2d
