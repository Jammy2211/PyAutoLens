from src.imaging import image as im
from src.imaging import mask as msk
from src.imaging import convolution
import numpy as np


class MaskedImage(im.AbstractImage):
    def __new__(cls, image, mask):
        return np.array(mask.masked_1d_array_from_2d_array(image)).view(cls)

    def __init__(self, image, mask):
        """
        An image that has been masked. Only data within the mask is kept. This data is kept in 1D with a corresponding
        array mapping data back to 2D.

        Parameters
        ----------
        image: im.Image
            A 2D image
        mask: msk.Mask
            A mask to be applied to the image
        """
        super().__init__(array=image,
                         effective_exposure_time=mask.masked_1d_array_from_2d_array(image.effective_exposure_time),
                         pixel_scale=image.pixel_scale,
                         psf=image.psf,
                         background_noise=mask.masked_1d_array_from_2d_array(image.background_noise),
                         poisson_noise=mask.masked_1d_array_from_2d_array(image.poisson_noise))

        self.coordinate_grid = mask.coordinate_grid
        self.image = image
        self.image_shape = image.shape
        self.mask = mask
        self.blurring_mask = mask.blurring_mask_for_kernel_shape(image.psf.shape)
        self.convolver_image = convolution.ConvolverImage(self.mask, self.blurring_mask, self.image.psf)
        self.convolver_mapping_matrix = convolution.ConvolverMappingMatrix(self.mask, self.image.psf)

    def __array_finalize__(self, obj):
        super(MaskedImage, self).__array_finalize__(obj)
        if isinstance(obj, MaskedImage):
            self.coordinate_grid = obj.coordinate_grid
            self.image = obj.image
            self.image_shape = obj.image_shape
            self.mask = obj.mask
            self.blurring_mask = obj.blurring_mask
            self.convolver_image = obj.convolver_image
            self.convolver_mapping_matrix = obj.convolver_mapping_matrix

    def map_to_2d(self, data):
        """
        Use mapper to map an input data-set from a *GridData* to its original 2D image.

        Parameters
        -----------
        data : ndarray
            The grid-data which is mapped to its 2D image.
        """
        data_2d = np.zeros(self.image_shape)

        for (i, pixel) in enumerate(self.mask.grid_to_pixel()):
            data_2d[pixel[0], pixel[1]] = data[i]

        return data_2d
